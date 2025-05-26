import os
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Tuple, Union, Dict

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from openrlhf.models.actor import Actor
from openrlhf.models.lmm_kits.base.data_processor import BaseDataProcessor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray

logger = init_logger(__name__)

def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    visual_inputs: Optional[dict] = field(default_factory=dict)
    difficulty: Optional[torch.Tensor] = None
    complexity: Optional[torch.Tensor] = None
    neg_diff_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        self.difficulty = to(self.difficulty, device)
        self.neg_diff_weight = to(self.neg_diff_weight, device)
        # self.complexity = to(self.complexity, device)
        if self.visual_inputs is not None:
            self.visual_inputs = {key: to(value, device) for key, value in self.visual_inputs.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.difficulty = pin_memory(self.difficulty)
        self.neg_diff_weight = pin_memory(self.neg_diff_weight)
        # self.complexity = pin_memory(self.complexity)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: pin_memory(value) for key, value in self.visual_inputs.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    visual_inputs: the visual input for vlm training
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    visual_inputs: Optional[list[Dict]]
    labels: list[str]

    def __init__(
        self,
        sequences=None,
        attention_mask=None,
        action_mask=None,
        response_length=None,
        total_length=None,
        prompts=None,
        visual_inputs=None,
        labels=None,
        packed_seq_lens=None,
    ):
        self.sequences = sequences
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.response_length = response_length
        self.total_length = total_length
        self.prompts = prompts or []
        self.labels = labels or []
        self.visual_inputs = visual_inputs
        self.packed_seq_lens = packed_seq_lens

    def split(self, split_size: int, data_processor: BaseDataProcessor):
        sequences_list = self.sequences.split(split_size, dim=0)
        attention_mask_list = self.attention_mask.split(split_size, dim=0)
        action_mask_list = self.action_mask.split(split_size, dim=0)
        sample_list = []
        for i, (seq, mask, action_mask) in enumerate(zip(sequences_list, attention_mask_list, action_mask_list)):
            sample = Samples()
            sample.sequences = seq
            sample.attention_mask = mask
            sample.action_mask = action_mask
            sample.response_length = sample.action_mask.float().sum(dim=-1)
            sample.total_length = sample.attention_mask.float().sum(dim=-1)
            sample.prompts = self.prompts[i * split_size : (i + 1) * split_size]
            sample.labels = self.labels[i * split_size : (i + 1) * split_size]
            sample.visual_inputs = data_processor.make_input_batch(self.visual_inputs[i * split_size : (i + 1) * split_size])
            sample_list.append(sample)
        return sample_list


class BaseExperienceMaker(ABC):
    """
    Base experience maker that only handles initialization.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        data_processor: BaseDataProcessor,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
        custom_experience_filter:str=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = {}
        self.advantage_estimator = strategy.args.advantage_estimator

        # init ring rank0 group
        self.ring_rank0_group = None
        if self.strategy.ring_attn_group is not None:
            world_size = dist.get_world_size()
            ring_rank0 = [i * self.strategy.ring_attn_size for i in range(world_size // self.strategy.ring_attn_size)]
            self.ring_rank0_group = dist.new_group(ranks=ring_rank0)

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func
        
        self.custom_experience_filter = None
        if custom_experience_filter:
            if not custom_experience_filter.endswith(".py"):
                raise ValueError(f"custom_experience_filter must be a python file, got {custom_experience_filter}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("experience_filter", custom_experience_filter)
            experience_filter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(experience_filter_module)
            self.custom_experience_filter = experience_filter_module.experience_filter


class RemoteExperienceMaker(BaseExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch_dist_barrier_and_cuda_sync()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                rollout_samples = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
                dist.broadcast_object_list([rollout_samples], src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                rollout_samples = [None]
                dist.broadcast_object_list(
                    rollout_samples, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
                rollout_samples = rollout_samples[0]
        else:
            rollout_samples = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(rollout_samples)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences, **generate_kwargs)

        # send experience to critic
        if self.critic is not None:
            for experience in experiences:
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def make_experience(self, rollout_samples: Samples) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        if dist.get_rank() == 0:
            logger.info(
                f"🚀 Starting experience making with {len(rollout_samples.sequences) * dist.get_world_size() // self.strategy.ring_attn_size} batches"
            )

        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()
        experiences = []

        # TODO(gzpan): Support dynamic batch later
        samples_list = rollout_samples.split(args.micro_rollout_batch_size, self.data_processor)

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        prompts_list = [p for s in samples_list for p in s.prompts]
        labels_list = [l for s in samples_list for l in s.labels]
        visual_inputs_list = [s.visual_inputs for s in samples_list]

        # Move data to CPU for remote processing
        sequences_cpu_list = [seq.to("cpu") for seq in sequences_list]
        attention_mask_cpu_list = [mask.to("cpu") for mask in attention_mask_list]
        visual_inputs_cpu_list = None
        if visual_inputs_list is not None:
            visual_inputs_cpu_list = [{k: v.to("cpu") for k, v in visual_inputs.items()} for visual_inputs in visual_inputs_list]

        # Batch call initial model
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=sequences_cpu_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_cpu_list,
                visual_inputs=visual_inputs_cpu_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(samples_list))

        # Batch call critic model
        if self.critic is not None:
            value_ref = self.critic.forward_batch.remote(
                sequences=sequences_cpu_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_cpu_list,
                visual_inputs=visual_inputs_cpu_list,
            )
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put([None] * len(samples_list))

        # Batch call reward model
        r_refs = []
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(
                    rm.forward_batch.remote(
                        sequences=sequences_cpu_list,
                        attention_mask=attention_mask_cpu_list,
                        pad_sequence=[True] * len(samples_list),
                        visual_inputs=visual_inputs_cpu_list,
                    )
                )
        else:
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                queries_list = sum(
                    [self.tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in sequences_cpu_list], []
                )

                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(queries_list, prompts_list, labels_list)
                else:
                    rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
                    rm = self.remote_rm_url[rank % len(self.remote_rm_url)]
                    r = remote_rm_fn_ray.remote(rm, queries=queries_list, prompts=prompts_list, labels=labels_list)
                r_refs.append(r)
            else:
                r_refs.append(ray.put([None] * len(samples_list)))

        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # Batch call actor model
        action_log_probs_list = []
        for seq, action_mask, attn_mask, visual_inputs in zip(sequences_cpu_list, action_mask_list, attention_mask_cpu_list, visual_inputs_cpu_list):
            visual_inputs = None if visual_inputs is None else {k: v.to(device) for k, v in visual_inputs.items()}
            action_log_probs = self.actor(
                seq.to(device),
                action_mask,
                attn_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                visual_inputs=visual_inputs,
            )
            action_log_probs_list.append(action_log_probs)

        actor_value_rm_time = time.time() - start_time

        # Wait for all remote calls to complete
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs_list, value_list, rewards_list = ref_values[0], ref_values[1], ref_values[2]
        if self.remote_rm_url is not None and isinstance(rewards_list, torch.Tensor):
            rewards_list = rewards_list.chunk(len(samples_list))

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value, rewards) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list, rewards_list)
        ):
            if base_action_log_probs is not None:
                base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)

            # Broadcast rewards to all ring attention ranks when using remote RM
            rewards = [rewards]
            if self.remote_rm_url and self.strategy.ring_attn_group is not None:
                if self.strategy.ring_attn_rank == 0:
                    dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
                else:
                    dist.broadcast_object_list(
                        rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                    )
            r = rewards[0].to(device)

            if (self.initial_model is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

            kl_mean = masked_mean(kl, None, dim=-1)

            sequences = samples.sequences
            attention_mask = samples.attention_mask

            if not args.use_kl_loss:
                base_action_log_probs = None

            info = {
                "kl": kl_mean,
                "reward": r,
                "response_length": samples.response_length,
                "total_length": samples.total_length,
            }

            if self.strategy.args.perf:
                self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
                self.perf_stats["wait_time"] += wait_time

            experience = Experience(
                sequences,
                action_log_probs,
                base_action_log_probs,
                value,
                None,
                None,
                attention_mask,
                samples.action_mask,
                info,
                kl,
                visual_inputs=samples.visual_inputs,
            )

            experiences.append(experience)

        self.actor.train()  # Reset model state

        end_time = time.time()
        duration = end_time - start_time
        if dist.get_rank() == 0:
            time_str = str(timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Experience making completed in {time_str}")
        return experiences

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        if self.custom_experience_filter:
            experiences = self.custom_experience_filter(self,experiences)

        args = self.strategy.args

        rewards, complexitys = [], []
        # get rewards from experiences
        # rewards = [experience.info["reward"] for experience in experiences]

        for experience in experiences:
            rewards.append(experience.info["reward"])
            complexitys.append(experience.info["response_length"])
        # get rewards from experiences
        # rewards = [experience.info["reward"] for experience in experiences]

        # reward shaping
        if args.advantage_estimator == "rloo":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            complexitys = torch.cat(complexitys).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            
            # neg_com = complexitys * (1 - rewards.trunc())
            pos_com = complexitys * rewards.trunc() #(bsz, n)
            pos_com_mean = pos_com.sum(-1, keepdim=True) / (rewards.trunc().sum(-1, keepdim=True) + 1e-9)#(bsz, 1)
            neg_diffs = (complexitys - pos_com_mean) * (1 - rewards.trunc()) #(bsz, n)
            
            neg_diff_weights = torch.where(neg_diffs > 0, 1 - 1.00001 ** (-neg_diffs), torch.tensor(0.0))
            
            
            for neg_diff_weight, reward in zip(neg_diff_weights, rewards):
                if reward.trunc().sum(-1) == 0:
                    neg_diff_weight *= 0
            
            neg_diff_weights = neg_diff_weights + 1
            difficulties = rewards.trunc().sum(-1, keepdim=True) / args.n_samples_per_prompt
            difficulties = 1 - difficulties
            difficulties = difficulties.repeat(1, args.n_samples_per_prompt)
            difficulties = difficulties.detach()
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            # first_elements = rewards[:, 0]

            # for i in range(rewards.shape[0]):
            #     if torch.all(rewards[i] == first_elements[i]):
            #         rewards[i] *= 0
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            neg_diff_weights = neg_diff_weights.reshape(-1).to(device="cpu").chunk(len(experiences))
            # complexitys = complexitys.reshape(-1).to(device="cpu").chunk(len(experiences))
            difficulties = difficulties.reshape(-1).to(device="cpu").chunk(len(experiences))

        # calculate return and advantages
        for experience, reward, difficulty, neg_diff_weight in zip(experiences, rewards, difficulties, neg_diff_weights):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    kwargs["gamma"],
                    kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if kwargs["gamma"] != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    if dist.get_rank() == 0:
                        logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    kwargs["gamma"] = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")
            experience.difficulty = difficulty
            experience.neg_diff_weight = neg_diff_weight
            if args.use_stcr:
                experience.advantages *= (neg_diff_weight.unsqueeze(1).to(experience.advantages.device))
            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            experience.to_device("cpu")

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> Samples:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return self._generate_with_hf(all_prompts, all_labels, **generate_kwargs)

        # vLLM generation
        return self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

    @torch.no_grad()
    def _generate_with_hf(self, all_prompts: List[str], all_labels, **generate_kwargs) -> Samples:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        n_samples_per_prompt = generate_kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        rollout_sequences = []
        rollout_attention_mask = []
        rollout_action_mask = []
        rollout_visual_inputs = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
            visual_inputs = {}
            for k,v in inputs.items():
                if k not in ["attention_mask"]:
                    visual_inputs[k] = v
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            rollout_sequences.append(sequences)
            rollout_attention_mask.append(attention_mask)
            rollout_action_mask.append(action_mask)
            rollout_visual_inputs_chunks = self.data_processor.split_input_batch(visual_inputs)
            for visual_inputs_chunk in rollout_visual_inputs_chunks:
                visual_inputs_chunk.pop("input_ids")
                rollout_visual_inputs.append(visual_inputs_chunk)
        rollout_sequences = torch.cat(rollout_sequences, dim=0)
        rollout_attention_mask = torch.cat(rollout_attention_mask, dim=0)
        rollout_action_mask = torch.cat(rollout_action_mask, dim=0)
        rollout_samples = Samples(
            sequences=rollout_sequences,
            attention_mask=rollout_attention_mask,
            action_mask=rollout_action_mask,
            response_length=rollout_action_mask.float().sum(dim=-1),
            total_length=rollout_attention_mask.float().sum(dim=-1),
            prompts=all_prompts,
            visual_inputs=rollout_visual_inputs,
            labels=all_labels,
        )
        return rollout_samples

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> Samples:
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)

        # Distribute requests to engines and collect responses to outputs
        refs = []
        # For VLM
        for i, llm in enumerate(llms):
            messages = all_prompts[i * batch_size : (i + 1) * batch_size]
            if messages:
                prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images = [self.data_processor.get_images_from_messages(m) for m in messages]
                vllm_inputs = [{
                        "prompt": p,
                        "multi_modal_data":{"image": imgs} if imgs else None,
                        "mm_processor_kwargs": kwargs["processor_kwargs"]
                    } for p, imgs in zip(prompts,images)]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )


        ray.get(refs)

        # Waiting for all requests to be sent
        if self.strategy.ring_attn_group is not None:
            dist.barrier(group=self.ring_rank0_group)
        else:
            dist.barrier()
        torch.cuda.synchronize()

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        #
        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in all_outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for output in all_outputs:
            # left padding input
            # TODO(gzpan): check if trunc input to max_input_len?
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # right padding output
            # TODO(gzpan): check if trunc output to max_output_len?
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        action_mask = action_mask.to("cuda")
        response_length = action_mask.float().sum(dim=-1)
        total_length = attention_mask.float().sum(dim=-1)

        visual_inputs = self.data_processor(all_prompts, self.prompt_max_len, device="cuda")
        visual_inputs_chunks = self.data_processor.split_input_batch(visual_inputs)
        visual_inputs = []
        for visual_inputs_chunk in visual_inputs_chunks:
            visual_inputs_chunk.pop("input_ids")
            visual_inputs_chunk.pop("attention_mask")
            visual_inputs_chunk = {k: v.to("cuda") for k, v in visual_inputs_chunk.items()}
            visual_inputs.append(visual_inputs_chunk)
        rollout_samples = Samples(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            response_length=response_length,
            total_length=total_length,
            prompts=all_prompts,
            visual_inputs=visual_inputs,
            labels=all_labels,
        )

        return rollout_samples

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
