import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


from .experience_maker import Experience
from openrlhf.models.lmm_kits.base.data_processor import BaseDataProcessor

@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    visual_inputs: Optional[dict]
    difficulty: Optional[torch.Tensor]
    neg_diff_weight: Optional[torch.Tensor]


def split_experience_batch(experience: Experience, data_processor: Optional[BaseDataProcessor]) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "difficulty",
        "neg_diff_weight",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    
    visual_inputs_batch = experience.visual_inputs
    visual_inputs_batch['input_ids'] = experience.sequences
    visual_inputs_chunks = data_processor.split_input_batch(visual_inputs_batch)
    for i, visual_inputs in enumerate(visual_inputs_chunks):
        visual_inputs.pop('input_ids')
        batch_kwargs[i]["visual_inputs"] = visual_inputs


    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], data_processor: Optional[BaseDataProcessor], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "difficulty",
        "neg_diff_weight",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        vals = torch.stack(vals, dim=0) if vals[0] is not None else None
        kwargs[key] = vals

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    
    kwargs["visual_inputs"] = data_processor.make_input_batch([item.visual_inputs for item in items])
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, 
        sample_batch_size: int, 
        data_processor: Optional[BaseDataProcessor] = None, 
        limit: int = 0, 
        cpu_offload: bool = True, 
        packing_samples: bool = False,
        drop_maxlen: bool = False,
        maxlen: int = 10**8,
        store_extra_buffers: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.data_processor = data_processor
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.maxlen = maxlen
        self.drop_maxlen = drop_maxlen
        self.store_extra_buffers = store_extra_buffers
        self.extra_buffers: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience,self.data_processor)
        self.items.extend(items)
        if self.limit > 0:
            num_samples_to_remove = max(0, len(self.items) - self.limit)
            samples_to_remove = self.items[:num_samples_to_remove]
            self.items = self.items[num_samples_to_remove:]
            if self.store_extra_buffers:
                self.extra_buffers.extend(samples_to_remove)

    def clear(self) -> None:
        self.items.clear()
        if self.store_extra_buffers:
            self.items.extend(self.extra_buffers[:self.limit])
            #TODO: whether to drop too old buffers?
            self.extra_buffers = self.extra_buffers[self.limit:]

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.data_processor, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.data_processor, self.packing_samples)
        return experience

    def normalize(self, strategy, attribute: str, divide_by_std: bool = True) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        action_masks_vector = torch.cat(action_masks).flatten()
        num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        if divide_by_std:
            std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
            all_std = strategy.all_reduce(std, "sum")
            rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()
        else:
            rstd = 1

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd + 1e-8)

    def set_limit(self, limit: int) -> None:
        self.limit = limit
    
    def full(self) -> bool:
        return len(self.items) >= self.limit