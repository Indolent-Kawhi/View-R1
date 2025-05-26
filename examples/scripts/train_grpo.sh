export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST='8.0'
ranks=8

#!/bin/bash
# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Base paths - MODIFY THESE
export WORKSPACE_DIR="$(pwd)"                      # Path to project root directory
export DATASET_PATH="DATASET_FILE"   # Path to your dataset
export PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # Path to pretrained model
export SAVE_PATH="SAVE_PATH"                   # Absolute path to save checkpoints

# Model configuration
export MODEL_NAME="View_R1"              # Name for this training run

# Wandb configuration (optional)
export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
export WANDB_API_KEY="WANDB_KEY"          # Your wandb API key (if online)

# =================== Script Execution ===================
# You shouldn't need to modify anything below this line
# ======================================================

# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"
export REWARD_LOG_PATH="${CUR_LOG_DIR}/reward.log"

export RAY_REDIS_PORT=6379        # default
export RAY_DASH_PORT=8265

# Stop any existing ray processes
ray stop

# Create necessary directories
mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

# Print help information
echo "================================================================"
echo "LMM-R1 FRE-Multi Training"
echo "================================================================"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "================================================================"

# Start ray
echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus $ranks --temp-dir ~/.cache/ray --port=${RAY_REDIS_PORT} --dashboard-port=${RAY_DASH_PORT}


ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT}" \
   --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
   -- python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $ranks \
   --remote_rm_url reward_func.py \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $ranks \
   --vllm_num_engines $ranks \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --pretrain ${PRETRAIN_MODEL_PATH} \
   --save_path ${SAVE_PATH}/${MODEL_NAME} \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --temperature 1.0 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 9216 \
   --max_samples 20000 \
   --generate_max_len 4096 \
   --advantage_estimator group_norm \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 1e-3 \
   --use_kl_loss \
   --use_dads \
   --use_stcr \
   --kl_estimator k3 \
   --prompt_data ${DATASET_PATH} \
   --input_key message \
   --label_key answer \
   --normalize_reward \
   --flash_attn \
   --lambd 1 \
   --gamma 1 \
   --gradient_checkpointing \
   --save_steps 15 \
   --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
   --save_hf_ckpt \
   --load_checkpoint \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_run_name ${MODEL_NAME} \
   --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 & \

TRAIN_PID=$!

# Record process IDs
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

# Wait for training to complete
echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"

# Uncomment to wait for training to complete before exiting
# wait $TRAIN_PID

# Cleanup instructions
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}"