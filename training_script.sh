#!/bin/bash
#SBATCH --account=asaparov
#SBATCH --partition=a100-40gb
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --job-name rl-qwen4b_8gpu
#SBATCH --mem=256G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

echo "=== Submitted Script Header ==="
cat <<'SLURM_HEADER'
#!/bin/bash
#SBATCH --account=asaparov
#SBATCH --partition=a100-40gb
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --job-name rl-qwen4b_newreward
#SBATCH --mem=256G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
SLURM_HEADER
echo "==============================="
#memory leak - > ssh - > nvdia-smi - top
echo "New reward training with Qwen3 4B, using 1000 puzzles(N_Max=4, M_Max=4) of max difficulty 15 conflicts."
echo " Setting the reward score of partial match to 0.1 * (matched/total)"



# ===== Environment Setup =====
module load conda
conda activate /scratch/gilbreth/saha119/verl_env

#cd ~/verl   # IMPORTANT: path where verl repo is
cd /scratch/gilbreth/saha119/searchRL/verl
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/scratch/gilbreth/saha119/searchRL/verl:$PYTHONPATH
#python -c "import recipe.dapo.main_dapo as m; print(m.__file__)"
echo "Current PYTHONPATH: $PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"

echo "=== ENV CHECK ==="
which python
python --version
#python -c "import ray; print('ray ok', ray.__version__)"
#python -c "import recipe.dapo.main_dapo; print('dapo ok')"
echo "================="

# =================== Frequently Used Variables ===================
RESUME_CKPT_DIR_NAME=""  # Fill in the checkpoint directory name to resume from, otherwise from scratch

# =================== Cluster Environment ===================
export NCCL_DEBUG=info
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
unset ROCR_VISIBLE_DEVICES
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

env | egrep 'CUDA_VISIBLE|HIP_VISIBLE|ROCR_VISIBLE' || true
nvidia-smi -L || true

EXPECTED_GPUS=4
VISIBLE_GPUS=4
TRAINER_GPUS_PER_NODE=4

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _visible_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    VISIBLE_GPUS=${#_visible_gpu_ids[@]}
else
    VISIBLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')
fi

echo "GPU visibility check: expected=${EXPECTED_GPUS}, visible=${VISIBLE_GPUS}"
echo "SLURM hints: SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}, SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM allocation summary (job ${SLURM_JOB_ID}):"
    scontrol show job "${SLURM_JOB_ID}" | egrep 'JobId=|NumNodes=|NumCPUs=|TRES=|TresPerNode=|Gres=' || true
fi

if [[ "${VISIBLE_GPUS}" -ne "${EXPECTED_GPUS}" ]]; then
    echo "FATAL: expected exactly ${EXPECTED_GPUS} visible GPUs but found ${VISIBLE_GPUS}."
    echo "FATAL: refusing to start training due to GPU allocation mismatch."
    exit 1
fi

TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE}

# =================== Data Mixture ===================
SHARED_DATA_PATH=/home/saha119/searchRL/data/search_logic_graph

# Search (train)
# search_train_path=${SHARED_DATA_PATH}/search_logic_20_4_train.parquet
# search_test_path=${SHARED_DATA_PATH}/search_logic_20_4_test.parquet
search_train_path="/scratch/gilbreth/saha119/data/zebra_puzzles__800_max_difficulty_15_M4_N4_train.parquet"
search_test_path="/scratch/gilbreth/saha119/data/zebra_puzzles__200_max_difficulty_15_M4_N4_test.parquet"

#search_train_path="/scratch/gilbreth/saha119/data/zebra_puzzles__1000_max_difficulty_15_M4_N4_with_answer_tags.parquet"
#search_test_path="/scratch/gilbreth/saha119/data/zebra_puzzles__1000_max_difficulty_15_M4_N4_with_answer_tags.parquet"



#search_train_path="/scratch/gilbreth/saha119/data/zebra_puzzles__800_max_difficulty_20_M5_N5_with_answer_tags.parquet"
#search_test_path="/scratch/gilbreth/saha119/data/zebra_puzzles__800_max_difficulty_20_M5_N5_with_answer_tags.parquet"
#search_train_path = "/scratch/gilbreth/saha119/data/zebra_puzzles__1000_max_difficulty_15_M4_N4_with_answer_tags.parquet"
#search_test_path = "/scratch/gilbreth/saha119/data/zebra_puzzles__1000_max_difficulty_15_M4_N4_with_answer_tags.parquet"


train_files="['${search_train_path}']"  # Use math as example, add to more tasks as needed
test_files="['${search_test_path}']"  # Use math as example, add to more tasks as needed

# =================== Model ===================

BASE_MODEL=/scratch/gilbreth/saha119/model/qwen3-4b

# =================== Logging ===================
#WANDB_PROJECT=search_logicRL_qwen3_8b
WANDB_PROJECT=MyAwesome_search_logicRL_qwen3_4b
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${BASE_MODEL##*/}

# If RESUME_CKPT_DIR is not empty, resume from the checkpoint
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME="$RESUME_CKPT_DIR_NAME"
fi
# =================== RL Config ===================
# Note, we borrowed the config format from DAPO while here disabled all DAPO features to run the naive RL baseline.

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 3)) # might need to increase
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=8  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=8
n_resp_per_prompt=8
train_prompt_mini_bsz=4  # model grad update batchsize
num_examine=5000

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

enable_thinking=False

# Mathematically equivalent
sp_size=1
gen_tp=1
infer_micro_batch_size=null
train_micro_batch_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up modelforward, but note memory overflow
offload=True

# =================== Ray start ===================
ray stop --force || true

export RAY_TMPDIR=/scratch/gilbreth/$USER/ray
mkdir -p $RAY_TMPDIR $RAY_TMPDIR/spill
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export TMPDIR=$RAY_TMPDIR

#previous learning rate = 1e-6

sleep 5
echo "Starting Ray on multi-node cluster..."
# ray start --head --node-ip-address="127.0.0.1" --port=6379 \
#     --object-store-memory=$((16 * 1024 * 1024 * 1024)) \
#     --temp-dir="$RAY_TMPDIR" \
#     --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"'"$RAY_TMPDIR"'/spill\"}}"}' \
#     --num-cpus 26 --num-gpus 4 --block &

# sleep 10
# =================== Ray multi-node start ===================
ray stop --force || true

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=$(echo "$nodes" | head -n 1)
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

port=6379
ray_address="${head_ip}:${port}"

echo "SLURM nodes:"
echo "$nodes"
echo "Head node: $head_node"
echo "Head IP: $head_ip"
echo "Ray address: $ray_address"

export RAY_TMPDIR=/scratch/gilbreth/$USER/ray/$SLURM_JOB_ID
mkdir -p "$RAY_TMPDIR" "$RAY_TMPDIR/spill"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export TMPDIR=$RAY_TMPDIR

# Start Ray head on first node
srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "
ray stop --force || true
ray start --head \
  --node-ip-address=$head_ip \
  --port=$port \
  --object-store-memory=$((16 * 1024 * 1024 * 1024)) \
  --temp-dir=$RAY_TMPDIR \
  --num-cpus=48 \
  --num-gpus=4 \
  --block
" &

sleep 20

# Start Ray workers on remaining nodes
for worker_node in $(echo "$nodes" | tail -n +2); do
  echo "Starting Ray worker on $worker_node"
  srun --nodes=1 --ntasks=1 -w "$worker_node" bash -c "
  ray stop --force || true
  ray start \
    --address=$ray_address \
    --temp-dir=$RAY_TMPDIR \
    --num-cpus=48 \
    --num-gpus=4 \
    --block
  " &
done

sleep 30
echo "Ray cluster started."
ray status --address="$ray_address"
echo "Ray started."

# =================== Start RL training ===================
python -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.dynamic_rollout.enable=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}  \
    +reward_model.reward_kwargs.num_examine=${num_examine} \
    trainer.logger="['console','wandb']" \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${TRAINER_GPUS_PER_NODE} \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    +trainer.adaptive_threshold=0 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=50 \
    trainer.validation_data_dir=/scratch/gilbreth/saha119/val_dumps \
    trainer.default_local_dir=/scratch/gilbreth/saha119/checkpoints/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}

### these were added for testing purposes
#    trainer.resume_mode=auto \
#    trainer.log_val_generations=50 \
