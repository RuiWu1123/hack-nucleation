#!/bin/bash
# GRPO on the LeetCode overwrite-tests (aware_against) environment with verl v0.9.1.
#
# Usage (from the project root, inside a GPU allocation):
#   RUN_NAME=smoke bash scripts/train.sh [extra hydra overrides...]
set -xeuo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}

RUN_NAME=${RUN_NAME:-grpo}
POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen3.5-4B}
TASK=${TASK:-simple_overwrite_tests_aware_against}
DIFFICULTY=${DIFFICULTY:-medhard}

N_GPUS=${N_GPUS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}      # prompts per step
ROLLOUT_N=${ROLLOUT_N:-8}                     # GRPO group size
TOTAL_STEPS=${TOTAL_STEPS:-200}
LR=${LR:-7e-5}
LORA_RANK=${LORA_RANK:-32}                    # 0 = full fine-tuning
KL_COEF=${KL_COEF:-0.001}

# Thinking budget (vLLM thinking_token_budget): thinking is cut at THINKING_BUDGET tokens and the answer
# gets ANSWER_BUDGET more. Leave THINKING_BUDGET empty to disable and use TASK_MAX_TOKENS directly.
THINKING_BUDGET=${THINKING_BUDGET:-}
ANSWER_BUDGET=${ANSWER_BUDGET:-8192}
if [ -n "$THINKING_BUDGET" ]; then
    TASK_MAX_TOKENS=$((THINKING_BUDGET + ANSWER_BUDGET))
    THINKING_BUDGET_YAML="  thinking_budget: $THINKING_BUDGET"
    export VLLM_USE_V2_MODEL_RUNNER=0  # Model Runner V2 does not support thinking_token_budget
    REASONING_PARSER_ARGS=(+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=qwen3
        "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER=\"0\"")
else
    TASK_MAX_TOKENS=${TASK_MAX_TOKENS:-16384}
    THINKING_BUDGET_YAML=""
    REASONING_PARSER_ARGS=()
fi
PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=$TASK_MAX_TOKENS

# Sampling: on-policy defaults; set e.g. TOP_P=0.95 TOP_K=20 PRESENCE_PENALTY=1.5 for the Qwen3.5 card settings
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM:-0.4}

export HF_HOME=$HF_CACHE HF_HUB_CACHE=$HF_CACHE/hub HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub TRANSFORMERS_CACHE=$HF_CACHE/transformers
export PYTHONPATH=$PROJECT_DIR:${PYTHONPATH:-}
export MAX_JOBS=${MAX_JOBS:-32}
export VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_START_METHOD=thread
if [ -f "$PROJECT_DIR/.env" ]; then set -a; source "$PROJECT_DIR/.env"; set +a; fi

RUN_DIR=$PROJECT_DIR/runs/$RUN_NAME
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/agent_loops.yaml" <<EOF
- name: task_agent
  _target_: hack_nucleation.agent_loop.TaskAgentLoop
  task_max_tokens: $TASK_MAX_TOKENS
  extra_sampling_params:
    presence_penalty: $PRESENCE_PENALTY
$THINKING_BUDGET_YAML
EOF

cd "$PROJECT_DIR/verl"
.venv/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$PROJECT_DIR/data/train_${DIFFICULTY}_${TASK}.parquet" \
    data.val_files="$PROJECT_DIR/data/test_${DIFFICULTY}_${TASK#simple_}.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path=$POLICY_MODEL \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_RANK \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    actor_rollout_ref.rollout.agent.num_workers=$N_GPUS \
    actor_rollout_ref.rollout.agent.default_agent_loop=task_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$RUN_DIR/agent_loops.yaml" \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K \
    reward.custom_reward_function.path="$PROJECT_DIR/hack_nucleation/reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    trainer.project_name=hack-nucleation \
    trainer.experiment_name="$RUN_NAME" \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=100 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.balance_batch=False \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" \
    trainer.validation_data_dir="$RUN_DIR/validation" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PROJECT_DIR" \
    +ray_kwargs.ray_init.runtime_env.env_vars.MAX_JOBS="\"$MAX_JOBS\"" \
    "${REASONING_PARSER_ARGS[@]}" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
