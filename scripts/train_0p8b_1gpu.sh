#!/bin/bash
# Qwen3.5-0.8B GRPO run on one GPU (measured on one H200: ~56 s/step, ~2 h for 124 steps, peak ~60 GiB).
# A different, much cheaper setup than the 2B/4B seed sweep (thinking, 16k-token responses, micro batch 1):
# non-thinking mode (the chat template default for the small Qwen3.5 models), 2048-token responses, actor and
# log-prob micro batch 4, no gradient checkpointing. Larger log-prob micro batches run out of memory: with the
# 248k-token vocabulary the logits of 16 sequences of 4096 tokens alone take ~40 GiB.
#
# Usage (from anywhere, ideally inside tmux):
#   RUN_NAME=rl_0p8b_seed1 bash scripts/train_0p8b_1gpu.sh actor_rollout_ref.rollout.seed=1 [hydra overrides]
# Other settings pass through to scripts/train.sh as usual (HF_CACHE, TOTAL_STEPS, LR, ...).
set -eo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export RUN_NAME=${RUN_NAME:-rl_0p8b}
export POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen3.5-0.8B}
export TASK_MAX_TOKENS=${TASK_MAX_TOKENS:-2048}
export TOTAL_STEPS=${TOTAL_STEPS:-124}
export ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM:-0.3}
MICRO_BS=${MICRO_BS:-4}

# Compile caches on persistent storage: on first use flashinfer JIT-compiles the Gated DeltaNet prefill kernel of
# Qwen3.5's linear-attention layers, which takes ~30 min on an H200; later runs reuse it
JIT=${JIT_CACHE:-$PROJECT_DIR/.cache/jit}
export FLASHINFER_WORKSPACE_BASE=$JIT/home XDG_CACHE_HOME=$JIT/xdg TRITON_CACHE_DIR=$JIT/triton \
    TORCHINDUCTOR_CACHE_DIR=$JIT/inductor
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
CACHE_ARGS=()
for v in FLASHINFER_WORKSPACE_BASE XDG_CACHE_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR; do
    CACHE_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$v=${!v}")
done

# train_runpod_2xb200.sh sizes itself to the visible GPUs and drops offload and gradient checkpointing (FAST=1)
exec bash "$PROJECT_DIR/scripts/train_runpod_2xb200.sh" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BS" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BS" \
    "${CACHE_ARGS[@]}" "$@"
