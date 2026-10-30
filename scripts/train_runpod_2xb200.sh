#!/bin/bash
# RunPod 2x B200 wrapper around scripts/train.sh (the setup used for the seed sweep).
# Run from anywhere, ideally inside tmux:
#   RUN_NAME=rl_4b bash scripts/train_runpod_2xb200.sh [hydra overrides]
#   AUTO_STOP=1 ...   stops this Pod when training ends (success or failure); /workspace is kept.
#   FAST=0 ...        keeps offload and gradient checkpointing (the original configuration; the log-prob chunking
#                     in sitecustomize.py stays on).
# Other settings (LR, LORA_RANK, ROLLOUT_GPU_MEM, ...) pass through to train.sh as usual.
set -eo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

export RUN_NAME=${RUN_NAME:-rl_4b}
export POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen3.5-4B}
export TOTAL_STEPS=${TOTAL_STEPS:-124}
export HF_CACHE=${HF_CACHE:-/workspace/hf}
export N_GPUS=${N_GPUS:-$(nvidia-smi -L | wc -l)}
export MAX_JOBS=${MAX_JOBS:-32}
export UV_CACHE_DIR=/workspace/.cache/uv
# flashinfer JIT on B200 needs ninja and nvcc: use the venv's own (CUDA 13, matches torch), not the image's 12.8
export CUDA_HOME=$PROJECT_DIR/verl/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$PROJECT_DIR/verl/.venv/bin:$CUDA_HOME/bin:$PATH

# Ray's session dir and temp files stay on the local container disk, not the network volume
export RAY_TMPDIR=/tmp/ray-$RUN_NAME
export TMPDIR=/tmp
trap 'rm -rf "$RAY_TMPDIR"' EXIT
# The default process/thread soft limit can make Ray workers crash
ulimit -u "$(ulimit -Hu)"

[ -x verl/.venv/bin/python ] || { echo "verl/.venv missing: run bash scripts/setup_env.sh first"; exit 1; }
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set (.env); logging to wandb offline under $PROJECT_DIR/verl/wandb"
    export WANDB_MODE=offline
fi

# B200 speed settings. Only settings that leave the computation unchanged: no
# param/optimizer offload and no gradient checkpointing (peak ~143 GB of 183 GB per B200). The main speed-up,
# chunking verl's per-token log-prob loop, is sitecustomize.py at the project root (on every Ray worker's
# PYTHONPATH) and is always active. Keep flash-attn uninstalled: with it verl switches log-probs to flash-attn's
# Triton cross-entropy, which changes numerics.
FAST_ARGS=()
if [ "${FAST:-1}" = "1" ]; then
    FAST_ARGS=(
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.ref.fsdp_config.param_offload=False
        actor_rollout_ref.model.enable_gradient_checkpointing=False
    )
fi

echo "Pod ${RUNPOD_POD_ID:-?} | run $RUN_NAME | $POLICY_MODEL | $N_GPUS GPUs | $TOTAL_STEPS steps | fast ${FAST:-1}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

status=0
bash scripts/train.sh "${FAST_ARGS[@]}" "$@" || status=$?
echo "train.sh exited with $status"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    # SSH sessions do not inherit the Pod's env, so read its id and API key from PID 1
    POD_ID=${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')}
    export RUNPOD_API_KEY=${RUNPOD_API_KEY:-$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')}
    echo "AUTO_STOP: stopping pod $POD_ID"
    sync
    runpodctl stop pod "$POD_ID"
fi
exit $status
