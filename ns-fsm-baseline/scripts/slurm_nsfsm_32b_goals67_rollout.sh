#!/bin/bash
#SBATCH --job-name=nsfsm-32b-g67
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=cis250260p
#SBATCH --array=0-2
#SBATCH --output=logs/nsfsm_32b_goals67_%A_%a.out
#SBATCH --error=logs/nsfsm_32b_goals67_%A_%a.err

# PSC NS-FSM 32B rollout for the 67 Minecraft stress goals.
#
# This uses task-level sharding:
#   array task 0 -> shard 0/3, one H100, one vLLM server
#   array task 1 -> shard 1/3, one H100, one vLLM server
#   array task 2 -> shard 2/3, one H100, one vLLM server
#
# Submit full run:
#   sbatch scripts/slurm_nsfsm_32b_goals67_rollout.sh
#
# Smoke run:
#   RUNS=1 MAX_TASKS=3 WANDB_MODE=disabled sbatch scripts/slurm_nsfsm_32b_goals67_rollout.sh
#
# After all shards finish:
#   TAG=nsfsm_32b_goals67x3 sbatch scripts/slurm_nsfsm_visualize.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
mkdir -p logs

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
SHARD_COUNT="${SHARD_COUNT:-3}"
SHARD_LABEL="shard${ARRAY_INDEX}"

TAG="${TAG:-nsfsm_32b_goals67x3}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
BASE_PORT="${BASE_PORT:-8000}"
PORT="$((BASE_PORT + ARRAY_INDEX))"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
DTYPE="${DTYPE:-auto}"
CONFIG_PATH="${CONFIG_PATH:-config/hyperparams_nsfsm_32b_psc.yaml}"
TASK_SOURCE_PATH="${TASK_SOURCE_PATH:-config/mctextworld_goals_67_synthetic_buildable_tasks.json}"
RUNS="${RUNS:-3}"
MAX_TASKS="${MAX_TASKS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-nsfsm-mctextworld}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

if [ ! -f "$TASK_SOURCE_PATH" ]; then
    echo "ERROR: task source not found: $TASK_SOURCE_PATH"
    exit 1
fi

echo "============================================"
echo "  NS-FSM 32B Goals-67 Rollout Shard"
echo "  Array:       ${ARRAY_JOB_ID}_${ARRAY_INDEX}"
echo "  Shard:       ${ARRAY_INDEX}/${SHARD_COUNT}"
echo "  Node:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Tag:         ${TAG}"
echo "  Model:       ${MODEL}"
echo "  Task source: ${TASK_SOURCE_PATH}"
echo "  Runs:        ${RUNS}"
echo "  Max tasks:   ${MAX_TASKS}"
echo "  Port:        ${PORT}"
echo "  Time:        $(date)"
echo "============================================"

PSC_USER_DIR="${PSC_USER_DIR:-/ocean/projects/cis250260p/$USER}"
export HF_HOME="${HF_HOME:-$PSC_USER_DIR/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PSC_USER_DIR/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PSC_USER_DIR/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PSC_USER_DIR/.cache/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PSC_USER_DIR/.cache}"
export PYTHONPATH="$ROOT/../MC-TextWorld:${PYTHONPATH:-}"
export WANDB_MODE
export NSFSM_LLM_API_BASE="http://localhost:${PORT}/v1"
export NSFSM_LLM_MODEL_NAME="$SERVED_MODEL_NAME"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-nsfsm}"

echo "Python: $(which python)"
echo "Conda:  ${CONDA_DEFAULT_ENV:-unknown}"

echo "[1/3] Starting vLLM server for ${SHARD_LABEL}..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype "$DTYPE" \
    > "logs/vllm_nsfsm_32b_goals67_${ARRAY_JOB_ID}_${ARRAY_INDEX}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[2/3] Waiting for vLLM readiness..."
MAX_WAIT="${MAX_WAIT:-1800}"
WAITED=0
until curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited early. Last log lines:"
        tail -80 "logs/vllm_nsfsm_32b_goals67_${ARRAY_JOB_ID}_${ARRAY_INDEX}.log" || true
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  waiting... ${WAITED}s"
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: vLLM did not become ready within ${MAX_WAIT}s"
        tail -80 "logs/vllm_nsfsm_32b_goals67_${ARRAY_JOB_ID}_${ARRAY_INDEX}.log" || true
        exit 1
    fi
done
echo "vLLM ready after ${WAITED}s"

echo "[3/3] Running shard ${SHARD_LABEL}..."
python scripts/run_nsfsm_full_rollout.py \
    --config "$CONFIG_PATH" \
    --tag "$TAG" \
    --runs "$RUNS" \
    --task-source-path "$TASK_SOURCE_PATH" \
    --max-tasks "$MAX_TASKS" \
    --task-shard-index "$ARRAY_INDEX" \
    --task-shard-count "$SHARD_COUNT" \
    --shard-label "$SHARD_LABEL" \
    --resume \
    --quiet \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "${TAG}-${SHARD_LABEL}" \
    --wandb-group "$TAG" \
    --wandb-mode "$WANDB_MODE"

echo "============================================"
echo "  Shard ${SHARD_LABEL} completed at $(date)"
echo "  Partial results: results/full/${TAG}"
echo "  Shard status:    results/analysis/${TAG}/rollout_status_${SHARD_LABEL}.json"
echo "============================================"
