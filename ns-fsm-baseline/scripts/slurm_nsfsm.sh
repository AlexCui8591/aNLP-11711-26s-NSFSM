#!/bin/bash
#SBATCH --job-name=nsfsm-minecraft
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/nsfsm_%j.out
#SBATCH --error=logs/nsfsm_%j.err

# SLURM script: NS-FSM Minecraft experiment on PSC Bridges-2.
# Usage:
#   sbatch scripts/slurm_nsfsm.sh
# Optional overrides:
#   TAG=minecraft_compare_small_vllm_r1 TASK_IDS=minecraft/stick,minecraft/wooden_pickaxe RUNS=1 sbatch scripts/slurm_nsfsm.sh
#   TAG=full_v1 TASK_IDS="" RUNS=15 sbatch scripts/slurm_nsfsm.sh
#   NSFSM_EXTRA_ARGS="--planner-only" sbatch scripts/slurm_nsfsm.sh

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_RUN:-0}" != "1" ]; then
    echo "ERROR: submit this script with sbatch, not bash."
    echo "Usage:"
    echo "  sbatch scripts/slurm_nsfsm.sh"
    echo
    echo "If you only want to debug on a login node, run:"
    echo "  ALLOW_LOGIN_RUN=1 bash scripts/slurm_nsfsm.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TAG="${TAG:-minecraft_compare_stick_vllm_r1}"
TASK_IDS="${TASK_IDS-minecraft/stick}"
RUNS="${RUNS:-1}"
CONFIG_PATH="${CONFIG_PATH:-config/hyperparams_psc.yaml}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
CACHE_OWNER="${CACHE_OWNER:-ezhang13}"
CACHE_ROOT="/ocean/projects/cis250260p/${CACHE_OWNER}/.cache"
MC_TEXTWORLD_PATH="${MC_TEXTWORLD_PATH:-/ocean/projects/cis250260p/ezhang13/aNLP-11711-26s-NSFSM/MC-TextWorld}"
JOB_ID="${SLURM_JOB_ID:-manual}"

echo "============================================"
echo "  NS-FSM Minecraft Experiment"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time:     $(date)"
echo "  Tag:      ${TAG}"
echo "  Task IDs: ${TASK_IDS}"
echo "  Runs:     ${RUNS}"
echo "============================================"

export HF_HOME="${CACHE_ROOT}/huggingface"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${CACHE_ROOT}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

if [ ! -d "${MC_TEXTWORLD_PATH}/mctextworld" ]; then
    echo "ERROR: MC-TextWorld path is not valid:"
    echo "  ${MC_TEXTWORLD_PATH}"
    echo "Expected:"
    echo "  ${MC_TEXTWORLD_PATH}/mctextworld"
    echo
    echo "If MC-TextWorld is elsewhere, submit with:"
    echo "  MC_TEXTWORLD_PATH=/path/to/MC-TextWorld sbatch scripts/slurm_nsfsm.sh"
    exit 1
fi

export PYTHONPATH="${MC_TEXTWORLD_PATH}:${PYTHONPATH:-}"

module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate nsfsm

echo "  Python:    $(which python)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo "  Project root: ${PROJECT_ROOT}"
echo "  MC-TextWorld: ${MC_TEXTWORLD_PATH}"

set -e

python -c "import sys; sys.path.insert(0, '${MC_TEXTWORLD_PATH}'); from mctextworld.simulator import Env; print('real MC-TextWorld ok')"

echo "[1/4] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${PORT}" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    &> "logs/vllm_nsfsm_${JOB_ID}.log" &

VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
echo "  vLLM PID: ${VLLM_PID}"

echo "[2/4] Waiting for vLLM to load model..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process died. Check logs/vllm_nsfsm_${JOB_ID}.log"
        tail -20 "logs/vllm_nsfsm_${JOB_ID}.log"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
        exit 1
    fi
    echo "  Waiting... (${WAITED}s)"
done
echo "  vLLM is ready! (took ${WAITED}s)"

echo "[3/4] Running NS-FSM experiment..."
python scripts/run_nsfsm_experiment.py \
    --dataset minecraft \
    --task-ids "${TASK_IDS}" \
    --runs "${RUNS}" \
    --tag "${TAG}" \
    --config "${CONFIG_PATH}" \
    --save-fsm-design \
    --resume \
    --quiet \
    ${NSFSM_EXTRA_ARGS:-}

echo "[4/4] Analyzing NS-FSM results..."
python scripts/analyze_nsfsm.py \
    --tag "${TAG}"

echo "============================================"
echo "  NS-FSM experiment completed at $(date)"
echo "============================================"
