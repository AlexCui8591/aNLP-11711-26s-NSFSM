#!/bin/bash
#SBATCH --job-name=nsfsm-robot-gpu
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/nsfsm_robotouille_%j.out
#SBATCH --error=logs/nsfsm_robotouille_%j.err

# SLURM script: NS-FSM Robotouille GPU run on PSC Bridges-2.
#
# Default mode launches a local vLLM server on one GPU and runs the first
# official asynchronous Robotouille task with NS-FSM action proposals.
#
# Usage:
#   sbatch scripts/slurm_nsfsm_robotouille.sh
#
# Optional overrides:
#   TAG=robotouille_async_qwen_r1 TASK_IDS=robotouille/asynchronous/0_cheese_chicken_sandwich RUNS=1 sbatch scripts/slurm_nsfsm_robotouille.sh
#   TASK_IDS=robotouille/asynchronous/5_potato_soup RUNS=1 sbatch scripts/slurm_nsfsm_robotouille.sh
#   ROBOTOUILLE_SEED=84 sbatch scripts/slurm_nsfsm_robotouille.sh
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct sbatch scripts/slurm_nsfsm_robotouille.sh
#   NSFSM_EXTRA_ARGS="--planner-only" sbatch scripts/slurm_nsfsm_robotouille.sh

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_RUN:-0}" != "1" ]; then
    echo "ERROR: submit this script with sbatch, not bash."
    echo "Usage:"
    echo "  sbatch scripts/slurm_nsfsm_robotouille.sh"
    echo
    echo "If you only want to debug on a login node, run:"
    echo "  ALLOW_LOGIN_RUN=1 bash scripts/slurm_nsfsm_robotouille.sh"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/scripts/run_nsfsm_experiment.py" ]; then
    ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/../scripts/run_nsfsm_experiment.py" ]; then
    ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
elif [ -f "${SUBMIT_DIR}/ns-fsm-baseline/scripts/run_nsfsm_experiment.py" ]; then
    ROOT="$(cd "${SUBMIT_DIR}/ns-fsm-baseline" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TAG="${TAG:-robotouille_async_qwen_r1}"
TASK_IDS="${TASK_IDS-robotouille/asynchronous/0_cheese_chicken_sandwich}"
RUNS="${RUNS:-1}"
CONFIG_PATH="${CONFIG_PATH:-config/hyperparams_psc.yaml}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
RESUME="${RESUME:-0}"
ROBOTOUILLE_SEED="${ROBOTOUILLE_SEED:-42}"
ROBOTOUILLE_ROOT="${ROBOTOUILLE_ROOT:-${PROJECT_ROOT}/Robotouille}"
ROBOTOUILLE_GT_PATH="${ROBOTOUILLE_GT_PATH:-${ROOT}/config/robotouille_ground_truth_asynchronous_fsm.json}"
CACHE_OWNER="${CACHE_OWNER:-ezhang13}"
CACHE_ROOT="/ocean/projects/cis250260p/${CACHE_OWNER}/.cache"
JOB_ID="${SLURM_JOB_ID:-manual}"

echo "============================================"
echo "  NS-FSM Robotouille GPU Experiment"
echo "  Node:         $(hostname)"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time:         $(date)"
echo "  Tag:          ${TAG}"
echo "  Task IDs:     ${TASK_IDS}"
echo "  Runs:         ${RUNS}"
echo "  Seed:         ${ROBOTOUILLE_SEED}"
echo "  Resume:       ${RESUME}"
echo "  Robotouille:  ${ROBOTOUILLE_ROOT}"
echo "  Ground truth: ${ROBOTOUILLE_GT_PATH}"
echo "============================================"

if [ ! -d "${ROBOTOUILLE_ROOT}" ]; then
    echo "ERROR: Robotouille repo path is not valid:"
    echo "  ${ROBOTOUILLE_ROOT}"
    echo
    echo "If Robotouille is elsewhere, submit with:"
    echo "  ROBOTOUILLE_ROOT=/path/to/Robotouille sbatch scripts/slurm_nsfsm_robotouille.sh"
    exit 1
fi

if [ ! -f "${ROBOTOUILLE_GT_PATH}" ]; then
    echo "ERROR: Robotouille ground-truth FSM file is missing:"
    echo "  ${ROBOTOUILLE_GT_PATH}"
    exit 1
fi

export HF_HOME="${CACHE_ROOT}/huggingface"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${CACHE_ROOT}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate nsfsm

echo "  Python:    $(which python)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo "  Model:     ${MODEL_NAME}"
echo "  Port:      ${PORT}"

export PYTHONPATH="${ROBOTOUILLE_ROOT}:${PYTHONPATH:-}"
set -e

RESUME_ARG=()
if [ "${RESUME}" = "1" ]; then
    RESUME_ARG=(--resume)
fi

echo "[1/3] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${PORT}" \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    &> "logs/vllm_robotouille_${JOB_ID}.log" &

VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
echo "  vLLM PID: ${VLLM_PID}"

echo "[2/3] Waiting for vLLM to load model..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process died. Check logs/vllm_robotouille_${JOB_ID}.log"
        tail -20 "logs/vllm_robotouille_${JOB_ID}.log"
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

echo "[3/3] Running NS-FSM Robotouille experiment..."
python scripts/run_nsfsm_experiment.py \
    --dataset robotouille \
    --robotouille-split asynchronous \
    --robotouille-root "${ROBOTOUILLE_ROOT}" \
    --robotouille-ground-truth-path "${ROBOTOUILLE_GT_PATH}" \
    --robotouille-seed "${ROBOTOUILLE_SEED}" \
    --task-ids "${TASK_IDS}" \
    --runs "${RUNS}" \
    --tag "${TAG}" \
    --llm-config "${CONFIG_PATH}" \
    --use-llm \
    --save-fsm-design \
    --quiet \
    "${RESUME_ARG[@]}" \
    ${NSFSM_EXTRA_ARGS:-}

echo "============================================"
echo "  NS-FSM Robotouille experiment completed at $(date)"
echo "  Results: ${ROOT}/results/full/${TAG}/robotouille/nsfsm"
echo "============================================"
