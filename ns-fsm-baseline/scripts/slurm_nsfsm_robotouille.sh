#!/bin/bash
#SBATCH --job-name=nsfsm-robot
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/nsfsm_robotouille_%j.out
#SBATCH --error=logs/nsfsm_robotouille_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line ${LINENO}: ${BASH_COMMAND} exited with ${rc}" >&2' ERR

# SLURM script: NS-FSM Robotouille run.
#
# Default mode runs the first official async task with the human-verified
# ground-truth FSM and runtime LLM inference. Planner-only mode is disabled
# for this Robotouille SLURM entrypoint.
#
# Usage:
#   sbatch scripts/slurm_nsfsm_robotouille.sh
#
# Optional overrides:
#   TAG=robotouille_async_first_r1 TASK_IDS=robotouille/asynchronous/0_cheese_chicken_sandwich RUNS=1 sbatch scripts/slurm_nsfsm_robotouille.sh
#   TASK_IDS=robotouille/asynchronous/5_potato_soup RUNS=3 sbatch scripts/slurm_nsfsm_robotouille.sh
#   ROBOTOUILLE_SEED=84 sbatch scripts/slurm_nsfsm_robotouille.sh
#   START_VLLM=0 RUNTIME_LLM_CONFIG=config/hyperparams_psc.yaml sbatch scripts/slurm_nsfsm_robotouille.sh
#   START_VLLM=1 MODEL_NAME=Qwen/Qwen2.5-7B-Instruct sbatch --partition=general --qos=normal --account=atalwalk --gres=gpu:1 scripts/slurm_nsfsm_robotouille.sh
#   BUILD_GT=1 GT_BUILD_MODE=branching sbatch scripts/slurm_nsfsm_robotouille.sh
#   BUILD_GT=1 USE_LLM_GT_GENERATOR=1 USE_LLM_GT_JUDGE=1 LLM_REQUIRED=1 GT_LLM_CONFIG=config/hyperparams_openai_gpt4o.yaml LAST_GOOD_GT_PATH=config/robotouille_ground_truth_asynchronous_fsm.json sbatch scripts/slurm_nsfsm_robotouille.sh
#   OFFICIAL_ASYNC=1 BUILD_GT=1 TAG=robotouille_async_nsfsm_official sbatch scripts/slurm_nsfsm_robotouille.sh
#
# Two supported comparison paths:
#   1. GPT-4o generates/judges the ground-truth state_list, then Qwen/vLLM
#      performs every runtime NS-FSM action/state inference.
#   2. The benchmark-derived deterministic builder creates the ground-truth
#      state_list, then Qwen/vLLM performs every runtime NS-FSM inference.

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

OFFICIAL_ASYNC="${OFFICIAL_ASYNC:-0}"
TAG="${TAG:-robotouille_async_first_gt}"
if [ "${OFFICIAL_ASYNC}" = "1" ]; then
    TASK_IDS="${TASK_IDS-}"
else
    TASK_IDS="${TASK_IDS-robotouille/asynchronous/0_cheese_chicken_sandwich}"
fi
RUNS="${RUNS:-1}"
RESUME="${RESUME:-0}"
BUILD_GT="${BUILD_GT:-0}"
GT_BUILD_MODE="${GT_BUILD_MODE:-branching}"
USE_LLM_GT_GENERATOR="${USE_LLM_GT_GENERATOR:-0}"
USE_LLM_GT_JUDGE="${USE_LLM_GT_JUDGE:-0}"
GT_GENERATOR_ATTEMPTS="${GT_GENERATOR_ATTEMPTS:-1}"
GT_JUDGE_ATTEMPTS="${GT_JUDGE_ATTEMPTS:-1}"
LLM_REQUIRED="${LLM_REQUIRED:-0}"
PLANNER_ONLY="${PLANNER_ONLY:-0}"
ROBOTOUILLE_SEED="${ROBOTOUILLE_SEED:-42}"
ROBOTOUILLE_SEEDS="${ROBOTOUILLE_SEEDS:-}"
MAX_STEP_MULTIPLIER="${MAX_STEP_MULTIPLIER:-}"
if [ "${OFFICIAL_ASYNC}" = "1" ]; then
    ROBOTOUILLE_SEEDS="${ROBOTOUILLE_SEEDS:-42,84,126,168,210,252,294,336,378,420}"
    MAX_STEP_MULTIPLIER="${MAX_STEP_MULTIPLIER:-1.5}"
fi
if [ -f "${PROJECT_ROOT}/../robotouille/robotouille/robotouille_env.py" ]; then
    DEFAULT_ROBOTOUILLE_ROOT="$(cd "${PROJECT_ROOT}/../robotouille" && pwd)"
else
    DEFAULT_ROBOTOUILLE_ROOT="${PROJECT_ROOT}/robotouille"
fi
ROBOTOUILLE_ROOT="${ROBOTOUILLE_ROOT:-${DEFAULT_ROBOTOUILLE_ROOT}}"
if [ "${BUILD_GT}" = "1" ]; then
    DEFAULT_ROBOTOUILLE_GT_PATH="${ROOT}/results/full/${TAG}/robotouille_ground_truth_asynchronous_fsm.json"
else
    DEFAULT_ROBOTOUILLE_GT_PATH="${ROOT}/config/robotouille_ground_truth_asynchronous_fsm.json"
fi
ROBOTOUILLE_GT_PATH="${ROBOTOUILLE_GT_PATH:-${DEFAULT_ROBOTOUILLE_GT_PATH}}"
LAST_GOOD_GT_PATH="${LAST_GOOD_GT_PATH:-${ROOT}/config/robotouille_ground_truth_asynchronous_fsm.json}"
GT_LLM_CONFIG="${GT_LLM_CONFIG:-config/hyperparams_openai_gpt4o.yaml}"
RUNTIME_LLM_CONFIG="${RUNTIME_LLM_CONFIG:-${LLM_CONFIG:-config/hyperparams_psc.yaml}}"
if [ "${PLANNER_ONLY}" = "0" ]; then
    START_VLLM="${START_VLLM:-1}"
else
    START_VLLM="${START_VLLM:-0}"
fi
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME}}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-auto}"
CONDA_ENV="${CONDA_ENV:-robotouille}"
JOB_ID="${SLURM_JOB_ID:-manual}"
CACHE_OWNER="${CACHE_OWNER:-${USER:-user}}"
if [ -n "${CACHE_ROOT:-}" ]; then
    CACHE_ROOT="${CACHE_ROOT}"
elif [ "${JOB_LOCAL_CACHE:-1}" = "1" ]; then
    CACHE_ROOT="/tmp/${CACHE_OWNER}/nsfsm_cache_${JOB_ID}"
elif [ -d "/ocean/projects/cis250260p" ]; then
    CACHE_ROOT="/ocean/projects/cis250260p/${CACHE_OWNER}/.cache"
else
    CACHE_ROOT="/tmp/${CACHE_OWNER}/nsfsm_cache_${JOB_ID}"
fi
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"

if [ "${PLANNER_ONLY}" = "1" ]; then
    echo "ERROR: planner-only mode is disabled for Robotouille SLURM runs."
    echo "Runtime action/state inference must use an LLM. Remove PLANNER_ONLY=1."
    exit 1
fi

echo "============================================"
echo "  NS-FSM Robotouille Experiment"
echo "  Node:         $(hostname)"
echo "  Time:         $(date)"
echo "  Tag:          ${TAG}"
echo "  Official async:${OFFICIAL_ASYNC}"
echo "  Task IDs:     ${TASK_IDS:-ALL}"
echo "  Runs:         ${RUNS}"
echo "  Seed:         ${ROBOTOUILLE_SEED}"
echo "  Seeds:        ${ROBOTOUILLE_SEEDS:-single-seed}"
echo "  Max step x:   ${MAX_STEP_MULTIPLIER:-adapter-default}"
echo "  Resume:       ${RESUME}"
echo "  Require LLM:  1"
echo "  Start vLLM:   ${START_VLLM}"
if [ "${START_VLLM}" = "1" ]; then
    echo "  vLLM model:   ${MODEL_NAME}"
    echo "  vLLM port:    ${PORT}"
    echo "  Cache root:   ${CACHE_ROOT}"
fi
echo "  HF cache:     ${HF_HOME}"
echo "  Conda env:    ${CONDA_ENV}"
echo "  Robotouille:  ${ROBOTOUILLE_ROOT}"
echo "  Ground truth: ${ROBOTOUILLE_GT_PATH}"
echo "  Build GT:     ${BUILD_GT}"
if [ "${BUILD_GT}" = "1" ]; then
    echo "  GT mode:      ${GT_BUILD_MODE}"
    echo "  LLM gen/jdg:  ${USE_LLM_GT_GENERATOR}/${USE_LLM_GT_JUDGE}"
    echo "  GT LLM config:${GT_LLM_CONFIG}"
    echo "  Last good GT: ${LAST_GOOD_GT_PATH}"
fi
echo "============================================"

if [ ! -d "${ROBOTOUILLE_ROOT}" ]; then
    echo "ERROR: Robotouille repo path is not valid:"
    echo "  ${ROBOTOUILLE_ROOT}"
    echo
    echo "If Robotouille is elsewhere, submit with:"
    echo "  ROBOTOUILLE_ROOT=/path/to/Robotouille sbatch scripts/slurm_nsfsm_robotouille.sh"
    exit 1
fi

if [ "${BUILD_GT}" != "1" ] && [ ! -f "${ROBOTOUILLE_GT_PATH}" ]; then
    echo "ERROR: Robotouille ground-truth FSM file is missing:"
    echo "  ${ROBOTOUILLE_GT_PATH}"
    exit 1
fi

PYTHON_BIN="$(command -v python || true)"
if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: python is not on PATH."
    echo "Activate the intended environment before submitting this job."
    exit 1
fi
echo "[env] Using pre-activated environment from submission context."
echo "  Python:    ${PYTHON_BIN}"
echo "  Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "  Conda prefix: ${CONDA_PREFIX:-unset}"

export PYTHONPATH="${ROBOTOUILLE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_robotouille_${USER:-user}}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
echo "[cache] Cache root: ${CACHE_ROOT}"
df -h "${CACHE_ROOT}" || true

VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID}" ]; then
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
    if [ "${CLEAN_JOB_CACHE:-1}" = "1" ] \
        && [ "${JOB_LOCAL_CACHE:-1}" = "1" ] \
        && [[ "${CACHE_ROOT}" == "/tmp/${CACHE_OWNER}/nsfsm_cache_"* ]]; then
        rm -rf "${CACHE_ROOT}"
    fi
}
trap cleanup EXIT

start_vllm_server() {
    export NSFSM_LLM_API_BASE="http://localhost:${PORT}/v1"
    export NSFSM_LLM_API_KEY="${NSFSM_LLM_API_KEY:-not-needed}"
    export NSFSM_LLM_MODEL_NAME="${SERVED_MODEL_NAME}"
    echo "[vLLM] Cache root: ${CACHE_ROOT}"
    df -h "${CACHE_ROOT}" || true

    if [ "${VLLM_PREFLIGHT:-1}" = "1" ]; then
        echo "[vLLM] Import preflight..."
        timeout "${VLLM_IMPORT_TIMEOUT:-600}" "${PYTHON_BIN}" -u - <<'PY'
import sys
print(f"python={sys.executable}", flush=True)
print("import torch...", flush=True)
import torch
print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
print("import vllm...", flush=True)
import vllm
print(f"vllm={getattr(vllm, '__version__', 'unknown')}", flush=True)
PY
    fi

    echo "[vLLM] Starting server..."
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --port "${PORT}" \
        --trust-remote-code \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --dtype "${DTYPE}" \
        > "logs/vllm_robotouille_${JOB_ID}.log" 2>&1 &
    VLLM_PID=$!
    echo "[vLLM] PID: ${VLLM_PID}"

    echo "[vLLM] Waiting for readiness..."
    MAX_WAIT="${MAX_WAIT:-7200}"
    WAITED=0
    until curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited early. Last log lines:"
            tail -80 "logs/vllm_robotouille_${JOB_ID}.log" || true
            exit 1
        fi
        sleep 10
        WAITED=$((WAITED + 10))
        echo "  waiting... ${WAITED}s"
        if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
            echo "ERROR: vLLM did not become ready within ${MAX_WAIT}s"
            tail -80 "logs/vllm_robotouille_${JOB_ID}.log" || true
            exit 1
        fi
    done
    echo "[vLLM] Ready after ${WAITED}s"
}

if [ "${BUILD_GT}" = "1" ]; then
    mkdir -p "$(dirname "${ROBOTOUILLE_GT_PATH}")"
    GT_ARGS=(
        --split asynchronous
        --mode "${GT_BUILD_MODE}"
        --output "${ROBOTOUILLE_GT_PATH}"
        --task-ids "${TASK_IDS}"
        --generator-attempts "${GT_GENERATOR_ATTEMPTS}"
        --judge-attempts "${GT_JUDGE_ATTEMPTS}"
    )
    if [ -f "${LAST_GOOD_GT_PATH}" ]; then
        GT_ARGS+=(--last-good-ground-truth "${LAST_GOOD_GT_PATH}")
    fi
    if [ "${USE_LLM_GT_GENERATOR}" = "1" ]; then
        GT_ARGS+=(--use-llm-generator --llm-config "${GT_LLM_CONFIG}")
    fi
    if [ "${USE_LLM_GT_JUDGE}" = "1" ]; then
        GT_ARGS+=(--use-llm-judge --llm-config "${GT_LLM_CONFIG}")
    fi
    if [ "${LLM_REQUIRED}" = "1" ]; then
        GT_ARGS+=(--llm-required)
    fi

    echo "[1/2] Building Robotouille ground-truth FSM..."
    "${PYTHON_BIN}" scripts/build_robotouille_ground_truth_fsm.py "${GT_ARGS[@]}"
fi

if [ "${START_VLLM}" = "1" ]; then
    start_vllm_server
fi

RESUME_ARG=()
if [ "${RESUME}" = "1" ]; then
    RESUME_ARG=(--resume)
fi

TASK_ARGS=()
if [ -n "${TASK_IDS}" ]; then
    TASK_ARGS=(--task-ids "${TASK_IDS}")
fi

SEED_ARGS=()
if [ -n "${ROBOTOUILLE_SEEDS}" ]; then
    SEED_ARGS=(--robotouille-seeds "${ROBOTOUILLE_SEEDS}")
else
    SEED_ARGS=(--robotouille-seed "${ROBOTOUILLE_SEED}")
fi

STEP_ARGS=()
if [ -n "${MAX_STEP_MULTIPLIER}" ]; then
    STEP_ARGS=(--max-step-multiplier "${MAX_STEP_MULTIPLIER}")
fi

DEFAULT_EXTRA_ARGS=(--save-fsm-design --quiet --use-llm --require-llm --llm-config "${RUNTIME_LLM_CONFIG}")

if [ "${BUILD_GT}" = "1" ]; then
    echo "[2/2] Running NS-FSM Robotouille experiment..."
else
    echo "[1/1] Running NS-FSM Robotouille experiment..."
fi
"${PYTHON_BIN}" scripts/run_nsfsm_experiment.py \
    --dataset robotouille \
    --robotouille-split asynchronous \
    --robotouille-root "${ROBOTOUILLE_ROOT}" \
    --robotouille-ground-truth-path "${ROBOTOUILLE_GT_PATH}" \
    "${SEED_ARGS[@]}" \
    "${STEP_ARGS[@]}" \
    "${TASK_ARGS[@]}" \
    --runs "${RUNS}" \
    --tag "${TAG}" \
    "${DEFAULT_EXTRA_ARGS[@]}" \
    "${RESUME_ARG[@]}" \
    ${NSFSM_EXTRA_ARGS:-}

echo "============================================"
echo "  NS-FSM Robotouille experiment completed at $(date)"
echo "  Results: ${ROOT}/results/full/${TAG}/robotouille/nsfsm"
echo "============================================"
