#!/bin/bash
#SBATCH --job-name=nsfsm-robot-arr
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --account=cis250260p
#SBATCH --array=0-99
#SBATCH --output=logs/nsfsm_robotouille_array_%A_%a.out
#SBATCH --error=logs/nsfsm_robotouille_array_%A_%a.err

set -euo pipefail

# Runs one official Robotouille asynchronous (task, seed) pair per array job.
# Runtime action/state inference is always LLM-backed; planner-only mode is
# disabled for this Robotouille entrypoint.
# Submit all 100 official runs:
#   sbatch scripts/slurm_nsfsm_robotouille_array.sh
#
# Build the branching GT once before submitting this array:
#   python scripts/build_robotouille_ground_truth_fsm.py \
#     --mode branching \
#     --output results/full/robotouille_async_nsfsm_array/robotouille_ground_truth_asynchronous_fsm.json

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/scripts/run_nsfsm_experiment.py" ]; then
    ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/ns-fsm-baseline/scripts/run_nsfsm_experiment.py" ]; then
    ROOT="$(cd "${SUBMIT_DIR}/ns-fsm-baseline" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TASKS=(
    "robotouille/asynchronous/0_cheese_chicken_sandwich"
    "robotouille/asynchronous/1_lettuce_chicken_sandwich"
    "robotouille/asynchronous/2_lettuce_tomato_fried_chicken_sandwich"
    "robotouille/asynchronous/3_tomato_burger_and_fries"
    "robotouille/asynchronous/4_onion_cheese_burger_and_fried_onion"
    "robotouille/asynchronous/5_potato_soup"
    "robotouille/asynchronous/6_onion_soup"
    "robotouille/asynchronous/7_tomato_soup_and_lettuce_chicken_sandwich"
    "robotouille/asynchronous/8_onion_tomato_soup_and_two_chicken_sandwich"
    "robotouille/asynchronous/9_onion_potato_soup_and_fried_onion_ring_lettuce_burger_and_onion_cheese_sandwich"
)
SEEDS=(42 84 126 168 210 252 294 336 378 420)

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_IDX=$((ARRAY_ID / ${#SEEDS[@]}))
SEED_IDX=$((ARRAY_ID % ${#SEEDS[@]}))
if [ "${TASK_IDX}" -ge "${#TASKS[@]}" ]; then
    echo "Array id ${ARRAY_ID} is outside the official task/seed grid."
    exit 1
fi

TAG="${TAG:-robotouille_async_nsfsm_array}"
TASK_ID="${TASKS[$TASK_IDX]}"
ROBOTOUILLE_SEED="${SEEDS[$SEED_IDX]}"
if [ -f "${PROJECT_ROOT}/../robotouille/robotouille/robotouille_env.py" ]; then
    DEFAULT_ROBOTOUILLE_ROOT="$(cd "${PROJECT_ROOT}/../robotouille" && pwd)"
else
    DEFAULT_ROBOTOUILLE_ROOT="${PROJECT_ROOT}/robotouille"
fi
ROBOTOUILLE_ROOT="${ROBOTOUILLE_ROOT:-${DEFAULT_ROBOTOUILLE_ROOT}}"
ROBOTOUILLE_GT_PATH="${ROBOTOUILLE_GT_PATH:-${ROOT}/results/full/${TAG}/robotouille_ground_truth_asynchronous_fsm.json}"
CONDA_ENV="${CONDA_ENV:-robotouille}"
PLANNER_ONLY="${PLANNER_ONLY:-0}"
RUNTIME_LLM_CONFIG="${RUNTIME_LLM_CONFIG:-${LLM_CONFIG:-config/hyperparams_psc.yaml}}"
MAX_STEP_MULTIPLIER="${MAX_STEP_MULTIPLIER:-1.5}"

if [ "${PLANNER_ONLY}" = "1" ]; then
    echo "ERROR: planner-only mode is disabled for Robotouille array runs."
    echo "Runtime action/state inference must use an LLM. Remove PLANNER_ONLY=1."
    exit 1
fi

if [ ! -f "${ROBOTOUILLE_GT_PATH}" ]; then
    echo "ERROR: Branching ground-truth FSM does not exist:"
    echo "  ${ROBOTOUILLE_GT_PATH}"
    echo "Build it once before submitting the array."
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load anaconda3 >/dev/null 2>&1 || true
fi

if [ -f /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh ]; then
    source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROBOTOUILLE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_robotouille_${USER:-user}_${SLURM_ARRAY_JOB_ID:-manual}_${ARRAY_ID}}"

EXTRA_ARGS=(--save-fsm-design --quiet --use-llm --require-llm --llm-config "${RUNTIME_LLM_CONFIG}")

echo "============================================"
echo "  NS-FSM Robotouille Array Job"
echo "  Tag:       ${TAG}"
echo "  Array ID:  ${ARRAY_ID}"
echo "  Task:      ${TASK_ID}"
echo "  Seed:      ${ROBOTOUILLE_SEED}"
echo "  GT:        ${ROBOTOUILLE_GT_PATH}"
echo "  LLM config:${RUNTIME_LLM_CONFIG}"
echo "============================================"

python scripts/run_nsfsm_experiment.py \
    --dataset robotouille \
    --robotouille-split asynchronous \
    --robotouille-root "${ROBOTOUILLE_ROOT}" \
    --robotouille-ground-truth-path "${ROBOTOUILLE_GT_PATH}" \
    --robotouille-seeds "${ROBOTOUILLE_SEED}" \
    --max-step-multiplier "${MAX_STEP_MULTIPLIER}" \
    --task-ids "${TASK_ID}" \
    --runs 1 \
    --tag "${TAG}" \
    "${EXTRA_ARGS[@]}" \
    ${NSFSM_EXTRA_ARGS:-}
