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

# SLURM script: NS-FSM Robotouille run.
#
# Default mode runs the first official async task with the human-verified
# ground-truth FSM in planner-only mode.
#
# Usage:
#   sbatch scripts/slurm_nsfsm_robotouille.sh
#
# Optional overrides:
#   TAG=robotouille_async_first_r1 TASK_IDS=robotouille/asynchronous/0_cheese_chicken_sandwich RUNS=1 sbatch scripts/slurm_nsfsm_robotouille.sh
#   TASK_IDS=robotouille/asynchronous/5_potato_soup RUNS=3 sbatch scripts/slurm_nsfsm_robotouille.sh
#   ROBOTOUILLE_SEED=84 sbatch scripts/slurm_nsfsm_robotouille.sh
#   NSFSM_EXTRA_ARGS="--use-llm --llm-config config/hyperparams_psc.yaml" sbatch scripts/slurm_nsfsm_robotouille.sh

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

TAG="${TAG:-robotouille_async_first_gt}"
TASK_IDS="${TASK_IDS-robotouille/asynchronous/0_cheese_chicken_sandwich}"
RUNS="${RUNS:-1}"
RESUME="${RESUME:-0}"
ROBOTOUILLE_SEED="${ROBOTOUILLE_SEED:-42}"
ROBOTOUILLE_ROOT="${ROBOTOUILLE_ROOT:-${PROJECT_ROOT}/Robotouille}"
ROBOTOUILLE_GT_PATH="${ROBOTOUILLE_GT_PATH:-${ROOT}/config/robotouille_ground_truth_asynchronous_fsm.json}"
JOB_ID="${SLURM_JOB_ID:-manual}"

echo "============================================"
echo "  NS-FSM Robotouille Experiment"
echo "  Node:         $(hostname)"
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

module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate nsfsm

echo "  Python:    $(which python)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"

export PYTHONPATH="${ROBOTOUILLE_ROOT}:${PYTHONPATH:-}"

RESUME_ARG=()
if [ "${RESUME}" = "1" ]; then
    RESUME_ARG=(--resume)
fi

DEFAULT_EXTRA_ARGS="--planner-only --save-fsm-design --quiet"

echo "[1/1] Running NS-FSM Robotouille experiment..."
python scripts/run_nsfsm_experiment.py \
    --dataset robotouille \
    --robotouille-split asynchronous \
    --robotouille-root "${ROBOTOUILLE_ROOT}" \
    --robotouille-ground-truth-path "${ROBOTOUILLE_GT_PATH}" \
    --robotouille-seed "${ROBOTOUILLE_SEED}" \
    --task-ids "${TASK_IDS}" \
    --runs "${RUNS}" \
    --tag "${TAG}" \
    ${DEFAULT_EXTRA_ARGS} \
    "${RESUME_ARG[@]}" \
    ${NSFSM_EXTRA_ARGS:-}

echo "============================================"
echo "  NS-FSM Robotouille experiment completed at $(date)"
echo "  Results: ${ROOT}/results/full/${TAG}/robotouille/nsfsm"
echo "============================================"
