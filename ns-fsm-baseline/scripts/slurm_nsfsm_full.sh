#!/bin/bash
#SBATCH --job-name=nsfsm-full
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/nsfsm_full_%j.out
#SBATCH --error=logs/nsfsm_full_%j.err

# Full NS-FSM Minecraft run on PSC Bridges-2.
#
# This mirrors the old full baseline scripts: run every Minecraft goal from
# config/goals_67.json with multiple runs, using vLLM via hyperparams_psc.yaml.
#
# Usage:
#   sbatch scripts/slurm_nsfsm_full.sh
#
# Optional overrides:
#   TAG=nsfsm_full_v2 RUNS=5 sbatch scripts/slurm_nsfsm_full.sh
#   GROUPS=Wooden RUNS=15 sbatch scripts/slurm_nsfsm_full.sh
#   TASK_IDS=minecraft/stick,minecraft/wooden_pickaxe RUNS=15 sbatch scripts/slurm_nsfsm_full.sh

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_RUN:-0}" != "1" ]; then
    echo "ERROR: submit this script with sbatch, not bash."
    echo "Usage:"
    echo "  sbatch scripts/slurm_nsfsm_full.sh"
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
cd "$ROOT"
mkdir -p logs

export TAG="${TAG:-full_v1}"
export RUNS="${RUNS:-15}"
export CONFIG_PATH="${CONFIG_PATH:-config/hyperparams_psc.yaml}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export PORT="${PORT:-8000}"
export MC_TEXTWORLD_PATH="${MC_TEXTWORLD_PATH:-/ocean/projects/cis250260p/ezhang13/aNLP-11711-26s-NSFSM/MC-TextWorld}"

# Empty TASK_IDS means run all tasks selected by the Minecraft adapter.
export TASK_IDS="${TASK_IDS-}"

if [ -n "${GROUPS:-}" ]; then
    export NSFSM_EXTRA_ARGS="${NSFSM_EXTRA_ARGS:-} --groups ${GROUPS}"
fi

echo "============================================"
echo "  NS-FSM Full Minecraft Experiment"
echo "  Tag:       ${TAG}"
echo "  Runs:      ${RUNS}"
echo "  Task IDs:  ${TASK_IDS:-ALL}"
echo "  Groups:    ${GROUPS:-ALL}"
echo "============================================"

bash scripts/slurm_nsfsm.sh
