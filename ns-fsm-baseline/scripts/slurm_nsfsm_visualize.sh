#!/bin/bash
#SBATCH --job-name=nsfsm-viz
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/nsfsm_viz_%j.out
#SBATCH --error=logs/nsfsm_viz_%j.err

# Generate combined CSV/PNG/Markdown after the 3-H100 shard array finishes.
#
# Submit manually:
#   TAG=nsfsm_32b_buildable_192x3 sbatch scripts/slurm_nsfsm_visualize.sh
#
# Or submit with dependency after the array job:
#   jid=$(sbatch --parsable scripts/slurm_nsfsm_32b_rollout.sh)
#   TAG=nsfsm_32b_buildable_192x3 sbatch --dependency=afterok:${jid} scripts/slurm_nsfsm_visualize.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
mkdir -p logs

TAG="${TAG:-nsfsm_32b_buildable_192x3}"
WANDB_PROJECT="${WANDB_PROJECT:-nsfsm-mctextworld}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-nsfsm}"

export WANDB_MODE
export PYTHONPATH="$ROOT/../MC-TextWorld:${PYTHONPATH:-}"

python scripts/visualize_nsfsm_rollout.py \
    --tag "$TAG" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "${TAG}-analysis" \
    --wandb-mode "$WANDB_MODE"

echo "Analysis written to results/analysis/${TAG}"
