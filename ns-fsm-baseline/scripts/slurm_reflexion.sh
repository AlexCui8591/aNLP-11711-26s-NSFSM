#!/bin/bash
#SBATCH --job-name=nsfsm-reflexion
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/reflexion_%j.out
#SBATCH --error=logs/reflexion_%j.err

# ============================================================
# SLURM script: Reflexion full experiment on PSC Bridges-2
# Usage: sbatch scripts/slurm_reflexion.sh
# Note: Reflexion needs more time (3 attempts per goal)
# ============================================================

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "$ROOT"
mkdir -p logs

echo "============================================"
echo "  Reflexion Full Experiment"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time: $(date)"
echo "============================================"

# ── Environment (all caches → ocean, avoid home quota) ───────
CACHE_OWNER="${CACHE_OWNER:-$USER}"
CACHE_ROOT="/ocean/projects/cis250260p/${CACHE_OWNER}/.cache"
MC_TEXTWORLD_PATH="${MC_TEXTWORLD_PATH:-${PROJECT_ROOT}/MC-TextWorld}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export PYTHONPATH="${MC_TEXTWORLD_PATH}:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"



# ── Conda (safe for non-interactive shell) ───────────────────
module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate nsfsm

# Verify environment
echo "  Python: $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  MC-TextWorld: $MC_TEXTWORLD_PATH"

set -e

# ── Start vLLM server ────────────────────────────────────────
echo "[1/3] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    &> logs/vllm_reflexion_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready (up to 10 min for first-time compile)
echo "[2/3] Waiting for vLLM to load model..."
MAX_WAIT=600
WAITED=0
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    # Check if vLLM process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check logs/vllm_reflexion_${SLURM_JOB_ID}.log"
        tail -20 logs/vllm_reflexion_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "  Waiting... (${WAITED}s)"
done
echo "  vLLM is ready! (took ${WAITED}s)"

# ── Run experiment ───────────────────────────────────────────
echo "[3/3] Running Reflexion experiment..."
python scripts/run_full_experiment.py \
    --agent reflexion \
    --config config/hyperparams_psc.yaml \
    --tag full_v1 \
    --resume \
    --quiet

echo "============================================"
echo "  Reflexion experiment completed at $(date)"
echo "============================================"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
