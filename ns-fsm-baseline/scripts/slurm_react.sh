#!/bin/bash
#SBATCH --job-name=nsfsm-react
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/react_%j.out
#SBATCH --error=logs/react_%j.err

# ============================================================
# SLURM script: ReAct full experiment on PSC Bridges-2
# Usage: sbatch scripts/slurm_react.sh
# ============================================================

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
mkdir -p logs

echo "============================================"
echo "  ReAct Full Experiment"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time: $(date)"
echo "============================================"

# ── Environment (all caches → ocean, avoid home quota) ───────
export HF_HOME=/ocean/projects/cis250260p/cuiz/.cache/huggingface
export VLLM_CACHE_ROOT=/ocean/projects/cis250260p/cuiz/.cache/vllm
export TRITON_CACHE_DIR=/ocean/projects/cis250260p/cuiz/.cache/triton
export TORCH_EXTENSIONS_DIR=/ocean/projects/cis250260p/cuiz/.cache/torch_extensions
export XDG_CACHE_HOME=/ocean/projects/cis250260p/cuiz/.cache
export PYTHONPATH=/ocean/projects/cis250260p/cuiz/aNLP-11711-26s-NSFSM/MC-TextWorld:$PYTHONPATH
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# ── Conda (safe for non-interactive shell) ───────────────────
module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate nsfsm

# Verify environment
echo "  Python: $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"

set -e

# ── Start vLLM server ────────────────────────────────────────
echo "[1/3] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    &> logs/vllm_react_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready (up to 10 min for first-time compile)
echo "[2/3] Waiting for vLLM to load model..."
MAX_WAIT=600
WAITED=0
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    # Check if vLLM process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check logs/vllm_react_${SLURM_JOB_ID}.log"
        tail -20 logs/vllm_react_${SLURM_JOB_ID}.log
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
echo "[3/3] Running ReAct experiment..."
python scripts/run_full_experiment.py \
    --agent react \
    --config config/hyperparams_psc.yaml \
    --tag full_v1 \
    --resume

echo "============================================"
echo "  ReAct experiment completed at $(date)"
echo "============================================"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
