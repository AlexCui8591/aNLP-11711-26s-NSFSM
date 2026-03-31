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

set -e

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

# ── Environment ──────────────────────────────────────────────
module load anaconda3
conda activate nsfsm

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

# Wait for vLLM to be ready
echo "[2/3] Waiting for vLLM to load model..."
MAX_WAIT=300  # 5 minutes max
WAITED=0
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
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
    --resume \
    --quiet

echo "============================================"
echo "  ReAct experiment completed at $(date)"
echo "============================================"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
