#!/bin/bash
# ============================================================
# PSC Bridges-2 — One-time environment setup
# Run this ONCE after cloning the repo:
#   bash scripts/psc_setup.sh
# ============================================================

set -e

echo "=== PSC Environment Setup ==="

# 1. Load modules
module load anaconda3
echo "[OK] Loaded anaconda3"

# 2. Create conda environment
if conda env list | grep -q "nsfsm"; then
    echo "[SKIP] Conda env 'nsfsm' already exists"
else
    echo "Creating conda env 'nsfsm' ..."
    conda create -n nsfsm python=3.10 -y
    echo "[OK] Created conda env"
fi

# 3. Activate and install dependencies
conda activate nsfsm

echo "Installing Python packages..."
pip install vllm openai pyyaml gymnasium 2>/dev/null || pip install vllm openai pyyaml gym

# 4. Check MC-TextWorld is accessible
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
MC_TW="$(dirname "$ROOT")/MC-TextWorld"

if [ -d "$MC_TW" ]; then
    echo "[OK] MC-TextWorld found at $MC_TW"
else
    echo "[WARN] MC-TextWorld NOT found at $MC_TW"
    echo "       Make sure MC-TextWorld is cloned alongside this repo"
fi

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Make sure MC-TextWorld is at: $MC_TW"
echo "  2. Submit experiment:  sbatch scripts/slurm_react.sh"
echo "  3. Then:               sbatch scripts/slurm_reflexion.sh"
