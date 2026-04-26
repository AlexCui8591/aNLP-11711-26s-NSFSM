#!/bin/bash
# ============================================================
# PSC one-time environment setup for NS-FSM full rollout.
#
# Run after cloning the repo on PSC:
#   bash scripts/psc_setup.sh
#
# This installs both experiment dependencies and PSC inference dependencies
# such as vLLM and W&B into the `nsfsm` conda environment.
# ============================================================

set -euo pipefail

echo "=== PSC Environment Setup ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
MC_TW="$(dirname "$ROOT")/MC-TextWorld"

if command -v module >/dev/null 2>&1; then
    module load anaconda3
    echo "[OK] Loaded anaconda3 via environment modules"
fi

CONDA_SH=""
if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

if [ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

if [ -z "$CONDA_SH" ] && [ -f /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh ]; then
    CONDA_SH=/opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
fi

if [ -z "$CONDA_SH" ]; then
    echo "[ERROR] Could not locate conda.sh. Activate conda manually and re-run."
    exit 1
fi

source "$CONDA_SH"
echo "[OK] Conda initialized from $CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "nsfsm"; then
    echo "[SKIP] Conda env 'nsfsm' already exists"
else
    echo "Creating conda env 'nsfsm' ..."
    conda create -n nsfsm python=3.10 -y
    echo "[OK] Created conda env"
fi

conda activate nsfsm
python -m pip install --upgrade pip setuptools wheel

echo "Installing Python packages from requirements-psc.txt ..."
python -m pip install -r "$ROOT/requirements-psc.txt"

if [ -d "$MC_TW" ]; then
    echo "[OK] MC-TextWorld found at $MC_TW"
else
    echo "[WARN] MC-TextWorld NOT found at $MC_TW"
    echo "       Expected sibling checkout: $MC_TW"
fi

echo ""
echo "Verifying imports ..."
python - <<'PY'
import importlib

packages = ["yaml", "numpy", "matplotlib", "networkx", "wandb", "vllm"]
for package in packages:
    importlib.import_module(package)
    print(f"[OK] import {package}")
PY

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Login to W&B if using online logging: wandb login"
echo "  2. Submit 3-H100 rollout:             sbatch scripts/slurm_nsfsm_32b_rollout.sh"
echo "  3. After all shards finish:           sbatch scripts/slurm_nsfsm_visualize.sh"
