# GPT API Reflexion Remote Run Guide

This guide explains how to run the non-NS-FSM GPT API Reflexion baseline on a
remote machine so the experiment can continue after your laptop is closed.

The target experiment is:

```text
67 Minecraft goals x 3 runs
Agent: Reflexion
Max attempts: 3
Model: configured API model, currently gpt-5-mini through CMU AI Gateway
No NS-FSM, no Datalog, no synthesized action hints
```

## 1. What You Need

Use a remote machine that stays online:

- Recommended: PSC or another university Slurm cluster CPU node.
- Alternative: a small AWS EC2 CPU instance with `tmux`.
- Not needed: GPU, vLLM, H100, CUDA.

The API runner uses these files:

```text
config/hyperparams_gpt4o_reflexion_api.yaml
config/goals_67.json
scripts/run_gpt4o_reflexion_full.py
src/reflexion_agent.py
src/react_agent.py
src/llm_interface.py
```

The runner expects `MC-TextWorld` to be a sibling directory of `ns-fsm-baseline`:

```text
project/
  ns-fsm-baseline/
  MC-TextWorld/
```

## 2. Prepare The Repo On The Remote Machine

Clone or copy the project onto the remote machine:

```bash
mkdir -p ~/project
cd ~/project

# Use your own repo URL / branch if different.
git clone https://github.com/AlexCui8591/aNLP-11711-26s-NSFSM.git ns-fsm-baseline

# MC-TextWorld must be next to ns-fsm-baseline.
# Clone or copy your existing MC-TextWorld directory here.
# Final layout should be: ~/project/MC-TextWorld
```

If these API-run changes are still local, commit and push them first, then pull
on the remote machine.

## 3. Create The Python Environment

On PSC or Linux:

```bash
cd ~/project/ns-fsm-baseline

conda create -n nsfsm-api python=3.11 -y
conda activate nsfsm-api

pip install -r requirements.txt
pip install gym gymnasium
```

Do not install `vllm` for the API baseline unless you also need local model
serving for another experiment.

Quick import check:

```bash
python - <<'PY'
import openai, yaml, gym
print("basic imports ok")
PY
```

MC-TextWorld check:

```bash
cd ~/project/ns-fsm-baseline
python - <<'PY'
import sys
sys.path.insert(0, "../MC-TextWorld")
from mctextworld.simulator import Env
print("MC-TextWorld import ok")
PY
```

## 4. Configure API Access

Do not put your API key in YAML or commit it to Git.

Set the key as an environment variable:

```bash
export NSFSM_LLM_API_KEY="YOUR_KEY_HERE"
```

The config file should keep:

```yaml
llm:
  model_name: "gpt-5-mini"
  backend: "api"
  api_base: "https://ai-gateway.andrew.cmu.edu"
  api_key: null
```

If CMU AI Gateway denies the model, select a model that works in the Gateway UI
and update `model_name`. The error usually looks like:

```text
team not allowed to access model ... Tried to access <model>
```

For your current Gateway UI, `gpt-5-mini` is a known working choice. If needed,
try one of the exact aliases shown in the error or UI, for example:

```yaml
model_name: "CompServ Cloud - AWS"
```

## 5. Smoke Test

Run a cheap smoke test before full execution:

```bash
cd ~/project/ns-fsm-baseline
conda activate nsfsm-api
export NSFSM_LLM_API_KEY="YOUR_KEY_HERE"

python scripts/run_gpt4o_reflexion_full.py \
  --tag gpt5mini_reflexion_smoke \
  --runs 1 \
  --max-tasks 3 \
  --resume \
  --quiet
```

Expected output:

```text
goals: 3
runs per goal: 1
planned runs: 3
...
Summary
completed: 3/3
```

The `gym` migration warning and `Action Library: 732` messages are expected and
can be ignored.

## 6. Full Run On PSC Slurm

The API baseline only needs CPU and network access. Use a CPU partition, not a
GPU partition.

Create a small Slurm script on the cluster, for example `run_api_reflexion.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=gpt-api-refl
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=cis250260p
#SBATCH --output=logs/gpt_api_reflexion_%j.out
#SBATCH --error=logs/gpt_api_reflexion_%j.err

set -euo pipefail

cd ~/project/ns-fsm-baseline
mkdir -p logs

module load anaconda3
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate nsfsm-api

if [ -z "${NSFSM_LLM_API_KEY:-}" ]; then
  echo "ERROR: set NSFSM_LLM_API_KEY before submitting this job."
  exit 1
fi

python scripts/run_gpt4o_reflexion_full.py \
  --tag gpt5mini_reflexion_goals67x3 \
  --runs 3 \
  --resume \
  --quiet
```

Submit it:

```bash
export NSFSM_LLM_API_KEY="YOUR_KEY_HERE"
sbatch --export=ALL run_api_reflexion.sbatch
```

Check status:

```bash
squeue -u $USER
tail -f logs/gpt_api_reflexion_<JOB_ID>.out
```

If the job stops or times out, resubmit the same command. The runner uses
`--resume`, so completed result files are skipped.

## 7. Optional Slurm Sharding

If the cluster allows multiple CPU jobs and you want faster execution, run three
shards:

```bash
for i in 0 1 2; do
  sbatch --export=ALL,SHARD_INDEX=$i,SHARD_COUNT=3 run_api_reflexion_shard.sbatch
done
```

Use this body for `run_api_reflexion_shard.sbatch`:

```bash
python scripts/run_gpt4o_reflexion_full.py \
  --tag gpt5mini_reflexion_goals67x3 \
  --runs 3 \
  --task-shard-index "${SHARD_INDEX}" \
  --task-shard-count "${SHARD_COUNT}" \
  --resume \
  --quiet
```

All shards write into the same tag directory, but each shard owns a disjoint
subset of goals.

## 8. Alternative: AWS EC2 + tmux

On a small CPU EC2 instance:

```bash
tmux new -s gpt-api

cd ~/project/ns-fsm-baseline
conda activate nsfsm-api
export NSFSM_LLM_API_KEY="YOUR_KEY_HERE"

python scripts/run_gpt4o_reflexion_full.py \
  --tag gpt5mini_reflexion_goals67x3 \
  --runs 3 \
  --resume \
  --quiet
```

Detach without stopping the run:

```text
Ctrl-b d
```

Reattach later:

```bash
tmux attach -t gpt-api
```

## 9. Outputs

Full-run outputs:

```text
results/full/gpt5mini_reflexion_goals67x3/
  experiment_config.json
  reflexion/
    <goal>_run01.json
    <goal>_run02.json
    <goal>_run03.json
  reflexion_summary.json
  combined_summary.json
```

A complete full run should have:

```text
67 goals x 3 runs = 201 result JSON files
```

Check completion:

```bash
find results/full/gpt5mini_reflexion_goals67x3/reflexion -name "*.json" | wc -l
```

Regenerate summary without making API calls:

```bash
python scripts/run_gpt4o_reflexion_full.py \
  --tag gpt5mini_reflexion_goals67x3 \
  --runs 3 \
  --summary-only
```

## 10. Common Problems

### `team not allowed to access model`

The model name in YAML is not allowed by your CMU Gateway team.

Fix:

```yaml
model_name: "gpt-5-mini"
```

or use the exact model alias shown in the Gateway UI.

### `Missing API key`

Set one of:

```bash
export NSFSM_LLM_API_KEY="YOUR_KEY_HERE"
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

### `No module named gym`

Install the environment dependency:

```bash
pip install gym gymnasium
```

### `No module named mctextworld`

Make sure the directory layout is:

```text
project/
  ns-fsm-baseline/
  MC-TextWorld/
```

### Laptop closed and run stopped

That means the process was still running locally. Move it to PSC Slurm or EC2
with `tmux`; local processes cannot survive laptop shutdown.

## 11. Safety Checklist Before Commit

Before pushing:

```bash
git status --short
git diff -- config/hyperparams_gpt4o_reflexion_api.yaml
```

Confirm the YAML contains:

```yaml
api_key: null
```

Never commit a real API key.
