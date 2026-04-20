#!/usr/bin/env python3
"""Run full NS-FSM rollout on the cleaned MC-TextWorld buildable task set.

Target experiment:
  192 dependency-graph-buildable tasks x 3 runs with a 32B model on PSC.

The script is intentionally resume-first:
  - each (task, run) result is written immediately;
  - progress/status JSON is refreshed after each episode;
  - W&B logs one row per completed episode when enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from datasets.minecraft import MinecraftAdapter
from llm_interface import LLMInterface
from nsfsm_agent import NSFSMAgent
from run_nsfsm_experiment import build_runtime_fsm, safe_name, summarize


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    apply_config_defaults(args, config)

    output_dir = ROOT / "results" / "full" / args.tag / "minecraft" / "nsfsm"
    analysis_dir = ROOT / "results" / "analysis" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    adapter = MinecraftAdapter(
        task_source="buildable",
        task_source_path=args.task_source_path or None,
    )
    tasks = select_tasks(adapter.list_tasks(), args)
    plan = build_plan(tasks, args.runs)

    wandb_run = init_wandb(args, config, len(tasks), len(plan))
    llm = LLMInterface(args.config)
    completed_before = find_completed(output_dir, plan) if args.resume else set()

    counters = {
        "planned_runs": len(plan),
        "completed_runs": 0,
        "successes": 0,
        "failed_runs": 0,
        "errors": 0,
        "total_steps": 0,
        "blocked_actions": 0,
        "fallback_actions": 0,
        "llm_valid_decisions": 0,
        "planner_decisions": 0,
    }
    start_time = time.time()
    results: list[dict[str, Any]] = []

    for index, (raw_task, run_id) in enumerate(plan, start=1):
        task_spec = adapter.to_task_spec(adapter.load_or_wrap(raw_task)).to_dict()
        path = result_path(output_dir, task_spec["task_id"], run_id)

        if (task_spec["task_id"], run_id) in completed_before and path.exists():
            result = read_json(path)
            results.append(result)
            update_counters(counters, result)
            continue

        episode_start = time.time()
        result = run_one_episode(args, adapter, llm, task_spec, run_id)
        result["metadata"].setdefault("rollout", {})
        result["metadata"]["rollout"].update(
            {
                "tag": args.tag,
                "global_index": index,
                "planned_runs": len(plan),
                "elapsed_seconds": round(time.time() - episode_start, 3),
                "model_name": config.get("llm", {}).get("model_name", ""),
            }
        )
        write_json(path, result)
        results.append(result)
        update_counters(counters, result)

        status = build_status(args, counters, start_time, current_task=task_spec["task_id"])
        write_json(output_dir / status_filename(args), status)
        write_json(analysis_dir / status_filename(args), status)
        append_progress_csv(analysis_dir / progress_filename(args), result, status)
        log_wandb_episode(wandb_run, result, status)

        if not args.quiet:
            print(
                f"[{counters['completed_runs']}/{len(plan)}] {task_spec['task_id']} "
                f"run={run_id} success={result.get('success')} "
                f"steps={result.get('total_steps')} "
                f"sr={status['success_rate']:.3f}"
            )

    write_summaries(args, output_dir, analysis_dir, results, counters, start_time)
    finish_wandb(wandb_run, counters, start_time)
    print(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "hyperparams_nsfsm_32b_psc.yaml"))
    parser.add_argument("--tag", default=f"nsfsm_32b_buildable_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--task-source-path", default="")
    parser.add_argument("--task-ids", default="", help="Comma-separated task ids to include.")
    parser.add_argument("--groups", default="", help="Comma-separated groups to include.")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-fsm-design", action="store_true")
    parser.add_argument(
        "--shard-label",
        default="",
        help="Optional label used for shard-specific status/progress/summary files.",
    )
    parser.add_argument(
        "--use-llm-fsm-designer",
        action="store_true",
        help="Use LLM-designed FSMs. Default uses the symbolic dependency-path template.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-mode", default="")
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config_defaults(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    env = config.get("env", {})
    nsfsm = config.get("nsfsm", {})
    wandb_cfg = config.get("wandb", {})
    if args.runs == 3 and env.get("num_runs"):
        args.runs = int(env["num_runs"])
    if not args.wandb_project:
        args.wandb_project = str(wandb_cfg.get("project") or "nsfsm-mctextworld")
    if not args.wandb_entity and wandb_cfg.get("entity"):
        args.wandb_entity = str(wandb_cfg["entity"])
    if not args.wandb_mode:
        args.wandb_mode = str(wandb_cfg.get("mode") or "online")
    if not args.wandb_group:
        args.wandb_group = args.tag
    if not args.save_fsm_design and bool(nsfsm.get("save_fsm_design", False)):
        args.save_fsm_design = True


def select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    task_ids = {item.strip() for item in args.task_ids.split(",") if item.strip()}
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    selected = []
    for task in tasks:
        short_id = str(task.get("task_id", "")).split("/", 1)[-1]
        if task_ids and task.get("task_id") not in task_ids and short_id not in task_ids:
            continue
        if groups and task.get("group") not in groups:
            continue
        selected.append(task)

    selected = sorted(selected, key=lambda item: str(item.get("task_id", "")))
    if args.task_shard_count > 1:
        selected = [
            task for idx, task in enumerate(selected)
            if idx % args.task_shard_count == args.task_shard_index
        ]
    if args.max_tasks:
        selected = selected[: args.max_tasks]
    return selected


def build_plan(tasks: list[dict[str, Any]], runs: int) -> list[tuple[dict[str, Any], int]]:
    return [(task, run_id) for task in tasks for run_id in range(1, runs + 1)]


def run_one_episode(
    args: argparse.Namespace,
    adapter: MinecraftAdapter,
    llm: LLMInterface,
    task_spec: dict[str, Any],
    run_id: int,
) -> dict[str, Any]:
    try:
        fsm, fsm_metadata = build_runtime_fsm(
            task_spec=task_spec,
            adapter=adapter,
            use_fixed_generic_fsm=not args.use_llm_fsm_designer,
            llm_config=args.config,
        )
        result = NSFSMAgent(
            task_spec=task_spec,
            adapter=adapter,
            fsm=fsm,
            llm=llm,
            planner_only=False,
            verbose=not args.quiet,
        ).run_episode()
        result["run_id"] = run_id
        result.setdefault("metadata", {})
        result["metadata"]["fsm_build"] = fsm_metadata
        if not args.save_fsm_design:
            result["metadata"].get("fsm", {}).pop("transitions", None)
        result["metadata"]["execution_metrics"] = execution_metrics(result)
        return result
    except Exception as exc:
        return error_result(task_spec, run_id, exc)


def error_result(task_spec: Mapping[str, Any], run_id: int, exc: Exception) -> dict[str, Any]:
    return {
        "dataset": task_spec.get("dataset", "minecraft"),
        "task_id": task_spec.get("task_id", ""),
        "task_type": task_spec.get("task_type", "symbolic_planning"),
        "run_id": run_id,
        "success": False,
        "termination": "runtime_error",
        "total_steps": 0,
        "blocked_action_count": 0,
        "fallback_action_count": 0,
        "trajectory": [],
        "metadata": {
            "task_spec": dict(task_spec),
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "execution_metrics": {
                "llm_action_count": 0,
                "llm_valid_decision_count": 0,
                "planner_action_count": 0,
            },
        },
    }


def execution_metrics(result: Mapping[str, Any]) -> dict[str, int]:
    trajectory = list(result.get("trajectory", []))
    llm_action_count = sum(
        1
        for step in trajectory
        if dict(step.get("proposal", {})).get("source") == "llm"
        and dict(step.get("proposal", {})).get("action")
    )
    llm_valid_decision_count = sum(1 for step in trajectory if step.get("decision_source") == "llm")
    planner_action_count = sum(1 for step in trajectory if step.get("decision_source") == "planner")
    return {
        "llm_action_count": llm_action_count,
        "llm_valid_decision_count": llm_valid_decision_count,
        "planner_action_count": planner_action_count,
    }


def result_path(output_dir: Path, task_id: str, run_id: int) -> Path:
    return output_dir / f"{safe_name(task_id)}_run{run_id:02d}.json"


def find_completed(output_dir: Path, plan: list[tuple[dict[str, Any], int]]) -> set[tuple[str, int]]:
    completed = set()
    for task, run_id in plan:
        task_id = str(task.get("task_id", ""))
        path = result_path(output_dir, task_id, run_id)
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if {"dataset", "task_id", "success", "trajectory"}.issubset(payload):
            completed.add((task_id, run_id))
    return completed


def update_counters(counters: dict[str, Any], result: Mapping[str, Any]) -> None:
    counters["completed_runs"] += 1
    counters["successes"] += int(bool(result.get("success", False)))
    counters["failed_runs"] += int(not bool(result.get("success", False)))
    counters["errors"] += int(result.get("termination") == "runtime_error")
    counters["total_steps"] += int(result.get("total_steps", 0))
    counters["blocked_actions"] += int(result.get("blocked_action_count", 0))
    counters["fallback_actions"] += int(result.get("fallback_action_count", 0))
    metrics = result.get("metadata", {}).get("execution_metrics", {})
    counters["llm_valid_decisions"] += int(metrics.get("llm_valid_decision_count", 0))
    counters["planner_decisions"] += int(metrics.get("planner_action_count", 0))


def build_status(
    args: argparse.Namespace,
    counters: Mapping[str, Any],
    start_time: float,
    current_task: str = "",
) -> dict[str, Any]:
    completed = int(counters["completed_runs"])
    planned = int(counters["planned_runs"])
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = planned - completed
    return {
        "tag": args.tag,
        "current_task": current_task,
        "planned_runs": planned,
        "completed_runs": completed,
        "remaining_runs": remaining,
        "successes": int(counters["successes"]),
        "failed_runs": int(counters["failed_runs"]),
        "errors": int(counters["errors"]),
        "success_rate": counters["successes"] / completed if completed else 0.0,
        "avg_steps": counters["total_steps"] / completed if completed else 0.0,
        "blocked_action_count": int(counters["blocked_actions"]),
        "fallback_action_count": int(counters["fallback_actions"]),
        "llm_valid_decisions": int(counters["llm_valid_decisions"]),
        "planner_decisions": int(counters["planner_decisions"]),
        "elapsed_seconds": round(elapsed, 3),
        "runs_per_second": rate,
        "eta_seconds": remaining / rate if rate > 0 else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_progress_csv(path: Path, result: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    fieldnames = [
        "completed_runs",
        "planned_runs",
        "task_id",
        "run_id",
        "success",
        "termination",
        "total_steps",
        "blocked_action_count",
        "fallback_action_count",
        "llm_valid_decision_count",
        "planner_action_count",
        "success_rate",
        "elapsed_seconds",
        "eta_seconds",
    ]
    exists = path.exists()
    metrics = result.get("metadata", {}).get("execution_metrics", {})
    row = {
        "completed_runs": status.get("completed_runs"),
        "planned_runs": status.get("planned_runs"),
        "task_id": result.get("task_id"),
        "run_id": result.get("run_id"),
        "success": result.get("success"),
        "termination": result.get("termination"),
        "total_steps": result.get("total_steps"),
        "blocked_action_count": result.get("blocked_action_count", 0),
        "fallback_action_count": result.get("fallback_action_count", 0),
        "llm_valid_decision_count": metrics.get("llm_valid_decision_count", 0),
        "planner_action_count": metrics.get("planner_action_count", 0),
        "success_rate": status.get("success_rate"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "eta_seconds": status.get("eta_seconds"),
    }
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summaries(
    args: argparse.Namespace,
    output_dir: Path,
    analysis_dir: Path,
    results: list[dict[str, Any]],
    counters: Mapping[str, Any],
    start_time: float,
) -> None:
    summary = summarize(results, "minecraft")
    summary["rollout"] = build_status(args, counters, start_time)
    write_json(output_dir / summary_filename(args, "minecraft_summary"), summary)
    write_json(output_dir.parents[1] / summary_filename(args, "combined_summary"), summary)
    write_task_csv(analysis_dir / csv_filename(args, "rollout_runs"), results)
    write_group_summary_csv(analysis_dir / csv_filename(args, "summary_by_group"), results)


def write_task_csv(path: Path, results: list[Mapping[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "run_id",
        "group",
        "success",
        "termination",
        "total_steps",
        "blocked_action_count",
        "fallback_action_count",
        "llm_valid_decision_count",
        "planner_action_count",
    ]
    rows = []
    for result in results:
        metadata = result.get("metadata", {})
        task_metadata = metadata.get("task_spec", {}).get("metadata", {})
        metrics = metadata.get("execution_metrics", {})
        rows.append(
            {
                "task_id": result.get("task_id"),
                "run_id": result.get("run_id"),
                "group": task_metadata.get("group", ""),
                "success": result.get("success"),
                "termination": result.get("termination"),
                "total_steps": result.get("total_steps"),
                "blocked_action_count": result.get("blocked_action_count", 0),
                "fallback_action_count": result.get("fallback_action_count", 0),
                "llm_valid_decision_count": metrics.get("llm_valid_decision_count", 0),
                "planner_action_count": metrics.get("planner_action_count", 0),
            }
        )
    write_csv(path, rows, fieldnames)


def write_group_summary_csv(path: Path, results: list[Mapping[str, Any]]) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        task_metadata = result.get("metadata", {}).get("task_spec", {}).get("metadata", {})
        groups[str(task_metadata.get("group", ""))].append(result)

    rows = []
    for group, items in sorted(groups.items()):
        runs = len(items)
        successes = sum(1 for item in items if item.get("success"))
        steps = sum(int(item.get("total_steps", 0)) for item in items)
        rows.append(
            {
                "group": group,
                "runs": runs,
                "successes": successes,
                "success_rate": successes / runs if runs else 0.0,
                "avg_steps": steps / runs if runs else 0.0,
            }
        )
    write_csv(path, rows, ["group", "runs", "successes", "success_rate", "avg_steps"])


def init_wandb(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    task_count: int,
    planned_runs: int,
) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install it or rerun without --wandb.") from exc

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or args.tag,
        group=args.wandb_group or None,
        config={
            "tag": args.tag,
            "task_count": task_count,
            "planned_runs": planned_runs,
            "runs_per_task": args.runs,
            "task_source": "buildable",
            "model_name": config.get("llm", {}).get("model_name", ""),
            "api_base": config.get("llm", {}).get("api_base", ""),
            "use_llm_fsm_designer": args.use_llm_fsm_designer,
            "task_shard_index": args.task_shard_index,
            "task_shard_count": args.task_shard_count,
            "shard_label": args.shard_label,
        },
    )


def log_wandb_episode(wandb_run: Any | None, result: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    if wandb_run is None:
        return
    metrics = result.get("metadata", {}).get("execution_metrics", {})
    task_metadata = result.get("metadata", {}).get("task_spec", {}).get("metadata", {})
    wandb_run.log(
        {
            "episode/success": int(bool(result.get("success", False))),
            "episode/total_steps": int(result.get("total_steps", 0)),
            "episode/blocked_action_count": int(result.get("blocked_action_count", 0)),
            "episode/fallback_action_count": int(result.get("fallback_action_count", 0)),
            "episode/llm_valid_decision_count": int(metrics.get("llm_valid_decision_count", 0)),
            "episode/planner_action_count": int(metrics.get("planner_action_count", 0)),
            "progress/completed_runs": int(status.get("completed_runs", 0)),
            "progress/success_rate": float(status.get("success_rate", 0.0)),
            "progress/avg_steps": float(status.get("avg_steps", 0.0)),
            "progress/eta_seconds": status.get("eta_seconds"),
            "task/group": task_metadata.get("group", ""),
            "task/id": result.get("task_id", ""),
            "task/run_id": result.get("run_id", ""),
        },
        step=int(status.get("completed_runs", 0)),
    )


def finish_wandb(wandb_run: Any | None, counters: Mapping[str, Any], start_time: float) -> None:
    if wandb_run is None:
        return
    completed = int(counters.get("completed_runs", 0))
    wandb_run.summary["completed_runs"] = completed
    wandb_run.summary["success_rate"] = counters.get("successes", 0) / completed if completed else 0.0
    wandb_run.summary["elapsed_seconds"] = time.time() - start_time
    wandb_run.finish()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def status_filename(args: argparse.Namespace) -> str:
    return f"rollout_status_{args.shard_label}.json" if args.shard_label else "rollout_status.json"


def progress_filename(args: argparse.Namespace) -> str:
    return f"rollout_progress_{args.shard_label}.csv" if args.shard_label else "rollout_progress.csv"


def summary_filename(args: argparse.Namespace, stem: str) -> str:
    suffix = f"_{args.shard_label}" if args.shard_label else ""
    return f"{stem}{suffix}.json"


def csv_filename(args: argparse.Namespace, stem: str) -> str:
    suffix = f"_{args.shard_label}" if args.shard_label else ""
    return f"{stem}{suffix}.csv"


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
