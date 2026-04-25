#!/usr/bin/env python
"""Run the GPT-4o Reflexion API baseline on the 67 Minecraft goals.

This is a non-NS-FSM comparison runner. It deliberately does not import or use
NSFSMAgent, FSM templates, Datalog facts, synthesized actions, or buildable task
sources. The model only sees the environment observation/history through the
ReAct/Reflexion prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MCTEXTWORLD = ROOT.parent / "MC-TextWorld"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(MCTEXTWORLD))


DEFAULT_CONFIG_PATH = ROOT / "config" / "hyperparams_gpt4o_reflexion_api.yaml"
GOALS_PATH = ROOT / "config" / "goals_67.json"
RESULTS_BASE = ROOT / "results" / "full"
AGENT_NAME = "reflexion"
BASELINE_NAME = "gpt4o_reflexion_api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--tag", default="gpt4o_reflexion_goals67x3")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--goals", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_api_key(config: dict[str, Any]) -> None:
    llm_config = config.get("llm", {})
    key = (
        os.environ.get("NSFSM_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or llm_config.get("api_key")
    )
    if not key:
        raise SystemExit(
            "ERROR: missing API key. Set NSFSM_LLM_API_KEY or OPENAI_API_KEY "
            "before running the GPT-4o Reflexion baseline."
        )


def load_goals(
    groups_filter: set[str] | None = None,
    goals_filter: set[str] | None = None,
) -> list[dict[str, str]]:
    with GOALS_PATH.open("r", encoding="utf-8") as f:
        grouped = json.load(f)

    goals: list[dict[str, str]] = []
    for group_name, group_data in grouped.items():
        if groups_filter and group_name not in groups_filter:
            continue
        for entry in group_data.get("goals", []):
            goal = str(entry["goal"])
            if goals_filter and goal not in goals_filter:
                continue
            goals.append(
                {
                    "goal": goal,
                    "group": str(group_name),
                    "type": str(entry.get("type", "")),
                    "instruction": str(entry.get("instruction", f"Obtain 1 {goal}.")),
                }
            )
    return goals


def select_goals(goals: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.max_tasks and args.max_tasks > 0:
        goals = goals[: args.max_tasks]

    if args.task_shard_count < 1:
        raise SystemExit("--task-shard-count must be >= 1")
    if not 0 <= args.task_shard_index < args.task_shard_count:
        raise SystemExit("--task-shard-index must be in [0, task_shard_count)")

    if args.task_shard_count == 1:
        return goals
    return [
        goal
        for index, goal in enumerate(goals)
        if index % args.task_shard_count == args.task_shard_index
    ]


def result_path(out_dir: Path, goal: str, run_id: int) -> Path:
    return out_dir / AGENT_NAME / f"{goal}_run{run_id:02d}.json"


def find_completed(out_dir: Path, goals: list[dict[str, str]], runs: int) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for goal_entry in goals:
        goal = goal_entry["goal"]
        for run_id in range(1, runs + 1):
            path = result_path(out_dir, goal, run_id)
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                continue
            if "success" in payload and "attempts" in payload:
                completed.add((goal, run_id))
    return completed


def build_plan(
    goals: list[dict[str, str]],
    runs: int,
    completed: set[tuple[str, int]],
) -> list[tuple[dict[str, str], int]]:
    return [
        (goal_entry, run_id)
        for goal_entry in goals
        for run_id in range(1, runs + 1)
        if (goal_entry["goal"], run_id) not in completed
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sanitize_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = dict(config.get("llm", {}))
    if llm_config.get("api_key"):
        llm_config["api_key"] = "***"
    return llm_config


def load_results(out_dir: Path, goals: list[dict[str, str]], runs: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for goal_entry in goals:
        goal = goal_entry["goal"]
        for run_id in range(1, runs + 1):
            path = result_path(out_dir, goal, run_id)
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
    return results


def summarize_results(results: list[dict[str, Any]], planned_runs: int) -> dict[str, Any]:
    successes = sum(1 for result in results if result.get("success"))
    attempts = [int(result.get("total_attempts") or len(result.get("attempts", []))) for result in results]
    first_attempt_failures = 0
    recovered_after_reflection = 0
    total_steps = 0

    group_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "successes": 0,
            "attempts": 0,
            "first_attempt_failures": 0,
            "recovered_after_reflection": 0,
            "total_steps": 0,
        }
    )

    for result in results:
        group = str(result.get("group", "unknown"))
        result_attempts = result.get("attempts", [])
        first_attempt_success = bool(
            result_attempts
            and result_attempts[0].get("result", {}).get("success")
        )
        recovered = bool(result.get("success")) and int(result.get("winning_attempt") or 0) > 1
        attempt_count = int(result.get("total_attempts") or len(result_attempts))
        step_count = sum(
            int(attempt.get("result", {}).get("total_steps", 0))
            for attempt in result_attempts
        )

        if result_attempts and not first_attempt_success:
            first_attempt_failures += 1
        if recovered:
            recovered_after_reflection += 1
        total_steps += step_count

        group_bucket = group_stats[group]
        group_bucket["runs"] += 1
        group_bucket["successes"] += int(bool(result.get("success")))
        group_bucket["attempts"] += attempt_count
        group_bucket["first_attempt_failures"] += int(bool(result_attempts and not first_attempt_success))
        group_bucket["recovered_after_reflection"] += int(recovered)
        group_bucket["total_steps"] += step_count

    by_group: dict[str, dict[str, Any]] = {}
    for group, stats in sorted(group_stats.items()):
        runs = stats["runs"]
        first_failures = stats["first_attempt_failures"]
        by_group[group] = {
            "runs": runs,
            "successes": stats["successes"],
            "success_rate": stats["successes"] / runs if runs else 0.0,
            "avg_attempts": stats["attempts"] / runs if runs else 0.0,
            "avg_total_steps": stats["total_steps"] / runs if runs else 0.0,
            "first_attempt_failures": first_failures,
            "recovered_after_reflection": stats["recovered_after_reflection"],
            "reflexion_recovery_rate": (
                stats["recovered_after_reflection"] / first_failures
                if first_failures
                else 0.0
            ),
        }

    completed_runs = len(results)
    return {
        "baseline": BASELINE_NAME,
        "agent": AGENT_NAME,
        "planned_runs": planned_runs,
        "completed_runs": completed_runs,
        "missing_runs": max(planned_runs - completed_runs, 0),
        "successes": successes,
        "success_rate": successes / completed_runs if completed_runs else 0.0,
        "avg_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
        "avg_total_steps": total_steps / completed_runs if completed_runs else 0.0,
        "first_attempt_failures": first_attempt_failures,
        "recovered_after_reflection": recovered_after_reflection,
        "reflexion_recovery_rate": (
            recovered_after_reflection / first_attempt_failures
            if first_attempt_failures
            else 0.0
        ),
        "by_group": by_group,
        "generated_at": datetime.now().isoformat(),
    }


def write_experiment_config(
    out_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    goals: list[dict[str, str]],
    runs: int,
) -> None:
    write_json(
        out_dir / "experiment_config.json",
        {
            "timestamp": datetime.now().isoformat(),
            "baseline": BASELINE_NAME,
            "agent": AGENT_NAME,
            "tag": args.tag,
            "num_goals": len(goals),
            "num_runs": runs,
            "planned_runs": len(goals) * runs,
            "goals": [goal["goal"] for goal in goals],
            "groups": sorted({goal["group"] for goal in goals}),
            "config_path": str(Path(args.config)),
            "llm_config": sanitize_llm_config(config),
            "uses_nsfsm": False,
            "uses_fsm": False,
            "uses_synthesized_actions": False,
            "task_shard_index": args.task_shard_index,
            "task_shard_count": args.task_shard_count,
        },
    )


def run_plan(
    plan: list[tuple[dict[str, str], int]],
    out_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    quiet: bool,
) -> None:
    from reflexion_agent import ReflexionAgent

    verbose = bool(config.get("logging", {}).get("verbose", False)) and not quiet
    agent = ReflexionAgent(config_path=str(config_path), verbose=verbose)
    model_name = str(config.get("llm", {}).get("model_name", ""))

    timings: list[float] = []
    for index, (goal_entry, run_id) in enumerate(plan, start=1):
        goal = goal_entry["goal"]
        started = time.time()
        if timings:
            avg = sum(timings) / len(timings)
            remaining = (len(plan) - index + 1) * avg
            eta = f"{remaining / 3600:.1f}h" if remaining >= 3600 else f"{remaining / 60:.0f}m"
        else:
            eta = "unknown"
        print(
            f"[{index}/{len(plan)}] {goal} run {run_id} "
            f"group={goal_entry['group']} eta={eta}"
        )

        try:
            result = agent.run(goal)
        except Exception as exc:
            result = {
                "goal": goal,
                "success": False,
                "winning_attempt": None,
                "total_attempts": 0,
                "attempts": [],
                "termination": "crash",
                "error": str(exc),
            }

        elapsed = time.time() - started
        timings.append(elapsed)
        result.update(
            {
                "baseline": BASELINE_NAME,
                "agent": AGENT_NAME,
                "model_name": model_name,
                "run_id": run_id,
                "group": goal_entry["group"],
                "type": goal_entry["type"],
                "instruction": goal_entry["instruction"],
                "elapsed_sec": round(elapsed, 2),
                "uses_nsfsm": False,
                "uses_fsm": False,
                "uses_synthesized_actions": False,
            }
        )
        write_json(result_path(out_dir, goal, run_id), result)

        status = "OK" if result.get("success") else "FAIL"
        print(
            f"  -> {status} attempts={result.get('total_attempts')} "
            f"winning={result.get('winning_attempt') or 'none'} "
            f"elapsed={elapsed:.1f}s"
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    runs = int(args.runs or config.get("env", {}).get("num_runs", 3))

    groups_filter = set(args.groups) if args.groups else None
    goals_filter = set(args.goals) if args.goals else None
    goals = load_goals(groups_filter=groups_filter, goals_filter=goals_filter)
    goals = select_goals(goals, args)
    if not goals:
        raise SystemExit("No goals matched the selected filters/shard.")

    out_dir = RESULTS_BASE / args.tag
    planned_runs = len(goals) * runs
    completed = find_completed(out_dir, goals, runs) if args.resume else set()
    plan = build_plan(goals, runs, completed)

    print("=" * 70)
    print("GPT-4o Reflexion API baseline")
    print(f"tag:              {args.tag}")
    print(f"goals:            {len(goals)}")
    print(f"runs per goal:    {runs}")
    print(f"planned runs:     {planned_runs}")
    print(f"completed runs:   {len(completed)}")
    print(f"remaining runs:   {len(plan)}")
    print(f"shard:            {args.task_shard_index}/{args.task_shard_count}")
    print(f"output:           {out_dir}")
    print("=" * 70)

    if not args.summary_only:
        require_api_key(config)

    write_experiment_config(out_dir, args, config, goals, runs)

    if not args.summary_only:
        if plan:
            run_plan(plan, out_dir, config, config_path, quiet=args.quiet)
        else:
            print("All selected runs are already complete.")

    results = load_results(out_dir, goals, runs)
    summary = summarize_results(results, planned_runs=planned_runs)
    write_json(out_dir / "reflexion_summary.json", summary)
    write_json(out_dir / "combined_summary.json", {"reflexion": summary})

    print("\nSummary")
    print(f"completed: {summary['completed_runs']}/{summary['planned_runs']}")
    print(f"success rate: {summary['success_rate']:.1%}")
    print(f"avg attempts: {summary['avg_attempts']:.2f}")
    print(f"recovery rate: {summary['reflexion_recovery_rate']:.1%}")
    print(f"saved: {out_dir / 'reflexion_summary.json'}")


if __name__ == "__main__":
    main()
