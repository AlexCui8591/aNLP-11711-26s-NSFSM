#!/usr/bin/env python3
"""Run NS-FSM over one or more tasks from a dataset adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from datetime import datetime
from typing import Any, Mapping

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from datasets.registry import get_adapter, registry_warning
from fsm_builder import FSMBuilder
from fsm_designer import LLMFSMDesigner
from nsfsm_agent import NSFSMAgent


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.dataset)
    warning = registry_warning(args.dataset, adapter)
    raw_tasks = select_tasks(adapter, args)
    if args.summary_only:
        print(json.dumps({"selected_tasks": len(raw_tasks)}, indent=2))
        return

    output_dir = os.path.join(ROOT, "results", "full", args.tag, adapter.dataset_name, "nsfsm")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    runtime_llm = None if args.planner_only else load_runtime_llm(args.config)

    for raw_task in raw_tasks:
        task_spec = adapter.to_task_spec(adapter.load_or_wrap(raw_task)).to_dict()
        if args.task_type:
            task_spec["task_type"] = args.task_type
        for run_idx in range(1, args.runs + 1):
            output_path = os.path.join(
                output_dir,
                f"{safe_name(task_spec['task_id'])}_run{run_idx:02d}.json",
            )
            if args.resume and os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue

            fsm, fsm_metadata = build_runtime_fsm(
                task_spec=task_spec,
                adapter=adapter,
                use_fixed_generic_fsm=args.use_fixed_generic_fsm,
                use_llm_fsm_design=args.use_llm_fsm_design,
                config_path=args.config,
            )
            result = NSFSMAgent(
                task_spec=task_spec,
                adapter=adapter,
                fsm=fsm,
                llm=runtime_llm,
                planner_only=args.planner_only,
                verbose=not args.quiet,
            ).run_episode()
            result["run_id"] = run_idx
            result["metadata"]["fsm_build"] = fsm_metadata
            if warning:
                result["metadata"]["dataset_warning"] = warning
            if not args.save_fsm_design:
                result["metadata"].get("fsm", {}).pop("transitions", None)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            results.append(result)
            if not args.quiet:
                print(
                    f"[run_nsfsm_experiment] {task_spec['task_id']} run={run_idx} "
                    f"success={result['success']} steps={result['total_steps']}"
                )

    summary = summarize(results, adapter.dataset_name)
    summary_path = os.path.join(output_dir, f"{adapter.dataset_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    combined_dir = os.path.join(ROOT, "results", "full", args.tag)
    os.makedirs(combined_dir, exist_ok=True)
    combined_path = os.path.join(combined_dir, "combined_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="generic")
    parser.add_argument("--task-type", default="")
    parser.add_argument("--task-ids", default="", help="Comma-separated task IDs.")
    parser.add_argument("--groups", default="", help="Comma-separated groups when supported.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to hyperparams YAML for LLM FSM design/runtime calls.",
    )
    parser.add_argument("--use-fixed-generic-fsm", action="store_true")
    parser.add_argument(
        "--use-llm-fsm-design",
        action="store_true",
        help=(
            "Use the LLM FSM designer even for datasets with grounded FSM rules. "
            "By default, Minecraft uses adapter-grounded dependency FSMs."
        ),
    )
    parser.add_argument("--save-fsm-design", action="store_true")
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument(
        "--instruction",
        default="Create a short plan for evaluating a model on a QA dataset.",
        help="Default generic instruction when dataset has no task list.",
    )
    return parser.parse_args()


def select_tasks(adapter: Any, args: argparse.Namespace) -> list[dict[str, Any] | str]:
    task_ids = {item.strip() for item in args.task_ids.split(",") if item.strip()}
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    tasks = adapter.list_tasks()

    if not tasks:
        if task_ids:
            return list(task_ids)
        return [args.instruction]

    selected = []
    for task in tasks:
        if task_ids and task.get("task_id") not in task_ids and task.get("goal") not in task_ids:
            continue
        if groups and task.get("group") not in groups:
            continue
        selected.append(task)
    if not selected and (task_ids or groups):
        raise ValueError(
            "No tasks matched the requested filters. "
            f"task_ids={sorted(task_ids) or 'ALL'} groups={sorted(groups) or 'ALL'}"
        )
    return selected


def build_runtime_fsm(
    task_spec: Mapping[str, Any],
    adapter: Any,
    use_fixed_generic_fsm: bool,
    use_llm_fsm_design: bool,
    config_path: str | None,
):
    builder = FSMBuilder()
    metadata: dict[str, Any] = {
        "source": "template" if use_fixed_generic_fsm else "llm_designer",
        "llm_fsm_error": None,
    }
    if use_fixed_generic_fsm:
        return builder.from_template(task_spec, adapter), metadata

    if _use_minecraft_grounded_rules(task_spec, adapter, use_llm_fsm_design):
        fsm = builder.from_template(task_spec, adapter)
        sequence = list(task_spec.get("metadata", {}).get("required_actions") or [])
        _assert_grounded_minecraft_fsm(fsm, sequence, task_spec)
        metadata.update(
            {
                "source": "minecraft_grounded_rules",
                "rule": "branching_dependency_dag_with_runtime_executable_filter",
                "required_action_count": len(sequence),
                "llm_fsm_error": None,
            }
        )
        return fsm, metadata

    try:
        proposal = LLMFSMDesigner(config_path=config_path).design_with_metadata(task_spec, adapter)
        fsm = builder.from_design(proposal.fsm_design, task_spec, adapter, allow_fallback=False)
        metadata.update(
            {
                "task_hash": proposal.task_hash,
                "cache_path": proposal.cache_path,
                "fallback_used": getattr(fsm, "validation", {}).get("fallback_used"),
            }
        )
        return fsm, metadata
    except Exception as exc:
        metadata["llm_fsm_error"] = str(exc)
        raise RuntimeError(
            "LLM FSM design failed validation and fallback FSM replacement is disabled."
        ) from exc


def _use_minecraft_grounded_rules(
    task_spec: Mapping[str, Any],
    adapter: Any,
    use_llm_fsm_design: bool,
) -> bool:
    return (
        not use_llm_fsm_design
        and str(task_spec.get("dataset")) == "minecraft"
        and getattr(adapter, "dataset_name", "") == "minecraft"
    )


def _assert_grounded_minecraft_fsm(
    fsm: Any,
    sequence: list[str],
    task_spec: Mapping[str, Any],
) -> None:
    if not sequence:
        raise ValueError(
            f"Minecraft grounded FSM requires a dependency action sequence for "
            f"{task_spec.get('task_id')}."
        )
    transitions = fsm.to_dict().get("transitions", [])
    transition_actions = [item.get("action") for item in transitions]
    sequence_set = set(sequence)
    transition_action_set = set(transition_actions)
    if not sequence_set.issubset(transition_action_set):
        raise ValueError(
            "Minecraft grounded FSM is missing dependency actions for "
            f"{task_spec.get('task_id')}: missing={sorted(sequence_set - transition_action_set)}"
        )
    extra_actions = transition_action_set - sequence_set
    if extra_actions:
        raise ValueError(
            "Minecraft grounded FSM contains actions outside the dependency "
            f"closure for {task_spec.get('task_id')}: {sorted(extra_actions)}"
        )
    final_action = sequence[-1]
    if not any(
        item.get("action") == final_action and item.get("next_state") in fsm.terminal_states
        for item in transitions
    ):
        raise ValueError(
            f"Minecraft grounded FSM final action does not reach DONE: {final_action}"
        )
    for state in fsm.to_dict().get("states", []):
        state_name = state.get("name")
        if state.get("terminal"):
            continue
        if not fsm.get_valid_transitions(state_name):
            raise ValueError(
                f"Minecraft grounded FSM state has no legal transitions: {state_name}"
            )


def load_runtime_llm(config_path: str | None):
    try:
        from llm_interface import LLMInterface
    except Exception as exc:
        raise RuntimeError(
            "LLMInterface could not be imported. Install optional LLM "
            "dependencies or run with --planner-only."
        ) from exc
    return LLMInterface(config_path)


def summarize(results: list[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for result in results if result.get("success"))
    total_steps = sum(int(result.get("total_steps", 0)) for result in results)
    blocked = sum(int(result.get("blocked_action_count", 0)) for result in results)
    fallback = sum(int(result.get("fallback_action_count", 0)) for result in results)
    goal_stats = _goal_stats(results)
    group_stats = _group_stats(goal_stats)
    return {
        "agent": "nsfsm",
        "dataset": dataset,
        "total_goals": len(goal_stats),
        "total_runs": total,
        "total_success": successes,
        "overall_success_rate": round(successes / total, 3) if total else 0.0,
        "group_stats": group_stats,
        "goal_stats": goal_stats,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "avg_steps": total_steps / total if total else 0.0,
        "blocked_action_count": blocked,
        "fallback_action_count": fallback,
        "results": [
            {
                "task_id": result.get("task_id"),
                "run_id": result.get("run_id"),
                "success": result.get("success"),
                "termination": result.get("termination"),
                "total_steps": result.get("total_steps"),
            }
            for result in results
        ],
    }


def _goal_stats(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        task_id = str(result.get("task_id") or "unknown")
        grouped.setdefault(task_id, []).append(result)

    rows = []
    for task_id, items in grouped.items():
        first = items[0]
        task_spec = first.get("metadata", {}).get("task_spec", {})
        metadata = task_spec.get("metadata", {}) if isinstance(task_spec, Mapping) else {}
        goal = str(metadata.get("goal_item") or task_id.split("/", 1)[-1])
        group = str(metadata.get("group") or "custom")
        n_runs = len(items)
        n_success = sum(1 for item in items if item.get("success"))
        step_values = [int(item.get("total_steps", 0)) for item in items]
        termination_distribution = Counter(
            str(item.get("termination") or "unknown") for item in items
        )
        error_distribution = Counter(
            _error_type(item) for item in items if not item.get("success")
        )
        rows.append(
            {
                "goal": goal,
                "group": group,
                "n_runs": n_runs,
                "n_success": n_success,
                "success_rate": round(n_success / n_runs, 3) if n_runs else 0.0,
                "avg_steps": round(sum(step_values) / n_runs, 1) if n_runs else 0.0,
                "blocked_action_count": sum(
                    int(item.get("blocked_action_count", 0)) for item in items
                ),
                "fallback_action_count": sum(
                    int(item.get("fallback_action_count", 0)) for item in items
                ),
                "error_distribution": dict(error_distribution),
                "termination_distribution": dict(termination_distribution),
            }
        )

    return sorted(rows, key=lambda row: (_group_order(row["group"]), row["goal"]))


def _group_stats(goal_stats: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in goal_stats:
        group = str(row.get("group") or "custom")
        bucket = grouped.setdefault(
            group,
            {
                "goals": 0,
                "total_runs": 0,
                "total_success": 0,
                "blocked_action_count": 0,
                "fallback_action_count": 0,
            },
        )
        bucket["goals"] += 1
        bucket["total_runs"] += int(row.get("n_runs", 0))
        bucket["total_success"] += int(row.get("n_success", 0))
        bucket["blocked_action_count"] += int(row.get("blocked_action_count", 0))
        bucket["fallback_action_count"] += int(row.get("fallback_action_count", 0))

    ordered = {}
    for group in sorted(grouped, key=_group_order):
        bucket = grouped[group]
        total_runs = bucket["total_runs"]
        bucket["success_rate"] = round(bucket["total_success"] / total_runs, 3) if total_runs else 0.0
        ordered[group] = bucket
    return ordered


def _error_type(result: Mapping[str, Any]) -> str:
    termination = str(result.get("termination") or "unknown")
    if termination == "no_valid_action":
        return "fsm_no_executable_action"
    if int(result.get("blocked_action_count", 0)) > 0:
        return "blocked_by_verifier"
    if termination == "max_steps":
        return "max_steps"
    return termination


def _group_order(group: str) -> tuple[int, str]:
    order = {
        "Wooden": 0,
        "Stone": 1,
        "Iron": 2,
        "Golden": 3,
        "Diamond": 4,
        "Redstone": 5,
        "Armor": 6,
    }
    return order.get(group, 99), group


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


if __name__ == "__main__":
    main()
