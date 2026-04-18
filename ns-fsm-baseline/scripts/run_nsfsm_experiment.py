#!/usr/bin/env python3
"""Run NS-FSM over one or more tasks from a dataset adapter."""

from __future__ import annotations

import argparse
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
            )
            result = NSFSMAgent(
                task_spec=task_spec,
                adapter=adapter,
                fsm=fsm,
                llm=None,
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
    parser.add_argument("--use-fixed-generic-fsm", action="store_true")
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
    return selected or tasks[:1]


def build_runtime_fsm(
    task_spec: Mapping[str, Any],
    adapter: Any,
    use_fixed_generic_fsm: bool,
):
    builder = FSMBuilder()
    metadata: dict[str, Any] = {
        "source": "template" if use_fixed_generic_fsm else "llm_designer",
        "llm_fsm_error": None,
    }
    if use_fixed_generic_fsm:
        return builder.from_template(task_spec, adapter), metadata

    try:
        proposal = LLMFSMDesigner().design_with_metadata(task_spec, adapter)
        fsm = builder.from_design(proposal.fsm_design, task_spec, adapter, allow_fallback=True)
        metadata.update(
            {
                "task_hash": proposal.task_hash,
                "cache_path": proposal.cache_path,
                "fallback_used": getattr(fsm, "validation", {}).get("fallback_used"),
            }
        )
        return fsm, metadata
    except Exception as exc:
        metadata["source"] = "template_after_llm_error"
        metadata["llm_fsm_error"] = str(exc)
        return builder.from_template(task_spec, adapter), metadata


def summarize(results: list[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for result in results if result.get("success"))
    total_steps = sum(int(result.get("total_steps", 0)) for result in results)
    blocked = sum(int(result.get("blocked_action_count", 0)) for result in results)
    fallback = sum(int(result.get("fallback_action_count", 0)) for result in results)
    return {
        "dataset": dataset,
        "total_runs": total,
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


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


if __name__ == "__main__":
    main()
