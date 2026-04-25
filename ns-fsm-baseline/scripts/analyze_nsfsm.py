#!/usr/bin/env python3
"""Analyze NS-FSM result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Mapping

ROOT = os.path.dirname(os.path.dirname(__file__))


def main() -> None:
    args = parse_args()
    result_root = args.results_dir or os.path.join(ROOT, "results", "full", args.tag)
    output_dir = os.path.join(ROOT, "results", "analysis", args.tag)
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(result_root)
    write_summary_by(results, "dataset", os.path.join(output_dir, "summary_by_dataset.csv"))
    write_summary_by(results, "task_type", os.path.join(output_dir, "summary_by_task_type.csv"))
    write_summary_by(results, "task_id", os.path.join(output_dir, "summary_by_task.csv"))
    write_events(results, "blocked", os.path.join(output_dir, "blocked_actions.csv"))
    write_events(results, "fallback", os.path.join(output_dir, "fallback_actions.csv"))
    write_fsm_validation(results, os.path.join(output_dir, "fsm_validation.csv"))
    write_datalog_violations(results, os.path.join(output_dir, "datalog_violations.csv"))
    write_report(results, os.path.join(output_dir, "combined_report.md"))

    print(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-dir", default="")
    return parser.parse_args()


def load_results(root: str) -> list[dict[str, Any]]:
    results = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if is_result(payload):
                payload["_path"] = path
                results.append(payload)
    return results


def is_result(payload: Mapping[str, Any]) -> bool:
    required = {"dataset", "task_id", "success", "trajectory"}
    return required.issubset(payload.keys())


def write_summary_by(results: list[Mapping[str, Any]], key: str, path: str) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result.get(key, ""))].append(result)

    rows = []
    for value, items in sorted(groups.items()):
        total = len(items)
        successes = sum(1 for item in items if item.get("success"))
        steps = sum(int(item.get("total_steps", 0)) for item in items)
        simulator_steps = sum(simulator_step_count(item) for item in items)
        blocked = sum(int(item.get("blocked_action_count", 0)) for item in items)
        fallback = sum(int(item.get("fallback_action_count", 0)) for item in items)
        rows.append(
            {
                key: value,
                "runs": total,
                "successes": successes,
                "success_rate": successes / total if total else 0,
                "avg_steps": steps / total if total else 0,
                "avg_simulator_steps": simulator_steps / total if total else 0,
                "blocked_action_rate": blocked / steps if steps else 0,
                "fallback_usage_rate": fallback / steps if steps else 0,
            }
        )
    write_csv(
        path,
        rows,
        [
            key,
            "runs",
            "successes",
            "success_rate",
            "avg_steps",
            "avg_simulator_steps",
            "blocked_action_rate",
            "fallback_usage_rate",
        ],
    )


def write_events(results: list[Mapping[str, Any]], event_type: str, path: str) -> None:
    rows = []
    key = "blocked_action_history" if event_type == "blocked" else "fallback_action_history"
    for result in results:
        fsm = result.get("metadata", {}).get("fsm", {})
        events = fsm.get(key, [])
        for event in events:
            rows.append(
                {
                    "dataset": result.get("dataset"),
                    "task_id": result.get("task_id"),
                    "run_id": result.get("run_id", ""),
                    "state": event.get("state"),
                    "action": event.get("action"),
                    "next_state": event.get("next_state", ""),
                    "reason": event.get("reason", ""),
                }
            )
    write_csv(path, rows, ["dataset", "task_id", "run_id", "state", "action", "next_state", "reason"])


def write_fsm_validation(results: list[Mapping[str, Any]], path: str) -> None:
    rows = []
    for result in results:
        build = result.get("metadata", {}).get("fsm_build", {})
        gt_generation = build.get("ground_truth_generation", {})
        rows.append(
            {
                "dataset": result.get("dataset"),
                "task_id": result.get("task_id"),
                "run_id": result.get("run_id", ""),
                "source": build.get("source", ""),
                "fallback_used": build.get("fallback_used", ""),
                "ground_truth_status": gt_generation.get("status", ""),
                "ground_truth_fallback_used": gt_generation.get("fallback_used", ""),
                "ground_truth_fallback_source": gt_generation.get("fallback_source", ""),
                "llm_fsm_error": build.get("llm_fsm_error", ""),
                "task_hash": build.get("task_hash", ""),
            }
        )
    write_csv(
        path,
        rows,
        [
            "dataset",
            "task_id",
            "run_id",
            "source",
            "fallback_used",
            "ground_truth_status",
            "ground_truth_fallback_used",
            "ground_truth_fallback_source",
            "llm_fsm_error",
            "task_hash",
        ],
    )


def write_datalog_violations(results: list[Mapping[str, Any]], path: str) -> None:
    rows = []
    for result in results:
        validation = result.get("metadata", {}).get("fsm", {}).get("verification", {})
        for item in validation.get("violations", []):
            rows.append(
                {
                    "dataset": result.get("dataset"),
                    "task_id": result.get("task_id"),
                    "run_id": result.get("run_id", ""),
                    "type": item.get("type"),
                    "message": item.get("message"),
                }
            )
    write_csv(path, rows, ["dataset", "task_id", "run_id", "type", "message"])


def write_report(results: list[Mapping[str, Any]], path: str) -> None:
    total = len(results)
    successes = sum(1 for result in results if result.get("success"))
    terminations = Counter(str(result.get("termination")) for result in results)
    blocked = sum(int(result.get("blocked_action_count", 0)) for result in results)
    fallback = sum(int(result.get("fallback_action_count", 0)) for result in results)
    steps = sum(int(result.get("total_steps", 0)) for result in results)
    simulator_steps = sum(simulator_step_count(result) for result in results)
    lines = [
        "# NS-FSM Analysis Report",
        "",
        f"- Total runs: {total}",
        f"- Successes: {successes}",
        f"- Success rate: {successes / total if total else 0:.3f}",
        f"- Average steps: {steps / total if total else 0:.3f}",
        f"- Average simulator steps: {simulator_steps / total if total else 0:.3f}",
        f"- Blocked action rate: {blocked / steps if steps else 0:.3f}",
        f"- Fallback usage rate: {fallback / steps if steps else 0:.3f}",
        "",
        "## Termination Distribution",
        "",
    ]
    for name, count in sorted(terminations.items()):
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for result in results[:20]:
        lines.append(f"- {result.get('task_id')} -> {result.get('_path')}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path: str, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def simulator_step_count(result: Mapping[str, Any]) -> int:
    return int(result.get("metadata", {}).get("adapter_summary", {}).get("total_steps", 0))


if __name__ == "__main__":
    main()
