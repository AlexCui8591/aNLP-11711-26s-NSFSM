#!/usr/bin/env python3
"""Run and audit the MC-TextWorld NS-FSM smoke checklist.

Default checklist tasks:
  stick, crafting_table, wooden_pickaxe, torch, iron_pickaxe

By default this script uses only config/mctextworld_ground_truth_buildable_tasks.json.
If a checklist task is absent from that cleaned task set, it is reported as
missing rather than silently falling back to goals_67.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.minecraft import MinecraftAdapter
from nsfsm_agent import NSFSMAgent
from run_nsfsm_experiment import build_llm, build_runtime_fsm, safe_name, summarize


DEFAULT_TASKS = ["stick", "crafting_table", "wooden_pickaxe", "torch", "iron_pickaxe"]
CSV_FIELDS = [
    "task_id",
    "run_id",
    "mode",
    "status",
    "success",
    "termination",
    "total_steps",
    "llm_really_proposed_action",
    "llm_action_count",
    "llm_valid_decision_count",
    "planner_action_count",
    "blocked_action_count",
    "fallback_action_count",
    "fsm_legal_actions_ok",
    "datalog_illegal_action_blocked",
    "fallback_advances",
    "fsm_seeded_by_required_actions",
    "direct_oracle_execution",
    "result_not_direct_oracle_plan",
    "task_source",
    "required_action_count",
    "executed_actions",
    "missing_reason",
]


def main() -> None:
    args = parse_args()
    if args.mode == "llm":
        args.use_llm = True
        args.planner_only = False
    else:
        args.use_llm = False
        args.planner_only = True

    output_dir = ROOT / "results" / "full" / args.tag / "minecraft" / "nsfsm"
    analysis_dir = ROOT / "results" / "analysis" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    missing_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    if args.report_only:
        results = load_existing_results(output_dir)
    else:
        results, missing_rows = run_checklist(args, output_dir)
        write_summary(results, output_dir)

    audit_rows = [audit_result(result) for result in results]
    audit_rows.extend(missing_rows)
    csv_path = analysis_dir / "smoke_checklist_audit.csv"
    report_path = analysis_dir / "smoke_checklist_report.md"
    write_audit_csv(csv_path, audit_rows)
    write_report(report_path, audit_rows, args)

    print(f"results: {output_dir}")
    print(f"audit:   {csv_path}")
    print(f"report:  {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["llm", "planner"],
        default="llm",
        help="llm checks model proposals; planner is a no-LLM pipeline sanity check.",
    )
    parser.add_argument(
        "--task-ids",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated checklist task ids.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tag", default=f"smoke_checklist_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--llm-config", default="", help="Optional YAML config path for LLMInterface.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use-llm-fsm-designer",
        action="store_true",
        help="Use the LLM FSM designer. Default uses the symbolic planning template.",
    )
    parser.add_argument(
        "--allow-synthetic-goals",
        action="store_true",
        help=(
            "Allow tasks absent from the cleaned buildable JSON to be loaded from "
            "mctextworld_task_discovery_report.json synthetic buildable records."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip execution and regenerate audit/report for an existing tag.",
    )
    return parser.parse_args()


def run_checklist(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = MinecraftAdapter(task_source="buildable")
    llm = build_llm(args)
    requested = parse_task_ids(args.task_ids)
    raw_tasks, missing_rows = resolve_tasks(adapter, requested, args.allow_synthetic_goals, args.mode)
    results: list[dict[str, Any]] = []

    for raw_task in raw_tasks:
        task_spec = adapter.to_task_spec(adapter.load_or_wrap(raw_task)).to_dict()
        for run_idx in range(1, args.runs + 1):
            output_path = output_dir / f"{safe_name(task_spec['task_id'])}_run{run_idx:02d}.json"
            if args.resume and output_path.exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue

            fsm, fsm_metadata = build_runtime_fsm(
                task_spec=task_spec,
                adapter=adapter,
                use_fixed_generic_fsm=not args.use_llm_fsm_designer,
                use_llm_fsm_design=args.use_llm_fsm_designer,
                llm_config=args.llm_config or None,
            )
            illegal_probe = probe_illegal_transition(fsm, task_spec)
            result = NSFSMAgent(
                task_spec=task_spec,
                adapter=adapter,
                fsm=fsm,
                llm=llm,
                planner_only=args.planner_only,
                verbose=not args.quiet,
            ).run_episode()
            result["run_id"] = run_idx
            result.setdefault("metadata", {})
            result["metadata"]["fsm_build"] = fsm_metadata
            result["metadata"]["smoke_checklist"] = {
                "mode": args.mode,
                "illegal_transition_probe": illegal_probe,
                "checklist_tasks": requested,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            results.append(result)

            if not args.quiet:
                print(
                    f"[smoke_checklist] {task_spec['task_id']} run={run_idx} "
                    f"success={result['success']} steps={result['total_steps']}"
                )

    return results, missing_rows


def parse_task_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_tasks(
    adapter: MinecraftAdapter,
    requested: list[str],
    allow_synthetic_goals: bool,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for task in adapter.list_tasks():
        task_id = str(task.get("task_id", ""))
        goal = str(task.get("goal", ""))
        by_id[task_id] = task
        by_id[goal] = task
        if task_id.startswith("minecraft/"):
            by_id[task_id.split("/", 1)[1]] = task

    tasks: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for task_id in requested:
        task = by_id.get(task_id) or by_id.get(f"minecraft/{task_id}")
        if task is None and allow_synthetic_goals:
            task = load_synthetic_buildable_task(task_id)
        if task is None:
            missing_rows.append(missing_audit_row(task_id, mode))
            continue
        tasks.append(task)
    return tasks, missing_rows


def load_synthetic_buildable_task(task_id: str) -> dict[str, Any] | None:
    report_path = ROOT / "config" / "mctextworld_task_discovery_report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    for record in report.get("goals_67_synthetic_records", []):
        if record.get("goal") != task_id or not record.get("dependency_graph_buildable"):
            continue
        actions = [str(action) for action in record.get("synthesized_actions", [])]
        return {
            "dataset": "minecraft",
            "task_id": f"minecraft/{task_id}",
            "goal": task_id,
            "goal_condition": {task_id: 1},
            "type": "dependency_graph_buildable_synthetic",
            "instruction": f"Obtain 1 {task_id}.",
            "group": record.get("group", "synthetic"),
            "max_steps": max(20, len(actions) + 5),
            "synthesized_actions": actions,
            "task_source": "synthetic_buildable_report",
        }
    return None


def missing_audit_row(task_id: str, mode: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "task_id": f"minecraft/{task_id}" if not task_id.startswith("minecraft/") else task_id,
            "run_id": "",
            "mode": mode,
            "status": "missing_from_buildable_set",
            "success": False,
            "missing_reason": (
                "Task is not present in mctextworld_ground_truth_buildable_tasks.json. "
                "Run with --allow-synthetic-goals to use synthetic buildable records."
            ),
        }
    )
    return row


def probe_illegal_transition(fsm: Any, task_spec: Mapping[str, Any]) -> dict[str, Any]:
    legal = set(fsm.get_valid_actions())
    candidates = [str(action) for action in task_spec.get("available_tools", [])]
    illegal_action = next((action for action in candidates if action not in legal), "__illegal_action__")
    check = fsm.verify_transition(illegal_action)
    return {
        "action": illegal_action,
        "blocked": not bool(check.get("valid", False)),
        "check": check,
    }


def audit_result(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(result.get("metadata", {}))
    task_spec = dict(metadata.get("task_spec", {}))
    spec_metadata = dict(task_spec.get("metadata", {}))
    smoke = dict(metadata.get("smoke_checklist", {}))
    probe = dict(smoke.get("illegal_transition_probe", {}))
    trajectory = list(result.get("trajectory", []))
    required_actions = [str(action) for action in spec_metadata.get("required_actions", [])]
    executed_actions = [str(step.get("action", "")) for step in trajectory if step.get("action")]

    llm_action_count = sum(
        1
        for step in trajectory
        if dict(step.get("proposal", {})).get("source") == "llm"
        and dict(step.get("proposal", {})).get("action")
    )
    llm_valid_decision_count = sum(1 for step in trajectory if step.get("decision_source") == "llm")
    planner_action_count = sum(1 for step in trajectory if step.get("decision_source") == "planner")
    fsm_legal_actions_ok = all(
        bool(dict(step.get("transition_check", {})).get("valid", False)) for step in trajectory
    )
    planner_steps = [step for step in trajectory if step.get("decision_source") == "planner"]
    fallback_advances = all(bool(step.get("success", False)) for step in planner_steps)
    direct_oracle_execution = bool(
        executed_actions
        and executed_actions == required_actions[: len(executed_actions)]
        and planner_action_count == len(executed_actions)
    )

    return {
        "task_id": result.get("task_id", ""),
        "run_id": result.get("run_id", ""),
        "mode": smoke.get("mode", "unknown"),
        "status": "completed",
        "success": bool(result.get("success", False)),
        "termination": result.get("termination", ""),
        "total_steps": int(result.get("total_steps", 0)),
        "llm_really_proposed_action": llm_action_count > 0,
        "llm_action_count": llm_action_count,
        "llm_valid_decision_count": llm_valid_decision_count,
        "planner_action_count": planner_action_count,
        "blocked_action_count": int(result.get("blocked_action_count", 0)),
        "fallback_action_count": int(result.get("fallback_action_count", 0)),
        "fsm_legal_actions_ok": fsm_legal_actions_ok,
        "datalog_illegal_action_blocked": bool(probe.get("blocked", False)),
        "fallback_advances": fallback_advances,
        "fsm_seeded_by_required_actions": bool(required_actions),
        "direct_oracle_execution": direct_oracle_execution,
        "result_not_direct_oracle_plan": not direct_oracle_execution,
        "task_source": spec_metadata.get("task_source", ""),
        "required_action_count": len(required_actions),
        "executed_actions": " ".join(executed_actions),
        "missing_reason": "",
    }


def load_existing_results(output_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(output_dir.glob("*_run*.json")):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def write_summary(results: list[Mapping[str, Any]], output_dir: Path) -> None:
    summary = summarize(results, "minecraft")
    summary_path = output_dir / "minecraft_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    combined_dir = output_dir.parents[1]
    combined_path = combined_dir / "combined_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def write_audit_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_report(path: Path, rows: list[Mapping[str, Any]], args: argparse.Namespace) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    missing = [row for row in rows if row.get("status") == "missing_from_buildable_set"]
    llm_check = (
        all_bool(completed, "llm_really_proposed_action") if args.mode == "llm" else "SKIP"
    )
    checks = [
        ("All requested tasks resolved", not missing),
        ("LLM proposed actions", llm_check),
        ("FSM legal actions accepted", all_bool(completed, "fsm_legal_actions_ok")),
        ("Datalog blocks injected illegal action", all_bool(completed, "datalog_illegal_action_blocked")),
        ("Fallback advances when used", all_bool(completed, "fallback_advances")),
        ("No direct oracle execution", all_bool(completed, "result_not_direct_oracle_plan")),
    ]

    lines = [
        "# MC-TextWorld Smoke Checklist Report",
        "",
        f"- mode: `{args.mode}`",
        f"- runs per task: `{args.runs}`",
        f"- requested tasks: `{', '.join(parse_task_ids(args.task_ids))}`",
        f"- completed rows: `{len(completed)}`",
        f"- missing rows: `{len(missing)}`",
        "",
        "## Checklist",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for label, passed in checks:
        lines.append(f"| {label} | `{passed}` |")

    lines.extend(
        [
            "",
            "## Per Task",
            "",
            "| task | status | success | steps | llm actions | planner actions | blocked | fallback | direct oracle | source |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {task} | {status} | `{success}` | {steps} | {llm} | {planner} | "
            "{blocked} | {fallback} | `{oracle}` | {source} |".format(
                task=row.get("task_id", ""),
                status=row.get("status", ""),
                success=row.get("success", ""),
                steps=row.get("total_steps", ""),
                llm=row.get("llm_action_count", ""),
                planner=row.get("planner_action_count", ""),
                blocked=row.get("blocked_action_count", ""),
                fallback=row.get("fallback_action_count", ""),
                oracle=row.get("direct_oracle_execution", ""),
                source=row.get("task_source", ""),
            )
        )
        if row.get("missing_reason"):
            lines.append(f"| {row.get('task_id', '')} reason | {row.get('missing_reason')} |  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `direct_oracle_execution=True` means every executed action came from planner fallback and matched the required action path.",
            "- `fsm_seeded_by_required_actions=True` is expected for the current symbolic planning template.",
            "- `datalog_illegal_action_blocked` is checked by injecting one invalid transition before the episode.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def all_bool(rows: list[Mapping[str, Any]], key: str) -> bool:
    return bool(rows) and all(bool(row.get(key, False)) for row in rows)


if __name__ == "__main__":
    main()
