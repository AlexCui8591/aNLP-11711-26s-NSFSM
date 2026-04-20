#!/usr/bin/env python3
"""Discover MC-TextWorld tasks with executable ground truth.

This script separates two notions that are easy to conflate:

1. official_plan_replayable:
   tasks.json + plans.json compile to action_lib action IDs and replay
   successfully under action preconditions/effects.

2. dependency_graph_buildable:
   the task goal can be satisfied by a backward-chained, quantity-aware
   executable action sequence synthesized from action_lib alone.

The second category is useful when plans.json is missing a task or contains a
high-level action name that does not exactly match action_lib.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
MCTEXTWORLD_DIR = ROOT / "MC-TextWorld" / "mctextworld"
BASELINE_CONFIG = ROOT / "ns-fsm-baseline" / "config"
DEFAULT_ACTION_LIB = MCTEXTWORLD_DIR / "action_lib.json"
DEFAULT_PLANS = MCTEXTWORLD_DIR / "plans.json"
DEFAULT_TASKS = MCTEXTWORLD_DIR / "tasks.json"
DEFAULT_GOALS_67 = BASELINE_CONFIG / "goals_67.json"
DEFAULT_OUTPUT_DIR = BASELINE_CONFIG

ACTION_TYPE_ORDER = {
    "mine": 0,
    "smelt": 1,
    "craft": 2,
    "no_op": 9,
}


@dataclass
class ReplayResult:
    ok: bool
    reason: str
    trace: list[dict[str, Any]]
    final_inventory: dict[str, int]
    failed_step: int | None = None
    failed_action: str | None = None


@dataclass
class SynthesisResult:
    ok: bool
    reason: str
    actions: list[str]
    trace: list[dict[str, Any]]
    final_inventory: dict[str, int]


def main() -> int:
    args = parse_args()
    action_lib = load_json(args.action_lib)
    tasks = load_json(args.tasks)
    plans = load_json(args.plans)
    goals_67_entries = load_goals_67_entries(args.goals_67) if args.goals_67 else []
    goals_67 = {entry["goal"] for entry in goals_67_entries}

    task_records, duplicate_task_ids = normalize_tasks(tasks)
    producers = build_producer_index(action_lib)

    records = []
    for task in task_records:
        task_id = task["task_id"]
        plan_steps = plans.get(task_id)
        official = replay_official_plan(task, plan_steps, action_lib, args.max_plan_ticks)
        synthesized = synthesize_task(task, action_lib, producers, args.max_synth_actions)
        records.append(
            {
                "task_id": task_id,
                "group": task["group"],
                "description": task.get("description", ""),
                "goal": task["goal"],
                "initial_inventory": task["initial_inventory"],
                "in_goals_67": task_id in goals_67,
                "has_plan": isinstance(plan_steps, list),
                "official_plan_replayable": official.ok,
                "official_plan_reason": official.reason,
                "official_plan_trace_len": len(official.trace),
                "official_plan_failed_step": official.failed_step,
                "official_plan_failed_action": official.failed_action,
                "dependency_graph_buildable": synthesized.ok,
                "dependency_graph_reason": synthesized.reason,
                "synthesized_action_count": len(synthesized.actions),
                "synthesized_actions": synthesized.actions,
            }
        )

    goals_67_synthetic_records = []
    for entry in goals_67_entries:
        synthesized = synthesize_task(
            {"goal": {entry["goal"]: 1}, "initial_inventory": {}},
            action_lib,
            producers,
            args.max_synth_actions,
        )
        goals_67_synthetic_records.append(
            {
                "goal": entry["goal"],
                "group": entry["group"],
                "dependency_graph_buildable": synthesized.ok,
                "dependency_graph_reason": synthesized.reason,
                "synthesized_action_count": len(synthesized.actions),
                "synthesized_actions": synthesized.actions,
                "present_in_tasks_json": any(record["task_id"] == entry["goal"] for record in records),
            }
        )

    summary = build_summary(records, duplicate_task_ids, plans, goals_67_synthetic_records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": summary,
        "records": records,
        "official_plan_replayable_tasks": unique_task_ids(
            record for record in records if record["official_plan_replayable"]
        ),
        "dependency_graph_buildable_tasks": unique_task_ids(
            record for record in records if record["dependency_graph_buildable"]
        ),
        "goals_67_dependency_graph_buildable_tasks": unique_task_ids(
            record
            for record in records
            if record["in_goals_67"] and record["dependency_graph_buildable"]
        ),
        "goals_67_official_plan_replayable_tasks": unique_task_ids(
            record
            for record in records
            if record["in_goals_67"] and record["official_plan_replayable"]
        ),
        "goals_67_synthetic_records": goals_67_synthetic_records,
        "goals_67_synthetic_dependency_graph_buildable_tasks": sorted(
            record["goal"]
            for record in goals_67_synthetic_records
            if record["dependency_graph_buildable"]
        ),
    }

    report_path = output_dir / "mctextworld_task_discovery_report.json"
    markdown_path = output_dir / "mctextworld_task_discovery_report.md"
    subset_path = output_dir / "mctextworld_ground_truth_buildable_tasks.json"

    write_json(report_path, report)
    write_json(
        subset_path,
        {
            "source": "MC-TextWorld tasks.json + action_lib.json",
            "criterion": "dependency_graph_buildable",
            "tasks": compact_unique_task_records(
                record for record in records if record["dependency_graph_buildable"]
            ),
        },
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {subset_path}")
    print(f"Wrote {markdown_path}")
    print(
        "Summary: "
        f"tasks={summary['tasks_total']} "
        f"official_replayable_unique={summary['unique_official_plan_replayable']} "
        f"dependency_graph_buildable_unique={summary['unique_dependency_graph_buildable']} "
        f"goals67_buildable_unique={summary['unique_goals_67_dependency_graph_buildable']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-lib", type=Path, default=DEFAULT_ACTION_LIB)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--goals-67", type=Path, default=DEFAULT_GOALS_67)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-plan-ticks", type=int, default=2048)
    parser.add_argument("--max-synth-actions", type=int, default=512)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_goals_67_entries(path: Path) -> list[dict[str, str]]:
    payload = load_json(path)
    entries = []
    for group_name, group_payload in payload.items():
        for entry in group_payload.get("goals", []):
            entries.append(
                {
                    "group": group_name,
                    "goal": str(entry.get("goal", "")),
                }
            )
    return entries


def normalize_tasks(tasks: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    seen = Counter()
    duplicates = []
    for task in tasks:
        task_id = str(task.get("task", "")).strip()
        if not task_id:
            continue
        seen[task_id] += 1
        if seen[task_id] > 1:
            duplicates.append(task_id)

        env = task.get("env", {}) or {}
        goal = {str(k): int(v) for k, v in (task.get("task_obj", {}) or {}).items()}
        records.append(
            {
                "task_id": task_id,
                "group": str(task.get("group", "unknown")),
                "description": str(task.get("description", "")),
                "initial_inventory": normalize_init_inventory(env.get("init_inventory", {}) or {}),
                "goal": goal,
            }
        )
    return records, sorted(set(duplicates))


def normalize_init_inventory(raw_inventory: Mapping[str, Any]) -> dict[str, int]:
    inventory: dict[str, int] = defaultdict(int)
    for key, value in raw_inventory.items():
        if isinstance(value, Mapping) and "type" in value:
            item = str(value["type"])
            qty = int(value.get("quantity", 1))
        else:
            item = str(key)
            qty = int(value)
        if qty > 0:
            inventory[item] += qty
    return dict(inventory)


def plan_action_id(step: Mapping[str, Any]) -> str:
    step_type = str(step.get("type", "")).strip()
    text = str(step.get("text", "")).strip()
    if step_type == "no_op" or text == "no_op":
        return "no_op"
    return f"{step_type}_{text}"


def goal_satisfied(goal: Mapping[str, int], inventory: Mapping[str, int]) -> bool:
    return all(int(inventory.get(item, 0)) >= int(qty) for item, qty in goal.items())


def has_requirements(variant: Mapping[str, Any], inventory: Mapping[str, int]) -> bool:
    for source in ("precondition", "tool"):
        for item, qty in variant.get(source, {}).items():
            if int(inventory.get(item, 0)) < int(qty):
                return False
    return True


def apply_variant(variant: Mapping[str, Any], inventory: Mapping[str, int]) -> dict[str, int]:
    updated = dict(inventory)
    for item, qty in variant.get("precondition", {}).items():
        updated[item] = int(updated.get(item, 0)) - int(qty)
        if updated[item] <= 0:
            updated.pop(item, None)
    for item, qty in variant.get("output", {}).items():
        updated[item] = int(updated.get(item, 0)) + int(qty)
    return updated


def replay_official_plan(
    task: Mapping[str, Any],
    steps: Any,
    action_lib: Mapping[str, list[Mapping[str, Any]]],
    max_ticks: int,
) -> ReplayResult:
    if not isinstance(steps, list):
        return ReplayResult(
            ok=False,
            reason="missing_plan",
            trace=[],
            final_inventory=dict(task["initial_inventory"]),
        )

    inventory = dict(task["initial_inventory"])
    task_goal = dict(task["goal"])
    pointer = 0
    trace: list[dict[str, Any]] = []

    for _tick in range(max_ticks):
        if task_goal and goal_satisfied(task_goal, inventory):
            return ReplayResult(True, "ok", trace, inventory)

        current_goal = {str(k): int(v) for k, v in (steps[pointer].get("goal", {}) or {}).items()}
        if current_goal and goal_satisfied(current_goal, inventory):
            pointer += 1

        if pointer >= len(steps):
            return ReplayResult(
                ok=False,
                reason="plan_exhausted_before_task_goal",
                trace=trace,
                final_inventory=inventory,
            )

        action_id = plan_action_id(steps[pointer])
        variants = action_lib.get(action_id)
        if not variants:
            return ReplayResult(
                ok=False,
                reason="unknown_action",
                trace=trace,
                final_inventory=inventory,
                failed_step=pointer,
                failed_action=action_id,
            )

        chosen = first_satisfied_variant(variants, inventory)
        if chosen is None:
            return ReplayResult(
                ok=False,
                reason=missing_requirements_message(action_id, variants, inventory),
                trace=trace,
                final_inventory=inventory,
                failed_step=pointer,
                failed_action=action_id,
            )

        before = dict(inventory)
        inventory = apply_variant(chosen, inventory)
        trace.append(
            {
                "plan_step": pointer,
                "action": action_id,
                "inventory_before": before,
                "inventory_after": dict(inventory),
            }
        )

    return ReplayResult(
        ok=False,
        reason=f"task_goal_not_reached_after_{max_ticks}_ticks",
        trace=trace,
        final_inventory=inventory,
        failed_step=pointer,
        failed_action=plan_action_id(steps[pointer]) if pointer < len(steps) else None,
    )


def first_satisfied_variant(
    variants: list[Mapping[str, Any]],
    inventory: Mapping[str, int],
) -> Mapping[str, Any] | None:
    for variant in variants:
        if has_requirements(variant, inventory):
            return variant
    return None


def missing_requirements_message(
    action_id: str,
    variants: list[Mapping[str, Any]],
    inventory: Mapping[str, int],
) -> str:
    best_parts = None
    for variant in variants:
        parts = []
        for source in ("tool", "precondition"):
            for item, qty in variant.get(source, {}).items():
                have = int(inventory.get(item, 0))
                need = int(qty)
                if have < need:
                    label = "tool" if source == "tool" else "material"
                    parts.append(f"{label}:{item}:{have}/{need}")
        if parts and (best_parts is None or len(parts) < len(best_parts)):
            best_parts = parts
    if best_parts:
        return f"missing_requirements:{action_id}:{';'.join(best_parts)}"
    return f"no_variant_matched:{action_id}"


def build_producer_index(
    action_lib: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, list[tuple[str, Mapping[str, Any]]]]:
    producers: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for action_id, variants in action_lib.items():
        for variant in variants:
            for item in variant.get("output", {}):
                producers[item].append((action_id, variant))

    for item, variants in producers.items():
        variants.sort(key=producer_sort_key)
    return producers


def producer_sort_key(entry: tuple[str, Mapping[str, Any]]) -> tuple[int, int, int, str]:
    action_id, variant = entry
    action_type = str(variant.get("type", "unknown"))
    prereq_count = len(variant.get("precondition", {})) + len(variant.get("tool", {}))
    block_cycle_penalty = 1 if any("block" in item for item in variant.get("precondition", {})) else 0
    return (ACTION_TYPE_ORDER.get(action_type, 5), block_cycle_penalty, prereq_count, action_id)


def synthesize_task(
    task: Mapping[str, Any],
    action_lib: Mapping[str, list[Mapping[str, Any]]],
    producers: Mapping[str, list[tuple[str, Mapping[str, Any]]]],
    max_actions: int,
) -> SynthesisResult:
    planner = BackwardPlanner(action_lib, producers, max_actions)
    return planner.plan(dict(task["goal"]), dict(task["initial_inventory"]))


class BackwardPlanner:
    def __init__(
        self,
        action_lib: Mapping[str, list[Mapping[str, Any]]],
        producers: Mapping[str, list[tuple[str, Mapping[str, Any]]]],
        max_actions: int,
    ):
        self.action_lib = action_lib
        self.producers = producers
        self.max_actions = max_actions
        self.actions: list[str] = []
        self.trace: list[dict[str, Any]] = []

    def plan(self, goal: dict[str, int], inventory: dict[str, int]) -> SynthesisResult:
        self.actions = []
        self.trace = []
        try:
            for item, qty in sorted(goal.items()):
                if not self.ensure_item(item, int(qty), inventory, []):
                    return SynthesisResult(False, f"cannot_satisfy:{item}:{qty}", self.actions, self.trace, inventory)
                if len(self.actions) > self.max_actions:
                    return SynthesisResult(False, "max_synth_actions_exceeded", self.actions, self.trace, inventory)
        except RecursionError:
            return SynthesisResult(False, "recursion_error", self.actions, self.trace, inventory)

        if not goal_satisfied(goal, inventory):
            return SynthesisResult(False, "final_goal_not_satisfied", self.actions, self.trace, inventory)
        return SynthesisResult(True, "ok", self.actions, self.trace, inventory)

    def ensure_item(
        self,
        item: str,
        qty: int,
        inventory: dict[str, int],
        stack: list[str],
    ) -> bool:
        if int(inventory.get(item, 0)) >= qty:
            return True
        if item in stack:
            return False
        if item not in self.producers:
            return False

        while int(inventory.get(item, 0)) < qty:
            need = qty - int(inventory.get(item, 0))
            if not self.apply_one_producer(item, need, inventory, stack):
                return False
            if len(self.actions) > self.max_actions:
                return False
        return True

    def apply_one_producer(
        self,
        item: str,
        need: int,
        inventory: dict[str, int],
        stack: list[str],
    ) -> bool:
        for action_id, variant in self.producers[item]:
            output_qty = int(variant.get("output", {}).get(item, 0))
            if output_qty <= 0:
                continue
            if variant_has_nonpositive_target_gain(item, variant):
                continue

            saved_inventory = deepcopy(inventory)
            saved_actions_len = len(self.actions)
            saved_trace_len = len(self.trace)

            if self.prepare_variant_requirements(variant, inventory, stack + [item]):
                repetitions = max(1, math.ceil(need / output_qty))
                ok = True
                for _ in range(repetitions):
                    if not self.prepare_variant_requirements(variant, inventory, stack + [item]):
                        ok = False
                        break
                    if not has_requirements(variant, inventory):
                        ok = False
                        break
                    before = dict(inventory)
                    updated = apply_variant(variant, inventory)
                    inventory.clear()
                    inventory.update(updated)
                    self.actions.append(action_id)
                    self.trace.append(
                        {
                            "action": action_id,
                            "inventory_before": before,
                            "inventory_after": dict(inventory),
                        }
                    )
                    if len(self.actions) > self.max_actions:
                        ok = False
                        break
                if ok:
                    return True

            inventory.clear()
            inventory.update(saved_inventory)
            del self.actions[saved_actions_len:]
            del self.trace[saved_trace_len:]

        return False

    def prepare_variant_requirements(
        self,
        variant: Mapping[str, Any],
        inventory: dict[str, int],
        stack: list[str],
    ) -> bool:
        for source in ("tool", "precondition"):
            for req_item, req_qty in sorted(variant.get(source, {}).items()):
                if not self.ensure_item(req_item, int(req_qty), inventory, stack):
                    return False
        return True


def variant_has_nonpositive_target_gain(item: str, variant: Mapping[str, Any]) -> bool:
    output = int(variant.get("output", {}).get(item, 0))
    consumed = int(variant.get("precondition", {}).get(item, 0))
    return output <= consumed


def build_summary(
    records: list[Mapping[str, Any]],
    duplicate_task_ids: list[str],
    plans: Mapping[str, Any],
    goals_67_synthetic_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_group: dict[str, dict[str, int]] = {}
    for record in records:
        group = record["group"]
        bucket = by_group.setdefault(
            group,
            {
                "tasks": 0,
                "official_plan_replayable": 0,
                "dependency_graph_buildable": 0,
                "goals_67": 0,
                "goals_67_dependency_graph_buildable": 0,
            },
        )
        bucket["tasks"] += 1
        if record["official_plan_replayable"]:
            bucket["official_plan_replayable"] += 1
        if record["dependency_graph_buildable"]:
            bucket["dependency_graph_buildable"] += 1
        if record["in_goals_67"]:
            bucket["goals_67"] += 1
        if record["in_goals_67"] and record["dependency_graph_buildable"]:
            bucket["goals_67_dependency_graph_buildable"] += 1

    task_ids = {record["task_id"] for record in records}
    official_ids = {
        record["task_id"] for record in records if record["official_plan_replayable"]
    }
    buildable_ids = {
        record["task_id"] for record in records if record["dependency_graph_buildable"]
    }
    goals_67_ids = {record["task_id"] for record in records if record["in_goals_67"]}
    goals_67_official_ids = {
        record["task_id"]
        for record in records
        if record["in_goals_67"] and record["official_plan_replayable"]
    }
    goals_67_buildable_ids = {
        record["task_id"]
        for record in records
        if record["in_goals_67"] and record["dependency_graph_buildable"]
    }
    return {
        "tasks_total": len(records),
        "unique_tasks_total": len(task_ids),
        "plans_total": len(plans),
        "duplicate_task_ids": duplicate_task_ids,
        "plans_without_task": sorted(set(plans) - task_ids),
        "tasks_without_plan": sorted(task_ids - set(plans)),
        "official_plan_replayable": sum(1 for record in records if record["official_plan_replayable"]),
        "unique_official_plan_replayable": len(official_ids),
        "dependency_graph_buildable": sum(1 for record in records if record["dependency_graph_buildable"]),
        "unique_dependency_graph_buildable": len(buildable_ids),
        "goals_67_total": sum(1 for record in records if record["in_goals_67"]),
        "unique_goals_67_total": len(goals_67_ids),
        "goals_67_official_plan_replayable": sum(
            1 for record in records if record["in_goals_67"] and record["official_plan_replayable"]
        ),
        "unique_goals_67_official_plan_replayable": len(goals_67_official_ids),
        "goals_67_dependency_graph_buildable": sum(
            1 for record in records if record["in_goals_67"] and record["dependency_graph_buildable"]
        ),
        "unique_goals_67_dependency_graph_buildable": len(goals_67_buildable_ids),
        "goals_67_synthetic_total": len(goals_67_synthetic_records),
        "goals_67_synthetic_dependency_graph_buildable": sum(
            1 for record in goals_67_synthetic_records if record["dependency_graph_buildable"]
        ),
        "goals_67_missing_from_tasks_json": sorted(
            record["goal"]
            for record in goals_67_synthetic_records
            if not record["present_in_tasks_json"]
        ),
        "by_group": dict(sorted(by_group.items())),
    }


def unique_task_ids(records: Any) -> list[str]:
    return sorted({record["task_id"] for record in records})


def compact_unique_task_records(records: Any) -> list[dict[str, Any]]:
    by_id = {}
    for record in records:
        by_id.setdefault(record["task_id"], compact_task_record(record))
    return [by_id[task_id] for task_id in sorted(by_id)]


def compact_task_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "group": record["group"],
        "goal": record["goal"],
        "in_goals_67": record["in_goals_67"],
        "synthesized_action_count": record["synthesized_action_count"],
        "synthesized_actions": record["synthesized_actions"],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    records = report["records"]
    buildable = [record for record in records if record["dependency_graph_buildable"]]
    official = [record for record in records if record["official_plan_replayable"]]
    goals67_buildable = [
        record for record in records if record["in_goals_67"] and record["dependency_graph_buildable"]
    ]

    lines = [
        "# MC-TextWorld Task Discovery",
        "",
        "## Criteria",
        "",
        "- `official_plan_replayable`: `tasks.json + plans.json` compile to known `action_lib.json` actions and replay successfully.",
        "- `dependency_graph_buildable`: a quantity-aware executable action sequence can be synthesized from `action_lib.json` for the task goal.",
        "",
        "## Summary",
        "",
        f"- Total tasks: `{summary['tasks_total']}`",
        f"- Unique task ids: `{summary['unique_tasks_total']}`",
        f"- Total plans: `{summary['plans_total']}`",
        f"- Official-plan replayable tasks: `{summary['unique_official_plan_replayable']}` unique ids",
        f"- Dependency-graph buildable tasks: `{summary['unique_dependency_graph_buildable']}` unique ids",
        f"- 67-goal tasks present in original tasks.json: `{summary['unique_goals_67_total']}` unique ids",
        f"- 67-goal official-plan replayable tasks: `{summary['unique_goals_67_official_plan_replayable']}` unique ids",
        f"- 67-goal dependency-graph buildable tasks: `{summary['unique_goals_67_dependency_graph_buildable']}` unique ids",
        f"- 67-goal synthetic dependency-graph buildable targets: `{summary['goals_67_synthetic_dependency_graph_buildable']}` / `{summary['goals_67_synthetic_total']}`",
        f"- 67-goal targets missing from original tasks.json: `{', '.join(summary['goals_67_missing_from_tasks_json'])}`",
        "",
        "## By Group",
        "",
        "| Group | Tasks | Official Replayable | Graph Buildable | Goals-67 | Goals-67 Buildable |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, row in summary["by_group"].items():
        lines.append(
            f"| {group} | {row['tasks']} | {row['official_plan_replayable']} | "
            f"{row['dependency_graph_buildable']} | {row['goals_67']} | "
            f"{row['goals_67_dependency_graph_buildable']} |"
        )

    lines.extend(
        [
            "",
            "## Dependency-Graph Buildable Tasks",
            "",
            render_task_list(buildable),
            "",
            "## Official-Plan Replayable Tasks",
            "",
            render_task_list(official),
            "",
            "## Goals-67 Dependency-Graph Buildable Tasks",
            "",
            render_task_list(goals67_buildable),
            "",
            "## Exclusions",
            "",
            "Tasks not listed under `Dependency-Graph Buildable Tasks` failed synthesis from `action_lib.json`.",
            "See `mctextworld_task_discovery_report.json` for per-task failure reasons.",
            "",
        ]
    )
    return "\n".join(lines)


def render_task_list(records: list[Mapping[str, Any]]) -> str:
    by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_group[record["group"]].add(record["task_id"])
    lines = []
    for group in sorted(by_group):
        tasks = ", ".join(sorted(by_group[group]))
        lines.append(f"- **{group}** ({len(by_group[group])}): {tasks}")
    return "\n".join(lines) if lines else "_None_"


if __name__ == "__main__":
    raise SystemExit(main())
