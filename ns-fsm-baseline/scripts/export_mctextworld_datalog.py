"""Export MC-TextWorld JSON data to Datalog-style fact files.

The runtime truth source is MC-TextWorld's action_lib.json.  Each action can
have multiple variants; every variant becomes one recipe id.  plans.json and
tasks.json are exported as task/plan facts for validation and analysis, not as
action validity rules.

Default output:
    MC-TextWorld/mctextworld/datalog_out/

Example:
    python ns-fsm-baseline/scripts/export_mctextworld_datalog.py --validate-plans
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
MCTEXTWORLD_DIR = REPO_ROOT / "MC-TextWorld" / "mctextworld"
DEFAULT_ACTION_LIB = MCTEXTWORLD_DIR / "action_lib.json"
DEFAULT_PLANS = MCTEXTWORLD_DIR / "plans.json"
DEFAULT_TASKS = MCTEXTWORLD_DIR / "tasks.json"
DEFAULT_OUTPUT_DIR = MCTEXTWORLD_DIR / "datalog_out"

ACTION_TYPES = {"mine", "craft", "smelt", "no_op"}


RULES_DL = """// MC-TextWorld action validity rules.
// These rules expect static facts exported from action_lib.json plus a runtime
// has.facts file containing the current inventory as: item<TAB>count.

.decl action_type(aid:symbol, kind:symbol)
.input action_type

.decl recipe(rid:symbol, aid:symbol)
.input recipe

.decl recipe_order(rid:symbol, idx:number)
.input recipe_order

.decl precondition(rid:symbol, item:symbol, needed:number)
.input precondition

.decl tool_need(rid:symbol, tool:symbol, needed:number)
.input tool_need

.decl output_fact(rid:symbol, item:symbol, count:number)
.input output_fact

.decl has(item:symbol, count:number)
.input has

.decl has_any(item:symbol)
has_any(Item) :- has(Item, _).

.decl precond_unmet(rid:symbol)
precond_unmet(RID) :-
    precondition(RID, Item, Needed),
    has(Item, Have),
    Have < Needed.
precond_unmet(RID) :-
    precondition(RID, Item, _),
    !has_any(Item).

.decl tool_unmet(rid:symbol)
tool_unmet(RID) :-
    tool_need(RID, Tool, Needed),
    has(Tool, Have),
    Have < Needed.
tool_unmet(RID) :-
    tool_need(RID, Tool, _),
    !has_any(Tool).

.decl recipe_ok(rid:symbol)
recipe_ok(RID) :-
    recipe(RID, _),
    !precond_unmet(RID),
    !tool_unmet(RID).

.decl valid_action(aid:symbol)
valid_action(AID) :-
    recipe(RID, AID),
    recipe_ok(RID).

.decl missing_precondition(aid:symbol, rid:symbol, item:symbol, needed:number, have:number)
missing_precondition(AID, RID, Item, Needed, Have) :-
    recipe(RID, AID),
    precondition(RID, Item, Needed),
    has(Item, Have),
    Have < Needed.
missing_precondition(AID, RID, Item, Needed, 0) :-
    recipe(RID, AID),
    precondition(RID, Item, Needed),
    !has_any(Item).

.decl missing_tool(aid:symbol, rid:symbol, tool:symbol, needed:number, have:number)
missing_tool(AID, RID, Tool, Needed, Have) :-
    recipe(RID, AID),
    tool_need(RID, Tool, Needed),
    has(Tool, Have),
    Have < Needed.
missing_tool(AID, RID, Tool, Needed, 0) :-
    recipe(RID, AID),
    tool_need(RID, Tool, Needed),
    !has_any(Tool).

.output recipe_ok
.output valid_action
.output missing_precondition
.output missing_tool
"""


SCHEMA_MD = """# MC-TextWorld Datalog Export

All `.facts` files are UTF-8, tab-separated, and headerless.

## Action domain facts

- `action.facts`: `aid`
- `action_type.facts`: `aid`, `type`
- `recipe.facts`: `rid`, `aid`
- `recipe_order.facts`: `rid`, `index`
- `precondition.facts`: `rid`, `item`, `count`
- `tool_need.facts`: `rid`, `tool`, `count`
- `output.facts`: `rid`, `item`, `count`
- `item.facts`: `item`
- `produces.facts`: `aid`, `item`, `count`

`rid` is generated as `<action_id>__<variant_index>`.

## Task and plan facts

- `task.facts`: `task_id`
- `task_group.facts`: `task_id`, `group`
- `task_biome.facts`: `task_id`, `biome`
- `goal.facts`: `task_id`, `item`, `count`
- `init_has.facts`: `task_id`, `item`, `count`
- `plan_task.facts`: `task_id`
- `plan_step.facts`: `task_id`, `step_index`, `action_id`
- `plan_subgoal.facts`: `task_id`, `step_index`, `item`, `count`

`plans.json` steps are high-level controller steps.  The optional replay
validator follows the plan-pointer behavior in `mctextworld/run.py`: before
each action, it checks the current step subgoal and advances the pointer by one
when that subgoal is already satisfied.
"""


@dataclass
class ExportReport:
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    plan_replay: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def symbol(value: Any, field_name: str) -> str:
    text = str(value).strip()
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    if not text:
        raise ValueError(f"empty symbol for {field_name}")
    return text


def count_value(value: Any, field_name: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if count <= 0:
        raise ValueError(f"{field_name} must be positive, got {count}")
    return count


def recipe_id(action_id: str, variant_index: int) -> str:
    return f"{action_id}__{variant_index:03d}"


def plan_action_id(step: Mapping[str, Any]) -> str:
    step_type = symbol(step.get("type", ""), "plan step type")
    text = symbol(step.get("text", ""), "plan step text")
    if step_type == "no_op" or text == "no_op":
        return "no_op"
    return f"{step_type}_{text}"


def normalize_init_inventory(raw_inventory: Mapping[str, Any]) -> dict[str, int]:
    inventory: dict[str, int] = defaultdict(int)
    for key, value in raw_inventory.items():
        if isinstance(value, Mapping) and "type" in value:
            item = symbol(value["type"], f"init_inventory[{key}].type")
            qty = count_value(value.get("quantity", 1), f"init_inventory[{key}].quantity")
            inventory[item] += qty
        else:
            item = symbol(key, "init_inventory item")
            qty = count_value(value, f"init_inventory[{key}]")
            inventory[item] += qty
    return dict(inventory)


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


def goal_satisfied(goal: Mapping[str, Any], inventory: Mapping[str, int]) -> bool:
    for item, qty in goal.items():
        if int(inventory.get(item, 0)) < int(qty):
            return False
    return True


def best_missing_message(action_id: str, variants: list[Mapping[str, Any]], inventory: Mapping[str, int]) -> str:
    best_parts: list[str] | None = None
    best_count = 10**9
    for variant in variants:
        parts = []
        for source in ("tool", "precondition"):
            for item, qty in variant.get(source, {}).items():
                have = int(inventory.get(item, 0))
                need = int(qty)
                if have < need:
                    label = "tool" if source == "tool" else "material"
                    parts.append(f"{label} {item} need {need}, have {have}")
        if parts and len(parts) < best_count:
            best_count = len(parts)
            best_parts = parts
    if best_parts:
        return f"{action_id}: " + "; ".join(best_parts)
    return f"{action_id}: no variant matched"


def export_action_facts(action_lib: Mapping[str, Any], report: ExportReport) -> dict[str, list[tuple[Any, ...]]]:
    facts: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    items: set[str] = set()
    type_counts: Counter[str] = Counter()
    variant_count = 0

    for raw_action_id, variants in action_lib.items():
        action_id = symbol(raw_action_id, "action id")
        if not isinstance(variants, list):
            report.errors.append(f"action {action_id} must map to a list of variants")
            continue
        if not variants:
            report.errors.append(f"action {action_id} has no variants")
            continue

        facts["action"].append((action_id,))

        seen_type: str | None = None
        for idx, variant in enumerate(variants):
            if not isinstance(variant, Mapping):
                report.errors.append(f"variant {action_id}[{idx}] must be an object")
                continue

            rid = recipe_id(action_id, idx)
            action_type = symbol(variant.get("type", "unknown"), f"{rid}.type")
            abbr = variant.get("abbr")
            if abbr is not None and str(abbr) != action_id:
                report.warnings.append(f"{rid}: abbr {abbr!r} does not match action id {action_id!r}")
            if action_type not in ACTION_TYPES:
                report.warnings.append(f"{rid}: unexpected action type {action_type!r}")
            if seen_type is None:
                seen_type = action_type
                facts["action_type"].append((action_id, action_type))
                type_counts[action_type] += 1
            elif seen_type != action_type:
                report.warnings.append(
                    f"{action_id}: variant {idx} has type {action_type!r}, first type is {seen_type!r}"
                )

            facts["recipe"].append((rid, action_id))
            facts["recipe_order"].append((rid, idx))
            variant_count += 1

            for fact_name, source_key in (
                ("precondition", "precondition"),
                ("tool_need", "tool"),
                ("output", "output"),
            ):
                raw_requirements = variant.get(source_key, {})
                if not isinstance(raw_requirements, Mapping):
                    report.errors.append(f"{rid}.{source_key} must be an object")
                    continue
                for raw_item, raw_count in raw_requirements.items():
                    item = symbol(raw_item, f"{rid}.{source_key} item")
                    count = count_value(raw_count, f"{rid}.{source_key}[{item}]")
                    facts[fact_name].append((rid, item, count))
                    items.add(item)
                    if fact_name == "output":
                        facts["produces"].append((action_id, item, count))

    for item in sorted(items):
        facts["item"].append((item,))

    report.counts.update(
        {
            "actions": len(action_lib),
            "recipe_variants": variant_count,
            "items": len(items),
        }
    )
    for action_type, count in sorted(type_counts.items()):
        report.counts[f"actions_type_{action_type}"] = count
    return facts


def export_task_facts(tasks: list[Mapping[str, Any]], report: ExportReport) -> tuple[dict[str, list[tuple[Any, ...]]], dict[str, dict[str, Any]]]:
    facts: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    task_by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            report.errors.append(f"tasks[{index}] must be an object")
            continue
        task_id = symbol(task.get("task", ""), f"tasks[{index}].task")
        if task_id in seen:
            report.warnings.append(f"duplicate task id in tasks.json: {task_id}")
        seen.add(task_id)

        group = symbol(task.get("group", "unknown"), f"{task_id}.group")
        env = task.get("env", {})
        if not isinstance(env, Mapping):
            report.errors.append(f"{task_id}.env must be an object")
            env = {}
        biome = symbol(env.get("biome", "unknown"), f"{task_id}.env.biome")
        init_inventory = normalize_init_inventory(env.get("init_inventory", {}) or {})
        task_obj = task.get("task_obj", {}) or {}
        if not isinstance(task_obj, Mapping):
            report.errors.append(f"{task_id}.task_obj must be an object")
            task_obj = {}

        facts["task"].append((task_id,))
        facts["task_group"].append((task_id, group))
        facts["task_biome"].append((task_id, biome))
        for item, qty in sorted(init_inventory.items()):
            facts["init_has"].append((task_id, item, qty))
        for raw_item, raw_qty in task_obj.items():
            item = symbol(raw_item, f"{task_id}.task_obj item")
            qty = count_value(raw_qty, f"{task_id}.task_obj[{item}]")
            facts["goal"].append((task_id, item, qty))

        task_by_id[task_id] = {
            "id": task_id,
            "group": group,
            "biome": biome,
            "init_inventory": init_inventory,
            "goal": {symbol(k, f"{task_id}.goal item"): int(v) for k, v in task_obj.items()},
        }

    report.counts["tasks"] = len(tasks)
    return facts, task_by_id


def export_plan_facts(plans: Mapping[str, Any], action_lib: Mapping[str, Any], report: ExportReport) -> dict[str, list[tuple[Any, ...]]]:
    facts: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    missing_actions: set[str] = set()

    for raw_task_id, steps in plans.items():
        task_id = symbol(raw_task_id, "plan task id")
        if not isinstance(steps, list):
            report.errors.append(f"plans[{task_id}] must be a list")
            continue
        facts["plan_task"].append((task_id,))
        for idx, step in enumerate(steps):
            if not isinstance(step, Mapping):
                report.errors.append(f"plans[{task_id}][{idx}] must be an object")
                continue
            action_id = plan_action_id(step)
            facts["plan_step"].append((task_id, idx, action_id))
            if action_id not in action_lib:
                missing_actions.add(action_id)
            goal = step.get("goal", {}) or {}
            if not isinstance(goal, Mapping):
                report.errors.append(f"plans[{task_id}][{idx}].goal must be an object")
                continue
            for raw_item, raw_qty in goal.items():
                item = symbol(raw_item, f"plans[{task_id}][{idx}].goal item")
                qty = count_value(raw_qty, f"plans[{task_id}][{idx}].goal[{item}]")
                facts["plan_subgoal"].append((task_id, idx, item, qty))

    if missing_actions:
        for action_id in sorted(missing_actions)[:25]:
            report.warnings.append(f"plan references unknown action: {action_id}")
        if len(missing_actions) > 25:
            report.warnings.append(f"... and {len(missing_actions) - 25} more unknown plan actions")

    report.counts["plans"] = len(plans)
    report.counts["plan_steps"] = sum(len(v) for v in plans.values() if isinstance(v, list))
    return facts


def replay_plans(
    plans: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    action_lib: Mapping[str, list[Mapping[str, Any]]],
    max_plan_ticks: int,
) -> dict[str, Any]:
    checked = 0
    passed = 0
    failed: list[dict[str, Any]] = []
    skipped_no_task: list[str] = []

    for task_id, steps in plans.items():
        if task_id not in tasks_by_id:
            skipped_no_task.append(task_id)
            continue
        if not isinstance(steps, list):
            continue
        checked += 1
        task = tasks_by_id[task_id]
        inventory = dict(task.get("init_inventory", {}))
        task_failed = None
        task_goal = task.get("goal", {})
        pointer = 0

        for tick in range(max_plan_ticks):
            if task_goal and goal_satisfied(task_goal, inventory):
                break

            if pointer >= len(steps):
                task_failed = {
                    "task": task_id,
                    "step": None,
                    "action": None,
                    "reason": "plan exhausted before task goal was satisfied",
                    "goal": task_goal,
                    "inventory": inventory,
                }
                break

            current_step = steps[pointer]
            current_goal = {str(k): int(v) for k, v in (current_step.get("goal", {}) or {}).items()}
            if current_goal and goal_satisfied(current_goal, inventory):
                pointer += 1
                if pointer >= len(steps):
                    task_failed = {
                        "task": task_id,
                        "step": None,
                        "action": None,
                        "reason": "plan exhausted after satisfied subgoal",
                        "goal": task_goal,
                        "inventory": inventory,
                    }
                    break

            step = steps[pointer]
            subgoal = {str(k): int(v) for k, v in (step.get("goal", {}) or {}).items()}
            action_id = plan_action_id(step)
            variants = action_lib.get(action_id)
            if not variants:
                task_failed = {
                    "task": task_id,
                    "step": pointer,
                    "action": action_id,
                    "reason": "unknown action",
                    "inventory": inventory,
                }
                break
            chosen = None
            for variant in variants:
                if has_requirements(variant, inventory):
                    chosen = variant
                    break
            if chosen is None:
                task_failed = {
                    "task": task_id,
                    "step": pointer,
                    "action": action_id,
                    "reason": best_missing_message(action_id, variants, inventory),
                    "subgoal": subgoal,
                    "inventory": inventory,
                }
                break
            inventory = apply_variant(chosen, inventory)
            if task_failed:
                break
        else:
            task_failed = {
                "task": task_id,
                "step": pointer if pointer < len(steps) else None,
                "action": plan_action_id(steps[pointer]) if pointer < len(steps) else None,
                "reason": f"task goal not reached after {max_plan_ticks} plan ticks",
                "goal": task_goal,
                "inventory": inventory,
            }

        if task_failed:
            failed.append(task_failed)
            continue
        if task_goal and not goal_satisfied(task_goal, inventory):
            failed.append(
                {
                    "task": task_id,
                    "step": None,
                    "action": None,
                    "reason": "final task goal not satisfied after replay",
                    "goal": task_goal,
                    "inventory": inventory,
                }
            )
            continue
        passed += 1

    return {
        "checked": checked,
        "passed": passed,
        "failed": failed,
        "skipped_no_task": skipped_no_task,
    }


def merge_facts(*fact_groups: Mapping[str, list[tuple[Any, ...]]]) -> dict[str, list[tuple[Any, ...]]]:
    merged: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for facts in fact_groups:
        for name, rows in facts.items():
            merged[name].extend(rows)
    return merged


def write_fact_file(path: Path, rows: Iterable[tuple[Any, ...]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write("\t".join(str(value) for value in row))
            f.write("\n")
            count += 1
    return count


def write_outputs(out_dir: Path, facts: Mapping[str, list[tuple[Any, ...]]], report: ExportReport) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_file: dict[str, int] = {}
    for fact_name, rows in sorted(facts.items()):
        rows_by_file[f"{fact_name}.facts"] = write_fact_file(out_dir / f"{fact_name}.facts", rows)

    (out_dir / "rules.dl").write_text(RULES_DL, encoding="utf-8")
    (out_dir / "SCHEMA.md").write_text(SCHEMA_MD, encoding="utf-8")

    report.counts["fact_files"] = len(rows_by_file)
    report.counts["fact_rows"] = sum(rows_by_file.values())
    payload = {
        "counts": report.counts,
        "rows_by_file": rows_by_file,
        "warnings": report.warnings,
        "errors": report.errors,
        "plan_replay": report.plan_replay,
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def validate_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("action-lib", args.action_lib),
        ("plans", args.plans),
        ("tasks", args.tasks),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-lib", type=Path, default=DEFAULT_ACTION_LIB)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validate-plans",
        action="store_true",
        help="Replay high-level plans against action_lib semantics and report failures.",
    )
    parser.add_argument(
        "--max-action-repeats",
        type=int,
        default=1024,
        help="Maximum simulated plan ticks during replay validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when warnings or plan replay failures are present.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    validate_paths(args)

    action_lib = load_json(args.action_lib)
    plans = load_json(args.plans)
    tasks = load_json(args.tasks)
    if not isinstance(action_lib, Mapping):
        raise TypeError("action_lib.json must be an object")
    if not isinstance(plans, Mapping):
        raise TypeError("plans.json must be an object")
    if not isinstance(tasks, list):
        raise TypeError("tasks.json must be a list")

    report = ExportReport()
    action_facts = export_action_facts(action_lib, report)
    task_facts, tasks_by_id = export_task_facts(tasks, report)
    plan_facts = export_plan_facts(plans, action_lib, report)

    plan_only = sorted(set(plans) - set(tasks_by_id))
    task_only = sorted(set(tasks_by_id) - set(plans))
    for task_id in plan_only[:25]:
        report.warnings.append(f"plans.json has no matching task in tasks.json: {task_id}")
    if len(plan_only) > 25:
        report.warnings.append(f"... and {len(plan_only) - 25} more plan-only tasks")
    for task_id in task_only[:25]:
        report.warnings.append(f"tasks.json has no matching plan in plans.json: {task_id}")
    if len(task_only) > 25:
        report.warnings.append(f"... and {len(task_only) - 25} more task-only tasks")

    if args.validate_plans:
        typed_action_lib = {
            str(action_id): list(variants)
            for action_id, variants in action_lib.items()
            if isinstance(variants, list)
        }
        report.plan_replay = replay_plans(
            plans,
            tasks_by_id,
            typed_action_lib,
            max_plan_ticks=args.max_action_repeats,
        )
        for failure in report.plan_replay.get("failed", [])[:25]:
            report.warnings.append(
                "plan replay failed for "
                f"{failure.get('task')} step {failure.get('step')}: {failure.get('reason')}"
            )
        remaining_failures = len(report.plan_replay.get("failed", [])) - 25
        if remaining_failures > 0:
            report.warnings.append(f"... and {remaining_failures} more plan replay failures")

    facts = merge_facts(action_facts, task_facts, plan_facts)
    write_outputs(args.out_dir, facts, report)

    print(f"Wrote Datalog facts to: {args.out_dir}")
    print(f"Actions: {report.counts.get('actions', 0)}")
    print(f"Recipe variants: {report.counts.get('recipe_variants', 0)}")
    print(f"Tasks: {report.counts.get('tasks', 0)}")
    print(f"Plan steps: {report.counts.get('plan_steps', 0)}")
    print(f"Fact rows: {report.counts.get('fact_rows', 0)}")
    if args.validate_plans:
        replay = report.plan_replay
        print(
            "Plan replay: "
            f"{replay.get('passed', 0)}/{replay.get('checked', 0)} passed, "
            f"{len(replay.get('failed', []))} failed, "
            f"{len(replay.get('skipped_no_task', []))} skipped"
        )
    if report.errors:
        print(f"Errors: {len(report.errors)}")
    if report.warnings:
        print(f"Warnings: {len(report.warnings)}")

    if report.errors:
        return 1
    if args.strict and (report.warnings or report.plan_replay.get("failed")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
