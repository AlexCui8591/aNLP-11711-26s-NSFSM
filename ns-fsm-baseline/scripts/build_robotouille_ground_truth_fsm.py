#!/usr/bin/env python3
"""Build stateful Robotouille ground-truth FSM files for NS-FSM.

The output is intentionally task-level and benchmark-stable:
- each state represents the next *kind* of work that should be done
- each state carries FSM-side allowed action templates
- runtime proposals are verified against the current FSM state's transition set
- simulator executable actions remain prompt-side context only

This follows the user's desired decomposition:
    FSM state -> what should be done next
    Environment -> what is executable right now
    Runtime verifier -> is the proposal in T(current_state)?
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPO_ROOT / "ns-fsm-baseline" / "config"


@dataclass
class ObjectRequirement:
    name: str
    object_id: int
    predicates: list[dict[str, Any]] = field(default_factory=list)
    needs_cut: bool = False
    needs_cook: bool = False
    needs_fry: bool = False
    add_to_water: bool = False
    serve_at_table: bool = False
    serve_on_table: bool = False
    must_remain_clear: bool = False

    @property
    def alias(self) -> str:
        return f"{self.name}__{self.object_id}"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = load_json(input_path)
    if payload.get("split") != args.split:
        raise ValueError(
            f"Input split mismatch: input={payload.get('split')} requested={args.split}"
        )

    llm_client = load_llm_client(args) if (args.use_llm_generator or args.use_llm_judge) else None
    llm_generator = llm_client if args.use_llm_generator else None
    llm_judge = llm_client if args.use_llm_judge else None
    last_good_tasks = load_last_good_task_map(args, output_path)
    selected_task_ids = {item.strip() for item in args.task_ids.split(",") if item.strip()}
    tasks = []
    judge_events = []
    for task in payload.get("tasks", []):
        if selected_task_ids and str(task.get("task_id")) not in selected_task_ids:
            continue
        built = build_task_fsm(task, mode=args.mode)
        generation_event: dict[str, Any] = {
            "llm_generator_used": False,
            "candidate_source": f"deterministic_{args.mode}",
        }
        if llm_generator is not None:
            built, generation_event = generate_llm_ground_truth_candidate(
                llm_generator=llm_generator,
                task=task,
                seed_candidate=built,
                attempts=args.generator_attempts,
            )
        fallback_candidate, fallback_source, fallback_events = select_fallback_candidate(
            task=task,
            last_good_tasks=last_good_tasks,
        )
        force_fallback_reason = None
        if args.use_llm_generator and not generation_event.get("accepted", False):
            force_fallback_reason = "llm_generation_failed"
        judged = judge_and_fallback_task_fsm(
            task=task,
            candidate=built,
            llm_judge=llm_judge,
            llm_required=args.llm_required,
            judge_attempts=args.judge_attempts,
            fallback_candidate=fallback_candidate,
            fallback_source=fallback_source,
            fallback_selection_events=fallback_events,
            candidate_event=generation_event,
            force_fallback_reason=force_fallback_reason,
        )
        tasks.append(judged)
        judge_events.append(judged.get("ground_truth_generation", {}))
    result = {
        "source": (
            "Robotouille official task JSON transformed into a stateful NS-FSM "
            "ground-truth file with guarded branching candidates and judge/fallback validation"
        ),
        "criterion": "guarded_state_list_with_per_action_next_states",
        "split": args.split,
        "task_count": len(tasks),
        "testing_seeds": payload.get("testing_seeds", []),
        "runtime_rule": (
            "final_legal_actions = state.guarded_transitions ∩ simulator.executable_actions"
        ),
        "generation": {
            "mode": args.mode,
            "llm_judge_requested": bool(args.use_llm_judge),
            "llm_generator_requested": bool(args.use_llm_generator),
            "llm_config": args.llm_config or "",
            "llm_required": bool(args.llm_required),
            "generator_attempts": args.generator_attempts,
            "judge_attempts": args.judge_attempts,
            "last_good_ground_truth": args.last_good_ground_truth or _default_last_good_path(output_path),
            "task_ids": sorted(selected_task_ids) if selected_task_ids else "ALL",
            "judge_events": judge_events,
        },
        "tasks": tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Wrote {output_path}")
    print(f"split={args.split} tasks={len(tasks)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["asynchronous", "synchronous"], default="asynchronous")
    parser.add_argument(
        "--mode",
        choices=["linear", "branching"],
        default="branching",
        help="Ground-truth FSM generation mode. branching adds guarded repair paths.",
    )
    parser.add_argument(
        "--input",
        default=str(BASELINE_CONFIG / "robotouille_ground_truth_asynchronous_tasks.json"),
    )
    parser.add_argument(
        "--output",
        default=str(BASELINE_CONFIG / "robotouille_ground_truth_asynchronous_fsm.json"),
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated task_ids to generate; default generates all tasks.",
    )
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument(
        "--use-llm-generator",
        "--llm-generate-ground-truth",
        action="store_true",
        help=(
            "Ask the configured LLM to generate a Robotouille ground-truth state_list "
            "candidate before validation/judging."
        ),
    )
    parser.add_argument(
        "--llm-config",
        default="",
        help="Optional hyperparams YAML for LLMInterface when LLM generation/judging is set.",
    )
    parser.add_argument(
        "--last-good-ground-truth",
        default="",
        help=(
            "Ground-truth FSM JSON used as the preferred fallback when LLM generation "
            "or LLM judge validation fails. Defaults to the existing output path when "
            "present, otherwise the repository default ground-truth file."
        ),
    )
    parser.add_argument(
        "--llm-required",
        action="store_true",
        help="Fail instead of falling back if the LLM judge cannot be called or rejects all candidates.",
    )
    parser.add_argument("--generator-attempts", type=int, default=1)
    parser.add_argument("--judge-attempts", type=int, default=1)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_llm_client(args: argparse.Namespace):
    try:
        import sys

        src = str(REPO_ROOT / "ns-fsm-baseline" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from llm_interface import LLMInterface

        return LLMInterface(args.llm_config or None)
    except Exception as exc:
        if args.llm_required:
            raise
        print(f"[llm] LLM unavailable; deterministic/last-good fallback will be used: {exc}")
        return None


def _default_last_good_path(output_path: Path) -> str:
    if output_path.exists():
        return str(output_path)
    default_path = BASELINE_CONFIG / "robotouille_ground_truth_asynchronous_fsm.json"
    return str(default_path) if default_path.exists() else ""


def load_last_good_task_map(
    args: argparse.Namespace,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    last_good_path = Path(args.last_good_ground_truth or _default_last_good_path(output_path))
    if not str(last_good_path) or not last_good_path.exists():
        return {}
    try:
        payload = load_json(last_good_path)
    except Exception as exc:
        if args.llm_required:
            raise
        print(f"[fallback] Could not load last-good ground truth {last_good_path}: {exc}")
        return {}
    return {
        str(task.get("task_id")): dict(task)
        for task in payload.get("tasks", [])
        if isinstance(task, Mapping) and task.get("task_id")
    }


def select_fallback_candidate(
    task: Mapping[str, Any],
    last_good_tasks: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Choose the best known-good fallback for a task.

    Preferred order:
    1. Existing last-good task from a ground-truth JSON file.
    2. The deterministic branching builder in this script.
    """

    task_id = str(task.get("task_id"))
    events: list[dict[str, Any]] = []

    last_good = last_good_tasks.get(task_id)
    if last_good is not None:
        candidate = json.loads(json.dumps(last_good))
        issues = ground_truth_validation_issues(candidate)
        events.append(
            {
                "source": "last_good_ground_truth",
                "task_id": task_id,
                "issues": issues,
                "selected": not issues,
            }
        )
        if not issues:
            return candidate, "last_good_ground_truth", events

    deterministic = build_task_fsm(task, mode="branching")
    issues = ground_truth_validation_issues(deterministic)
    events.append(
        {
            "source": "deterministic_branching",
            "task_id": task_id,
            "issues": issues,
            "selected": not issues,
        }
    )
    return deterministic, "deterministic_branching", events


def build_task_fsm(task: Mapping[str, Any], mode: str = "branching") -> dict[str, Any]:
    requirements = collect_object_requirements(task)
    state_list: list[dict[str, Any]] = []

    root_clusters: list[str] = []
    pending_cluster_starts: list[str] = []

    def add_state(
        name: str,
        kind: str,
        phase_goal: str,
        fsm_allowed_actions: list[dict[str, Any]],
        completion_condition: list[str],
        next_state: str | None = None,
        root_cluster: bool = False,
        notes: list[str] | None = None,
        parallel_subgoals: list[str] | None = None,
        transitions: list[dict[str, Any]] | None = None,
    ) -> None:
        if transitions is not None:
            fsm_allowed_actions = [
                dict(transition["action"])
                for transition in transitions
                if isinstance(transition.get("action"), Mapping)
            ]
        state = {
            "name": name,
            "kind": kind,
            "phase_goal": phase_goal,
            "fsm_allowed_actions": fsm_allowed_actions,
            "completion_condition": completion_condition,
        }
        if transitions is not None:
            state["transitions"] = transitions
        if next_state:
            state["next_state_on_completion"] = next_state
        if notes:
            state["notes"] = notes
        if parallel_subgoals:
            state["parallel_subgoals"] = parallel_subgoals
        state_list.append(state)
        if root_cluster:
            state["root_cluster"] = True
            root_clusters.append(name)
            pending_cluster_starts.append(name)

    soup_needed = any(
        goal.get("predicate") in {"in", "isboiling", "container_at", "addedto"}
        for goal in task.get("goal_predicates", [])
    )
    if soup_needed:
        build_soup_base_cluster(add_state)

    for req in requirements:
        if req.needs_cut:
            build_cut_cluster(req, add_state)
        if req.needs_cook:
            if mode == "branching":
                build_branching_cook_cluster(req, add_state)
            else:
                build_cook_cluster(req, add_state)
        if req.needs_fry:
            if mode == "branching":
                build_branching_fry_cluster(req, add_state)
            else:
                build_fry_cluster(req, add_state)

    if soup_needed:
        for req in requirements:
            if req.add_to_water:
                build_add_to_water_cluster(req, add_state)
        build_fill_bowl_cluster(add_state)

    for req in ordered_serve_requirements(requirements):
        if req.serve_at_table or req.serve_on_table:
            build_serve_cluster(req, add_state)

    verify_name = "VERIFY_GOAL"
    done_name = "DONE"
    add_state(
        name=verify_name,
        kind="verify",
        phase_goal="Check that all Robotouille goal predicates are satisfied.",
        fsm_allowed_actions=[
            {
                "template": "verify_goal_complete",
                "constraint": "All official goal predicates must evaluate to true in the simulator state.",
            }
        ],
        completion_condition=["all goal predicates satisfied"],
        next_state=done_name,
        root_cluster=True,
    )
    add_state(
        name=done_name,
        kind="terminal",
        phase_goal="Task complete.",
        fsm_allowed_actions=[],
        completion_condition=["terminal"],
    )

    annotate_async_parallelism(state_list)

    return {
        "task_id": task["task_id"],
        "dataset": "robotouille",
        "group": task.get("group", "unknown"),
        "environment_name": task.get("environment_name"),
        "goal_description": task.get("goal_description"),
        "goal_predicates": task.get("goal_predicates", []),
        "optimal_steps": task.get("optimal_steps"),
        "testing_seeds": task.get("testing_seeds", []),
        "verified_plan_status": f"stateful_ground_truth_{mode}_ready",
        "fsm_generation_mode": mode,
        "state_list": state_list,
    }


def collect_object_requirements(task: Mapping[str, Any]) -> list[ObjectRequirement]:
    raw_goals = task.get("goal_predicates", [])
    items = task.get("items", [])
    item_traits = defaultdict(set)
    for item in items:
        item_traits[str(item.get("name"))].update(str(pred) for pred in item.get("predicates", []))

    reqs: dict[tuple[str, int], ObjectRequirement] = {}
    synthetic_counter = defaultdict(int)
    for goal in raw_goals:
        predicate = str(goal.get("predicate", ""))
        args = [str(arg) for arg in goal.get("args", [])]
        ids = [int(item) for item in goal.get("ids", [])]
        if not args:
            continue

        if predicate == "container_at":
            name = args[0]
            object_id = ids[0] if ids else _next_id(synthetic_counter, name)
            req = reqs.setdefault((name, object_id), ObjectRequirement(name=name, object_id=object_id))
            req.serve_at_table = True
            req.predicates.append(dict(goal))
            continue

        if predicate == "isboiling" and args[0] == "water":
            name = "water"
            object_id = ids[0] if ids else _next_id(synthetic_counter, name)
            req = reqs.setdefault((name, object_id), ObjectRequirement(name=name, object_id=object_id))
            req.predicates.append(dict(goal))
            continue

        name = args[0]
        object_id = ids[0] if ids else _next_id(synthetic_counter, name)
        req = reqs.setdefault((name, object_id), ObjectRequirement(name=name, object_id=object_id))
        req.predicates.append(dict(goal))
        if predicate == "iscut":
            req.needs_cut = True
        elif predicate == "iscooked":
            req.needs_cook = True
        elif predicate == "isfried":
            req.needs_fry = True
        elif predicate == "addedto":
            req.add_to_water = True
        elif predicate == "item_at":
            req.serve_at_table = True
        elif predicate == "item_on":
            req.serve_on_table = True
        elif predicate == "clear":
            req.must_remain_clear = True

    # Fryable-if-cut items should explicitly keep the cut step if they appear in fry goals.
    for req in reqs.values():
        if req.needs_fry and "isfryableifcut" in item_traits.get(req.name, set()):
            req.needs_cut = True

    return sorted(reqs.values(), key=lambda item: (_priority(item), item.name, item.object_id))


def _next_id(counter: defaultdict[str, int], name: str) -> int:
    counter[name] += 1
    return counter[name]


def _priority(req: ObjectRequirement) -> int:
    if req.name == "water":
        return 0
    if req.needs_cut and not (req.needs_cook or req.needs_fry):
        return 1
    if req.needs_cook:
        return 2
    if req.needs_fry:
        return 3
    if req.add_to_water:
        return 4
    if req.serve_on_table or req.serve_at_table:
        return 5
    return 9


def ordered_serve_requirements(
    requirements: list[ObjectRequirement],
) -> list[ObjectRequirement]:
    return sorted(
        [req for req in requirements if req.serve_at_table or req.serve_on_table],
        key=lambda req: (_serve_priority(req), req.name, req.object_id),
    )


def _serve_priority(req: ObjectRequirement) -> int:
    if req.serve_on_table:
        return 0
    if req.must_remain_clear:
        return 3
    if req.needs_cook or req.needs_fry or req.needs_cut:
        return 1
    return 2


def build_cut_cluster(req: ObjectRequirement, add_state) -> None:
    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_CUT",
        f"PICK_UP_{alias}_FOR_CUT",
        f"NAVIGATE_TO_BOARD_WITH_{alias}",
        f"PLACE_{alias}_ON_BOARD_FOR_CUT",
        f"CUT_{alias}",
    ]
    add_state(
        names[0],
        "navigate",
        f"Reach the station containing {alias} so it can be cut.",
        [_move_action(f"station containing {alias}")],
        [f"robot is at station containing {alias}"],
        next_state=names[1],
        root_cluster=True,
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for cutting.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to a cutting board.",
        [_move_action("cutting board while holding item")],
        [f"robot is at board with {alias}"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "place-item",
        f"Place {alias} on the cutting board.",
        [_place_item_action(alias, "board")],
        [f"{alias} is on a board"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "cut",
        f"Cut {alias} until the goal predicate is satisfied.",
        [
            {
                "template": "cut",
                "item": alias,
                "target": "board",
                "constraint": "Repeat as needed until the simulator marks the item as cut.",
            }
        ],
        [f"iscut({alias})"],
    )


def build_cook_cluster(req: ObjectRequirement, add_state) -> None:
    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_COOK",
        f"PICK_UP_{alias}_FOR_COOK",
        f"NAVIGATE_TO_STOVE_WITH_{alias}",
        f"PLACE_{alias}_ON_STOVE",
        f"START_COOK_{alias}",
        f"{alias}_COOKING_BRANCH",
    ]
    add_state(
        names[0],
        "navigate",
        f"Reach the station containing {alias} so it can be cooked.",
        [_move_action(f"station containing {alias}")],
        [f"robot is at station containing {alias}"],
        next_state=names[1],
        root_cluster=True,
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for cooking.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to a stove.",
        [_move_action("stove while holding item")],
        [f"robot is at stove with {alias}"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "place-item",
        f"Place {alias} on the stove.",
        [_place_item_action(alias, "stove")],
        [f"{alias} is on stove"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "cook",
        f"Start cooking {alias}.",
        [
            {
                "template": "cook",
                "item": alias,
                "target": "stove",
                "constraint": "Only start cooking once the item is on the stove and not already cooking.",
            }
        ],
        [f"iscooking({alias}) or iscooked({alias})"],
        next_state=names[5],
    )
    add_state(
        names[5],
        "async-branch",
        f"Preserve {alias}'s cooking process while making progress on other pending subgoals.",
        [
            {
                "template": "wait",
                "constraint": f"Wait if {alias} is the only pending useful subgoal or if cooking is about to finish.",
            },
            {
                "template": "advance_pending_subgoal",
                "constraint": f"Only switch to another pending subgoal if it does not interrupt cooking {alias}.",
            },
        ],
        [f"iscooked({alias})"],
        notes=[
            "This is the canonical asynchronous branch state.",
            "Runtime verifier should forbid removing the item before it is fully cooked.",
        ],
    )


def build_branching_cook_cluster(req: ObjectRequirement, add_state) -> None:
    """Build a cook cluster that can repair common randomized-layout blockers.

    The linear ground-truth path assumes a free stove. Robotouille seed
    randomization can place irrelevant items on the stove, so the first state
    branches between clearing a blocked stove and taking the normal cook path.
    """

    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_COOK",
        f"PICK_UP_{alias}_FOR_COOK",
        f"NAVIGATE_TO_STOVE_WITH_{alias}",
        f"PLACE_{alias}_ON_STOVE",
        f"START_COOK_{alias}",
        f"{alias}_COOKING_BRANCH",
    ]
    clear_names = [
        f"PICK_UP_STOVE_BLOCKER_FOR_{alias}",
        f"NAVIGATE_TO_CLEAR_TABLE_WITH_STOVE_BLOCKER_FOR_{alias}",
        f"PLACE_STOVE_BLOCKER_ON_TABLE_FOR_{alias}",
    ]
    add_state(
        names[0],
        "guarded-cook-entry",
        f"Prepare an empty stove, then reach {alias} so it can be cooked.",
        [],
        [f"stove is ready for {alias}", f"robot is at station containing {alias}"],
        root_cluster=True,
        notes=[
            "This state deliberately branches before picking up the cook target.",
            "If every stove is blocked by another item, clear the stove first.",
        ],
        transitions=[
            _transition(
                _move_action("occupied stove"),
                clear_names[0],
                f"A stove is occupied by an item other than {alias}.",
                guard=f"stove_blocked_for:{alias}",
            ),
            _transition(
                _move_action(f"station containing {alias}"),
                names[1],
                f"At least one stove can accept {alias}; move to the item.",
                guard=f"stove_ready_for:{alias}",
            ),
        ],
    )
    add_state(
        clear_names[0],
        "clear-blocker",
        f"Pick up the top item blocking the stove before cooking {alias}.",
        [
            {
                "template": "pick-up-item",
                "target": "occupied stove",
                "constraint": "Pick up any clear item currently sitting on a stove.",
            }
        ],
        ["robot holds stove blocker"],
        next_state=clear_names[1],
    )
    add_state(
        clear_names[1],
        "navigate",
        f"Move the stove blocker to an empty table so {alias} can be cooked.",
        [_move_action("empty table while holding item")],
        ["robot is at an empty table with stove blocker"],
        next_state=clear_names[2],
    )
    add_state(
        clear_names[2],
        "place-item",
        f"Place the stove blocker on a table, then re-check stove readiness for {alias}.",
        [
            {
                "template": "place-item",
                "target": "table",
                "constraint": "Place the held blocker on an empty table; item identity is runtime-bound.",
            }
        ],
        ["stove blocker is no longer on stove"],
        next_state=names[0],
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for cooking.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to an empty stove.",
        [_move_action("empty stove while holding item")],
        [f"robot is at an empty stove with {alias}"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "place-item",
        f"Place {alias} directly on an empty stove.",
        [
            {
                "template": "place-item",
                "item": alias,
                "target": "empty stove",
                "constraint": (
                    f"The stove must be empty so {alias} is directly on the stove, "
                    "not stacked on another item."
                ),
            }
        ],
        [f"{alias} is directly on stove"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "cook",
        f"Start cooking {alias}.",
        [
            {
                "template": "cook",
                "item": alias,
                "target": "stove",
                "constraint": "Only start cooking once the item is directly on the stove.",
            }
        ],
        [f"iscooking({alias}) or iscooked({alias})"],
        next_state=names[5],
    )
    add_state(
        names[5],
        "async-branch",
        f"Preserve {alias}'s cooking process while making progress on other pending subgoals.",
        [
            {
                "template": "wait",
                "constraint": f"Wait if {alias} is the only pending useful subgoal or if cooking is about to finish.",
            },
            {
                "template": "advance_pending_subgoal",
                "constraint": f"Only switch to another pending subgoal if it does not interrupt cooking {alias}.",
            },
        ],
        [f"iscooked({alias})"],
        notes=[
            "This is the canonical asynchronous branch state.",
            "Runtime verifier should forbid removing the item before it is fully cooked.",
        ],
    )


def build_fry_cluster(req: ObjectRequirement, add_state) -> None:
    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_FRY",
        f"PICK_UP_{alias}_FOR_FRY",
        f"NAVIGATE_TO_FRYER_WITH_{alias}",
        f"PLACE_{alias}_ON_FRYER",
        f"START_FRY_{alias}",
        f"{alias}_FRYING_BRANCH",
    ]
    add_state(
        names[0],
        "navigate",
        f"Reach the station containing {alias} so it can be fried.",
        [_move_action(f"station containing {alias}")],
        [f"robot is at station containing {alias}"],
        next_state=names[1],
        root_cluster=True,
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for frying.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to a fryer.",
        [_move_action("fryer while holding item")],
        [f"robot is at fryer with {alias}"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "place-item",
        f"Place {alias} on the fryer.",
        [_place_item_action(alias, "fryer")],
        [f"{alias} is on fryer"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "fry",
        f"Start frying {alias}.",
        [
            {
                "template": "fry",
                "item": alias,
                "target": "fryer",
                "constraint": "Only fry once the item is ready for frying and placed on the fryer.",
            }
        ],
        [f"isfried({alias}) or frying in progress"],
        next_state=names[5],
    )
    add_state(
        names[5],
        "async-branch",
        f"Preserve {alias}'s frying process while advancing independent subgoals.",
        [
            {
                "template": "wait",
                "constraint": f"Wait if {alias} is the only pending useful subgoal or if frying is about to finish.",
            },
            {
                "template": "advance_pending_subgoal",
                "constraint": f"Only switch to another pending subgoal if it does not interrupt frying {alias}.",
            },
        ],
        [f"isfried({alias})"],
        notes=[
            "Runtime verifier should forbid removing the item before it is fully fried.",
        ],
    )


def build_branching_fry_cluster(req: ObjectRequirement, add_state) -> None:
    """Build a fry cluster with a guarded path for occupied fryers."""

    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_FRY",
        f"PICK_UP_{alias}_FOR_FRY",
        f"NAVIGATE_TO_FRYER_WITH_{alias}",
        f"PLACE_{alias}_ON_FRYER",
        f"START_FRY_{alias}",
        f"{alias}_FRYING_BRANCH",
    ]
    clear_names = [
        f"PICK_UP_FRYER_BLOCKER_FOR_{alias}",
        f"NAVIGATE_TO_CLEAR_TABLE_WITH_FRYER_BLOCKER_FOR_{alias}",
        f"PLACE_FRYER_BLOCKER_ON_TABLE_FOR_{alias}",
    ]
    add_state(
        names[0],
        "guarded-fry-entry",
        f"Prepare an empty fryer, then reach {alias} so it can be fried.",
        [],
        [f"fryer is ready for {alias}", f"robot is at station containing {alias}"],
        root_cluster=True,
        notes=[
            "This state branches before picking up the fry target.",
            "If every fryer is blocked by another item, clear the fryer first.",
        ],
        transitions=[
            _transition(
                _move_action("occupied fryer"),
                clear_names[0],
                f"A fryer is occupied by an item other than {alias}.",
                guard=f"fryer_blocked_for:{alias}",
            ),
            _transition(
                _move_action(f"station containing {alias}"),
                names[1],
                f"At least one fryer can accept {alias}; move to the item.",
                guard=f"fryer_ready_for:{alias}",
            ),
        ],
    )
    add_state(
        clear_names[0],
        "clear-blocker",
        f"Pick up the top item blocking the fryer before frying {alias}.",
        [
            {
                "template": "pick-up-item",
                "target": "occupied fryer",
                "constraint": "Pick up any clear item currently sitting on a fryer.",
            }
        ],
        ["robot holds fryer blocker"],
        next_state=clear_names[1],
    )
    add_state(
        clear_names[1],
        "navigate",
        f"Move the fryer blocker to an empty table so {alias} can be fried.",
        [_move_action("empty table while holding item")],
        ["robot is at an empty table with fryer blocker"],
        next_state=clear_names[2],
    )
    add_state(
        clear_names[2],
        "place-item",
        f"Place the fryer blocker on a table, then re-check fryer readiness for {alias}.",
        [
            {
                "template": "place-item",
                "target": "table",
                "constraint": "Place the held blocker on an empty table; item identity is runtime-bound.",
            }
        ],
        ["fryer blocker is no longer on fryer"],
        next_state=names[0],
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for frying.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to an empty fryer.",
        [_move_action("empty fryer while holding item")],
        [f"robot is at an empty fryer with {alias}"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "place-item",
        f"Place {alias} directly on an empty fryer.",
        [
            {
                "template": "place-item",
                "item": alias,
                "target": "empty fryer",
                "constraint": (
                    f"The fryer must be empty so {alias} is directly on the fryer, "
                    "not stacked on another item."
                ),
            }
        ],
        [f"{alias} is directly on fryer"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "fry",
        f"Start frying {alias}.",
        [
            {
                "template": "fry",
                "item": alias,
                "target": "fryer",
                "constraint": "Only fry once the item is ready for frying and directly on the fryer.",
            }
        ],
        [f"isfried({alias}) or frying in progress"],
        next_state=names[5],
    )
    add_state(
        names[5],
        "async-branch",
        f"Preserve {alias}'s frying process while advancing independent subgoals.",
        [
            {
                "template": "wait",
                "constraint": f"Wait if {alias} is the only pending useful subgoal or if frying is about to finish.",
            },
            {
                "template": "advance_pending_subgoal",
                "constraint": f"Only switch to another pending subgoal if it does not interrupt frying {alias}.",
            },
        ],
        [f"isfried({alias})"],
        notes=[
            "Runtime verifier should forbid removing the item before it is fully fried.",
        ],
    )


def build_soup_base_cluster(add_state) -> None:
    names = [
        "NAVIGATE_TO_POT_FOR_SOUP_BASE",
        "PICK_UP_POT_FOR_SOUP_BASE",
        "NAVIGATE_TO_SINK_WITH_POT",
        "PLACE_POT_ON_SINK",
        "FILL_POT_WITH_WATER",
        "PICK_UP_POT_WITH_WATER",
        "NAVIGATE_TO_STOVE_WITH_POT",
        "PLACE_POT_ON_STOVE",
        "START_BOIL_WATER",
        "WATER_BOILING_BRANCH",
    ]
    add_state(
        names[0],
        "navigate",
        "Reach the pot so soup preparation can start.",
        [_move_action("station containing pot")],
        ["robot is at station containing pot"],
        next_state=names[1],
        root_cluster=True,
    )
    add_state(
        names[1],
        "pick-up-container",
        "Pick up the pot.",
        [
            {
                "template": "pick-up-container",
                "container": "pot",
                "constraint": "Hands must be free and the robot must be at the pot station.",
            }
        ],
        ["robot has pot"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        "Bring the pot to the sink.",
        [_move_action("sink while holding pot")],
        ["robot is at sink with pot"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "place-container",
        "Place the pot on the sink.",
        [
            {
                "template": "place-container",
                "container": "pot",
                "target": "sink",
                "constraint": "The sink station must accept the pot.",
            }
        ],
        ["pot is at sink"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "fill-pot",
        "Fill the pot with water.",
        [
            {
                "template": "fill-pot",
                "container": "pot",
                "target": "sink",
                "constraint": "The pot must be empty and placed on the sink.",
            }
        ],
        ["pot contains water"],
        next_state=names[5],
    )
    add_state(
        names[5],
        "pick-up-container",
        "Pick up the water-filled pot.",
        [
            {
                "template": "pick-up-container",
                "container": "pot",
                "constraint": "Hands must be free and the pot must be at the sink.",
            }
        ],
        ["robot has pot with water"],
        next_state=names[6],
    )
    add_state(
        names[6],
        "navigate",
        "Bring the pot to an empty stove.",
        [_move_action("empty stove while holding water-filled pot")],
        ["robot is at empty stove with pot"],
        next_state=names[7],
    )
    add_state(
        names[7],
        "place-container",
        "Place the pot on the stove.",
        [
            {
                "template": "place-container",
                "container": "pot",
                "target": "empty stove",
                "constraint": "The stove must be empty so the pot is directly on the stove.",
            }
        ],
        ["pot is on stove"],
        next_state=names[8],
    )
    add_state(
        names[8],
        "boil-water",
        "Start boiling the water in the pot.",
        [
            {
                "template": "boil-water",
                "container": "pot",
                "meal": "water",
                "target": "stove",
                "constraint": "Water must already be in the pot and the pot must be on the stove.",
            }
        ],
        ["isboiling(water) or boiling in progress"],
        next_state=names[9],
    )
    add_state(
        names[9],
        "async-branch",
        "Preserve boiling water while advancing independent ingredient subgoals.",
        [
            {"template": "wait", "constraint": "Wait if boiling is the only remaining useful process."},
            {
                "template": "advance_pending_subgoal",
                "constraint": "Only switch to ingredient preparation or serving steps that do not interrupt boiling.",
            },
        ],
        ["isboiling(water)"],
        notes=[
            "Runtime verifier should prevent removing the pot if that would break the soup pipeline.",
        ],
    )


def build_add_to_water_cluster(req: ObjectRequirement, add_state) -> None:
    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_SOUP",
        f"PICK_UP_{alias}_FOR_SOUP",
        f"NAVIGATE_TO_POT_WITH_{alias}",
        f"ADD_{alias}_TO_POT",
    ]
    add_state(
        names[0],
        "navigate",
        f"Reach {alias} so it can be added to the soup.",
        [_move_action(f"station containing {alias}")],
        [f"robot is at station containing {alias}"],
        next_state=names[1],
        root_cluster=True,
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for the soup.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to the pot.",
        [_move_action("pot on stove while holding ingredient")],
        [f"robot is at pot with {alias}"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "add-to",
        f"Add {alias} into the soup pot.",
        [
            {
                "template": "add-to",
                "item": alias,
                "container": "pot",
                "constraint": "The pot must contain water and the robot must be holding the ingredient.",
            }
        ],
        [f"addedto({alias}, water)"],
    )


def build_fill_bowl_cluster(add_state) -> None:
    names = [
        "NAVIGATE_TO_POT_FOR_BOWL_FILL",
        "PICK_UP_POT_FOR_BOWL_FILL",
        "NAVIGATE_TO_BOWL_WITH_POT",
        "FILL_BOWL_WITH_SOUP",
        "NAVIGATE_TO_BOWL_FOR_SERVE",
        "PICK_UP_BOWL_FOR_SERVE",
        "NAVIGATE_TO_TABLE_WITH_BOWL",
        "PLACE_BOWL_ON_TABLE",
    ]
    add_state(
        names[0],
        "navigate",
        "Reach the pot so soup can be transferred to the bowl.",
        [_move_action("pot location")],
        ["robot is at pot"],
        next_state=names[1],
        root_cluster=True,
    )
    add_state(
        names[1],
        "pick-up-container",
        "Pick up the pot for bowl filling.",
        [
            {
                "template": "pick-up-container",
                "container": "pot",
                "constraint": "Hands must be free and the robot must be at the pot.",
            }
        ],
        ["robot has pot"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        "Bring the pot to the bowl.",
        [_move_action("bowl while holding pot")],
        ["robot is at bowl with pot"],
        next_state=names[3],
    )
    add_state(
        names[3],
        "fill-bowl",
        "Transfer soup from the pot into the bowl.",
        [
            {
                "template": "fill-bowl",
                "source_container": "pot",
                "target_container": "bowl",
                "constraint": "The bowl must be empty and the pot must contain the soup.",
            }
        ],
        ["in(water, bowl)"],
        next_state=names[4],
    )
    add_state(
        names[4],
        "navigate",
        "Reach the bowl so it can be served to the table.",
        [_move_action("station containing bowl")],
        ["robot is at bowl"],
        next_state=names[5],
    )
    add_state(
        names[5],
        "pick-up-container",
        "Pick up the filled bowl.",
        [
            {
                "template": "pick-up-container",
                "container": "bowl",
                "constraint": "Hands must be free and the bowl must be filled.",
            }
        ],
        ["robot has bowl"],
        next_state=names[6],
    )
    add_state(
        names[6],
        "navigate",
        "Bring the filled bowl to a serving table.",
        [_move_action("serving table while holding bowl")],
        ["robot is at serving table with bowl"],
        next_state=names[7],
    )
    add_state(
        names[7],
        "place-container",
        "Place the bowl on a serving table.",
        [
            {
                "template": "place-container",
                "container": "bowl",
                "target": "table",
                "constraint": "Choose a free serving table for the soup.",
            }
        ],
        ["container_at(bowl, table)"],
    )


def build_serve_cluster(req: ObjectRequirement, add_state) -> None:
    alias = req.alias
    names = [
        f"NAVIGATE_TO_{alias}_FOR_SERVE",
        f"PICK_UP_{alias}_FOR_SERVE",
        f"NAVIGATE_TO_TABLE_WITH_{alias}",
        f"PLACE_{alias}_ON_TABLE",
    ]
    add_state(
        names[0],
        "guarded-serve-entry",
        f"Reach or pick up {alias} so it can be served to a table.",
        [],
        [f"robot is at station containing {alias}"],
        root_cluster=True,
        transitions=[
            _transition(
                _pickup_item_action(alias),
                names[2],
                f"Robot is already at the station containing {alias}; pick it up.",
                guard=f"robot_at_station_containing:{alias}",
            ),
            _transition(
                _move_action(f"station containing {alias}"),
                names[1],
                f"Robot is not at {alias}; navigate to its station.",
                guard=f"not_robot_at_station_containing:{alias}",
            ),
        ],
    )
    add_state(
        names[1],
        "pick-up-item",
        f"Pick up {alias} for serving.",
        [_pickup_item_action(alias)],
        [f"robot has {alias}"],
        next_state=names[2],
    )
    add_state(
        names[2],
        "navigate",
        f"Bring {alias} to a serving table.",
        [_move_action("serving table while holding item")],
        [f"robot is at serving table with {alias}"],
        next_state=names[3],
    )
    serve_template = "place-item"
    serve_target_mode = "on" if req.serve_on_table else "at"
    notes = []
    if req.must_remain_clear:
        notes.append(
            f"{alias} is marked clear in the goal, so it should be placed as the top-most visible item."
        )
    add_state(
        names[3],
        serve_template,
        f"Serve {alias} on the target table.",
        [
            {
                "template": serve_template,
                "item": alias,
                "target": "table",
                "table_mode": serve_target_mode,
                "constraint": (
                    "Choose the serving table that keeps the dish decomposition reachable "
                    "and respects any top/clear requirement."
                ),
            }
        ],
        [f"{serve_target_mode}table({alias})"],
        notes=notes,
    )


def annotate_async_parallelism(state_list: list[dict[str, Any]]) -> None:
    future_roots: list[str] = []
    for state in reversed(state_list):
        name = state["name"]
        if state.get("kind") == "async-branch":
            branch_item = name.rsplit("_", 2)[0]
            own_serve_root = f"NAVIGATE_TO_{branch_item}_FOR_SERVE"
            state["parallel_subgoals"] = [
                root for root in reversed(future_roots[:8]) if root != own_serve_root
            ]
        if state.get("root_cluster"):
            future_roots.append(name)


def judge_and_fallback_task_fsm(
    task: Mapping[str, Any],
    candidate: dict[str, Any],
    llm_judge: Any | None,
    llm_required: bool,
    judge_attempts: int,
    fallback_candidate: Mapping[str, Any] | None = None,
    fallback_source: str = "deterministic_branching",
    fallback_selection_events: list[dict[str, Any]] | None = None,
    candidate_event: Mapping[str, Any] | None = None,
    force_fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Validate a generated FSM with an optional LLM judge and deterministic fallback."""

    candidate = json.loads(json.dumps(candidate))
    validation_issues = ground_truth_validation_issues(candidate)
    event: dict[str, Any] = {
        "candidate_mode": candidate.get("fsm_generation_mode"),
        "candidate_source": (candidate_event or {}).get("candidate_source", "unknown"),
        "candidate_event": dict(candidate_event or {}),
        "validation_issues": validation_issues,
        "deterministic_issues": deterministic_ground_truth_issues(candidate),
        "llm_judge_used": False,
        "llm_verdict": None,
        "fallback_used": False,
        "fallback_source": None,
        "fallback_selection_events": list(fallback_selection_events or []),
    }
    if force_fallback_reason:
        if llm_required:
            raise RuntimeError(
                f"LLM ground-truth generation failed for {task.get('task_id')}: "
                f"{force_fallback_reason}; event={event}"
            )
        return apply_ground_truth_fallback(
            task=task,
            fallback_candidate=fallback_candidate,
            fallback_source=fallback_source,
            event=event,
            reason=force_fallback_reason,
        )

    if validation_issues:
        if llm_required and event["candidate_event"].get("llm_generator_used"):
            raise RuntimeError(
                f"LLM-generated ground truth failed validation for {task.get('task_id')}: "
                f"{validation_issues}"
            )
        return apply_ground_truth_fallback(
            task=task,
            fallback_candidate=fallback_candidate,
            fallback_source=fallback_source,
            event=event,
            reason="candidate_validation_failed",
        )

    if llm_judge is None:
        candidate["ground_truth_generation"] = {
            **event,
            "status": "validation_pass_no_llm_judge",
        }
        return candidate

    event["llm_judge_used"] = True
    last_error = None
    judge_attempt_events: list[dict[str, Any]] = []
    for attempt_idx in range(1, max(1, int(judge_attempts)) + 1):
        try:
            judge_response = call_llm_ground_truth_judge(llm_judge, task, candidate)
            verdict = str(judge_response.get("verdict", "")).lower()
            judge_attempt_events.append(
                {
                    "attempt": attempt_idx,
                    "verdict": verdict,
                    "score": judge_response.get("score"),
                    "issues": judge_response.get("issues", []),
                    "repair_hints": judge_response.get("repair_hints", []),
                }
            )
            event["raw_llm_judge"] = judge_response
            event["llm_verdict"] = verdict
            event["llm_score"] = judge_response.get("score")
            event["llm_issues"] = judge_response.get("issues", [])
            patched = judge_response.get("patched_state_list")
            if verdict == "pass":
                candidate["ground_truth_generation"] = {
                    **event,
                    "judge_attempts": judge_attempt_events,
                    "status": "llm_judge_pass",
                }
                return candidate
            if isinstance(patched, list):
                patched_candidate = json.loads(json.dumps(candidate))
                patched_candidate["state_list"] = patched
                patched_candidate["fsm_generation_mode"] = "llm_judge_repaired"
                patched_candidate["verified_plan_status"] = "stateful_ground_truth_llm_judge_repaired"
                patched_issues = ground_truth_validation_issues(patched_candidate)
                if not patched_issues:
                    patched_candidate["ground_truth_generation"] = {
                        **event,
                        "judge_attempts": judge_attempt_events,
                        "status": "llm_judge_repaired",
                        "fallback_used": False,
                    }
                    return patched_candidate
                event["patched_issues"] = patched_issues
        except Exception as exc:
            last_error = str(exc)
            event.setdefault("llm_errors", []).append(last_error)
            judge_attempt_events.append({"attempt": attempt_idx, "error": last_error})

    if llm_required:
        raise RuntimeError(
            f"LLM judge did not approve {task.get('task_id')}; last_error={last_error}; event={event}"
        )

    event["judge_attempts"] = judge_attempt_events
    return apply_ground_truth_fallback(
        task=task,
        fallback_candidate=fallback_candidate,
        fallback_source=fallback_source,
        event=event,
        reason="llm_judge_failed",
    )


def apply_ground_truth_fallback(
    task: Mapping[str, Any],
    fallback_candidate: Mapping[str, Any] | None,
    fallback_source: str,
    event: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    fallback = (
        json.loads(json.dumps(fallback_candidate))
        if fallback_candidate is not None
        else build_task_fsm(task, mode="branching")
    )
    fallback_issues = ground_truth_validation_issues(fallback)
    if fallback_issues:
        raise RuntimeError(
            f"Fallback ground truth failed validation for {task.get('task_id')}: "
            f"source={fallback_source} issues={fallback_issues}"
        )

    status = (
        "fallback_to_last_good_ground_truth"
        if fallback_source == "last_good_ground_truth"
        else "fallback_to_deterministic_branching_ground_truth"
    )
    fallback["ground_truth_generation"] = {
        **dict(event),
        "status": status,
        "fallback_used": True,
        "fallback_reason": reason,
        "fallback_source": fallback_source,
        "fallback_issues": [],
    }
    return fallback


def generate_llm_ground_truth_candidate(
    llm_generator: Any,
    task: Mapping[str, Any],
    seed_candidate: Mapping[str, Any],
    attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the LLM to generate a ground-truth candidate and validate it locally."""

    event: dict[str, Any] = {
        "llm_generator_used": True,
        "candidate_source": "llm_generated_ground_truth",
        "accepted": False,
        "attempts": [],
    }
    last_candidate: dict[str, Any] | None = None
    for attempt_idx in range(1, max(1, int(attempts)) + 1):
        try:
            response = call_llm_ground_truth_generator(llm_generator, task, seed_candidate)
            candidate = normalize_llm_ground_truth_candidate(response, task, seed_candidate)
            issues = ground_truth_validation_issues(candidate)
            last_candidate = candidate
            event["attempts"].append(
                {
                    "attempt": attempt_idx,
                    "parsed": True,
                    "issues": issues,
                    "state_count": len(candidate.get("state_list", [])),
                }
            )
            if not issues:
                event["accepted"] = True
                event["accepted_attempt"] = attempt_idx
                return candidate, event
        except Exception as exc:
            event["attempts"].append(
                {
                    "attempt": attempt_idx,
                    "parsed": False,
                    "error": str(exc),
                }
            )

    if last_candidate is not None:
        return last_candidate, event

    invalid = json.loads(json.dumps(seed_candidate))
    invalid["state_list"] = []
    invalid["fsm_generation_mode"] = "llm_generation_failed"
    invalid["verified_plan_status"] = "stateful_ground_truth_llm_generation_failed"
    return invalid, event


def call_llm_ground_truth_generator(
    llm_generator: Any,
    task: Mapping[str, Any],
    seed_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "You generate Robotouille NS-FSM ground-truth JSON candidates. "
        "Return JSON only and do not include code fences. The output will be "
        "rejected unless it passes deterministic schema/reachability checks and "
        "a separate LLM judge."
    )
    compact_task = {
        "task_id": task.get("task_id"),
        "environment_name": task.get("environment_name"),
        "goal_description": task.get("goal_description"),
        "goal_predicates": task.get("goal_predicates", []),
        "items": task.get("items", []),
        "stations": task.get("stations", []),
        "candidate_action_skeleton": task.get("candidate_action_skeleton", []),
        "optimal_steps": task.get("optimal_steps"),
    }
    user_prompt = (
        "Generate a Robotouille ground-truth FSM candidate for this task.\n\n"
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "task_id": "same task id",\n'
        '  "verified_plan_status": "stateful_ground_truth_llm_candidate",\n'
        '  "fsm_generation_mode": "llm_generated",\n'
        '  "state_list": [\n'
        "    {\n"
        '      "name": "STATE_NAME",\n'
        '      "kind": "navigate | pick-up-item | place-item | cook | fry | async-branch | verify | terminal | ...",\n'
        '      "phase_goal": "short operational goal",\n'
        '      "fsm_allowed_actions": [{"template": "move", "target": "station containing item__id"}],\n'
        '      "completion_condition": ["short condition"],\n'
        '      "next_state_on_completion": "NEXT_STATE",\n'
        '      "root_cluster": false,\n'
        '      "transitions": null\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Hard requirements:\n"
        "- Preserve aliases like chicken__3 from goal ids.\n"
        "- Include DONE as a terminal state and VERIFY_GOAL before DONE.\n"
        "- Every non-terminal state must expose fsm_allowed_actions or guarded transitions.\n"
        "- For cooking, branch before pickup: clear an occupied stove if needed, otherwise navigate to the item.\n"
        "- Place cook targets and pots only on an empty stove.\n"
        "- For frying, branch before pickup: clear an occupied fryer if needed, otherwise navigate to the item.\n"
        "- Place fry targets only on an empty fryer.\n"
        "- Serving item states should support already-at-item pickup as a guarded branch.\n"
        "- Async cook branches finish on iscooked(alias); async fry branches finish on isfried(alias).\n\n"
        "The deterministic seed below is valid local style. You may reuse it, but "
        "return a complete state_list, not a diff.\n\n"
        f"Task:\n{json.dumps(compact_task, indent=2, sort_keys=True)}\n\n"
        f"Deterministic seed candidate:\n{json.dumps(seed_candidate, indent=2, sort_keys=True)}"
    )
    raw = llm_generator.generate(system_prompt, user_prompt)
    try:
        import sys

        src = str(REPO_ROOT / "ns-fsm-baseline" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from fsm_designer import parse_json_object

        parsed = parse_json_object(raw)
    except Exception:
        parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM ground-truth generator response must be a JSON object.")
    parsed["_raw_llm_response"] = raw
    return parsed


def normalize_llm_ground_truth_candidate(
    generated: Mapping[str, Any],
    task: Mapping[str, Any],
    seed_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    payload: Mapping[str, Any] = generated
    if isinstance(generated.get("tasks"), list):
        matching = [
            item
            for item in generated["tasks"]
            if isinstance(item, Mapping) and str(item.get("task_id")) == str(task.get("task_id"))
        ]
        payload = matching[0] if matching else generated["tasks"][0]
    elif isinstance(generated.get("ground_truth"), Mapping):
        payload = generated["ground_truth"]

    state_list = payload.get("state_list")
    if not isinstance(state_list, list) or not state_list:
        raise ValueError("LLM ground-truth candidate must include a non-empty state_list.")

    candidate = json.loads(json.dumps(seed_candidate))
    candidate["state_list"] = json.loads(json.dumps(state_list))
    candidate["task_id"] = str(task.get("task_id"))
    candidate["dataset"] = "robotouille"
    candidate["group"] = task.get("group", candidate.get("group", "unknown"))
    candidate["environment_name"] = task.get("environment_name", candidate.get("environment_name"))
    candidate["goal_description"] = task.get("goal_description", candidate.get("goal_description"))
    candidate["goal_predicates"] = task.get("goal_predicates", candidate.get("goal_predicates", []))
    candidate["optimal_steps"] = task.get("optimal_steps", candidate.get("optimal_steps"))
    candidate["testing_seeds"] = task.get("testing_seeds", candidate.get("testing_seeds", []))
    candidate["verified_plan_status"] = str(
        payload.get("verified_plan_status") or "stateful_ground_truth_llm_candidate"
    )
    candidate["fsm_generation_mode"] = str(payload.get("fsm_generation_mode") or "llm_generated")
    if payload.get("generation_notes"):
        candidate["llm_generation_notes"] = payload.get("generation_notes")
    if generated.get("_raw_llm_response"):
        candidate.setdefault("llm_generation_metadata", {})["raw_response"] = generated["_raw_llm_response"]
    return candidate


def call_llm_ground_truth_judge(
    llm_judge: Any,
    task: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict verifier for Robotouille NS-FSM ground-truth files. "
        "Return JSON only. Do not include code fences. Your job is to judge "
        "whether the state graph supports guarded alternatives for randomized "
        "layouts, especially blocked stoves/fryers and already-at-item cases."
    )
    compact_task = {
        "task_id": task.get("task_id"),
        "environment_name": task.get("environment_name"),
        "goal_description": task.get("goal_description"),
        "goal_predicates": task.get("goal_predicates", []),
        "items": task.get("items", []),
        "stations": task.get("stations", []),
        "optimal_steps": task.get("optimal_steps"),
    }
    user_prompt = (
        "Judge this Robotouille ground-truth FSM candidate.\n\n"
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "verdict": "pass" | "fail",\n'
        '  "score": 0.0,\n'
        '  "issues": ["short issue"],\n'
        '  "repair_hints": ["short hint"],\n'
        '  "patched_state_list": null\n'
        "}\n\n"
        "Pass only if non-terminal states have explicit outgoing transitions, "
        "cook setup can clear a blocked stove before picking up the cook target, "
        "fry setup can clear a blocked fryer before picking up the fry target, "
        "stove/fryer placement requires an empty destination, and serving can "
        "pick up an item when the robot is already at the item's station.\n\n"
        f"Task:\n{json.dumps(compact_task, indent=2, sort_keys=True)}\n\n"
        f"Candidate:\n{json.dumps(candidate, indent=2, sort_keys=True)}"
    )
    raw = llm_judge.generate(system_prompt, user_prompt)
    try:
        import sys

        src = str(REPO_ROOT / "ns-fsm-baseline" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from fsm_designer import parse_json_object

        parsed = parse_json_object(raw)
    except Exception:
        parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM judge response must be a JSON object.")
    return parsed


def ground_truth_validation_issues(candidate: Mapping[str, Any]) -> list[str]:
    issues = deterministic_ground_truth_issues(candidate)
    issues.extend(formal_ground_truth_issues(candidate))
    return list(dict.fromkeys(issues))


def formal_ground_truth_issues(candidate: Mapping[str, Any]) -> list[str]:
    try:
        import sys

        src = str(REPO_ROOT / "ns-fsm-baseline" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from datasets.robotouille import RobotouilleAdapter
        from fsm_validator import FSMDesignValidator

        adapter = RobotouilleAdapter()
        task_spec = adapter.to_task_spec(candidate).to_dict()
        design = adapter.build_grounded_fsm_design(task_spec)
        result = FSMDesignValidator().validate(
            design,
            task_spec=task_spec,
            adapter=adapter,
            allow_fallback=False,
        )
    except Exception as exc:
        return [f"formal_validation_error: {exc}"]

    if result.get("valid"):
        return []
    errors = [str(item) for item in result.get("errors", [])]
    return [f"formal_validation_failed: {error}" for error in errors]


def deterministic_ground_truth_issues(candidate: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    states = candidate.get("state_list", [])
    if not isinstance(states, list) or not states:
        return ["state_list is missing or empty"]
    state_names = {str(state.get("name")) for state in states if isinstance(state, Mapping)}
    if "DONE" not in state_names:
        issues.append("DONE terminal state is missing")

    for state in states:
        if not isinstance(state, Mapping):
            issues.append("state_list contains a non-object state")
            continue
        name = str(state.get("name", ""))
        kind = str(state.get("kind", ""))
        if not name:
            issues.append("state missing name")
        if kind != "terminal" and not state_transition_specs(state):
            issues.append(f"{name} has no outgoing action specs")
        for transition in state.get("transitions", []) or []:
            next_state = str(transition.get("next_state", ""))
            if next_state and next_state not in state_names:
                issues.append(f"{name} transition targets unknown state {next_state}")
            action = transition.get("action")
            if not isinstance(action, Mapping):
                issues.append(f"{name} transition has non-object action")

    for state in states:
        name = str(state.get("name", ""))
        if name.startswith("NAVIGATE_TO_") and name.endswith("_FOR_COOK"):
            guards = {
                str(transition.get("guard", ""))
                for transition in state.get("transitions", []) or []
            }
            if not any(guard.startswith("stove_blocked_for:") for guard in guards):
                issues.append(f"{name} lacks a blocked-stove repair transition")
            if not any(guard.startswith("stove_ready_for:") for guard in guards):
                issues.append(f"{name} lacks a stove-ready normal transition")
        if name.startswith("PLACE_") and name.endswith("_ON_STOVE"):
            actions = state_transition_specs(state)
            if not any(str(action.get("target")) == "empty stove" for action in actions):
                issues.append(f"{name} does not require an empty stove")
        if name.startswith("NAVIGATE_TO_") and name.endswith("_FOR_FRY"):
            guards = {
                str(transition.get("guard", ""))
                for transition in state.get("transitions", []) or []
            }
            if not any(guard.startswith("fryer_blocked_for:") for guard in guards):
                issues.append(f"{name} lacks a blocked-fryer repair transition")
            if not any(guard.startswith("fryer_ready_for:") for guard in guards):
                issues.append(f"{name} lacks a fryer-ready normal transition")
        if name.startswith("PLACE_") and name.endswith("_ON_FRYER"):
            actions = state_transition_specs(state)
            if not any(str(action.get("target")) == "empty fryer" for action in actions):
                issues.append(f"{name} does not require an empty fryer")
        if (
            state.get("kind") == "guarded-serve-entry"
            and name.startswith("NAVIGATE_TO_")
            and name.endswith("_FOR_SERVE")
        ):
            guards = {
                str(transition.get("guard", ""))
                for transition in state.get("transitions", []) or []
            }
            if not any(guard.startswith("robot_at_station_containing:") for guard in guards):
                issues.append(f"{name} lacks already-at-item pickup branch")
    return issues


def state_transition_specs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(state.get("transitions"), list):
        return [
            dict(transition.get("action"))
            for transition in state.get("transitions", [])
            if isinstance(transition, Mapping) and isinstance(transition.get("action"), Mapping)
        ]
    return [
        dict(action)
        for action in state.get("fsm_allowed_actions", [])
        if isinstance(action, Mapping)
    ]


def _transition(
    action: Mapping[str, Any],
    next_state: str,
    condition: str,
    guard: str | None = None,
) -> dict[str, Any]:
    transition = {
        "action": dict(action),
        "next_state": next_state,
        "condition": condition,
    }
    if guard:
        transition["guard"] = guard
    return transition


def _move_action(target: str) -> dict[str, Any]:
    return {
        "template": "move",
        "target": target,
        "constraint": "Only grounded move actions that make progress toward this target are allowed.",
    }


def _pickup_item_action(alias: str) -> dict[str, Any]:
    return {
        "template": "pick-up-item",
        "item": alias,
        "constraint": "Hands must be free and the robot must already be at the item's station.",
    }


def _place_item_action(alias: str, target: str) -> dict[str, Any]:
    return {
        "template": "place-item",
        "item": alias,
        "target": target,
        "constraint": f"The robot must be holding {alias} and be at the destination {target}.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
