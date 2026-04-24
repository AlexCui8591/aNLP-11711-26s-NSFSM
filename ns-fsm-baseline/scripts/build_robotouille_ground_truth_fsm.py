#!/usr/bin/env python3
"""Build stateful Robotouille ground-truth FSM files for NS-FSM.

The output is intentionally task-level and benchmark-stable:
- each state represents the next *kind* of work that should be done
- each state carries FSM-side allowed action templates
- runtime executable actions should still be intersected with simulator-valid
  grounded actions

This follows the user's desired decomposition:
    FSM state -> what should be done next
    Environment -> what is executable right now
    Final legal actions = intersection
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

    tasks = [build_task_fsm(task) for task in payload.get("tasks", [])]
    result = {
        "source": (
            "Robotouille official task JSON transformed into a stateful NS-FSM "
            "ground-truth file"
        ),
        "criterion": "state_list_with_fsm_allowed_actions",
        "split": args.split,
        "task_count": len(tasks),
        "testing_seeds": payload.get("testing_seeds", []),
        "runtime_rule": (
            "final_legal_actions = state.fsm_allowed_actions ∩ simulator.executable_actions"
        ),
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
        "--input",
        default=str(BASELINE_CONFIG / "robotouille_ground_truth_asynchronous_tasks.json"),
    )
    parser.add_argument(
        "--output",
        default=str(BASELINE_CONFIG / "robotouille_ground_truth_asynchronous_fsm.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_task_fsm(task: Mapping[str, Any]) -> dict[str, Any]:
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
    ) -> None:
        state = {
            "name": name,
            "kind": kind,
            "phase_goal": phase_goal,
            "fsm_allowed_actions": fsm_allowed_actions,
            "completion_condition": completion_condition,
        }
        if next_state:
            state["next_state_on_completion"] = next_state
        if notes:
            state["notes"] = notes
        if parallel_subgoals:
            state["parallel_subgoals"] = parallel_subgoals
        state_list.append(state)
        if root_cluster:
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
            build_cook_cluster(req, add_state)
        if req.needs_fry:
            build_fry_cluster(req, add_state)

    if soup_needed:
        for req in requirements:
            if req.add_to_water:
                build_add_to_water_cluster(req, add_state)
        build_fill_bowl_cluster(add_state)

    for req in requirements:
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
        "verified_plan_status": "stateful_ground_truth_ready",
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
        "Bring the pot to the stove.",
        [_move_action("stove while holding water-filled pot")],
        ["robot is at stove with pot"],
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
                "target": "stove",
                "constraint": "The stove must accept the pot.",
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
        "navigate",
        f"Reach {alias} so it can be served to a table.",
        [_move_action(f"station containing {alias}")],
        [f"robot is at station containing {alias}"],
        next_state=names[1],
        root_cluster=True,
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
            state["parallel_subgoals"] = list(reversed(future_roots[:8]))
        if name.startswith(
            (
                "NAVIGATE_TO_",
                "NAVIGATE_TO_POT_FOR_SOUP_BASE",
                "NAVIGATE_TO_POT_FOR_BOWL_FILL",
                "VERIFY_GOAL",
            )
        ):
            future_roots.append(name)


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
