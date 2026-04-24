#!/usr/bin/env python3
"""Extract Robotouille benchmark tasks into NS-FSM ground-truth JSON.

This script reads Robotouille's official evaluation configs and environment
JSON files, then emits a task file that mirrors the Minecraft buildable-task
JSON shape.  The output is intentionally conservative: generated action
sequences are marked as candidate skeletons until a human verifies them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTOUILLE_ROOT = REPO_ROOT / "Robotouille"
BASELINE_CONFIG = REPO_ROOT / "ns-fsm-baseline" / "config"

DEFAULT_EXPERIMENTS = {
    "asynchronous": ROBOTOUILLE_ROOT
    / "conf"
    / "experiments"
    / "ReAct"
    / "asynchronous"
    / "last-reasoning-action-mpc.yaml",
    "synchronous": ROBOTOUILLE_ROOT
    / "conf"
    / "experiments"
    / "ReAct"
    / "synchronous"
    / "last-reasoning-action-mpc.yaml",
}


def main() -> int:
    args = parse_args()
    split = args.split
    experiment_path = Path(args.experiment_config or DEFAULT_EXPERIMENTS[split])
    environments_dir = Path(args.environments_dir)
    domain_path = Path(args.domain)
    output_path = Path(args.output)

    env_names, optimal_steps, testing_seeds = parse_official_config(experiment_path)
    domain = load_json(domain_path)
    domain_actions = [str(action["name"]) for action in domain.get("action_defs", [])]

    tasks = []
    for idx, env_name in enumerate(env_names):
        env_path = environments_dir / f"{env_name}.json"
        env = load_json(env_path)
        task = build_task_record(
            env_name=env_name,
            split=split,
            env=env,
            env_path=env_path,
            optimal_steps=optimal_steps[idx] if idx < len(optimal_steps) else None,
            testing_seeds=testing_seeds,
            domain_actions=domain_actions,
        )
        tasks.append(task)

    payload = {
        "source": (
            "Robotouille official ReAct evaluation config + environment JSON + "
            "domain/robotouille.json"
        ),
        "criterion": "candidate_async_task_graph_needs_human_verification",
        "split": split,
        "task_count": len(tasks),
        "testing_seeds": testing_seeds,
        "tasks": tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote {output_path}")
    print(f"split={split} tasks={len(tasks)} seeds={len(testing_seeds)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["asynchronous", "synchronous"], default="asynchronous")
    parser.add_argument("--experiment-config", default="")
    parser.add_argument(
        "--environments-dir",
        default=str(ROBOTOUILLE_ROOT / "environments" / "env_generator" / "examples"),
    )
    parser.add_argument("--domain", default=str(ROBOTOUILLE_ROOT / "domain" / "robotouille.json"))
    parser.add_argument(
        "--output",
        default=str(BASELINE_CONFIG / "robotouille_ground_truth_asynchronous_tasks.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_official_config(path: Path) -> tuple[list[str], list[int], list[int]]:
    text = path.read_text(encoding="utf-8")
    env_names = [str(item) for item in _extract_quoted_list(text, "environment_names")]
    optimal_steps = [int(item) for item in _extract_number_list(text, "optimal_steps")]
    testing_seeds = [int(item) for item in _extract_number_list(text, "testing_seeds")]
    if not env_names:
        raise ValueError(f"No environment_names found in {path}")
    if optimal_steps and len(optimal_steps) != len(env_names):
        raise ValueError(
            f"optimal_steps length does not match environment_names in {path}: "
            f"{len(optimal_steps)} vs {len(env_names)}"
        )
    return env_names, optimal_steps, testing_seeds


def _extract_block(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}\s*:\s*\[(.*?)\]", text, re.DOTALL)
    return match.group(1) if match else ""


def _extract_quoted_list(text: str, key: str) -> list[str]:
    block = _extract_block(text, key)
    return re.findall(r'"([^"]+)"', block)


def _extract_number_list(text: str, key: str) -> list[int]:
    block = _extract_block(text, key)
    return [int(item) for item in re.findall(r"-?\d+", block)]


def build_task_record(
    env_name: str,
    split: str,
    env: Mapping[str, Any],
    env_path: Path,
    optimal_steps: int | None,
    testing_seeds: list[int],
    domain_actions: list[str],
) -> dict[str, Any]:
    goal_predicates = normalize_goal_predicates(env.get("goal", []))
    items = normalize_entities(env.get("items", []))
    stations = normalize_entities(env.get("stations", []))
    containers = normalize_entities(env.get("containers", []))
    meals = normalize_entities(env.get("meals", []))

    skeleton = derive_candidate_skeleton(goal_predicates, items, stations, env.get("config", {}))

    return {
        "task_id": f"robotouille/{env_name}",
        "dataset": "robotouille",
        "group": split,
        "environment_name": env_name,
        "environment_path": str(env_path.relative_to(REPO_ROOT)),
        "goal_description": str(env.get("goal_description", "")),
        "goal_predicates": goal_predicates,
        "stations": stations,
        "items": items,
        "containers": containers,
        "meals": meals,
        "config": env.get("config", {}),
        "domain_actions": domain_actions,
        "optimal_steps": optimal_steps,
        "testing_seeds": testing_seeds,
        "fsm_mode": "robotouille_human_verified_task_graph",
        "verified_plan_status": "needs_human_verification",
        "human_verified_actions": [],
        "candidate_action_skeleton": skeleton,
        "candidate_state_graph": build_candidate_state_graph(skeleton, split),
    }


def normalize_goal_predicates(raw_goals: Any) -> list[dict[str, Any]]:
    goals = []
    for raw_goal in raw_goals or []:
        goals.append(
            {
                "predicate": str(raw_goal.get("predicate", "")),
                "args": [str(arg) for arg in raw_goal.get("args", [])],
                "ids": [int(item) for item in raw_goal.get("ids", [])],
            }
        )
    return goals


def normalize_entities(raw_entities: Any) -> list[dict[str, Any]]:
    entities = []
    for raw in raw_entities or []:
        item = dict(raw)
        if "predicates" in item:
            item["predicates"] = [str(pred) for pred in item.get("predicates", [])]
        entities.append(item)
    return entities


def derive_candidate_skeleton(
    goal_predicates: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
    stations: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Produce conservative subgoal-level actions for human verification."""

    item_traits = {
        str(item.get("name")): set(str(pred) for pred in item.get("predicates", []))
        for item in items
    }
    station_names = {str(station.get("name")) for station in stations}
    has_board = "board" in station_names
    has_stove = "stove" in station_names
    has_fryer = "fryer" in station_names

    skeleton: list[dict[str, Any]] = []
    seen: set[str] = set()
    for goal in goal_predicates:
        predicate = str(goal.get("predicate", ""))
        args = [str(arg) for arg in goal.get("args", [])]
        if not args:
            continue
        item = args[0]
        key = f"{predicate}:{','.join(args)}"
        if key in seen:
            continue
        seen.add(key)

        if predicate == "iscooked" and has_stove:
            skeleton.append(
                {
                    "subgoal": f"cook {item}",
                    "requires_async_wait": True,
                    "legal_action_templates": [
                        f"move_to_{item}",
                        f"pick_up_{item}",
                        "move_to_stove",
                        f"place_{item}_on_stove",
                        f"cook_{item}",
                        f"wait_until_{item}_cooked",
                    ],
                    "source_goal": goal,
                }
            )
        elif predicate == "isfried" and has_fryer:
            if "isfryableifcut" in item_traits.get(item, set()) and has_board:
                skeleton.append(
                    {
                        "subgoal": f"cut {item} before frying",
                        "requires_async_wait": False,
                        "legal_action_templates": [
                            f"move_to_{item}",
                            f"pick_up_{item}",
                            "move_to_board",
                            f"place_{item}_on_board",
                            f"cut_{item}",
                        ],
                        "source_goal": goal,
                    }
                )
            skeleton.append(
                {
                    "subgoal": f"fry {item}",
                    "requires_async_wait": True,
                    "legal_action_templates": [
                        f"move_to_{item}",
                        f"pick_up_{item}",
                        "move_to_fryer",
                        f"place_{item}_on_fryer",
                        f"fry_{item}",
                        f"wait_until_{item}_fried",
                    ],
                    "source_goal": goal,
                }
            )
        elif predicate == "iscut" and has_board:
            num_cuts = _config_lookup(config.get("num_cuts", {}), item, default=3)
            skeleton.append(
                {
                    "subgoal": f"cut {item}",
                    "requires_async_wait": False,
                    "legal_action_templates": [
                        f"move_to_{item}",
                        f"pick_up_{item}",
                        "move_to_board",
                        f"place_{item}_on_board",
                    ]
                    + [f"cut_{item}" for _ in range(num_cuts)],
                    "source_goal": goal,
                }
            )
        elif predicate == "atop" and len(args) >= 2:
            top, bottom = args[0], args[1]
            skeleton.append(
                {
                    "subgoal": f"stack {top} on {bottom}",
                    "requires_async_wait": False,
                    "legal_action_templates": [
                        f"move_to_{top}",
                        f"pick_up_{top}",
                        f"move_to_{bottom}",
                        f"stack_{top}_on_{bottom}",
                    ],
                    "source_goal": goal,
                }
            )
        elif predicate in {"item_at", "item_on"} and len(args) >= 2:
            target = args[1]
            skeleton.append(
                {
                    "subgoal": f"place {item} at {target}",
                    "requires_async_wait": False,
                    "legal_action_templates": [
                        f"move_to_{item}",
                        f"pick_up_{item}",
                        f"move_to_{target}",
                        f"place_{item}_on_{target}",
                    ],
                    "source_goal": goal,
                }
            )
        else:
            skeleton.append(
                {
                    "subgoal": f"satisfy {predicate}({', '.join(args)})",
                    "requires_async_wait": False,
                    "legal_action_templates": [],
                    "source_goal": goal,
                    "note": "No heuristic action skeleton; fill manually.",
                }
            )
    return skeleton


def _config_lookup(raw: Any, item: str, default: int) -> int:
    if not isinstance(raw, Mapping):
        return default
    return int(raw.get(item, raw.get("default", default)))


def build_candidate_state_graph(
    skeleton: list[Mapping[str, Any]],
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    graph: dict[str, list[dict[str, Any]]] = {"START": [], "DONE": []}
    previous_state = "START"
    for idx, subgoal in enumerate(skeleton, start=1):
        state = f"SUBGOAL_{idx}"
        actions = [str(action) for action in subgoal.get("legal_action_templates", [])]
        graph.setdefault(previous_state, [])
        if actions:
            graph[previous_state].append(
                {
                    "action": actions[0],
                    "next_state": state,
                    "condition": f"Begin candidate subgoal: {subgoal.get('subgoal')}",
                }
            )
        graph[state] = [
            {
                "action": action,
                "next_state": state,
                "condition": f"Candidate legal action for {subgoal.get('subgoal')}",
            }
            for action in actions
        ]
        if subgoal.get("requires_async_wait") and split == "asynchronous":
            graph[state].append(
                {
                    "action": "continue_independent_subtask_while_waiting",
                    "next_state": state,
                    "condition": "Async branch: preserve cooking/frying state while doing useful work.",
                }
            )
        previous_state = state

    if previous_state != "START":
        graph.setdefault(previous_state, []).append(
            {
                "action": "verify_goal_complete",
                "next_state": "DONE",
                "condition": "All candidate subgoals have been satisfied.",
            }
        )
    return graph


if __name__ == "__main__":
    raise SystemExit(main())
