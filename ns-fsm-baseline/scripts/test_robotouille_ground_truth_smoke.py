#!/usr/bin/env python3
"""Smoke-test Robotouille ground-truth FSM files against the current NS-FSM stack.

This is intentionally a *formal-layer* test, not full Robotouille execution.
It checks whether one Robotouille task can be compiled into the current
FSMDesign schema, pass FSMDesignValidator + DatalogVerifier, and instantiate a
RuntimeFSM with sensible legal actions at the initial state.

Why this exists:
- the repo does not yet have a RobotouilleAdapter wired into run_nsfsm_experiment.py
- but we still want a real "under the NS-FSM pipeline" test of the ground-truth
  file format and formal validation logic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fsm import RuntimeFSM
from fsm_validator import FSMDesignValidator
from datasets.robotouille import RobotouilleAdapter


DEFAULT_GROUND_TRUTH = ROOT / "config" / "robotouille_ground_truth_asynchronous_fsm.json"


def main() -> int:
    args = parse_args()
    payload = load_json(Path(args.ground_truth))
    task = select_task(payload, args.task_id)

    print(f"[1/4] Loaded task: {task['task_id']}")
    print(f"      Goal: {task.get('goal_description', '')}")
    print(f"      States: {len(task.get('state_list', []))}")

    validate_state_list_schema(task)
    print("[2/4] Ground-truth state_list schema looks complete.")

    task_spec, design, adapter = compile_to_fsm_design(task)
    print("[3/4] Compiled task into current FSMDesign schema.")
    print(f"      Initial state: {design['initial_state']}")
    print(f"      Terminal states: {design['terminal_states']}")
    print(f"      Unique actions: {len(task_spec['available_tools'])}")

    validator = FSMDesignValidator()
    result = validator.validate(
        design,
        task_spec=task_spec,
        adapter=adapter,
        allow_fallback=False,
    )
    print("[4/4] FSMDesignValidator result:")
    print(f"      valid={result['valid']}")
    if result["warnings"]:
        print("      warnings:")
        for item in result["warnings"][:10]:
            print(f"        - {item}")
    if result["errors"]:
        print("      errors:")
        for item in result["errors"][:20]:
            print(f"        - {item}")
        return 1

    runtime_fsm = RuntimeFSM(result["fsm_design"], task_spec=task_spec, adapter=adapter)
    initial_actions = runtime_fsm.get_valid_actions()
    print("\nRuntimeFSM smoke:")
    print(f"  current_state={runtime_fsm.current_state}")
    print(f"  legal_actions_at_initial={len(initial_actions)}")
    for action in initial_actions[:12]:
        print(f"    - {action}")

    print(
        "\nPASS: ground-truth file is compatible with the current formal NS-FSM layer.\n"
        "NOTE: this validates the adapter-compiled FSM and formal layer; it does not\n"
        "reset or step the external Robotouille simulator."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument(
        "--task-id",
        default="robotouille/asynchronous/0_cheese_chicken_sandwich",
        help="Robotouille task_id to smoke-test.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_task(payload: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    for task in payload.get("tasks", []):
        if str(task.get("task_id")) == task_id:
            return dict(task)
    raise SystemExit(f"Task not found in ground-truth file: {task_id}")


def validate_state_list_schema(task: Mapping[str, Any]) -> None:
    required = {"name", "kind", "phase_goal", "fsm_allowed_actions", "completion_condition"}
    states = task.get("state_list", [])
    if not isinstance(states, list) or not states:
        raise SystemExit("state_list is empty or missing.")
    for idx, state in enumerate(states):
        missing = required - set(state)
        if missing:
            raise SystemExit(
                f"State {idx} ({state.get('name')}) is missing required fields: {sorted(missing)}"
            )
        if not isinstance(state.get("fsm_allowed_actions"), list):
            raise SystemExit(f"State {state.get('name')} has non-list fsm_allowed_actions.")
        if not isinstance(state.get("completion_condition"), list):
            raise SystemExit(f"State {state.get('name')} has non-list completion_condition.")


def compile_to_fsm_design(
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], RobotouilleAdapter]:
    adapter = RobotouilleAdapter()
    task_spec = adapter.to_task_spec(task).to_dict()
    design = adapter.build_grounded_fsm_design(task_spec)
    return task_spec, design, adapter


def _next_state_name(states: list[Mapping[str, Any]], idx: int) -> str | None:
    if idx + 1 < len(states):
        return str(states[idx + 1]["name"])
    return None


def canonical_action_name(action: Mapping[str, Any]) -> str:
    parts = [str(action.get("template", "")).strip()]
    for key in ("item", "target", "container", "meal", "source_container", "target_container"):
        value = str(action.get(key, "")).strip()
        if value:
            parts.append(f"{key}={value}")
    table_mode = str(action.get("table_mode", "")).strip()
    if table_mode:
        parts.append(f"table_mode={table_mode}")
    return "|".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
