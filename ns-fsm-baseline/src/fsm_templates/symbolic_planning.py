"""FSM template for symbolic dependency-path planning tasks."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from ..datasets.base import TaskSpec, task_spec_to_dict
    from .base import FSMTemplate
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict
    from fsm_templates.base import FSMTemplate


class SymbolicPlanningFSM(FSMTemplate):
    name = "symbolic_planning"

    def build_design(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
    ) -> dict[str, Any]:
        spec = task_spec_to_dict(task_spec)
        metadata = spec.get("metadata", {})
        sequence = list(metadata.get("optimal_sequence") or metadata.get("required_actions") or [])
        if not sequence and adapter is not None:
            goal = metadata.get("goal_item")
            try:
                sequence = list(adapter._optimal_sequence(goal)) if goal else []
            except Exception:
                sequence = []
        if not sequence:
            sequence = [str(action) for action in spec.get("available_tools", [])[:1]]
        if metadata.get("grounded_fsm_mode") == "branching_dependency_dag":
            return _branching_dependency_design(
                actions=sequence,
                success_signals=spec.get("success_criteria", []),
                risk_note=(
                    "Minecraft grounded FSM generated as a branching dependency DAG. "
                    "Runtime legal actions are intersected with MC-TextWorld executable actions."
                ),
            )
        return _linear_design(
            actions=sequence,
            success_signals=spec.get("success_criteria", []),
            risk_note="Symbolic planning FSM generated from dependency action path.",
        )


def _linear_design(
    actions: list[str],
    success_signals: list[str],
    risk_note: str,
) -> dict[str, Any]:
    actions = [str(action) for action in actions if str(action)]
    if not actions:
        return {
            "states": ["START", "DONE"],
            "initial_state": "START",
            "terminal_states": ["DONE"],
            "transitions_by_state": {"START": [], "DONE": []},
            "fallback_policy": {"on_invalid_action": "retry_with_valid_transition"},
            "success_signals": success_signals,
            "risk_notes": [risk_note, "No dependency actions were available."],
        }

    middle_states = [f"STEP_{idx}" for idx in range(1, len(actions))]
    states = ["START"] + middle_states + ["DONE"]
    transitions: dict[str, list[dict[str, str]]] = {}
    for idx, action in enumerate(actions):
        source = "START" if idx == 0 else f"STEP_{idx}"
        target = "DONE" if idx == len(actions) - 1 else f"STEP_{idx + 1}"
        transitions[source] = [
            {
                "action": action,
                "next_state": target,
                "condition": f"Execute dependency-path action {idx + 1}: {action}.",
            }
        ]
    transitions["DONE"] = []
    return {
        "states": states,
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "transitions_by_state": transitions,
        "fallback_policy": {
            "on_invalid_action": "choose_next_dependency_action",
            "on_dead_end": "retry_from_last_dependency_state",
        },
        "success_signals": success_signals,
        "risk_notes": [risk_note],
    }


def _branching_dependency_design(
    actions: list[str],
    success_signals: list[str],
    risk_note: str,
) -> dict[str, Any]:
    """Build a compact branching FSM over grounded dependency actions.

    The FSM statically permits every dependency action from START/IN_PROGRESS,
    while the runtime intersects this list with the current MC-TextWorld
    executable-action set. This gives each state multiple safe possible paths
    without allowing actions outside the verified dependency closure.
    """

    actions = list(dict.fromkeys(str(action) for action in actions if str(action)))
    if not actions:
        return {
            "states": ["START", "DONE"],
            "initial_state": "START",
            "terminal_states": ["DONE"],
            "transitions_by_state": {"START": [], "DONE": []},
            "fallback_policy": {"on_invalid_action": "fail_no_dependency_actions"},
            "success_signals": success_signals,
            "risk_notes": [risk_note, "No dependency actions were available."],
        }

    final_action = actions[-1]
    transitions = {
        "START": _branching_options(actions, final_action),
        "IN_PROGRESS": _branching_options(actions, final_action),
        "DONE": [],
    }
    return {
        "states": ["START", "IN_PROGRESS", "DONE"],
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "transitions_by_state": transitions,
        "fallback_policy": {
            "on_invalid_action": "retry_with_current_executable_dependency_action",
            "on_dead_end": "fail_fast_no_executable_dependency_action",
        },
        "success_signals": success_signals,
        "risk_notes": [risk_note],
    }


def _branching_options(actions: list[str], final_action: str) -> list[dict[str, str]]:
    options = []
    for action in actions:
        next_state = "DONE" if action == final_action else "IN_PROGRESS"
        options.append(
            {
                "action": action,
                "next_state": next_state,
                "condition": (
                    "Action is in the grounded dependency closure and is "
                    "currently executable in MC-TextWorld."
                ),
            }
        )
    return options
