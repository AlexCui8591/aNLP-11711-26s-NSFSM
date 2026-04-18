"""Level 1 structured scenario adapter.

Structured scenarios sit between free-form user instructions and full benchmark
records: the user provides a task, a finite action list, and optionally a
success condition. This adapter can also build a simple linear FSM from that
action list.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping

try:
    from .base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict


class StructuredScenarioAdapter(DatasetAdapter):
    """Adapter for user-provided task/action/success-condition schemas."""

    dataset_name = "structured"

    def __init__(self):
        self.task_spec: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def list_tasks(self) -> list[dict[str, Any]]:
        return []

    def load_or_wrap(self, raw_input: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_input, str):
            return {
                "task": raw_input.strip(),
                "actions": [],
                "success_condition": "A final answer or artifact is produced.",
            }

        raw = dict(raw_input)
        task = raw.get("task") or raw.get("instruction")
        if not task:
            raise ValueError("StructuredScenarioAdapter requires a task/instruction.")
        raw["task"] = str(task).strip()
        raw["actions"] = [str(action).strip() for action in raw.get("actions", []) if str(action).strip()]
        raw.setdefault("success_condition", "A final answer or artifact is produced.")
        return raw

    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        raw = self.load_or_wrap(raw_task)
        task_hash = hashlib.sha1(
            (raw["task"] + repr(raw.get("actions", []))).encode("utf-8")
        ).hexdigest()[:12]
        actions = list(raw.get("actions", []))
        success_condition = str(raw.get("success_condition", "")).strip()
        available_tools = list(dict.fromkeys(actions + ["finalize"]))

        return TaskSpec(
            dataset=self.dataset_name,
            task_id=raw.get("task_id", f"structured/{task_hash}"),
            task_type=raw.get("task_type", "structured_tool_use"),
            instruction=raw["task"],
            initial_state={
                "completed_actions": [],
                "current_index": 0,
                "finalized": False,
            },
            goal_condition={"success_condition": success_condition, "finalized": True},
            available_tools=available_tools,
            max_steps=int(raw.get("max_steps", max(10, len(actions) + 3))),
            success_criteria=[success_condition] if success_condition else [],
            metadata={
                **dict(raw.get("metadata", {})),
                "actions": actions,
                "success_condition": success_condition,
            },
        )

    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        self.task_spec = task_spec_to_dict(task_spec)
        self.state = deepcopy(self.task_spec.get("initial_state", {}))
        self.state.setdefault("completed_actions", [])
        self.state.setdefault("current_index", 0)
        self.state.setdefault("finalized", False)
        self.state.setdefault("step_count", 0)
        self.state.setdefault("last_action", None)
        self.state.setdefault("last_action_status", "not_started")
        self.state.setdefault("last_error", None)
        self.history = []
        return self.get_observation()

    def step(self, action: str | Mapping[str, Any]) -> StepResult:
        if self.task_spec is None:
            raise RuntimeError("Call reset(task_spec) before step(action).")

        action_name, payload = self._split_action(action)
        actions = self._actions(self.task_spec)
        current_index = int(self.state.get("current_index", 0))
        expected = actions[current_index] if current_index < len(actions) else "finalize"
        info: dict[str, Any] = {
            "action": action_name,
            "success": False,
            "message": "",
        }

        if action_name == "finalize":
            if current_index >= len(actions):
                self.state["finalized"] = True
                self.state["final_answer"] = payload.get("answer") or payload.get("summary") or "Finalized."
                info.update({"success": True, "message": "Structured scenario finalized."})
            else:
                info["message"] = f"Cannot finalize before completing expected action: {expected}"
            self._record(action_name, info)
            return StepResult(self.get_observation(), self.is_done(self.state, self.task_spec), info)

        if action_name != expected:
            info["message"] = f"Expected next action '{expected}', got '{action_name}'."
            self._record(action_name, info)
            return StepResult(self.get_observation(), self.is_done(self.state, self.task_spec), info)

        self.state.setdefault("completed_actions", []).append(action_name)
        self.state["current_index"] = current_index + 1
        info.update({"success": True, "message": f"Completed structured action: {action_name}"})
        self._record(action_name, info)
        return StepResult(self.get_observation(), self.is_done(self.state, self.task_spec), info)

    def get_observation(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def is_done(
        self,
        state: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any],
    ) -> bool:
        spec = task_spec_to_dict(task_spec)
        return bool(state.get("finalized")) or int(state.get("step_count", 0)) >= int(
            spec.get("max_steps", 30)
        )

    def get_available_tools(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        spec = task_spec_to_dict(task_spec)
        actions = self._actions(spec)
        index = int(state.get("current_index", 0))
        if index < len(actions):
            return [actions[index]]
        return ["finalize"]

    def normalize_action(
        self,
        raw_action: str | Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> str | None:
        action_name, _ = self._split_action(raw_action)
        legal_names = [
            str(item.get("action", "")) if isinstance(item, Mapping) else str(item)
            for item in legal_actions
        ]
        if action_name in legal_names:
            return action_name
        lowered = {name.lower(): name for name in legal_names}
        return lowered.get(action_name.lower())

    def format_state_for_prompt(self, state: Mapping[str, Any]) -> str:
        return "\n".join(
            [
                f"completed_actions: {state.get('completed_actions', [])}",
                f"current_index: {state.get('current_index', 0)}",
                f"finalized: {bool(state.get('finalized'))}",
                f"last_action: {state.get('last_action')}",
                f"last_action_status: {state.get('last_action_status')}",
            ]
        )

    def summarize_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "success": bool(state.get("finalized")),
            "total_steps": int(state.get("step_count", 0)),
            "termination": "success" if state.get("finalized") else "max_steps_or_incomplete",
            "completed_actions": list(state.get("completed_actions", [])),
            "history": deepcopy(self.history),
        }

    def build_fsm_design(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        """Build a deterministic linear FSM from the structured action list."""

        spec = task_spec_to_dict(task_spec)
        actions = self._actions(spec)
        if not actions:
            return {
                "states": ["START", "DONE"],
                "initial_state": "START",
                "terminal_states": ["DONE"],
                "transitions_by_state": {
                    "START": [
                        {
                            "action": "finalize",
                            "next_state": "DONE",
                            "condition": "No explicit actions were provided; final output is ready.",
                        }
                    ],
                    "DONE": [],
                },
                "fallback_policy": {"on_invalid_action": "retry_with_valid_transition"},
                "success_signals": spec.get("success_criteria", []),
                "risk_notes": ["Linear FSM generated from structured user action list."],
            }

        states = ["START"] + [f"ACTION_{idx + 1}_{self._state_slug(action)}" for idx, action in enumerate(actions)] + ["DONE"]
        transitions: dict[str, list[dict[str, str]]] = {}
        transitions["START"] = [
            {
                "action": actions[0],
                "next_state": states[1],
                "condition": f"Begin by executing {actions[0]}.",
            }
        ]
        for idx, action in enumerate(actions[1:], start=1):
            transitions[states[idx]] = [
                {
                    "action": action,
                    "next_state": states[idx + 1],
                    "condition": f"Previous action completed; execute {action}.",
                }
            ]
        transitions[states[-2]] = [
            {
                "action": "finalize",
                "next_state": "DONE",
                "condition": "All provided actions are complete and success condition can be reported.",
            }
        ]
        transitions["DONE"] = []

        return {
            "states": states,
            "initial_state": "START",
            "terminal_states": ["DONE"],
            "transitions_by_state": transitions,
            "fallback_policy": {"on_invalid_action": "retry_with_valid_transition"},
            "success_signals": spec.get("success_criteria", []),
            "risk_notes": ["Linear FSM generated from structured user action list."],
        }

    def _record(self, action_name: str, info: Mapping[str, Any]) -> None:
        self.state["step_count"] = int(self.state.get("step_count", 0)) + 1
        self.state["last_action"] = action_name
        self.state["last_action_status"] = "success" if info.get("success") else "failed"
        self.state["last_error"] = None if info.get("success") else info.get("message", "")
        self.history.append(
            {
                "step": self.state["step_count"],
                "action": action_name,
                "success": bool(info.get("success")),
                "message": info.get("message", ""),
                "state": self.get_observation(),
            }
        )

    @staticmethod
    def _split_action(action: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        if isinstance(action, Mapping):
            name = str(action.get("action") or action.get("name") or "").strip()
            payload = dict(action.get("payload") or {})
            return name, payload
        return str(action).strip(), {}

    @staticmethod
    def _actions(task_spec: Mapping[str, Any]) -> list[str]:
        metadata = task_spec.get("metadata", {})
        return [str(action) for action in metadata.get("actions", [])]

    @staticmethod
    def _state_slug(value: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in value.upper()).strip("_") or "STEP"
