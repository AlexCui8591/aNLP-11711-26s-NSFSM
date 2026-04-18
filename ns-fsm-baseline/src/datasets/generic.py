"""Generic natural-language scenario adapter.

This is the Level 0 path: a user can provide only an instruction, and the
pipeline can still create a TaskSpec, design an FSM, and run workflow-level
actions without a benchmark-specific environment.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping

try:
    from .base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict


GENERIC_TOOLS = [
    "read_task",
    "inspect_context",
    "ask_clarifying_question",
    "write_plan",
    "execute_step",
    "check_progress",
    "revise",
    "finalize",
]


class GenericScenarioAdapter(DatasetAdapter):
    """Adapter for arbitrary user-provided scenarios."""

    dataset_name = "generic"

    def __init__(self, repo_root: str | None = None):
        self.repo_root = repo_root
        self.task_spec: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def list_tasks(self) -> list[dict[str, Any]]:
        return []

    def load_or_wrap(self, raw_input: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_input, str):
            return {
                "instruction": raw_input.strip(),
                "metadata": {"source": "user_instruction"},
            }

        raw = dict(raw_input)
        instruction = raw.get("instruction") or raw.get("task") or raw.get("prompt")
        if not instruction:
            raise ValueError("GenericScenarioAdapter requires an instruction/task string.")
        raw["instruction"] = str(instruction).strip()
        raw.setdefault("metadata", {})
        raw["metadata"].setdefault("source", "user_instruction")
        return raw

    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        raw = self.load_or_wrap(raw_task)
        instruction = raw["instruction"]
        task_hash = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]
        metadata = dict(raw.get("metadata", {}))
        if self.repo_root:
            metadata.setdefault("repo_root", self.repo_root)

        return TaskSpec(
            dataset=self.dataset_name,
            task_id=raw.get("task_id", f"scenario/{task_hash}"),
            task_type=raw.get("task_type", "generic_tool_use"),
            instruction=instruction,
            initial_state=self._initial_state(),
            goal_condition=raw.get("goal_condition", {"finalized": True}),
            available_tools=list(raw.get("available_tools", GENERIC_TOOLS)),
            max_steps=int(raw.get("max_steps", 30)),
            success_criteria=list(
                raw.get("success_criteria", ["A final answer or artifact is produced."])
            ),
            metadata=metadata,
        )

    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        self.task_spec = task_spec_to_dict(task_spec)
        self.state = deepcopy(self.task_spec.get("initial_state") or self._initial_state())
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
        available = self.get_available_tools(self.task_spec, self.state)
        info: dict[str, Any] = {
            "action": action_name,
            "success": False,
            "message": "",
        }

        if action_name not in available:
            info["message"] = f"Unknown generic action: {action_name}"
            self._record(action_name, info)
            return StepResult(self.get_observation(), self.is_done(self.state, self.task_spec), info)

        handler = getattr(self, f"_handle_{action_name}", None)
        if handler is None:
            info["message"] = f"No handler is implemented for action: {action_name}"
            self._record(action_name, info)
            return StepResult(self.get_observation(), self.is_done(self.state, self.task_spec), info)

        message = handler(payload)
        info.update({"success": True, "message": message})
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
        return list(spec.get("available_tools") or GENERIC_TOOLS)

    def normalize_action(
        self,
        raw_action: str | Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> str | None:
        action_name, _ = self._split_action(raw_action)
        legal_names = []
        for item in legal_actions:
            if isinstance(item, Mapping):
                legal_names.append(str(item.get("action", "")))
            else:
                legal_names.append(str(item))

        if action_name in legal_names:
            return action_name

        lowered = {name.lower(): name for name in legal_names}
        return lowered.get(action_name.lower())

    def format_state_for_prompt(self, state: Mapping[str, Any]) -> str:
        completed = state.get("steps_completed", [])
        lines = [
            f"task_read: {bool(state.get('task_read'))}",
            f"context_gathered: {bool(state.get('context_gathered'))}",
            f"plan_created: {bool(state.get('plan_created'))}",
            f"checked: {bool(state.get('checked'))}",
            f"finalized: {bool(state.get('finalized'))}",
            f"steps_completed: {len(completed)}",
            f"last_action: {state.get('last_action')}",
            f"last_action_status: {state.get('last_action_status')}",
        ]
        if state.get("last_error"):
            lines.append(f"last_error: {state.get('last_error')}")
        return "\n".join(lines)

    def summarize_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "success": bool(state.get("finalized")),
            "total_steps": int(state.get("step_count", 0)),
            "termination": "success" if state.get("finalized") else "max_steps_or_incomplete",
            "history": deepcopy(self.history),
        }

    def _handle_read_task(self, payload: Mapping[str, Any]) -> str:
        self.state["task_read"] = True
        return "Task instruction has been read."

    def _handle_inspect_context(self, payload: Mapping[str, Any]) -> str:
        self.state["context_gathered"] = True
        note = payload.get("note") or payload.get("summary")
        if note:
            self.state.setdefault("context_notes", []).append(str(note))
        return "Context has been inspected."

    def _handle_ask_clarifying_question(self, payload: Mapping[str, Any]) -> str:
        question = payload.get("question", "Clarification needed.")
        self.state.setdefault("clarifying_questions", []).append(str(question))
        return "Clarifying question recorded."

    def _handle_write_plan(self, payload: Mapping[str, Any]) -> str:
        self.state["plan_created"] = True
        self.state["plan"] = payload.get("plan") or payload.get("summary") or "Plan drafted."
        return "Plan has been drafted."

    def _handle_execute_step(self, payload: Mapping[str, Any]) -> str:
        step_summary = payload.get("summary") or payload.get("step") or "Executed one workflow step."
        self.state.setdefault("steps_completed", []).append(str(step_summary))
        return "One workflow step has been executed."

    def _handle_check_progress(self, payload: Mapping[str, Any]) -> str:
        self.state["checked"] = True
        self.state["progress_summary"] = payload.get("summary", "Progress checked.")
        return "Progress has been checked."

    def _handle_revise(self, payload: Mapping[str, Any]) -> str:
        self.state["plan_created"] = True
        self.state["checked"] = False
        revision = payload.get("revision") or payload.get("summary") or "Revision applied."
        self.state.setdefault("revisions", []).append(str(revision))
        return "Plan or work product has been revised."

    def _handle_finalize(self, payload: Mapping[str, Any]) -> str:
        self.state["finalized"] = True
        self.state["final_answer"] = payload.get("answer") or payload.get("summary") or "Finalized."
        return "Final output has been produced."

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
    def _initial_state() -> dict[str, Any]:
        return {
            "task_read": False,
            "context_gathered": False,
            "plan_created": False,
            "steps_completed": [],
            "checked": False,
            "finalized": False,
        }

    @staticmethod
    def _split_action(action: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        if isinstance(action, Mapping):
            name = str(action.get("action") or action.get("name") or "").strip()
            payload = dict(action.get("payload") or {})
            return name, payload
        return str(action).strip(), {}
