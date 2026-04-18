"""Deterministic fallback planner for NS-FSM."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from .datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict


class Planner:
    """Choose a safe fallback action when the LLM proposal is blocked."""

    def next_action(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        transition_options: list[dict[str, Any]] | None = None,
        blocked_reason: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        spec = task_spec_to_dict(task_spec)
        history = history or []
        transition_options = transition_options or self._transition_options(legal_actions)
        legal_names = [str(option["action"]) for option in transition_options]
        if not legal_names:
            return None

        task_type = str(spec.get("task_type", "generic_tool_use"))
        if task_type == "code_repair":
            action = self._code_repair_action(state, legal_names)
            reason = "code_repair_policy"
        elif task_type == "symbolic_planning":
            action = self._symbolic_action(spec, legal_names, history)
            reason = "symbolic_dependency_path_policy"
        else:
            action = self._generic_action(state, legal_names, history)
            reason = "generic_progress_policy"

        action = action if action in legal_names else legal_names[0]
        next_state = self._next_state_for_action(action, transition_options)
        return {
            "action": action,
            "next_state": next_state,
            "reason": reason,
            "fallback_used": True,
            "blocked_reason": dict(blocked_reason or {}),
        }

    def _generic_action(
        self,
        state: Mapping[str, Any],
        legal_names: list[str],
        history: list[dict[str, Any]],
    ) -> str:
        if "read_task" in legal_names and not state.get("task_read"):
            return "read_task"
        if "inspect_context" in legal_names and not state.get("context_gathered"):
            return "inspect_context"
        if "write_plan" in legal_names and not state.get("plan_created"):
            return "write_plan"
        if "finalize" in legal_names and state.get("checked"):
            return "finalize"
        if "execute_step" in legal_names and state.get("plan_created"):
            return "execute_step"
        if (
            "check_progress" in legal_names
            and state.get("steps_completed")
            and not state.get("checked")
        ):
            return "check_progress"
        if "revise" in legal_names and self._recent_failure(history):
            return "revise"
        return legal_names[0]

    def _symbolic_action(
        self,
        task_spec: Mapping[str, Any],
        legal_names: list[str],
        history: list[dict[str, Any]],
    ) -> str:
        sequence = list(
            task_spec.get("metadata", {}).get("required_actions")
            or task_spec.get("metadata", {}).get("optimal_sequence")
            or []
        )
        completed = [entry.get("action") for entry in history if entry.get("success", True)]
        for action in sequence:
            if action in legal_names and action not in completed:
                return str(action)
        return legal_names[0]

    def _code_repair_action(
        self,
        state: Mapping[str, Any],
        legal_names: list[str],
    ) -> str:
        ordered_policy = [
            ("setup_repo", not state.get("repo_ready")),
            ("read_problem_statement", not state.get("issue_read")),
            ("search_code", not state.get("candidate_files")),
            ("open_file", bool(state.get("candidate_files")) and not state.get("opened_files")),
            ("edit_file", not state.get("patch_exists")),
            ("run_tests", state.get("patch_exists") and not state.get("tests_run")),
            ("analyze_failure", state.get("last_test_status") == "failed"),
            ("finalize_patch", state.get("last_test_status") == "passed"),
        ]
        for action, should_take in ordered_policy:
            if should_take and action in legal_names:
                return action
        return legal_names[0]

    @staticmethod
    def _recent_failure(history: list[dict[str, Any]]) -> bool:
        if not history:
            return False
        recent = history[-1]
        if "success" in recent:
            return not bool(recent["success"])
        info = recent.get("info", {})
        return not bool(info.get("success", True))

    @staticmethod
    def _transition_options(
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        options = []
        for item in legal_actions:
            if isinstance(item, Mapping):
                options.append(dict(item))
            else:
                options.append({"action": str(item), "next_state": None})
        return options

    @staticmethod
    def _next_state_for_action(
        action: str,
        transition_options: list[dict[str, Any]],
    ) -> str | None:
        for option in transition_options:
            if option.get("action") == action:
                return option.get("next_state")
        return None
