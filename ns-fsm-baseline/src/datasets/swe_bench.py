"""SWE-bench-style code repair adapter.

This MVP adapter models high-level code-repair workflow state. It intentionally
does not run arbitrary shell commands or edit files directly; those operations
should be added later behind explicit safety gates.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping

try:
    from .base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict


SWE_BENCH_TOOLS = [
    "setup_repo",
    "read_problem_statement",
    "search_code",
    "open_file",
    "edit_file",
    "run_tests",
    "analyze_failure",
    "finalize_patch",
]


class SWEBenchAdapter(DatasetAdapter):
    """High-level adapter for SWE-bench-like code repair tasks."""

    dataset_name = "swe_bench"

    def __init__(self):
        self.task_spec: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def list_tasks(self) -> list[dict[str, Any]]:
        return []

    def load_or_wrap(self, raw_input: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_input, str):
            return {
                "problem_statement": raw_input.strip(),
                "repo": "unknown/unknown",
            }

        raw = dict(raw_input)
        problem = (
            raw.get("problem_statement")
            or raw.get("issue")
            or raw.get("instruction")
            or raw.get("task")
        )
        if not problem:
            raise ValueError("SWEBenchAdapter requires a problem_statement/issue.")
        raw["problem_statement"] = str(problem)
        raw.setdefault("repo", "unknown/unknown")
        raw.setdefault("instance_id", raw.get("task_id", self._hash(problem)))
        return raw

    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        raw = self.load_or_wrap(raw_task)
        instance_id = str(raw.get("instance_id") or self._hash(raw["problem_statement"]))
        return TaskSpec(
            dataset=self.dataset_name,
            task_id=instance_id,
            task_type="code_repair",
            instruction=str(raw["problem_statement"]),
            initial_state={
                "repo_ready": False,
                "issue_read": False,
                "candidate_files": [],
                "opened_files": [],
                "modified_files": [],
                "patch_exists": False,
                "tests_run": [],
                "last_test_status": "not_run",
                "failure_analysis": "",
                "final_patch_ready": False,
            },
            goal_condition={"final_patch_ready": True},
            available_tools=list(SWE_BENCH_TOOLS),
            max_steps=int(raw.get("max_steps", 80)),
            success_criteria=["A patch is ready and marked final."],
            metadata={
                "repo": raw.get("repo"),
                "base_commit": raw.get("base_commit", ""),
                "problem_statement": raw["problem_statement"],
                "test_command": raw.get("test_command", ""),
                "candidate_files_hint": raw.get("candidate_files", []),
            },
        )

    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        self.task_spec = task_spec_to_dict(task_spec)
        self.state = deepcopy(self.task_spec.get("initial_state", {}))
        self.state.setdefault("step_count", 0)
        self.state.setdefault("last_action", None)
        self.state.setdefault("last_action_status", "not_started")
        self.state.setdefault("last_error", None)
        self.history = []
        return self.get_observation()

    def step(self, action: str | Mapping[str, Any]) -> StepResult:
        if self.task_spec is None:
            raise RuntimeError("Call reset(task_spec) before SWEBenchAdapter.step(action).")
        action_name, payload = self._split_action(action)
        if action_name not in SWE_BENCH_TOOLS:
            info = {"success": False, "message": f"Unknown SWE-bench action: {action_name}"}
            self._record(action_name, info)
            return StepResult(self.get_observation(), self.is_done(self.state, self.task_spec), info)

        handler = getattr(self, f"_handle_{action_name}")
        success, message = handler(payload)
        info = {"success": success, "message": message}
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
        return bool(state.get("final_patch_ready")) or int(state.get("step_count", 0)) >= int(
            spec.get("max_steps", 80)
        )

    def get_available_tools(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        if not state.get("repo_ready"):
            return ["setup_repo"]
        if not state.get("issue_read"):
            return ["read_problem_statement"]
        if not state.get("candidate_files"):
            return ["search_code"]
        if not state.get("opened_files"):
            return ["open_file", "search_code"]
        if not state.get("patch_exists"):
            return ["edit_file", "search_code"]
        if state.get("last_test_status") in ("not_run", ""):
            return ["run_tests"]
        if state.get("last_test_status") == "failed":
            return ["analyze_failure", "edit_file", "search_code"]
        if state.get("last_test_status") == "passed":
            return ["finalize_patch"]
        return list(SWE_BENCH_TOOLS)

    def normalize_action(
        self,
        raw_action: str | Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> str | None:
        action_name, _payload = self._split_action(raw_action)
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
                f"repo_ready: {bool(state.get('repo_ready'))}",
                f"issue_read: {bool(state.get('issue_read'))}",
                f"candidate_files: {state.get('candidate_files', [])}",
                f"opened_files: {state.get('opened_files', [])}",
                f"modified_files: {state.get('modified_files', [])}",
                f"patch_exists: {bool(state.get('patch_exists'))}",
                f"last_test_status: {state.get('last_test_status')}",
                f"final_patch_ready: {bool(state.get('final_patch_ready'))}",
            ]
        )

    def summarize_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "success": bool(state.get("final_patch_ready")),
            "total_steps": int(state.get("step_count", 0)),
            "termination": "success" if state.get("final_patch_ready") else "max_steps_or_incomplete",
            "history": deepcopy(self.history),
        }

    def _handle_setup_repo(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        self.state["repo_ready"] = True
        return True, "Repository marked ready."

    def _handle_read_problem_statement(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        self.state["issue_read"] = True
        return True, "Problem statement read."

    def _handle_search_code(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        hints = self.task_spec.get("metadata", {}).get("candidate_files_hint", [])
        candidates = list(hints) or ["UNKNOWN_RELEVANT_FILE.py"]
        self.state["candidate_files"] = list(dict.fromkeys(self.state.get("candidate_files", []) + candidates))
        return True, "Candidate files identified."

    def _handle_open_file(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        candidates = self.state.get("candidate_files", [])
        if not candidates:
            return False, "No candidate files available to open."
        filename = payload.get("file") or candidates[0]
        self.state["opened_files"] = list(dict.fromkeys(self.state.get("opened_files", []) + [filename]))
        return True, f"Opened file: {filename}"

    def _handle_edit_file(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        if not self.state.get("opened_files"):
            return False, "No opened file available to edit."
        filename = payload.get("file") or self.state["opened_files"][-1]
        self.state["modified_files"] = list(dict.fromkeys(self.state.get("modified_files", []) + [filename]))
        self.state["patch_exists"] = True
        self.state["last_test_status"] = "not_run"
        return True, f"Patch marked for file: {filename}"

    def _handle_run_tests(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        if not self.state.get("patch_exists"):
            return False, "Cannot run tests before a patch exists."
        command = payload.get("command") or self.task_spec.get("metadata", {}).get("test_command") or "mock_tests"
        self.state.setdefault("tests_run", []).append(command)
        self.state["last_test_status"] = payload.get("status", "passed")
        return True, f"Tests recorded with status: {self.state['last_test_status']}"

    def _handle_analyze_failure(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        if self.state.get("last_test_status") != "failed":
            return False, "Failure analysis requires failed tests."
        self.state["failure_analysis"] = payload.get("summary", "Analyze failing tests and revise patch.")
        return True, "Failure analysis recorded."

    def _handle_finalize_patch(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        if not self.state.get("patch_exists"):
            return False, "Cannot finalize without a patch."
        if self.state.get("last_test_status") != "passed":
            return False, "Cannot finalize before tests pass."
        self.state["final_patch_ready"] = True
        return True, "Patch finalized."

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
    def _hash(value: str) -> str:
        return "swe_bench/" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
