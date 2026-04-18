"""Workflow FSM template for code-repair / SWE-bench-style tasks."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from ..datasets.base import TaskSpec, task_spec_to_dict
    from .base import FSMTemplate
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict
    from fsm_templates.base import FSMTemplate


class CodeRepairFSM(FSMTemplate):
    name = "code_repair"

    def build_design(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
    ) -> dict[str, Any]:
        spec = task_spec_to_dict(task_spec)
        return {
            "states": [
                "START",
                "SETUP",
                "UNDERSTAND_TASK",
                "LOCALIZE",
                "INSPECT",
                "PATCH",
                "TEST",
                "REVISE",
                "DONE",
            ],
            "initial_state": "START",
            "terminal_states": ["DONE"],
            "transitions_by_state": {
                "START": [
                    {
                        "action": "setup_repo",
                        "next_state": "SETUP",
                        "condition": "Repository is not ready.",
                    }
                ],
                "SETUP": [
                    {
                        "action": "read_problem_statement",
                        "next_state": "UNDERSTAND_TASK",
                        "condition": "Repository is ready and issue should be read.",
                    }
                ],
                "UNDERSTAND_TASK": [
                    {
                        "action": "search_code",
                        "next_state": "LOCALIZE",
                        "condition": "Need candidate files related to the issue.",
                    }
                ],
                "LOCALIZE": [
                    {
                        "action": "open_file",
                        "next_state": "INSPECT",
                        "condition": "Candidate files exist and should be inspected.",
                    },
                    {
                        "action": "search_code",
                        "next_state": "LOCALIZE",
                        "condition": "Need more candidate files.",
                    },
                ],
                "INSPECT": [
                    {
                        "action": "edit_file",
                        "next_state": "PATCH",
                        "condition": "Relevant code has been inspected.",
                    },
                    {
                        "action": "search_code",
                        "next_state": "LOCALIZE",
                        "condition": "Opened file was not relevant enough.",
                    },
                ],
                "PATCH": [
                    {
                        "action": "run_tests",
                        "next_state": "TEST",
                        "condition": "Patch exists and should be validated.",
                    }
                ],
                "TEST": [
                    {
                        "action": "finalize_patch",
                        "next_state": "DONE",
                        "condition": "Tests passed and patch is ready.",
                    },
                    {
                        "action": "analyze_failure",
                        "next_state": "REVISE",
                        "condition": "Tests failed or more diagnosis is needed.",
                    },
                ],
                "REVISE": [
                    {
                        "action": "edit_file",
                        "next_state": "PATCH",
                        "condition": "Failure analysis suggests a patch change.",
                    },
                    {
                        "action": "search_code",
                        "next_state": "LOCALIZE",
                        "condition": "Need to relocalize after failed tests.",
                    },
                ],
                "DONE": [],
            },
            "fallback_policy": {
                "on_invalid_action": "follow_code_repair_loop",
                "on_dead_end": "return_to_localize_or_revise",
            },
            "success_signals": spec.get("success_criteria", ["final patch ready"]),
            "risk_notes": ["Workflow-level code repair FSM; no shell execution policy here."],
        }
