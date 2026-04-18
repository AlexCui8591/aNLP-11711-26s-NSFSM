"""Fallback workflow FSM for arbitrary tool-use tasks."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from ..datasets.base import TaskSpec
    from ..fsm_validator import build_generic_tool_use_fsm_design
    from .base import FSMTemplate
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec
    from fsm_validator import build_generic_tool_use_fsm_design
    from fsm_templates.base import FSMTemplate


class GenericToolUseFSM(FSMTemplate):
    name = "generic_tool_use"

    def build_design(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
    ) -> dict[str, Any]:
        return build_generic_tool_use_fsm_design(task_spec)
