"""Base classes and registry for reusable FSM templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Type

try:
    from ..datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict


class FSMTemplate(ABC):
    """Base class for task-type-specific FSM templates."""

    name = "base"

    @abstractmethod
    def build_design(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
    ) -> dict[str, Any]:
        """Build an FSMDesign dict from the task spec."""

    def initial_phase(self, task_spec: TaskSpec | Mapping[str, Any]) -> str:
        return self.build_design(task_spec)["initial_state"]

    def get_legal_actions(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
        adapter: Any | None = None,
    ) -> list[str]:
        design = self.build_design(task_spec, adapter)
        current = str(state.get("phase") or design["initial_state"])
        return [
            str(option["action"])
            for option in design.get("transitions_by_state", {}).get(current, [])
        ]

    def next_phase(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
        action: str,
        info: Mapping[str, Any] | None = None,
    ) -> str:
        design = self.build_design(task_spec)
        current = str(state.get("phase") or design["initial_state"])
        for option in design.get("transitions_by_state", {}).get(current, []):
            if option.get("action") == action:
                return str(option.get("next_state"))
        return current

    def is_terminal(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> bool:
        design = self.build_design(task_spec)
        phase = str(state.get("phase") or design["initial_state"])
        return phase in set(design.get("terminal_states", []))


TEMPLATE_REGISTRY: dict[str, Type[FSMTemplate]] = {}


def register_template(task_type: str, template_cls: Type[FSMTemplate]) -> None:
    if not task_type:
        raise ValueError("task_type cannot be empty.")
    TEMPLATE_REGISTRY[task_type] = template_cls


def get_template(task_type: str | None) -> FSMTemplate:
    template_cls = TEMPLATE_REGISTRY.get(task_type or "")
    if template_cls is None:
        template_cls = TEMPLATE_REGISTRY.get("generic_tool_use")
    if template_cls is None:
        raise KeyError("No generic_tool_use template registered.")
    return template_cls()


def list_templates() -> list[str]:
    return sorted(TEMPLATE_REGISTRY)


def as_task_dict(task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
    return task_spec_to_dict(task_spec)
