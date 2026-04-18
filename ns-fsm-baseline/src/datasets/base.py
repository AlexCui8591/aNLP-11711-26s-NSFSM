"""Shared dataset adapter contract for the generic NS-FSM pipeline.

The core pipeline should not know whether a task came from a benchmark, a JSON
scenario, or a one-line user instruction. Adapters translate those inputs into a
common TaskSpec and expose a small step/reset interface for execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TaskSpec:
    """Common task representation used by FSM design and verification."""

    dataset: str
    task_id: str
    task_type: str
    instruction: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    goal_condition: dict[str, Any] = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    max_steps: int = 30
    success_criteria: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    """Result returned by adapter.step."""

    state: dict[str, Any]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        return self.state, self.done, self.info


def task_spec_to_dict(task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a TaskSpec dataclass or mapping into a mutable dict."""

    if isinstance(task_spec, TaskSpec):
        return task_spec.to_dict()
    return dict(task_spec)


class DatasetAdapter(ABC):
    """Base class for both user-scenario and benchmark adapters."""

    dataset_name: str = "base"

    @abstractmethod
    def list_tasks(self) -> list[dict[str, Any]]:
        """Return raw task records known to this adapter."""

    @abstractmethod
    def load_or_wrap(self, raw_input: str | Mapping[str, Any]) -> dict[str, Any]:
        """Load a benchmark task or wrap a user-provided scenario."""

    @abstractmethod
    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        """Convert a raw task into the common TaskSpec format."""

    @abstractmethod
    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        """Initialize adapter state for one task."""

    @abstractmethod
    def step(self, action: str | Mapping[str, Any]) -> StepResult:
        """Execute one high-level action."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Return the current adapter-visible observation/state."""

    @abstractmethod
    def is_done(
        self,
        state: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any],
    ) -> bool:
        """Return whether the task is complete."""

    @abstractmethod
    def get_available_tools(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        """Return adapter-level actions/tools that may be exposed to the FSM."""

    @abstractmethod
    def normalize_action(
        self,
        raw_action: str | Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> str | None:
        """Map raw model output to one canonical legal action, if possible."""

    @abstractmethod
    def format_state_for_prompt(self, state: Mapping[str, Any]) -> str:
        """Render compact state text for LLM prompts."""

    @abstractmethod
    def summarize_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Return a normalized result summary for logging/analysis."""
