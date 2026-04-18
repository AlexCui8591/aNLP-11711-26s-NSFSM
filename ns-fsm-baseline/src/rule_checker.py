"""Rule checker for NS-FSM action proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

try:
    from .datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict


@dataclass
class LegalityResult:
    legal: bool
    reason_type: str
    message: str
    matched_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleChecker:
    """Check workflow, dataset, and budget constraints before execution."""

    def check(
        self,
        proposed_action: str | None,
        legal_actions: list[str] | list[dict[str, Any]],
        state: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any] | None = None,
        adapter: Any | None = None,
        transition_check: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = task_spec_to_dict(task_spec) if task_spec is not None else {}
        legal_names = self._legal_names(legal_actions)

        if not proposed_action:
            return LegalityResult(
                legal=False,
                reason_type="parse_error",
                message="No proposed action was provided.",
            ).to_dict()

        if spec and int(state.get("step_count", state.get("step", 0))) >= int(
            spec.get("max_steps", 10**9)
        ):
            return LegalityResult(
                legal=False,
                reason_type="budget_exceeded",
                message="Step budget has been exhausted.",
                matched_action=proposed_action,
            ).to_dict()

        if proposed_action not in legal_names:
            return LegalityResult(
                legal=False,
                reason_type="not_in_legal_actions",
                message=f"Action '{proposed_action}' is not in T(current_state).",
                matched_action=proposed_action,
                metadata={"legal_actions": legal_names},
            ).to_dict()

        if adapter is not None and spec:
            adapter_actions = self._adapter_actions(adapter, spec, state)
            if adapter_actions and proposed_action not in adapter_actions:
                return LegalityResult(
                    legal=False,
                    reason_type="dataset_rule_violation",
                    message=(
                        f"Action '{proposed_action}' is not currently executable "
                        "according to the dataset adapter."
                    ),
                    matched_action=proposed_action,
                    metadata={"adapter_actions": adapter_actions[:50]},
                ).to_dict()

        if transition_check is not None and not transition_check.get("valid", False):
            return LegalityResult(
                legal=False,
                reason_type="invalid_state_transition",
                message="Datalog rejected the proposed state transition.",
                matched_action=proposed_action,
                metadata={"transition_check": dict(transition_check)},
            ).to_dict()

        return LegalityResult(
            legal=True,
            reason_type="ok",
            message="Action is legal.",
            matched_action=proposed_action,
        ).to_dict()

    @staticmethod
    def _legal_names(legal_actions: list[str] | list[dict[str, Any]]) -> list[str]:
        names = []
        for item in legal_actions:
            if isinstance(item, Mapping):
                name = str(item.get("action", ""))
            else:
                name = str(item)
            if name:
                names.append(name)
        return list(dict.fromkeys(names))

    @staticmethod
    def _adapter_actions(
        adapter: Any,
        task_spec: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        try:
            return [str(action) for action in adapter.get_available_tools(task_spec, state)]
        except Exception:
            return []
