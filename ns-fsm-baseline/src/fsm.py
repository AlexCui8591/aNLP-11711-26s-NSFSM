"""Runtime FSM representation for NS-FSM."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

try:
    from .datalog_verifier import DatalogVerifier
    from .datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datalog_verifier import DatalogVerifier
    from datasets.base import TaskSpec, task_spec_to_dict


@dataclass
class FSMState:
    name: str
    terminal: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class FSMAction:
    name: str
    metadata: dict[str, Any] | None = None


@dataclass
class FSMTransition:
    source_state: str
    action: str
    next_state: str
    condition: str = ""


class RuntimeFSM:
    """Runtime control object backed by a validated FSMDesign."""

    def __init__(
        self,
        fsm_design: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any] | None = None,
        adapter: Any | None = None,
        verifier: DatalogVerifier | None = None,
    ):
        self.fsm_design = deepcopy(dict(fsm_design))
        self.task_spec = task_spec_to_dict(task_spec) if task_spec is not None else {}
        self.adapter = adapter
        self.verifier = verifier or DatalogVerifier()
        self.verification = self.verifier.verify(self.fsm_design, self.task_spec, adapter)
        self.current_state = str(self.fsm_design["initial_state"])
        self.visited_states: list[str] = [self.current_state]
        self.action_history: list[dict[str, Any]] = []
        self.blocked_action_history: list[dict[str, Any]] = []
        self.fallback_action_history: list[dict[str, Any]] = []

    @property
    def terminal_states(self) -> set[str]:
        return {str(state) for state in self.fsm_design.get("terminal_states", [])}

    @property
    def states(self) -> list[FSMState]:
        terminals = self.terminal_states
        return [
            FSMState(name=str(state), terminal=str(state) in terminals)
            for state in self.fsm_design.get("states", [])
        ]

    @property
    def actions(self) -> list[FSMAction]:
        return [FSMAction(name=str(action)) for action in self.fsm_design.get("actions", [])]

    @property
    def transitions(self) -> list[FSMTransition]:
        transitions = []
        for item in self.fsm_design.get("transition_facts", []):
            transitions.append(
                FSMTransition(
                    source_state=str(item["source_state"]),
                    action=str(item["action"]),
                    next_state=str(item["next_state"]),
                    condition=str(item.get("condition", "")),
                )
            )
        return transitions

    def reset(self) -> str:
        self.current_state = str(self.fsm_design["initial_state"])
        self.visited_states = [self.current_state]
        self.action_history = []
        self.blocked_action_history = []
        self.fallback_action_history = []
        return self.current_state

    def is_terminal(self, state: str | None = None) -> bool:
        return (state or self.current_state) in self.terminal_states

    def get_valid_actions(self, state: str | None = None) -> list[str]:
        return self.verifier.get_valid_actions(state or self.current_state)

    def get_valid_transitions(self, state: str | None = None) -> list[dict[str, str]]:
        return self.verifier.get_valid_transitions(state or self.current_state)

    def verify_transition(
        self,
        action: str,
        proposed_next_state: str | None = None,
        current_state: str | None = None,
    ) -> dict[str, Any]:
        return self.verifier.verify_transition(
            current_state or self.current_state,
            action,
            proposed_next_state,
        )

    def update(
        self,
        action: str,
        proposed_next_state: str | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        check = self.verify_transition(action, proposed_next_state)
        if not check["valid"]:
            self.record_blocked_action(action, check)
            return {"updated": False, "check": check, "current_state": self.current_state}

        next_state = proposed_next_state or check.get("inferred_next_state")
        if next_state is None and check.get("expected_next_states"):
            next_state = check["expected_next_states"][0]
        previous = self.current_state
        self.current_state = str(next_state)
        self.visited_states.append(self.current_state)
        self.action_history.append(
            {
                "source_state": previous,
                "action": action,
                "next_state": self.current_state,
                "info": dict(info or {}),
            }
        )
        return {"updated": True, "check": check, "current_state": self.current_state}

    def record_blocked_action(
        self,
        action: str | None,
        reason: Mapping[str, Any],
    ) -> None:
        self.blocked_action_history.append(
            {
                "state": self.current_state,
                "action": action,
                "reason": deepcopy(dict(reason)),
            }
        )

    def record_fallback_action(
        self,
        action: str,
        next_state: str | None,
        reason: str,
    ) -> None:
        self.fallback_action_history.append(
            {
                "state": self.current_state,
                "action": action,
                "next_state": next_state,
                "reason": reason,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state,
            "visited_states": list(self.visited_states),
            "states": [asdict(state) for state in self.states],
            "actions": [asdict(action) for action in self.actions],
            "transitions": [asdict(transition) for transition in self.transitions],
            "blocked_action_history": deepcopy(self.blocked_action_history),
            "fallback_action_history": deepcopy(self.fallback_action_history),
            "verification": deepcopy(self.verification),
        }
