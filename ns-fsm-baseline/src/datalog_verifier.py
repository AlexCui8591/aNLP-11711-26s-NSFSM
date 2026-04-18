"""Dependency-light Datalog-style verifier for NS-FSM designs.

This MVP keeps the formal layer in plain Python. It materializes Datalog-style
facts, then computes the two closures the pipeline needs most: states reachable
from the initial state, and states that can eventually reach a terminal state.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping

try:
    from .datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict


class DatalogVerifier:
    """Formal verifier for generated FSMDesign objects."""

    def __init__(self):
        self.fsm_design: dict[str, Any] = {}
        self.task_spec: dict[str, Any] = {}
        self.facts: list[tuple[Any, ...]] = []
        self.derived: list[tuple[Any, ...]] = []
        self.violations: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.transitions_by_state: dict[str, list[dict[str, str]]] = {}
        self.legal_actions_by_state: dict[str, list[str]] = {}
        self.states: set[str] = set()
        self.terminal_states: set[str] = set()
        self.actions: set[str] = set()

    def verify(
        self,
        fsm_design: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any] | None = None,
        adapter: Any | None = None,
    ) -> dict[str, Any]:
        """Return facts, derived predicates, violations, and warnings."""

        self._reset()
        self.fsm_design = deepcopy(dict(fsm_design))
        self.task_spec = task_spec_to_dict(task_spec) if task_spec is not None else {}

        self.states = {str(state) for state in self.fsm_design.get("states", [])}
        initial_state = str(self.fsm_design.get("initial_state", ""))
        self.terminal_states = {
            str(state) for state in self.fsm_design.get("terminal_states", [])
        }
        adapter_actions = self._adapter_actions(self.task_spec, adapter)

        for state in sorted(self.states):
            self.facts.append(("state", state))
        if initial_state:
            self.facts.append(("initial", initial_state))
        for state in sorted(self.terminal_states):
            self.facts.append(("terminal", state))
        for action in sorted(adapter_actions):
            self.facts.append(("adapter_action", action))

        duplicate_edges: set[tuple[str, str, str]] = set()
        seen_edges: set[tuple[str, str, str]] = set()
        raw_transitions = self.fsm_design.get("transitions_by_state", {})
        if isinstance(raw_transitions, Mapping):
            for source_state, options in raw_transitions.items():
                source = str(source_state)
                if not isinstance(options, list):
                    self._violation(
                        "invalid_transition_list",
                        f"transitions_by_state[{source}] must be a list.",
                        state=source,
                    )
                    continue

                self.transitions_by_state.setdefault(source, [])
                self.legal_actions_by_state.setdefault(source, [])
                for option in options:
                    if not isinstance(option, Mapping):
                        self._violation(
                            "invalid_transition_option",
                            f"Transition option in {source} must be an object.",
                            state=source,
                        )
                        continue

                    action = str(option.get("action", "")).strip()
                    target = str(option.get("next_state", "")).strip()
                    condition = str(option.get("condition", "")).strip()
                    normalized = {
                        "action": action,
                        "next_state": target,
                        "condition": condition,
                    }
                    self.transitions_by_state[source].append(normalized)
                    if action:
                        self.actions.add(action)
                        self.legal_actions_by_state[source].append(action)
                        self.facts.append(("action", action))
                        self.facts.append(("legal", source, action))
                    self.facts.append(("transition", source, action, target))

                    edge = (source, action, target)
                    if edge in seen_edges:
                        duplicate_edges.add(edge)
                    seen_edges.add(edge)

        for action in sorted(self.actions):
            self.facts.append(("action", action))

        self._check_static_facts(initial_state, adapter_actions)
        reachable = self._reachable_states(initial_state)
        can_reach_terminal = self._terminal_reachable_states()

        for state in sorted(reachable):
            self.derived.append(("reachable", state))
        for state in sorted(can_reach_terminal):
            self.derived.append(("can_reach_terminal", state))
        for source, options in sorted(self.transitions_by_state.items()):
            for option in options:
                if (
                    source in self.states
                    and option["next_state"] in self.states
                    and option["action"] in self.actions
                    and option["action"] in self.legal_actions_by_state.get(source, [])
                ):
                    self.derived.append(
                        (
                            "valid_transition",
                            source,
                            option["action"],
                            option["next_state"],
                        )
                    )

        self._check_global_consistency(initial_state, reachable, can_reach_terminal)
        self._check_premature_finalize(initial_state)

        for edge in sorted(duplicate_edges):
            self._warning(
                "duplicate_transition",
                f"Duplicate transition exists: {edge[0]} --{edge[1]}--> {edge[2]}",
                state=edge[0],
                action=edge[1],
                next_state=edge[2],
            )

        return {
            "ok": not self.violations,
            "facts": list(dict.fromkeys(self.facts)),
            "derived": list(dict.fromkeys(self.derived)),
            "violations": deepcopy(self.violations),
            "warnings": deepcopy(self.warnings),
        }

    def get_valid_actions(self, current_state: str) -> list[str]:
        """Return Actions(s), the projection of T(s) onto action names."""

        return list(dict.fromkeys(self.legal_actions_by_state.get(current_state, [])))

    def get_valid_transitions(self, current_state: str) -> list[dict[str, str]]:
        """Return T(s), valid action/next_state options for the current state."""

        return deepcopy(self.transitions_by_state.get(current_state, []))

    def verify_transition(
        self,
        current_state: str,
        action: str,
        proposed_next_state: str | None = None,
    ) -> dict[str, Any]:
        """Post-hoc runtime verification for an LLM-proposed transition."""

        violations: list[dict[str, Any]] = []
        expected = self.get_valid_transitions(current_state)
        expected_for_action = [
            option for option in expected if option.get("action") == action
        ]

        if current_state not in self.states:
            violations.append(
                self._make_issue(
                    "unknown_current_state",
                    f"Current state is not in FSM states: {current_state}",
                    state=current_state,
                )
            )
            return self._transition_result(False, violations, expected_for_action)

        if not expected_for_action:
            violations.append(
                self._make_issue(
                    "invalid_current_action",
                    f"Action '{action}' is not legal in current state '{current_state}'.",
                    state=current_state,
                    action=action,
                )
            )
            return self._transition_result(False, violations, expected_for_action)

        if proposed_next_state is None:
            if len(expected_for_action) == 1:
                inferred = expected_for_action[0]["next_state"]
                return {
                    "valid": True,
                    "violations": [],
                    "expected_next_states": [inferred],
                    "inferred_next_state": inferred,
                }
            violations.append(
                self._make_issue(
                    "ambiguous_next_state",
                    f"Action '{action}' has multiple possible next states; LLM must propose one.",
                    state=current_state,
                    action=action,
                )
            )
            return self._transition_result(False, violations, expected_for_action)

        if proposed_next_state not in self.states:
            violations.append(
                self._make_issue(
                    "unknown_next_state",
                    f"Proposed next state is not in FSM states: {proposed_next_state}",
                    state=current_state,
                    action=action,
                    next_state=proposed_next_state,
                )
            )
            return self._transition_result(False, violations, expected_for_action)

        valid_targets = {option["next_state"] for option in expected_for_action}
        if proposed_next_state not in valid_targets:
            violations.append(
                self._make_issue(
                    "invalid_next_state",
                    (
                        f"Action '{action}' from '{current_state}' cannot transition to "
                        f"'{proposed_next_state}'."
                    ),
                    state=current_state,
                    action=action,
                    next_state=proposed_next_state,
                )
            )
            return self._transition_result(False, violations, expected_for_action)

        return {
            "valid": True,
            "violations": [],
            "expected_next_states": sorted(valid_targets),
            "inferred_next_state": proposed_next_state,
        }

    def _reset(self) -> None:
        self.fsm_design = {}
        self.task_spec = {}
        self.facts = []
        self.derived = []
        self.violations = []
        self.warnings = []
        self.transitions_by_state = {}
        self.legal_actions_by_state = {}
        self.states = set()
        self.terminal_states = set()
        self.actions = set()

    def _check_static_facts(self, initial_state: str, adapter_actions: set[str]) -> None:
        if initial_state and initial_state not in self.states:
            self._violation(
                "unknown_initial_state",
                f"Initial state '{initial_state}' is not listed in states.",
                state=initial_state,
            )

        for terminal in sorted(self.terminal_states):
            if terminal not in self.states:
                self._violation(
                    "unknown_terminal_state",
                    f"Terminal state '{terminal}' is not listed in states.",
                    state=terminal,
                )

        for source, options in self.transitions_by_state.items():
            if source not in self.states:
                self._violation(
                    "unknown_source_state",
                    f"Transition source state '{source}' is not listed in states.",
                    state=source,
                )
            for option in options:
                action = option.get("action", "")
                target = option.get("next_state", "")
                if not action:
                    self._violation(
                        "unknown_action",
                        f"Transition from '{source}' has an empty action.",
                        state=source,
                    )
                if target not in self.states:
                    self._violation(
                        "unknown_target_state",
                        f"Transition target state '{target}' is not listed in states.",
                        state=source,
                        action=action,
                        next_state=target,
                    )
                if action and action not in self.legal_actions_by_state.get(source, []):
                    self._violation(
                        "illegal_transition_action",
                        f"Action '{action}' is not legal in state '{source}'.",
                        state=source,
                        action=action,
                    )
                if adapter_actions and action and action not in adapter_actions:
                    self._violation(
                        "action_not_allowed_by_adapter",
                        f"Action '{action}' is not exposed by the adapter/task spec.",
                        state=source,
                        action=action,
                    )

    def _check_global_consistency(
        self,
        initial_state: str,
        reachable: set[str],
        can_reach_terminal: set[str],
    ) -> None:
        if initial_state in self.states and not (reachable & self.terminal_states):
            self._violation(
                "initial_cannot_reach_terminal",
                "Initial state cannot reach any terminal state.",
                state=initial_state,
            )

        for state in sorted(self.states - reachable):
            self._warning(
                "unreachable_state",
                f"State '{state}' is not reachable from the initial state.",
                state=state,
            )

        for state in sorted(reachable):
            if state not in self.terminal_states and state not in can_reach_terminal:
                self._violation(
                    "dead_end_state",
                    f"Reachable non-terminal state '{state}' cannot reach a terminal state.",
                    state=state,
                )

        for state in sorted(self.states):
            if (
                state not in self.terminal_states
                and not self.legal_actions_by_state.get(state)
            ):
                self._violation(
                    "no_legal_action",
                    f"Non-terminal state '{state}' has no legal outgoing actions.",
                    state=state,
                )

        for state in sorted(self.terminal_states):
            if self.legal_actions_by_state.get(state):
                self._warning(
                    "terminal_has_outgoing_actions",
                    f"Terminal state '{state}' has outgoing actions.",
                    state=state,
                )

    def _check_premature_finalize(self, initial_state: str) -> None:
        if "read_task" not in self.actions or "finalize" not in self.actions:
            return

        not_read_reachable = set()
        frontier = [initial_state]
        while frontier:
            state = frontier.pop()
            if state in not_read_reachable:
                continue
            not_read_reachable.add(state)
            for option in self.transitions_by_state.get(state, []):
                if option.get("action") == "read_task":
                    continue
                target = option.get("next_state")
                if target and target not in not_read_reachable:
                    frontier.append(target)

        for state in sorted(not_read_reachable):
            if "finalize" in self.legal_actions_by_state.get(state, []):
                self._violation(
                    "premature_finalize",
                    f"State '{state}' allows finalize before read_task has happened.",
                    state=state,
                    action="finalize",
                )

    def _reachable_states(self, initial_state: str) -> set[str]:
        if initial_state not in self.states:
            return set()

        reachable = {initial_state}
        changed = True
        while changed:
            changed = False
            for source, options in self.transitions_by_state.items():
                if source not in reachable:
                    continue
                for option in options:
                    target = option.get("next_state")
                    if target in self.states and target not in reachable:
                        reachable.add(target)
                        changed = True
        return reachable

    def _terminal_reachable_states(self) -> set[str]:
        reverse_graph: dict[str, set[str]] = defaultdict(set)
        for source, options in self.transitions_by_state.items():
            for option in options:
                target = option.get("next_state")
                if source in self.states and target in self.states:
                    reverse_graph[target].add(source)

        can_reach_terminal = set(self.terminal_states & self.states)
        changed = True
        while changed:
            changed = False
            for target, sources in reverse_graph.items():
                if target not in can_reach_terminal:
                    continue
                for source in sources:
                    if source not in can_reach_terminal:
                        can_reach_terminal.add(source)
                        changed = True
        return can_reach_terminal

    @staticmethod
    def _adapter_actions(
        task_spec: Mapping[str, Any],
        adapter: Any | None,
    ) -> set[str]:
        actions = {str(action) for action in task_spec.get("available_tools", []) if str(action)}
        if actions or adapter is None:
            return actions

        try:
            state = task_spec.get("initial_state", {})
            return {str(action) for action in adapter.get_available_tools(task_spec, state)}
        except Exception:
            return set()

    def _violation(self, issue_type: str, message: str, **kwargs: Any) -> None:
        self.violations.append(self._make_issue(issue_type, message, **kwargs))

    def _warning(self, issue_type: str, message: str, **kwargs: Any) -> None:
        issue = self._make_issue(issue_type, message, **kwargs)
        issue["severity"] = "warning"
        self.warnings.append(issue)

    @staticmethod
    def _make_issue(issue_type: str, message: str, **kwargs: Any) -> dict[str, Any]:
        issue = {
            "type": issue_type,
            "severity": "hard",
            "message": message,
        }
        issue.update({key: value for key, value in kwargs.items() if value is not None})
        return issue

    @staticmethod
    def _transition_result(
        valid: bool,
        violations: list[dict[str, Any]],
        expected_for_action: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "violations": violations,
            "expected_next_states": sorted(
                {option["next_state"] for option in expected_for_action}
            ),
        }
