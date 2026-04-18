"""Schema and formal validation for LLM-generated FSM designs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from .datalog_verifier import DatalogVerifier
    from .datasets.base import TaskSpec, task_spec_to_dict
    from .fsm_designer import parse_json_object
except ImportError:  # pragma: no cover - supports direct src execution
    from datalog_verifier import DatalogVerifier
    from datasets.base import TaskSpec, task_spec_to_dict
    from fsm_designer import parse_json_object


REQUIRED_FSM_FIELDS = {
    "states",
    "initial_state",
    "terminal_states",
    "transitions_by_state",
    "fallback_policy",
    "success_signals",
    "risk_notes",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fsm_design: dict[str, Any] | None = None
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "fsm_design": self.fsm_design,
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
        }


class FSMDesignValidator:
    """Validate FSMDesign schema, derive transition facts, and call Datalog."""

    def __init__(self, datalog_verifier: DatalogVerifier | None = None):
        self.datalog_verifier = datalog_verifier or DatalogVerifier()

    def validate(
        self,
        raw_design: str | Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any] | None = None,
        adapter: Any | None = None,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        spec = task_spec_to_dict(task_spec) if task_spec is not None else {}
        errors: list[str] = []
        warnings: list[str] = []

        try:
            design = parse_json_object(raw_design) if isinstance(raw_design, str) else deepcopy(dict(raw_design))
        except Exception as exc:
            errors.append(f"json_parse_error: {exc}")
            return self._maybe_fallback(errors, warnings, spec, adapter, allow_fallback)

        schema_errors, schema_warnings = self._validate_schema(design, spec, adapter)
        errors.extend(schema_errors)
        warnings.extend(schema_warnings)

        if errors:
            return self._maybe_fallback(errors, warnings, spec, adapter, allow_fallback)

        derived_design = self._derive_structures(design)
        datalog_result = self.datalog_verifier.verify(derived_design, spec, adapter)
        datalog_errors = [
            f"{item['type']}: {item['message']}"
            for item in datalog_result.get("violations", [])
        ]
        datalog_warnings = [
            f"{item['type']}: {item['message']}"
            for item in datalog_result.get("warnings", [])
        ]
        errors.extend(datalog_errors)
        warnings.extend(datalog_warnings)

        if errors:
            return self._maybe_fallback(
                errors,
                warnings,
                spec,
                adapter,
                allow_fallback,
                datalog_result=datalog_result,
            )

        return ValidationResult(
            valid=True,
            errors=[],
            warnings=warnings,
            fsm_design=derived_design,
            fallback_used=False,
            metadata={"datalog": datalog_result},
        ).to_dict()

    def _validate_schema(
        self,
        design: Mapping[str, Any],
        task_spec: Mapping[str, Any],
        adapter: Any | None,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        missing = sorted(REQUIRED_FSM_FIELDS - set(design))
        if missing:
            errors.append(f"missing_required_fields: {missing}")

        if "actions" in design:
            warnings.append("actions field is ignored; derive actions from transitions_by_state.")
        if "legal_actions_by_state" in design:
            warnings.append(
                "legal_actions_by_state field is ignored; derive it from transitions_by_state."
            )
        if "transitions" in design:
            warnings.append("transitions field is ignored; transitions_by_state is canonical.")

        states = design.get("states", [])
        if not isinstance(states, list) or not states:
            errors.append("states must be a non-empty list.")
            states = []
        state_names = [str(state) for state in states]
        state_set = set(state_names)
        if len(state_names) != len(state_set):
            errors.append("states must be unique.")

        initial_state = str(design.get("initial_state", ""))
        if not initial_state:
            errors.append("initial_state is required.")
        elif initial_state not in state_set:
            errors.append(f"initial_state does not exist in states: {initial_state}")

        terminal_states = design.get("terminal_states", [])
        if not isinstance(terminal_states, list) or not terminal_states:
            errors.append("terminal_states must be a non-empty list.")
            terminal_states = []
        for terminal in terminal_states:
            if str(terminal) not in state_set:
                errors.append(f"terminal_state does not exist in states: {terminal}")

        transitions = design.get("transitions_by_state", {})
        if not isinstance(transitions, Mapping) or not transitions:
            errors.append("transitions_by_state must be a non-empty object.")
            transitions = {}

        adapter_actions = self._adapter_actions(task_spec, adapter)
        actions_seen: set[str] = set()
        for state, options in transitions.items():
            state_name = str(state)
            if state_name not in state_set:
                errors.append(f"transitions_by_state key is not a known state: {state_name}")
            if not isinstance(options, list):
                errors.append(f"transitions_by_state[{state_name}] must be a list.")
                continue
            for option in options:
                if not isinstance(option, Mapping):
                    errors.append(f"transition option in {state_name} must be an object.")
                    continue
                for field_name in ("action", "next_state", "condition"):
                    if field_name not in option:
                        errors.append(
                            f"transition option in {state_name} missing field: {field_name}"
                        )
                action = str(option.get("action", "")).strip()
                next_state = str(option.get("next_state", "")).strip()
                if action:
                    actions_seen.add(action)
                if next_state and next_state not in state_set:
                    errors.append(
                        f"transition from {state_name} targets unknown state: {next_state}"
                    )
                if adapter_actions and action and action not in adapter_actions:
                    errors.append(
                        f"action is not allowed by adapter/task spec: {action}"
                    )

        terminal_set = {str(state) for state in terminal_states}
        for state in state_names:
            options = transitions.get(state, [])
            if state not in terminal_set and not options:
                errors.append(f"non-terminal state has no transitions: {state}")

        if state_names and transitions:
            path_errors = self._path_checks(
                states=state_set,
                initial_state=initial_state,
                terminal_states=terminal_set,
                transitions=transitions,
            )
            errors.extend(path_errors)

        if "read_task" in actions_seen and "finalize" in actions_seen:
            premature = self._premature_finalize_states(initial_state, transitions)
            for state in premature:
                errors.append(f"finalize is legal before read_task at state: {state}")

        return errors, warnings

    def _derive_structures(self, design: Mapping[str, Any]) -> dict[str, Any]:
        derived = deepcopy(dict(design))
        transitions = derived.get("transitions_by_state", {})
        actions: list[str] = []
        legal_by_state: dict[str, list[str]] = {}
        transition_facts: list[dict[str, str]] = []

        for state in derived.get("states", []):
            state_name = str(state)
            options = transitions.get(state_name, [])
            legal_by_state[state_name] = []
            for option in options:
                action = str(option["action"])
                next_state = str(option["next_state"])
                legal_by_state[state_name].append(action)
                actions.append(action)
                transition_facts.append(
                    {
                        "source_state": state_name,
                        "action": action,
                        "next_state": next_state,
                        "condition": str(option.get("condition", "")),
                    }
                )

        derived["actions"] = sorted(set(actions))
        derived["legal_actions_by_state"] = {
            state: list(dict.fromkeys(state_actions))
            for state, state_actions in legal_by_state.items()
        }
        derived["transition_facts"] = transition_facts
        return derived

    def _maybe_fallback(
        self,
        errors: list[str],
        warnings: list[str],
        task_spec: Mapping[str, Any],
        adapter: Any | None,
        allow_fallback: bool,
        datalog_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not allow_fallback:
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                fsm_design=None,
                fallback_used=False,
                metadata={"datalog": datalog_result or {}},
            ).to_dict()

        fallback_design = build_generic_tool_use_fsm_design(task_spec)
        fallback_result = self.validate(
            fallback_design,
            task_spec=task_spec,
            adapter=adapter,
            allow_fallback=False,
        )
        fallback_result["fallback_used"] = True
        fallback_result["metadata"].setdefault("original_errors", errors)
        fallback_result["metadata"].setdefault("original_warnings", warnings)
        if datalog_result is not None:
            fallback_result["metadata"].setdefault("original_datalog", datalog_result)
        if fallback_result["valid"]:
            fallback_result["warnings"] = warnings + [
                "Invalid FSM proposal replaced with fallback FSM."
            ] + fallback_result["warnings"]
            fallback_result["errors"] = []
        else:
            fallback_result["errors"] = errors + [
                "Fallback FSM also failed validation."
            ] + fallback_result["errors"]
        return fallback_result

    @staticmethod
    def _path_checks(
        states: set[str],
        initial_state: str,
        terminal_states: set[str],
        transitions: Mapping[str, Any],
    ) -> list[str]:
        if initial_state not in states:
            return []
        reachable = {initial_state}
        changed = True
        while changed:
            changed = False
            for source, options in transitions.items():
                if str(source) not in reachable or not isinstance(options, list):
                    continue
                for option in options:
                    if not isinstance(option, Mapping):
                        continue
                    target = str(option.get("next_state", ""))
                    if target in states and target not in reachable:
                        reachable.add(target)
                        changed = True
        if not (reachable & terminal_states):
            return ["initial_state has no path to a terminal state."]
        return []

    @staticmethod
    def _premature_finalize_states(
        initial_state: str,
        transitions: Mapping[str, Any],
    ) -> list[str]:
        not_read_reachable: set[str] = set()
        frontier = [initial_state]
        while frontier:
            state = frontier.pop()
            if state in not_read_reachable:
                continue
            not_read_reachable.add(state)
            for option in transitions.get(state, []):
                if not isinstance(option, Mapping):
                    continue
                if option.get("action") == "read_task":
                    continue
                target = str(option.get("next_state", ""))
                if target and target not in not_read_reachable:
                    frontier.append(target)

        premature = []
        for state in sorted(not_read_reachable):
            for option in transitions.get(state, []):
                if isinstance(option, Mapping) and option.get("action") == "finalize":
                    premature.append(state)
        return premature

    @staticmethod
    def _adapter_actions(task_spec: Mapping[str, Any], adapter: Any | None) -> set[str]:
        actions = {str(action) for action in task_spec.get("available_tools", []) if str(action)}
        if actions or adapter is None:
            return actions
        try:
            return {
                str(action)
                for action in adapter.get_available_tools(
                    task_spec,
                    task_spec.get("initial_state", {}),
                )
            }
        except Exception:
            return set()


def build_generic_tool_use_fsm_design(
    task_spec: TaskSpec | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fallback FSM for arbitrary user scenarios.

    If the task exposes a non-generic custom action list, generate a simple
    linear fallback over those actions so validation still respects the adapter.
    """

    spec = task_spec_to_dict(task_spec) if task_spec is not None else {}
    available_tools = [str(action) for action in spec.get("available_tools", [])]
    generic_tools = {
        "read_task",
        "inspect_context",
        "ask_clarifying_question",
        "write_plan",
        "execute_step",
        "check_progress",
        "revise",
        "finalize",
    }
    available_set = set(available_tools)
    if available_tools and not generic_tools.issubset(available_set):
        available_non_finalize = [tool for tool in available_tools if tool != "finalize"]
        if available_non_finalize:
            return _build_linear_fallback(available_non_finalize, spec)
        return _build_finalize_only_fallback(spec)

    return {
        "states": [
            "START",
            "READ_TASK",
            "GATHER_CONTEXT",
            "MAKE_PLAN",
            "EXECUTE_STEP",
            "CHECK_PROGRESS",
            "REVISE",
            "FINALIZE",
            "DONE",
        ],
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "transitions_by_state": {
            "START": [
                {
                    "action": "read_task",
                    "next_state": "READ_TASK",
                    "condition": "The task instruction must be read first.",
                }
            ],
            "READ_TASK": [
                {
                    "action": "inspect_context",
                    "next_state": "GATHER_CONTEXT",
                    "condition": "More context is needed before planning.",
                },
                {
                    "action": "write_plan",
                    "next_state": "MAKE_PLAN",
                    "condition": "The task is clear enough to draft a plan.",
                },
            ],
            "GATHER_CONTEXT": [
                {
                    "action": "inspect_context",
                    "next_state": "GATHER_CONTEXT",
                    "condition": "More context is still needed.",
                },
                {
                    "action": "write_plan",
                    "next_state": "MAKE_PLAN",
                    "condition": "Enough context has been gathered.",
                },
            ],
            "MAKE_PLAN": [
                {
                    "action": "execute_step",
                    "next_state": "EXECUTE_STEP",
                    "condition": "A plan exists and work can start.",
                },
                {
                    "action": "inspect_context",
                    "next_state": "GATHER_CONTEXT",
                    "condition": "The plan revealed missing context.",
                },
            ],
            "EXECUTE_STEP": [
                {
                    "action": "check_progress",
                    "next_state": "CHECK_PROGRESS",
                    "condition": "A step was executed and should be checked.",
                },
                {
                    "action": "revise",
                    "next_state": "REVISE",
                    "condition": "Execution found a problem requiring revision.",
                },
            ],
            "CHECK_PROGRESS": [
                {
                    "action": "execute_step",
                    "next_state": "EXECUTE_STEP",
                    "condition": "More work remains.",
                },
                {
                    "action": "revise",
                    "next_state": "REVISE",
                    "condition": "Progress check found an issue.",
                },
                {
                    "action": "finalize",
                    "next_state": "FINALIZE",
                    "condition": "The task appears complete.",
                },
            ],
            "REVISE": [
                {
                    "action": "execute_step",
                    "next_state": "EXECUTE_STEP",
                    "condition": "Revision is complete and work should continue.",
                },
                {
                    "action": "write_plan",
                    "next_state": "MAKE_PLAN",
                    "condition": "The plan should be rewritten before continuing.",
                },
            ],
            "FINALIZE": [
                {
                    "action": "finalize",
                    "next_state": "DONE",
                    "condition": "Final answer or artifact is ready.",
                }
            ],
            "DONE": [],
        },
        "fallback_policy": {
            "on_invalid_action": "retry_with_valid_transition",
            "on_dead_end": "reset_to_last_valid_state",
        },
        "success_signals": spec.get("success_criteria", ["final output produced"]),
        "risk_notes": ["Fallback workflow FSM; not benchmark-specific."],
    }


def _build_linear_fallback(actions: list[str], task_spec: Mapping[str, Any]) -> dict[str, Any]:
    has_finalize = "finalize" in task_spec.get("available_tools", [])
    intermediate_count = len(actions) if has_finalize else max(len(actions) - 1, 0)
    states = ["START"] + [f"STEP_{idx + 1}" for idx in range(intermediate_count)] + ["DONE"]
    transitions: dict[str, list[dict[str, str]]] = {}

    for idx, action in enumerate(actions):
        source = "START" if idx == 0 else f"STEP_{idx}"
        if idx == len(actions) - 1 and not has_finalize:
            next_state = "DONE"
        else:
            next_state = f"STEP_{idx + 1}"
        transitions[source] = [
            {
                "action": action,
                "next_state": next_state,
                "condition": f"Execute available task action: {action}.",
            }
        ]

    final_source = f"STEP_{len(actions)}"
    final_action = "finalize" if has_finalize else actions[-1]
    if has_finalize:
        transitions[final_source] = [
            {
                "action": final_action,
                "next_state": "DONE",
                "condition": "All custom actions are complete.",
            }
        ]
    transitions.setdefault("DONE", [])
    return {
        "states": states,
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "transitions_by_state": transitions,
        "fallback_policy": {"on_invalid_action": "retry_with_valid_transition"},
        "success_signals": task_spec.get("success_criteria", ["final output produced"]),
        "risk_notes": ["Linear fallback generated from task_spec.available_tools."],
    }


def _build_finalize_only_fallback(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "states": ["START", "DONE"],
        "initial_state": "START",
        "terminal_states": ["DONE"],
        "transitions_by_state": {
            "START": [
                {
                    "action": "finalize",
                    "next_state": "DONE",
                    "condition": "Only finalize is exposed by the task spec.",
                }
            ],
            "DONE": [],
        },
        "fallback_policy": {"on_invalid_action": "retry_with_valid_transition"},
        "success_signals": task_spec.get("success_criteria", ["final output produced"]),
        "risk_notes": ["Finalize-only fallback generated from task_spec.available_tools."],
    }
