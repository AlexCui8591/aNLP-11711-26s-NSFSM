"""Generic NS-FSM runtime agent."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

try:
    from .context_manager import ContextManager
    from .datasets.base import TaskSpec, task_spec_to_dict
    from .fsm import RuntimeFSM
    from .planner import Planner
    from .prompts import build_nsfsm_prompt
    from .rule_checker import RuleChecker
except ImportError:  # pragma: no cover - supports direct src execution
    from context_manager import ContextManager
    from datasets.base import TaskSpec, task_spec_to_dict
    from fsm import RuntimeFSM
    from planner import Planner
    from prompts import build_nsfsm_prompt
    from rule_checker import RuleChecker


class NSFSMAgent:
    """Run one task under FSM/Datalog action control."""

    def __init__(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any,
        fsm: RuntimeFSM,
        llm: Any | None = None,
        rule_checker: RuleChecker | None = None,
        planner: Planner | None = None,
        context_manager: ContextManager | None = None,
        planner_only: bool = False,
        require_llm: bool = False,
        max_llm_retries: int = 1,
        max_blocked_repeats: int = 3,
        verbose: bool = False,
    ):
        self.task_spec = task_spec_to_dict(task_spec)
        self.adapter = adapter
        self.fsm = fsm
        self.llm = llm
        self.rule_checker = rule_checker or RuleChecker()
        self.planner = planner or Planner()
        self.context_manager = context_manager or ContextManager()
        self.planner_only = planner_only
        self.require_llm = require_llm
        self.max_llm_retries = max_llm_retries
        self.max_blocked_repeats = max_blocked_repeats
        self.verbose = verbose
        self.trajectory: list[dict[str, Any]] = []
        self.blocked_actions: list[dict[str, Any]] = []
        self.fallback_actions: list[dict[str, Any]] = []
        self.current_executable_actions: list[str] = []
        self.current_verified_transition_options: list[dict[str, Any]] = []
        self.current_verified_actions: list[str] = []
        if self.require_llm and (self.planner_only or self.llm is None):
            raise ValueError("require_llm=True requires an LLM client and planner_only=False.")

    def run_episode(self) -> dict[str, Any]:
        state = self.adapter.reset(self.task_spec)
        self.fsm.reset()
        self.trajectory = []
        self.blocked_actions = []
        self.fallback_actions = []
        termination = "max_steps"
        max_steps = int(self.task_spec.get("max_steps", 30))

        for step_idx in range(max_steps):
            if self.adapter.is_done(state, self.task_spec):
                termination = "success"
                break
            if self.fsm.is_terminal():
                termination = "fsm_terminal"
                break

            transition_options, candidate_actions = self._runtime_transition_options(state)
            if not transition_options or not candidate_actions:
                termination = "no_valid_action"
                break

            context_packet = self.context_manager.build_packet(
                task_spec=self.task_spec,
                fsm=self.fsm,
                adapter=self.adapter,
                adapter_state=state,
                history=self.trajectory,
                blocked_history=self.blocked_actions,
                fallback_history=self.fallback_actions,
                transition_options=transition_options,
                legal_actions=candidate_actions,
                executable_actions=self.current_executable_actions,
                verified_actions=self.current_verified_actions,
            )

            decision, proposal = self._verified_decision_with_retries(
                context_packet=context_packet,
                state=state,
                legal_actions=candidate_actions,
                transition_options=transition_options,
            )
            if decision is None:
                termination = "no_valid_action"
                break

            previous_fsm_state = self.fsm.current_state
            step_result = self.adapter.step(
                {
                    "action": decision["action"],
                    "payload": decision.get("payload", {}),
                }
            )
            state = step_result.state
            info = dict(step_result.info)
            adapter_success = bool(info.get("success", True))

            fsm_update = {"updated": False, "current_state": self.fsm.current_state}
            if adapter_success:
                preferred_next_state = info.get("fsm_next_state") or decision.get("next_state")
                fsm_update = self.fsm.update(
                    decision["action"],
                    preferred_next_state,
                    info,
                )

            record = {
                "step": step_idx + 1,
                "fsm_state_before": previous_fsm_state,
                "fsm_state_after": self.fsm.current_state,
                "action": decision["action"],
                "next_state": decision.get("next_state"),
                "decision_source": decision.get("source", "unknown"),
                "forced_choice": bool(decision.get("forced_choice", False)),
                "blocked": decision.get("blocked", False),
                "success": adapter_success,
                "message": info.get("message", ""),
                "proposal": proposal,
                "rule_check": decision.get("rule_check", {}),
                "transition_check": decision.get("transition_check", {}),
                "fsm_update": fsm_update,
                "info": info,
                "adapter_state": state,
            }
            self.trajectory.append(record)

            if self.verbose:
                print(
                    f"[NS-FSM] step={step_idx + 1} state={previous_fsm_state} "
                    f"action={decision['action']} -> {self.fsm.current_state} "
                    f"success={adapter_success}"
                )

            if step_result.done or self.adapter.is_done(state, self.task_spec):
                termination = "success" if self._adapter_success(state) else "max_steps"
                break
            if self._blocked_dead_loop():
                termination = "repeated_blocked_actions"
                break
        else:
            termination = "max_steps"

        summary = self.adapter.summarize_result(state)
        success = bool(summary.get("success", termination == "success"))
        if success:
            termination = "success"

        return {
            "dataset": self.task_spec.get("dataset"),
            "task_id": self.task_spec.get("task_id"),
            "task_type": self.task_spec.get("task_type"),
            "success": success,
            "total_steps": len(self.trajectory),
            "termination": termination,
            "blocked_action_count": len(self.blocked_actions),
            "fallback_action_count": len(self.fallback_actions),
            "trajectory": self.trajectory,
            "metadata": {
                "task_spec": self.task_spec,
                "adapter_summary": summary,
                "fsm": self.fsm.to_dict(),
                "planner_only": self.planner_only,
                "require_llm": self.require_llm,
                "verification_mode": "posthoc_generate_then_verify",
            },
        }

    def _propose_action(self, context_packet: Mapping[str, Any]) -> dict[str, Any]:
        if self.planner_only or self.llm is None:
            if self.require_llm:
                raise RuntimeError("LLM inference is required, but no runtime LLM is configured.")
            return {
                "thought": "Planner-only mode or no LLM available.",
                "action": None,
                "next_state": None,
                "payload": {},
                "source": "planner",
                "raw": "",
            }

        system_prompt, user_prompt = build_nsfsm_prompt(dict(context_packet))
        try:
            raw = self.llm.generate(system_prompt, user_prompt)
            parsed = parse_nsfsm_response(raw)
            parsed["source"] = "llm"
            parsed["raw"] = raw
            return parsed
        except Exception as exc:
            if self.require_llm:
                raise RuntimeError(f"LLM inference is required but generate() failed: {exc}") from exc
            return {
                "thought": f"LLM failed; using planner fallback. Error: {exc}",
                "action": None,
                "next_state": None,
                "payload": {},
                "source": "planner",
                "raw": "",
                "llm_error": str(exc),
            }

    def _verified_decision_with_retries(
        self,
        context_packet: Mapping[str, Any],
        state: Mapping[str, Any],
        legal_actions: list[str],
        transition_options: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.planner_only or self.llm is None:
            if self.require_llm:
                raise RuntimeError("LLM inference is required, but planner-only mode is active.")
            proposal = self._planner_proposal(
                state=state,
                legal_actions=legal_actions,
                transition_options=transition_options,
                blocked_reason={},
            )
            decision, blocked_reason = self._try_llm_decision(
                proposal=proposal,
                state=state,
                legal_actions=legal_actions,
                transition_options=transition_options,
            )
            if decision is not None:
                return decision, proposal

            self._record_blocked_action(proposal, blocked_reason)
            decision = self._planner_fsm_fallback(
                state=state,
                blocked_reason=blocked_reason,
                record_block=True,
                source="forced_choice",
            )
            return (decision, proposal)

        last_proposal: dict[str, Any] = {}
        last_blocked_reason: dict[str, Any] = {}
        attempts = self.max_llm_retries + 1
        for attempt_idx in range(1, attempts + 1):
            retry_packet = dict(context_packet)
            if last_blocked_reason:
                retry_packet["last_verification_error"] = last_blocked_reason
            proposal = self._propose_action(retry_packet)
            proposal["attempt"] = attempt_idx
            last_proposal = proposal

            decision, blocked_reason = self._try_llm_decision(
                proposal=proposal,
                state=state,
                legal_actions=legal_actions,
                transition_options=transition_options,
            )
            if decision is not None:
                return decision, proposal

            last_blocked_reason = blocked_reason
            self._record_blocked_action(proposal, blocked_reason)

        planner_decision = self._planner_fsm_fallback(
            state=state,
            blocked_reason=last_blocked_reason,
            record_block=True,
            source="forced_choice",
        )
        if planner_decision is not None:
            planner_decision["forced_after_llm_retries"] = True
        return planner_decision, last_proposal

    def _runtime_transition_options(
        self,
        state: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if hasattr(self.adapter, "set_runtime_fsm_state"):
            try:
                self.adapter.set_runtime_fsm_state(self.fsm.current_state)
            except Exception:
                pass
        static_options = self.fsm.get_valid_transitions()

        try:
            executable = set(self.adapter.get_available_tools(self.task_spec, state))
        except Exception:
            executable = set()

        self.current_executable_actions = sorted(executable)
        if executable:
            self.current_verified_transition_options = [
                dict(option)
                for option in static_options
                if str(option.get("action")) in executable
            ]
        else:
            self.current_verified_transition_options = (
                static_options
                if self.task_spec.get("dataset") not in {"minecraft", "robotouille"}
                else []
            )
        self.current_verified_actions = list(
            dict.fromkeys(
                str(option.get("action"))
                for option in self.current_verified_transition_options
                if str(option.get("action"))
            )
        )
        return static_options, self.fsm.get_valid_actions()

    def _try_llm_decision(
        self,
        proposal: Mapping[str, Any],
        state: Mapping[str, Any],
        legal_actions: list[str],
        transition_options: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        proposed_action = proposal.get("action")
        normalized_action = None
        if proposed_action:
            normalized_action = self.adapter.normalize_action(
                proposed_action,
                self._normalization_candidates(legal_actions),
            )

        proposed_next = self._normalize_next_state(proposal.get("next_state"))
        transition_check = (
            self._verify_transition_with_next_fallback(
                normalized_action,
                proposed_next,
                transition_options,
            )
            if normalized_action
            else {"valid": False, "violations": [{"type": "parse_error"}]}
        )
        rule_check = self.rule_checker.check(
            normalized_action,
            legal_actions,
            state,
            self.task_spec,
            self.adapter,
            transition_check,
        )

        if rule_check["legal"] and transition_check.get("valid"):
            return {
                "action": normalized_action,
                "next_state": transition_check.get("inferred_next_state") or proposed_next,
                "payload": proposal.get("payload", {}),
                "source": proposal.get("source", "llm"),
                "blocked": False,
                "rule_check": rule_check,
                "transition_check": transition_check,
            }, {}

        blocked_reason = {
            "proposal": dict(proposal),
            "rule_check": rule_check,
            "transition_check": transition_check,
        }
        return None, blocked_reason

    def _planner_proposal(
        self,
        state: Mapping[str, Any],
        legal_actions: list[str],
        transition_options: list[dict[str, Any]],
        blocked_reason: Mapping[str, Any],
    ) -> dict[str, Any]:
        fallback = self.planner.next_action(
            self.task_spec,
            state,
            legal_actions,
            self.trajectory,
            transition_options,
            blocked_reason,
        )
        if fallback is None:
            return {
                "thought": "Planner could not propose an action.",
                "action": None,
                "next_state": None,
                "payload": {},
                "source": "planner",
                "raw": "",
            }
        return {
            "thought": fallback.get("reason", "planner proposal"),
            "action": fallback.get("action"),
            "next_state": fallback.get("next_state"),
            "payload": {},
            "source": "planner",
            "raw": "",
            "planner_proposal": fallback,
        }

    def _record_blocked_action(
        self,
        proposal: Mapping[str, Any],
        blocked_reason: Mapping[str, Any],
    ) -> None:
        rule_check = dict(blocked_reason.get("rule_check", {}))
        self.blocked_actions.append(
            {
                "step": len(self.trajectory) + 1,
                "state": self.fsm.current_state,
                "action": proposal.get("action"),
                "success": False,
                "message": rule_check.get("message", "blocked"),
                "reason": blocked_reason,
            }
        )
        self.fsm.record_blocked_action(proposal.get("action"), blocked_reason)

    def _planner_decision(
        self,
        state: Mapping[str, Any],
        legal_actions: list[str],
        transition_options: list[dict[str, Any]],
        blocked_reason: Mapping[str, Any],
        record_block: bool,
    ) -> dict[str, Any] | None:
        fallback = self.planner.next_action(
            self.task_spec,
            state,
            legal_actions,
            self.trajectory,
            transition_options,
            blocked_reason,
        )
        if fallback is None:
            return None

        fallback_action = self.adapter.normalize_action(fallback["action"], legal_actions)
        fallback_transition = self.fsm.verify_transition(
            fallback_action,
            fallback.get("next_state"),
        )
        fallback_rule = self.rule_checker.check(
            fallback_action,
            legal_actions,
            state,
            self.task_spec,
            self.adapter,
            fallback_transition,
        )
        if not fallback_action or not fallback_rule["legal"] or not fallback_transition.get("valid"):
            self.blocked_actions.append(
                {
                    "step": len(self.trajectory) + 1,
                    "state": self.fsm.current_state,
                    "action": fallback.get("action"),
                    "success": False,
                    "message": "Planner fallback was also illegal.",
                    "reason": {
                        "rule_check": fallback_rule,
                        "transition_check": fallback_transition,
                    },
                }
            )
            return None

        self.fallback_actions.append(
            {
                "step": len(self.trajectory) + 1,
                "state": self.fsm.current_state,
                "action": fallback_action,
                "next_state": fallback.get("next_state"),
                "success": True,
                "message": fallback.get("reason", "planner fallback"),
                "reason": fallback,
            }
        )
        self.fsm.record_fallback_action(
            fallback_action,
            fallback.get("next_state"),
            fallback.get("reason", "planner fallback"),
        )
        return {
            "action": fallback_action,
            "next_state": fallback.get("next_state")
            or fallback_transition.get("inferred_next_state"),
            "payload": {},
            "source": "planner",
            "blocked": record_block,
            "rule_check": fallback_rule,
            "transition_check": fallback_transition,
        }

    def _planner_verified_fallback(
        self,
        state: Mapping[str, Any],
        blocked_reason: Mapping[str, Any],
        record_block: bool,
        source: str = "planner",
    ) -> dict[str, Any] | None:
        decision = self._planner_decision(
            state=state,
            legal_actions=self.current_verified_actions,
            transition_options=self.current_verified_transition_options,
            blocked_reason=blocked_reason,
            record_block=record_block,
        )
        if decision is not None:
            decision["source"] = source
            decision["forced_choice"] = source == "forced_choice"
        return decision

    def _planner_fsm_fallback(
        self,
        state: Mapping[str, Any],
        blocked_reason: Mapping[str, Any],
        record_block: bool,
        source: str = "planner",
    ) -> dict[str, Any] | None:
        transition_options = self.fsm.get_valid_transitions()
        if not transition_options:
            return None

        fallback = self.planner.next_action(
            self.task_spec,
            state,
            [option["action"] for option in transition_options],
            self.trajectory,
            transition_options,
            blocked_reason,
        )
        if fallback is None:
            option = transition_options[0]
            fallback = {
                "action": option.get("action"),
                "next_state": option.get("next_state"),
                "reason": "forced_fsm_transition_policy",
            }

        action = self.adapter.normalize_action(
            fallback.get("action"),
            [option["action"] for option in transition_options],
        ) or str(fallback.get("action") or "")
        if not action:
            return None

        next_state = self._normalize_next_state(fallback.get("next_state"))
        transition_check = self._verify_transition_with_next_fallback(
            action,
            next_state,
            transition_options,
        )
        if not transition_check.get("valid"):
            return None

        selected_next_state = transition_check.get("inferred_next_state") or next_state
        self.fallback_actions.append(
            {
                "step": len(self.trajectory) + 1,
                "state": self.fsm.current_state,
                "action": action,
                "next_state": selected_next_state,
                "success": True,
                "message": fallback.get("reason", "forced FSM fallback"),
                "reason": fallback,
            }
        )
        self.fsm.record_fallback_action(
            action,
            selected_next_state,
            fallback.get("reason", "forced FSM fallback"),
        )
        return {
            "action": action,
            "next_state": selected_next_state,
            "payload": {},
            "source": source,
            "forced_choice": source == "forced_choice",
            "blocked": record_block,
            "rule_check": {
                "legal": True,
                "reason_type": "forced_fsm_choice",
                "message": (
                    "Repeated proposals failed post-hoc FSM verification; "
                    "selected a valid FSM transition to keep the episode running."
                ),
                "matched_action": action,
                "metadata": {
                    "fsm_candidate_actions": [
                        str(option.get("action")) for option in transition_options
                    ],
                    "executable_actions": list(self.current_executable_actions),
                },
            },
            "transition_check": transition_check,
        }

    def _normalization_candidates(self, legal_actions: list[str]) -> list[str]:
        candidates = list(legal_actions)
        candidates.extend(str(action) for action in self.task_spec.get("available_tools", []))
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _normalize_next_state(self, proposed_next: Any) -> str | None:
        if proposed_next is None:
            return None
        if hasattr(self.adapter, "normalize_state_name"):
            try:
                normalized = self.adapter.normalize_state_name(str(proposed_next))
                if normalized:
                    return normalized
            except Exception:
                pass
        return str(proposed_next).strip() or None

    def _verify_transition_with_next_fallback(
        self,
        action: str,
        proposed_next: str | None,
        transition_options: list[dict[str, Any]],
    ) -> dict[str, Any]:
        check = self.fsm.verify_transition(action, proposed_next)
        if check.get("valid"):
            return check

        fallback_next = self._next_state_for_action(action, self.current_verified_transition_options)
        if fallback_next is None:
            fallback_next = self._next_state_for_action(action, transition_options)
        if fallback_next is None and proposed_next is not None:
            inferred = self.fsm.verify_transition(action, None)
            if inferred.get("valid"):
                inferred["next_state_fallback_used"] = True
                inferred["original_next_state"] = proposed_next
                return inferred
            return check
        if fallback_next is None or fallback_next == proposed_next:
            return check

        fallback_check = self.fsm.verify_transition(action, fallback_next)
        if fallback_check.get("valid"):
            fallback_check["next_state_fallback_used"] = True
            fallback_check["original_next_state"] = proposed_next
        return fallback_check if fallback_check.get("valid") else check

    @staticmethod
    def _next_state_for_action(
        action: str,
        transition_options: list[dict[str, Any]],
    ) -> str | None:
        for option in transition_options:
            if str(option.get("action")) == action:
                next_state = str(option.get("next_state", "")).strip()
                if next_state:
                    return next_state
        return None

    def _adapter_success(self, state: Mapping[str, Any]) -> bool:
        try:
            return bool(self.adapter.summarize_result(state).get("success", False))
        except Exception:
            return False

    def _blocked_dead_loop(self) -> bool:
        if len(self.blocked_actions) < self.max_blocked_repeats:
            return False
        recent = self.blocked_actions[-self.max_blocked_repeats :]
        pairs = {(entry.get("state"), entry.get("action")) for entry in recent}
        return len(pairs) == 1


def parse_nsfsm_response(text: str) -> dict[str, Any]:
    """Parse JSON or Thought/Action/Next State NS-FSM output."""

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            return {
                "thought": str(payload.get("thought", "")),
                "action": payload.get("action"),
                "next_state": payload.get("next_state") or payload.get("nextState"),
                "payload": dict(payload.get("payload") or {}),
            }
        except json.JSONDecodeError:
            pass

    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\nAction:|\nNext State:|\Z)",
        stripped,
        re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(r"Action:\s*([^\n]+)", stripped, re.IGNORECASE)
    next_match = re.search(r"Next State:\s*([^\n]+)", stripped, re.IGNORECASE)
    return {
        "thought": thought_match.group(1).strip() if thought_match else "",
        "action": action_match.group(1).strip() if action_match else None,
        "next_state": next_match.group(1).strip() if next_match else None,
        "payload": {},
    }
