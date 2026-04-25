"""Unified bounded context manager for NS-FSM.

There is deliberately no retriever or RAG memory here. The context manager only
packages task, FSM, current T(s), adapter state, and compact histories into one
bounded packet for the model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

try:
    from .datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict


class ContextManager:
    """Build one bounded context packet for each NS-FSM LLM call."""

    def __init__(self, max_chars: int = 12000, recent_k: int = 8):
        self.max_chars = max_chars
        self.recent_k = recent_k
        self.compressed_memory = ""

    def build_packet(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        fsm: Any,
        adapter: Any,
        adapter_state: Mapping[str, Any],
        history: list[dict[str, Any]] | None = None,
        blocked_history: list[dict[str, Any]] | None = None,
        fallback_history: list[dict[str, Any]] | None = None,
        transition_options: list[dict[str, Any]] | None = None,
        legal_actions: list[str] | None = None,
        executable_actions: list[str] | None = None,
        verified_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        spec = task_spec_to_dict(task_spec)
        history = history or []
        blocked_history = blocked_history or []
        fallback_history = fallback_history or []

        if len(history) > self.recent_k:
            self.compressed_memory = self._summarize_history(history[:-self.recent_k])

        packet = {
            "task": {
                "dataset": spec.get("dataset", ""),
                "task_id": spec.get("task_id", ""),
                "task_type": spec.get("task_type", ""),
                "instruction": spec.get("instruction", ""),
                "success_criteria": spec.get("success_criteria", []),
            },
            "fsm_state": fsm.current_state,
            "transition_options": (
                deepcopy(transition_options)
                if transition_options is not None
                else fsm.get_valid_transitions()
            ),
            "legal_actions": (
                list(legal_actions)
                if legal_actions is not None
                else fsm.get_valid_actions()
            ),
            "executable_actions": list(executable_actions or []),
            "verified_actions": list(verified_actions or []),
            "fsm_summary": self._fsm_summary(fsm),
            "recent_history": deepcopy(history[-self.recent_k :]),
            "blocked_history": deepcopy(blocked_history[-self.recent_k :]),
            "fallback_history": deepcopy(fallback_history[-self.recent_k :]),
            "adapter_state_summary": adapter.format_state_for_prompt(adapter_state),
            "compressed_memory": self.compressed_memory or "(none)",
        }
        return self._bound_packet(packet)

    def _bound_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Trim histories first while preserving current state and T(s)."""

        while self._packet_size(packet) > self.max_chars and packet["recent_history"]:
            removed = packet["recent_history"].pop(0)
            self.compressed_memory = self._append_memory(
                self.compressed_memory,
                self._short_record(removed),
            )
            packet["compressed_memory"] = self.compressed_memory

        while self._packet_size(packet) > self.max_chars and packet["blocked_history"]:
            packet["blocked_history"].pop(0)

        while self._packet_size(packet) > self.max_chars and packet["fallback_history"]:
            packet["fallback_history"].pop(0)

        if self._packet_size(packet) > self.max_chars:
            summary = packet.get("adapter_state_summary", "")
            packet["adapter_state_summary"] = summary[: max(500, self.max_chars // 5)]

        return packet

    @staticmethod
    def _fsm_summary(fsm: Any) -> str:
        design = getattr(fsm, "fsm_design", {})
        states = design.get("states", [])
        terminals = design.get("terminal_states", [])
        action_count = len(design.get("actions", []))
        transition_count = len(design.get("transition_facts", []))
        return (
            f"states={len(states)}, actions={action_count}, "
            f"transitions={transition_count}, terminal_states={terminals}"
        )

    @staticmethod
    def _summarize_history(history: list[dict[str, Any]]) -> str:
        if not history:
            return ""
        total = len(history)
        successes = sum(1 for entry in history if entry.get("success") is True)
        failures = sum(1 for entry in history if entry.get("success") is False)
        last_actions = [
            str(entry.get("action", "?")) for entry in history[-5:]
        ]
        return (
            f"Older history summarized: {total} steps, "
            f"{successes} successes, {failures} failures, "
            f"recent older actions={last_actions}."
        )

    @staticmethod
    def _append_memory(existing: str, addition: str) -> str:
        if not existing or existing == "(none)":
            return addition
        return f"{existing}\n{addition}"

    @staticmethod
    def _short_record(entry: Mapping[str, Any]) -> str:
        return (
            f"Step {entry.get('step', '?')}: action={entry.get('action')}, "
            f"success={entry.get('success')}, state={entry.get('fsm_state_before')}"
        )

    @staticmethod
    def _packet_size(packet: Mapping[str, Any]) -> int:
        return len(str(packet))
