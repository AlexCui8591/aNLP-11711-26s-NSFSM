"""LLM-assisted FSM proposal stage for NS-FSM.

The designer is intentionally only a proposal generator. Its output must be
validated before any agent uses it to constrain actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from .datasets.base import TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict


FSM_DESIGNER_SYSTEM_PROMPT = """You are designing a workflow-level FSM for an NS-FSM agent.
Return JSON only.
Do not include code fences.
Do not invent external tools unless they are provided.
Every state must be a short uppercase identifier.
Every non-terminal state must have at least one outgoing transition.
Use transitions_by_state as the only source of transition information.
Every transition option must include action, next_state, and condition.
Every transitions_by_state key must be a known state.
Every transition next_state must be a known state.
Do not output separate actions, legal_actions_by_state, or transitions fields.
Include a safe fallback_policy, success_signals, and risk_notes.
"""


FSM_DESIGNER_USER_TEMPLATE = """Design a task-specific FSM for this NS-FSM task.

TaskSpec:
{task_spec_json}

Available tools/actions:
{tools_json}

Success criteria:
{success_criteria_json}

Benchmark or adapter metadata:
{metadata_json}

Required JSON schema:
{{
  "states": ["START", "...", "DONE"],
  "initial_state": "START",
  "terminal_states": ["DONE"],
  "transitions_by_state": {{
    "START": [
      {{"action": "read_task", "next_state": "UNDERSTAND_TASK", "condition": "task is read"}}
    ],
    "DONE": []
  }},
  "fallback_policy": {{
    "on_invalid_action": "retry_with_valid_transition",
    "on_dead_end": "fall_back_to_generic_tool_use_fsm"
  }},
  "success_signals": ["final output produced"],
  "risk_notes": ["workflow-level FSM only"]
}}
"""


@dataclass
class FSMProposal:
    """FSM proposal plus raw LLM metadata for debugging."""

    fsm_design: dict[str, Any]
    raw_response: str
    task_hash: str
    cache_path: str | None = None


class LLMFSMDesigner:
    """Call the configured LLM once to propose a task-specific FSM."""

    def __init__(
        self,
        llm: Any | None = None,
        cache_dir: str | None = None,
        force_refresh: bool = False,
        config_path: str | None = None,
    ):
        self.llm = llm
        self.config_path = config_path
        self.force_refresh = force_refresh
        if cache_dir is None:
            root = os.path.dirname(os.path.dirname(__file__))
            cache_dir = os.path.join(root, "results", "fsm_designs")
        self.cache_dir = cache_dir

    def design(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
        user_tools: list[str] | None = None,
        success_criteria: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a proposed FSMDesign dict.

        The returned design has not been formally validated. Call
        FSMDesignValidator before using it in the runtime loop.
        """

        proposal = self.design_with_metadata(
            task_spec=task_spec,
            adapter=adapter,
            user_tools=user_tools,
            success_criteria=success_criteria,
        )
        return proposal.fsm_design

    def design_with_metadata(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
        user_tools: list[str] | None = None,
        success_criteria: list[str] | None = None,
    ) -> FSMProposal:
        spec = task_spec_to_dict(task_spec)
        tools = user_tools or self._available_tools(spec, adapter)
        criteria = success_criteria or list(spec.get("success_criteria", []))
        task_hash = self.task_hash(spec, tools)
        cache_path = os.path.join(self.cache_dir, f"{task_hash}.json")

        if not self.force_refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return FSMProposal(
                    fsm_design=cached["fsm_design"],
                    raw_response=cached.get("raw_response", ""),
                    task_hash=task_hash,
                    cache_path=cache_path,
                )

        llm = self.llm or self._load_llm_interface()
        system_prompt, user_prompt = self.build_prompt(spec, tools, criteria)
        raw_response = llm.generate(system_prompt, user_prompt)
        fsm_design = parse_json_object(raw_response)
        fsm_design.setdefault("metadata", {})
        fsm_design["metadata"].update(
            {
                "task_hash": task_hash,
                "raw_llm_response": raw_response,
                "source": "llm_fsm_designer",
            }
        )
        self._write_cache(
            cache_path,
            {
                "task_hash": task_hash,
                "task_spec": spec,
                "tools": tools,
                "raw_response": raw_response,
                "fsm_design": fsm_design,
            },
        )
        return FSMProposal(
            fsm_design=fsm_design,
            raw_response=raw_response,
            task_hash=task_hash,
            cache_path=cache_path,
        )

    def build_prompt(
        self,
        task_spec: Mapping[str, Any],
        tools: list[str],
        success_criteria: list[str],
    ) -> tuple[str, str]:
        metadata = dict(task_spec.get("metadata", {}))
        metadata.pop("raw_llm_response", None)
        user_prompt = FSM_DESIGNER_USER_TEMPLATE.format(
            task_spec_json=json.dumps(task_spec, indent=2, sort_keys=True),
            tools_json=json.dumps(tools, indent=2),
            success_criteria_json=json.dumps(success_criteria, indent=2),
            metadata_json=json.dumps(metadata, indent=2, sort_keys=True),
        )
        return FSM_DESIGNER_SYSTEM_PROMPT, user_prompt

    @staticmethod
    def task_hash(task_spec: Mapping[str, Any], tools: list[str] | None = None) -> str:
        payload = {
            "dataset": task_spec.get("dataset"),
            "task_id": task_spec.get("task_id"),
            "task_type": task_spec.get("task_type"),
            "instruction": task_spec.get("instruction"),
            "available_tools": tools or task_spec.get("available_tools", []),
            "success_criteria": task_spec.get("success_criteria", []),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    @staticmethod
    def _available_tools(task_spec: Mapping[str, Any], adapter: Any | None) -> list[str]:
        state = task_spec.get("initial_state", {})
        if adapter is not None:
            try:
                return list(adapter.get_available_tools(task_spec, state))
            except Exception:
                pass
        return list(task_spec.get("available_tools", []))

    @staticmethod
    def _read_cache(cache_path: str) -> dict[str, Any] | None:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_cache(cache_path: str, payload: Mapping[str, Any]) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _load_llm_interface(self) -> Any:
        try:
            try:
                from .llm_interface import LLMInterface
            except ImportError:  # pragma: no cover - supports direct src execution
                from llm_interface import LLMInterface
        except Exception as exc:
            raise RuntimeError(
                "LLMInterface could not be imported. Install optional LLM "
                "dependencies or run with --use-fixed-generic-fsm."
            ) from exc
        return LLMInterface(self.config_path)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw LLM output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM FSM proposal did not contain a JSON object.")
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM FSM proposal must be a JSON object.")
    return parsed
