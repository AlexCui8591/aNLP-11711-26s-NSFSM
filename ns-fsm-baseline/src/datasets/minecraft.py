"""MC-TextWorld / Minecraft benchmark adapter.

This adapter is optional. Core NS-FSM code should work without importing it; use
it only when running Minecraft-style symbolic planning benchmarks.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Mapping

try:
    from ..minecraft_fallback import MinecraftFallbackSimulator, load_action_library
except ImportError:  # pragma: no cover - supports direct src execution
    from minecraft_fallback import MinecraftFallbackSimulator, load_action_library

try:
    from .base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict


class MinecraftAdapter(DatasetAdapter):
    """Adapter that wraps the existing MCTextWorldWrapper."""

    dataset_name = "minecraft"

    def __init__(
        self,
        goals_path: str | None = None,
        max_steps: int | None = None,
    ):
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.goals_path = goals_path or os.path.join(self.root, "config", "goals_67.json")
        self.max_steps_override = max_steps
        self._goals_cache: list[dict[str, Any]] | None = None
        self._actions_cache: list[str] | None = None
        self.task_spec: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.trajectory: list[dict[str, Any]] = []
        self._fallback_env = False
        self.env = None
        self.parser = None
        self.ground_truth = None
        self._fallback_simulator = None

    def list_tasks(self) -> list[dict[str, Any]]:
        if self._goals_cache is not None:
            return deepcopy(self._goals_cache)

        with open(self.goals_path, "r", encoding="utf-8") as f:
            grouped = json.load(f)

        tasks: list[dict[str, Any]] = []
        for group, payload in grouped.items():
            group_max_steps = int(payload.get("max_steps", 100))
            for entry in payload.get("goals", []):
                goal = entry["goal"]
                tasks.append(
                    {
                        **entry,
                        "dataset": self.dataset_name,
                        "task_id": f"minecraft/{goal}",
                        "group": group,
                        "max_steps": self.max_steps_override or min(group_max_steps, 100),
                    }
                )
        self._goals_cache = tasks
        return deepcopy(tasks)

    def load_or_wrap(self, raw_input: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_input, str):
            task_id = raw_input.strip()
            goal = task_id.split("/", 1)[-1]
            for task in self.list_tasks():
                if task["goal"] == goal or task["task_id"] == task_id:
                    return task
            return {
                "dataset": self.dataset_name,
                "task_id": f"minecraft/{goal}",
                "goal": goal,
                "type": "craft",
                "instruction": f"Obtain 1 {goal}.",
                "group": "custom",
                "max_steps": self.max_steps_override or 100,
            }

        raw = dict(raw_input)
        if "goal" not in raw:
            task_id = str(raw.get("task_id", "minecraft/custom"))
            raw["goal"] = task_id.split("/", 1)[-1]
        raw.setdefault("dataset", self.dataset_name)
        raw.setdefault("task_id", f"minecraft/{raw['goal']}")
        raw.setdefault("instruction", f"Obtain 1 {raw['goal']}.")
        raw.setdefault("type", "craft")
        raw.setdefault("group", "custom")
        raw.setdefault("max_steps", self.max_steps_override or 100)
        return raw

    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        raw = self.load_or_wrap(raw_task)
        goal = str(raw["goal"])
        optimal_sequence = self._optimal_sequence(goal)

        return TaskSpec(
            dataset=self.dataset_name,
            task_id=str(raw.get("task_id", f"minecraft/{goal}")),
            task_type="symbolic_planning",
            instruction=str(raw.get("instruction", f"Obtain 1 {goal}.")),
            initial_state={"inventory": {}, "goal": goal, "step": 0},
            goal_condition={f"inventory.{goal}": ">=1"},
            available_tools=self._all_actions(),
            max_steps=int(raw.get("max_steps", self.max_steps_override or 100)),
            success_criteria=[f"Inventory contains at least 1 {goal}."],
            metadata={
                "goal_item": goal,
                "group": raw.get("group", "custom"),
                "goal_type": raw.get("type", "craft"),
                "optimal_sequence": optimal_sequence,
                "required_actions": optimal_sequence,
            },
        )

    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        self.task_spec = task_spec_to_dict(task_spec)
        goal = self.task_spec.get("metadata", {}).get("goal_item")
        self.trajectory = []
        try:
            wrapper_cls = self._get_wrapper_cls()
            self.env = wrapper_cls(max_steps=int(self.task_spec.get("max_steps", 100)))
            self.state = self.env.reset(goal)
            self._fallback_env = False
        except Exception as exc:
            self.env = None
            self._fallback_env = True
            self.state = {
                "inventory": {},
                "position": [0, 0, 0],
                "biome": "plains",
                "step": 0,
                "max_steps": int(self.task_spec.get("max_steps", 100)),
                "goal": goal,
                "fallback_env_reason": str(exc),
            }
        return self.get_observation()

    def step(self, action: str | Mapping[str, Any]) -> StepResult:
        if self.task_spec is None:
            raise RuntimeError("Call reset(task_spec) before MinecraftAdapter.step(action).")
        action_name = self._action_name(action)
        if self.env is None:
            return self._fallback_step(action_name)
        obs, done, info = self.env.step(action_name)
        self.state = obs
        return StepResult(self.get_observation(), done, info)

    def get_observation(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def is_done(
        self,
        state: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any],
    ) -> bool:
        spec = task_spec_to_dict(task_spec)
        goal = spec.get("metadata", {}).get("goal_item")
        inventory = state.get("inventory", {})
        return bool(goal and inventory.get(goal, 0) >= 1) or int(state.get("step", 0)) >= int(
            spec.get("max_steps", 100)
        )

    def get_available_tools(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        inventory = dict(state.get("inventory", {}))
        if self._fallback_env:
            return self._candidate_actions(inventory)
        if inventory or self.env is not None:
            try:
                parser = self._get_parser()
                return parser.get_candidate_actions(inventory)
            except Exception:
                pass
        spec = task_spec_to_dict(task_spec)
        return list(spec.get("available_tools") or self._all_actions())

    def normalize_action(
        self,
        raw_action: str | Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> str | None:
        action_name = self._action_name(raw_action)
        legal_names = [
            str(item.get("action", "")) if isinstance(item, Mapping) else str(item)
            for item in legal_actions
        ]
        if action_name in legal_names:
            return action_name

        try:
            parsed, _reason = self._get_parser().parse(f"Action: {action_name}")
            if parsed in legal_names:
                return parsed
        except Exception:
            pass

        lowered = {name.lower(): name for name in legal_names}
        if action_name.lower() in lowered:
            return lowered[action_name.lower()]
        return None

    def format_state_for_prompt(self, state: Mapping[str, Any]) -> str:
        inventory = state.get("inventory", {})
        inventory_text = ", ".join(f"{item}: {qty}" for item, qty in sorted(inventory.items()))
        if not inventory_text:
            inventory_text = "(empty)"
        return "\n".join(
            [
                f"goal: {state.get('goal')}",
                f"step: {state.get('step')} / {state.get('max_steps')}",
                f"inventory: {inventory_text}",
            ]
        )

    def summarize_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        trajectory = self.env.get_trajectory() if self.env is not None else {
            "total_steps": state.get("step", 0),
            "final_success": self.is_done(state, self.task_spec or {}),
            "steps": self.trajectory,
        }
        return {
            "dataset": self.dataset_name,
            "success": bool(trajectory.get("final_success", False)),
            "total_steps": int(trajectory.get("total_steps", state.get("step", 0))),
            "termination": "success" if trajectory.get("final_success") else "max_steps_or_incomplete",
            "trajectory": trajectory.get("steps", []),
        }

    def _optimal_sequence(self, goal: str) -> list[str]:
        try:
            return list(self._get_ground_truth().get_optimal_sequence(goal))
        except Exception:
            return []

    def _get_parser(self):
        if self.parser is None:
            try:
                from ..action_parser import ActionParser
            except ImportError:  # pragma: no cover
                from action_parser import ActionParser
            self.parser = ActionParser()
        return self.parser

    def _all_actions(self) -> list[str]:
        if self._actions_cache is not None:
            return list(self._actions_cache)

        try:
            raw = self._action_library()
            actions = sorted(action for action in raw if action != "no_op")
        except Exception:
            try:
                actions = sorted(action for action in self._get_parser().valid_actions if action != "no_op")
            except Exception:
                actions = []
        self._actions_cache = actions
        return list(actions)

    def _fallback_step(self, action_name: str) -> StepResult:
        inventory_before = deepcopy(self.state.get("inventory", {}))
        success, message, inventory_after = self._apply_action(action_name, inventory_before)
        self.state["inventory"] = inventory_after
        self.state["step"] = int(self.state.get("step", 0)) + 1
        goal = self.state.get("goal")
        goal_achieved = bool(goal and inventory_after.get(goal, 0) >= 1)
        timeout = self.state["step"] >= int(self.state.get("max_steps", 100))
        done = goal_achieved or timeout
        info = {
            "success": success,
            "message": message,
            "goal_achieved": goal_achieved,
            "timeout": timeout,
            "fallback_env": True,
        }
        self.trajectory.append(
            {
                "step": self.state["step"],
                "action": action_name,
                "success": success,
                "inventory_before": inventory_before,
                "inventory_after": deepcopy(inventory_after),
                "message": message,
                "done": done,
            }
        )
        return StepResult(self.get_observation(), done, info)

    def _candidate_actions(self, inventory: Mapping[str, int]) -> list[str]:
        return self._get_fallback_simulator().candidate_actions(inventory)

    def _apply_action(
        self,
        action_name: str,
        inventory: Mapping[str, int],
    ) -> tuple[bool, str, dict[str, int]]:
        return self._get_fallback_simulator().step(action_name, inventory)

    def _action_library(self) -> dict[str, list[dict[str, Any]]]:
        return self._get_fallback_simulator().action_library

    def _get_fallback_simulator(self) -> MinecraftFallbackSimulator:
        if self._fallback_simulator is None:
            summary_path = os.path.join(self.root, "config", "action_lib_summary.json")
            self._fallback_simulator = MinecraftFallbackSimulator(load_action_library(summary_path))
        return self._fallback_simulator

    def _get_ground_truth(self):
        if self.ground_truth is None:
            try:
                from ..ground_truth import GroundTruth
            except ImportError:  # pragma: no cover
                from ground_truth import GroundTruth
            self.ground_truth = GroundTruth()
        return self.ground_truth

    @staticmethod
    def _get_wrapper_cls():
        try:
            from ..env_wrapper import MCTextWorldWrapper
        except ImportError:  # pragma: no cover
            from env_wrapper import MCTextWorldWrapper
        return MCTextWorldWrapper

    @staticmethod
    def _action_name(action: str | Mapping[str, Any]) -> str:
        if isinstance(action, Mapping):
            return str(action.get("action") or action.get("name") or "").strip()
        return str(action).strip()
