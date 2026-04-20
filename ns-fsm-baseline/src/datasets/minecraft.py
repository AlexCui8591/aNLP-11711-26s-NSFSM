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
        task_source: str = "buildable",
        task_source_path: str | None = None,
    ):
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.task_source = task_source
        self.goals_path = goals_path or os.path.join(self.root, "config", "goals_67.json")
        self.task_source_path = task_source_path or os.path.join(
            self.root,
            "config",
            "mctextworld_ground_truth_buildable_tasks.json",
        )
        self.max_steps_override = max_steps
        self._goals_cache: list[dict[str, Any]] | None = None
        self._actions_cache: list[str] | None = None
        self.task_spec: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.trajectory: list[dict[str, Any]] = []
        self.env = None
        self.parser = None
        self.ground_truth = None

    def list_tasks(self) -> list[dict[str, Any]]:
        if self._goals_cache is not None:
            return deepcopy(self._goals_cache)

        if self.task_source == "buildable":
            self._goals_cache = self._list_buildable_tasks()
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

    def _list_buildable_tasks(self) -> list[dict[str, Any]]:
        with open(self.task_source_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tasks: list[dict[str, Any]] = []
        for entry in payload.get("tasks", []):
            raw_goal = entry.get("goal", {})
            if not isinstance(raw_goal, Mapping) or not raw_goal:
                continue
            goal, quantity = next(iter(raw_goal.items()))
            synthesized_actions = [str(action) for action in entry.get("synthesized_actions", [])]
            task_id = str(entry.get("task_id") or goal)
            max_steps = self.max_steps_override or max(20, len(synthesized_actions) + 5)
            prefixed_task_id = task_id if task_id.startswith("minecraft/") else f"minecraft/{task_id}"
            tasks.append(
                {
                    **entry,
                    "dataset": self.dataset_name,
                    "task_id": prefixed_task_id,
                    "goal": str(goal),
                    "goal_condition": {str(goal): int(quantity)},
                    "type": "dependency_graph_buildable",
                    "instruction": f"Obtain {int(quantity)} {goal}.",
                    "group": entry.get("group", "buildable"),
                    "max_steps": max_steps,
                    "synthesized_actions": synthesized_actions,
                    "task_source": "buildable",
                }
            )
        return tasks

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
        raw_goal = raw.get("goal")
        if isinstance(raw_goal, Mapping) and raw_goal:
            goal, quantity = next(iter(raw_goal.items()))
            raw["goal"] = str(goal)
            raw.setdefault("goal_condition", {str(goal): int(quantity)})
        if "goal" not in raw:
            task_id = str(raw.get("task_id", "minecraft/custom"))
            raw["goal"] = task_id.split("/", 1)[-1]
        raw.setdefault("dataset", self.dataset_name)
        raw["task_id"] = str(raw.get("task_id", f"minecraft/{raw['goal']}"))
        if not raw["task_id"].startswith("minecraft/"):
            raw["task_id"] = f"minecraft/{raw['task_id']}"
        raw.setdefault("instruction", f"Obtain 1 {raw['goal']}.")
        raw.setdefault("type", "craft")
        raw.setdefault("group", "custom")
        raw.setdefault(
            "max_steps",
            self.max_steps_override or max(20, len(raw.get("synthesized_actions", [])) + 5),
        )
        return raw

    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        raw = self.load_or_wrap(raw_task)
        goal = str(raw["goal"])
        optimal_sequence = [str(action) for action in raw.get("synthesized_actions", [])]
        if not optimal_sequence:
            optimal_sequence = self._optimal_sequence(goal)
        goal_quantity = int(raw.get("goal_condition", {}).get(goal, 1))

        return TaskSpec(
            dataset=self.dataset_name,
            task_id=str(raw.get("task_id", f"minecraft/{goal}")),
            task_type="symbolic_planning",
            instruction=str(raw.get("instruction", f"Obtain 1 {goal}.")),
            initial_state={"inventory": {}, "goal": goal, "step": 0},
            goal_condition={f"inventory.{goal}": f">={goal_quantity}"},
            available_tools=self._all_actions(),
            max_steps=int(raw.get("max_steps", self.max_steps_override or 100)),
            success_criteria=[f"Inventory contains at least {goal_quantity} {goal}."],
            metadata={
                "goal_item": goal,
                "goal_quantity": goal_quantity,
                "goal_condition": {goal: goal_quantity},
                "group": raw.get("group", "custom"),
                "goal_type": raw.get("type", "craft"),
                "task_source": raw.get("task_source", self.task_source),
                "optimal_sequence": optimal_sequence,
                "required_actions": optimal_sequence,
                "grounded_fsm_mode": "branching_dependency_dag",
            },
        )

    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        self.task_spec = task_spec_to_dict(task_spec)
        goal = self.task_spec.get("metadata", {}).get("goal_item")
        self.trajectory = []
        wrapper_cls = self._get_wrapper_cls()
        self.env = wrapper_cls(max_steps=int(self.task_spec.get("max_steps", 100)))
        self.state = self.env.reset(goal)
        return self.get_observation()

    def step(self, action: str | Mapping[str, Any]) -> StepResult:
        if self.task_spec is None:
            raise RuntimeError("Call reset(task_spec) before MinecraftAdapter.step(action).")
        action_name = self._action_name(action)
        if self.env is None:
            raise RuntimeError("MC-TextWorld environment is not initialized.")
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
        goal_quantity = int(spec.get("metadata", {}).get("goal_quantity", 1))
        inventory = state.get("inventory", {})
        return (
            bool(goal and inventory.get(goal, 0) >= goal_quantity)
            or int(state.get("step", 0)) >= int(spec.get("max_steps", 100))
        )

    def get_available_tools(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        inventory = dict(state.get("inventory", {}))
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
        goal_quantity = int((self.task_spec or {}).get("metadata", {}).get("goal_quantity", 1))
        goal_achieved = bool(goal and inventory_after.get(goal, 0) >= goal_quantity)
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
        candidates = []
        for action_name, variants in self._action_library().items():
            if action_name == "no_op":
                continue
            if any(self._has_requirements(variant, inventory) for variant in variants):
                candidates.append(action_name)
        return sorted(candidates)

    def _apply_action(
        self,
        action_name: str,
        inventory: Mapping[str, int],
    ) -> tuple[bool, str, dict[str, int]]:
        variants = self._action_library().get(action_name)
        if not variants:
            return False, f"Unknown action '{action_name}'.", dict(inventory)

        for variant in variants:
            if not self._has_requirements(variant, inventory):
                continue
            updated = dict(inventory)
            for item, qty in variant.get("precondition", {}).items():
                updated[item] = updated.get(item, 0) - int(qty)
                if updated[item] <= 0:
                    updated.pop(item, None)
            for item, qty in variant.get("output", {}).items():
                updated[item] = updated.get(item, 0) + int(qty)
            return True, "Action executed in fallback simulator.", updated

        return False, self._missing_requirements_message(action_name, variants, inventory), dict(inventory)

    @staticmethod
    def _has_requirements(
        variant: Mapping[str, Any],
        inventory: Mapping[str, int],
    ) -> bool:
        for item, qty in variant.get("precondition", {}).items():
            if int(inventory.get(item, 0)) < int(qty):
                return False
        for item, qty in variant.get("tool", {}).items():
            if int(inventory.get(item, 0)) < int(qty):
                return False
        return True

    @staticmethod
    def _missing_requirements_message(
        action_name: str,
        variants: list[Mapping[str, Any]],
        inventory: Mapping[str, int],
    ) -> str:
        if not variants:
            return f"Unknown action '{action_name}'."
        variant = variants[0]
        missing = []
        for source in ("tool", "precondition"):
            for item, qty in variant.get(source, {}).items():
                have = int(inventory.get(item, 0))
                need = int(qty)
                if have < need:
                    missing.append(f"{item} need {need}, have {have}")
        if not missing:
            return f"Action '{action_name}' failed."
        return f"Failed {action_name}: " + "; ".join(missing)

    def _action_library(self) -> dict[str, list[dict[str, Any]]]:
        summary_path = os.path.join(self.root, "config", "action_lib_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
