"""Robotouille benchmark adapter for the NS-FSM pipeline.

This adapter consumes the human-verified/stateful Robotouille ground-truth FSM
JSON, runs the official Robotouille simulator, and exposes current executable
actions as canonical NS-FSM action names.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import math
import os
import re
import sys
import types
from typing import Any, Mapping

try:
    from .base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict
except ImportError:  # pragma: no cover - supports direct src execution
    from base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict


def canonical_action_name(action: Mapping[str, Any]) -> str:
    """Encode one template action into a stable action string."""

    parts = [str(action.get("template", "")).strip()]
    for key in (
        "item",
        "target",
        "container",
        "meal",
        "source_container",
        "target_container",
    ):
        value = str(action.get(key, "")).strip()
        if value:
            parts.append(f"{key}={value}")
    table_mode = str(action.get("table_mode", "")).strip()
    if table_mode:
        parts.append(f"table_mode={table_mode}")
    return "|".join(parts)


class RobotouilleAdapter(DatasetAdapter):
    """State-aware adapter that runs Robotouille tasks under NS-FSM control."""

    dataset_name = "robotouille"

    def __init__(
        self,
        ground_truth_path: str | None = None,
        split: str = "asynchronous",
        robotouille_root: str | None = None,
        seed: int | None = None,
    ):
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.project_root = os.path.dirname(self.root)
        self.split = split
        self.robotouille_root = robotouille_root or os.path.join(self.project_root, "Robotouille")
        default_name = f"robotouille_ground_truth_{split}_fsm.json"
        self.ground_truth_path = ground_truth_path or os.path.join(
            self.root,
            "config",
            default_name,
        )
        self.seed_override = seed
        self._tasks_cache: list[dict[str, Any]] | None = None
        self._task_by_id: dict[str, dict[str, Any]] = {}
        self.task_spec: dict[str, Any] | None = None
        self.raw_task: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.current_fsm_state: str | None = None
        self.completed_root_clusters: set[str] = set()
        self.branch_exit_target: str | None = None
        self.current_valid_actions: list[tuple[Any, dict[str, Any]]] = []
        self.current_valid_action_strings: list[str] = []
        self._current_matches: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._last_match_reason: dict[str, str] = {}
        self._alias_map: dict[str, str] = {}
        self.serving_table_target: str | None = None
        self.env = None

    def list_tasks(self) -> list[dict[str, Any]]:
        payload = self._load_tasks_payload()
        tasks = [deepcopy(task) for task in payload.get("tasks", [])]
        self._task_by_id = {str(task["task_id"]): deepcopy(task) for task in tasks}
        return tasks

    def load_or_wrap(self, raw_input: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_input, str):
            task_id = raw_input.strip()
            if not self._task_by_id:
                self.list_tasks()
            if task_id in self._task_by_id:
                return deepcopy(self._task_by_id[task_id])
            raise ValueError(f"Unknown Robotouille task_id: {task_id}")
        raw = dict(raw_input)
        task_id = str(raw.get("task_id", "")).strip()
        if task_id and not task_id.startswith("robotouille/"):
            raw["task_id"] = f"robotouille/{task_id}"
        raw.setdefault("dataset", self.dataset_name)
        return raw

    def to_task_spec(self, raw_task: Mapping[str, Any]) -> TaskSpec:
        raw = self.load_or_wrap(raw_task)
        compiled = self._compile_ground_truth(raw)
        optimal_steps = raw.get("optimal_steps")
        max_steps = int(math.ceil(float(optimal_steps or 40) * 1.5))
        if raw.get("fsm_generation_mode") == "branching":
            max_steps = max(
                max_steps,
                int(math.ceil(float(optimal_steps or 40) * 2.5)),
                len(compiled.get("state_order", [])) + 10,
            )
        available_tools = list(compiled["actions"])
        task_seed = (
            int(self.seed_override)
            if self.seed_override is not None
            else int((raw.get("testing_seeds") or [42])[0])
        )
        metadata = {
            "group": raw.get("group", self.split),
            "environment_name": raw.get("environment_name"),
            "testing_seeds": raw.get("testing_seeds", []),
            "grounded_fsm_mode": "robotouille_human_verified_task_graph",
            "ground_truth_task": deepcopy(raw),
            "compiled_fsm_design": deepcopy(compiled["design"]),
            "compiled_state_map": deepcopy(compiled["state_map"]),
            "compiled_state_order": list(compiled["state_order"]),
            "root_cluster_starts": list(compiled["root_cluster_starts"]),
            "root_cluster_end_states": dict(compiled["root_cluster_end_states"]),
            "state_to_root_cluster": dict(compiled["state_to_root_cluster"]),
            "async_branch_map": deepcopy(compiled["async_branch_map"]),
            "task_seed": task_seed,
            "required_actions": list(available_tools),
        }
        return TaskSpec(
            dataset=self.dataset_name,
            task_id=str(raw["task_id"]),
            task_type="symbolic_planning",
            instruction=str(raw.get("goal_description", "")),
            initial_state={
                "environment_name": raw.get("environment_name"),
                "split": raw.get("group", self.split),
                "task_seed": task_seed,
                "fsm_state": compiled["design"]["initial_state"],
                "step_count": 0,
            },
            goal_condition={"goal_satisfied": True},
            available_tools=available_tools,
            max_steps=max_steps,
            success_criteria=[str(raw.get("goal_description", "Complete the Robotouille task."))],
            metadata=metadata,
        )

    def reset(self, task_spec: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
        self.task_spec = task_spec_to_dict(task_spec)
        self.raw_task = deepcopy(self.task_spec.get("metadata", {}).get("ground_truth_task", {}))
        self.history = []
        self.completed_root_clusters = set()
        self.current_fsm_state = str(self.task_spec.get("initial_state", {}).get("fsm_state", ""))
        self.active_async_branch = None
        self.branch_entry_state = None
        self.branch_exit_target = None
        self._current_matches = {}
        self._last_match_reason = {}
        self.serving_table_target = self._goal_serving_table_target()
        environment_name = str(self.task_spec.get("metadata", {}).get("environment_name", ""))
        seed = self.task_spec.get("metadata", {}).get("task_seed")
        env = self._create_robotouille_env(environment_name, seed=seed)
        obs, _info = env.reset()
        self.env = env
        self._alias_map = self._build_runtime_alias_map()
        self.state = {
            "observation": obs,
            "environment_name": environment_name,
            "step_count": 0,
            "goal_satisfied": bool(env.current_state.is_goal_reached()),
            "fsm_state": self.current_fsm_state,
        }
        self._refresh_valid_actions()
        return self.get_observation()

    def step(self, action: str | Mapping[str, Any]) -> StepResult:
        if self.task_spec is None:
            raise RuntimeError("Call reset(task_spec) before RobotouilleAdapter.step(action).")
        action_name = self._action_name(action)
        info: dict[str, Any] = {"success": True, "message": ""}
        done = False

        if action_name.startswith("advance_pending_subgoal::"):
            target_state = action_name.split("::", 1)[1]
            self.current_fsm_state = target_state
            info["message"] = f"Switched to pending subgoal: {target_state}"
            info["fsm_next_state"] = target_state
            self.state["fsm_state"] = self.current_fsm_state
            return StepResult(self.get_observation(), False, info)

        if action_name.startswith("skip_completed::"):
            target_state = str(self._skip_target(action_name.split("::", 1)[1]) or "")
            if not target_state:
                info["success"] = False
                info["message"] = f"No skip target is available for {action_name}."
                return StepResult(self.get_observation(), False, info)
            self.current_fsm_state = target_state
            info["message"] = f"Skipped completed subgoal: {action_name}"
            info["fsm_next_state"] = target_state
            self.state["fsm_state"] = self.current_fsm_state
            return StepResult(self.get_observation(), False, info)

        if action_name == "verify_goal_complete":
            goal_satisfied = bool(self.env and self.env.current_state.is_goal_reached())
            self.state["goal_satisfied"] = goal_satisfied
            info["success"] = goal_satisfied
            info["message"] = "Goal verified." if goal_satisfied else "Goal predicates not satisfied yet."
            if goal_satisfied:
                info["fsm_next_state"] = "DONE"
                done = True
            return StepResult(self.get_observation(), done, info)

        match = self._current_matches.get(action_name)
        if match is None:
            self._refresh_valid_actions()
            match = self._current_matches.get(action_name)
        if match is None:
            info["success"] = False
            info["message"] = self._last_match_reason.get(
                action_name,
                f"Action '{action_name}' is not currently executable in Robotouille.",
            )
            return StepResult(self.get_observation(), False, info)

        obs, _reward, done, _env_info = self.env.step([match])
        self.state["observation"] = obs
        self.state["step_count"] = int(self.state.get("step_count", 0)) + 1
        self.state["goal_satisfied"] = bool(self.env.current_state.is_goal_reached())
        next_state = self._post_action_fsm_next_state(action_name)
        if next_state:
            info["fsm_next_state"] = next_state
            self.current_fsm_state = next_state
        self.state["fsm_state"] = self.current_fsm_state
        self._refresh_valid_actions()
        return StepResult(self.get_observation(), done or self.state["goal_satisfied"], info)

    def get_observation(self) -> dict[str, Any]:
        observation = deepcopy(self.state)
        if self._alias_map:
            observation["alias_map"] = dict(self._alias_map)
        return observation

    def is_done(
        self,
        state: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any],
    ) -> bool:
        spec = task_spec_to_dict(task_spec)
        return bool(state.get("goal_satisfied")) or int(state.get("step_count", 0)) >= int(
            spec.get("max_steps", 100)
        )

    def get_available_tools(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> list[str]:
        spec = task_spec_to_dict(task_spec)
        self.task_spec = spec
        self.current_fsm_state = str(self.current_fsm_state or state.get("fsm_state") or "")
        if self.env is None:
            return list(spec.get("available_tools", []))

        self._refresh_valid_actions()
        state_map = spec.get("metadata", {}).get("compiled_state_map", {})
        current = state_map.get(self.current_fsm_state, {})
        if not current:
            return []

        if current.get("kind") == "terminal":
            return []
        if current.get("kind") == "verify":
            return ["verify_goal_complete"]

        tools: list[str] = []
        skip_action = f"skip_completed::{self.current_fsm_state}"
        has_skip_transition = any(
            option.get("action") == skip_action
            for option in self.task_spec.get("metadata", {})
            .get("compiled_fsm_design", {})
            .get("transitions_by_state", {})
            .get(self.current_fsm_state, [])
        )
        if has_skip_transition and (
            self.current_fsm_state in self.completed_root_clusters
            or self._state_completion_satisfied(current)
        ):
            self.completed_root_clusters.add(self.current_fsm_state)
            return [skip_action]

        if current.get("kind") == "async-branch":
            if self._async_branch_done(current):
                exit_target = current.get("branch_exit_target")
                if exit_target:
                    self.branch_exit_target = exit_target
                    tools.append("wait")
            else:
                for target_state in current.get("parallel_subgoals", []):
                    if self._subgoal_available_from_branch(target_state):
                        tools.append(f"advance_pending_subgoal::{target_state}")
                if self._current_matches.get("wait") is not None:
                    tools.append("wait")
            return tools

        transitions = self._state_transition_specs(current)
        for transition in transitions:
            action = transition.get("action", {})
            if not isinstance(action, Mapping):
                continue
            if not self._guard_satisfied(str(transition.get("guard", "")).strip()):
                continue
            canonical = canonical_action_name(action)
            if canonical in self._current_matches:
                tools.append(canonical)

        if transitions:
            return list(dict.fromkeys(tools))

        for action in current.get("fsm_allowed_actions", []):
            canonical = canonical_action_name(action)
            if canonical in self._current_matches:
                tools.append(canonical)
        return list(dict.fromkeys(tools))

    def normalize_action(
        self,
        raw_action: str | Mapping[str, Any],
        legal_actions: list[str] | list[dict[str, Any]],
    ) -> str | None:
        action_name = self._clean_model_text(self._action_name(raw_action))
        legal_names = [
            str(item.get("action", "")) if isinstance(item, Mapping) else str(item)
            for item in legal_actions
        ]
        return self._match_model_name(
            action_name,
            legal_names,
            include_runtime_action_descriptions=True,
            include_action_variants=True,
        )

    def normalize_state_name(self, raw_state: str | None) -> str | None:
        state_name = self._clean_model_text(raw_state or "")
        state_names = list((self.task_spec or {}).get("metadata", {}).get("compiled_state_map", {}))
        return self._match_model_name(
            state_name,
            state_names,
            include_runtime_action_descriptions=False,
            include_action_variants=False,
        )

    def format_state_for_prompt(self, state: Mapping[str, Any]) -> str:
        valid = ", ".join(self.current_valid_action_strings[:10]) or "(none)"
        return "\n".join(
            [
                f"environment: {state.get('environment_name')}",
                f"fsm_state: {state.get('fsm_state')}",
                f"goal_satisfied: {bool(state.get('goal_satisfied'))}",
                f"step_count: {state.get('step_count')}",
                f"observation: {state.get('observation', '')}",
                f"valid_actions: {valid}",
            ]
        )

    def summarize_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "success": bool(state.get("goal_satisfied")),
            "total_steps": int(state.get("step_count", 0)),
            "termination": (
                "success"
                if state.get("goal_satisfied")
                else "max_steps_or_incomplete"
            ),
            "environment_name": state.get("environment_name"),
        }

    def set_runtime_fsm_state(self, state_name: str) -> None:
        self.current_fsm_state = str(state_name)
        if self.state:
            self.state["fsm_state"] = self.current_fsm_state

    def build_grounded_fsm_design(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
    ) -> dict[str, Any]:
        spec = task_spec_to_dict(task_spec)
        return deepcopy(spec.get("metadata", {}).get("compiled_fsm_design", {}))

    def _load_tasks_payload(self) -> dict[str, Any]:
        if self._tasks_cache is not None:
            return {"tasks": deepcopy(self._tasks_cache)}
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tasks = [dict(task) for task in payload.get("tasks", [])]
        self._tasks_cache = tasks
        return {"tasks": deepcopy(tasks)}

    def _compile_ground_truth(self, raw_task: Mapping[str, Any]) -> dict[str, Any]:
        states = [dict(state) for state in raw_task.get("state_list", [])]
        state_order = [str(state["name"]) for state in states]
        state_map = {str(state["name"]): dict(state) for state in states}
        explicit_roots = [str(state["name"]) for state in states if state.get("root_cluster")]
        root_cluster_starts = explicit_roots or [
            name
            for name in state_order
            if name.startswith(
                (
                    "NAVIGATE_TO_",
                    "NAVIGATE_TO_POT_FOR_SOUP_BASE",
                    "NAVIGATE_TO_POT_FOR_BOWL_FILL",
                    "VERIFY_GOAL",
                )
            )
        ]

        root_cluster_end_states: dict[str, str] = {}
        state_to_root_cluster: dict[str, str] = {}
        root_cluster_set = set(root_cluster_starts)
        for start in root_cluster_starts:
            members, ends = self._cluster_members_and_ends(start, state_map, root_cluster_set)
            for member in members:
                state_to_root_cluster[member] = start
            root_cluster_end_states[start] = sorted(ends or {start})[0]

        async_branch_map: dict[str, dict[str, Any]] = {}
        for state in states:
            name = str(state["name"])
            if state.get("kind") != "async-branch":
                continue
            branch_item = name.rsplit("_", 2)[0]
            exit_target = self._default_branch_exit_target(branch_item, state_map)
            async_branch_map[name] = {
                "branch_item": branch_item,
                "completion_predicate": self._async_completion_predicate(name),
                "exit_target": exit_target,
                "parallel_subgoals": list(state.get("parallel_subgoals", [])),
            }

        for branch_name, branch_info in async_branch_map.items():
            for start_state in branch_info.get("parallel_subgoals", []):
                end_state = root_cluster_end_states.get(start_state)
                if end_state and end_state in state_map:
                    state_map[end_state]["return_branch_state"] = branch_name

        actions: set[str] = set()
        transitions_by_state: dict[str, list[dict[str, str]]] = {}
        next_root_by_root: dict[str, str] = {}
        for idx, root in enumerate(root_cluster_starts):
            if idx + 1 < len(root_cluster_starts):
                next_root_by_root[root] = root_cluster_starts[idx + 1]
        for state in states:
            name = str(state["name"])
            state = state_map.get(name, state)
            kind = str(state.get("kind", ""))
            if kind == "terminal":
                transitions_by_state[name] = []
                continue
            if kind == "verify":
                transitions_by_state[name] = [
                    {
                        "action": "verify_goal_complete",
                        "next_state": "DONE",
                        "condition": "All official goal predicates are satisfied.",
                    }
                ]
                actions.add("verify_goal_complete")
                continue

            options: list[dict[str, str]] = []
            if kind == "async-branch":
                branch_info = async_branch_map.get(name, {})
                options.append(
                    {
                        "action": "wait",
                        "next_state": branch_info.get("exit_target") or name,
                        "condition": "Stay in branch or auto-exit if the async process finished.",
                    }
                )
                actions.add("wait")
                for target_state in branch_info.get("parallel_subgoals", []):
                    action_name = f"advance_pending_subgoal::{target_state}"
                    actions.add(action_name)
                    options.append(
                        {
                            "action": action_name,
                            "next_state": target_state,
                            "condition": "Switch to one pending safe subgoal.",
                        }
                    )
                transitions_by_state[name] = options
                continue

            explicit_transitions = self._state_transition_specs(state)
            if explicit_transitions:
                for transition in explicit_transitions:
                    action = transition.get("action", {})
                    if not isinstance(action, Mapping):
                        continue
                    action_name = canonical_action_name(action)
                    next_state = str(transition.get("next_state", "")).strip()
                    if not next_state:
                        next_state = str(state.get("next_state_on_completion", "")).strip()
                    actions.add(action_name)
                    options.append(
                        {
                            "action": action_name,
                            "next_state": next_state,
                            "condition": str(
                                transition.get("condition")
                                or transition.get("guard")
                                or "Ground-truth guarded transition."
                            ),
                        }
                    )
                if name in root_cluster_starts and name in next_root_by_root:
                    skip_action = f"skip_completed::{name}"
                    actions.add(skip_action)
                    options.append(
                        {
                            "action": skip_action,
                            "next_state": next_root_by_root[name],
                            "condition": "Skip a root subgoal that was completed while inside an async branch.",
                        }
                    )
                transitions_by_state[name] = options
                continue

            next_state = str(state.get("next_state_on_completion", "")).strip()
            if not next_state:
                root_start = state_to_root_cluster.get(name)
                return_branch = str(state.get("return_branch_state", "")).strip()
                if return_branch:
                    next_state = return_branch
                elif root_start:
                    next_state = next_root_by_root.get(root_start, "DONE")
                else:
                    next_state = "DONE"
            for action in state.get("fsm_allowed_actions", []):
                action_name = canonical_action_name(action)
                actions.add(action_name)
                options.append(
                    {
                        "action": action_name,
                        "next_state": next_state,
                        "condition": "Ground-truth state-local action template.",
                    }
                )
            if name in root_cluster_starts and name in next_root_by_root:
                skip_action = f"skip_completed::{name}"
                actions.add(skip_action)
                options.append(
                    {
                        "action": skip_action,
                        "next_state": next_root_by_root[name],
                        "condition": "Skip a root subgoal that was completed while inside an async branch.",
                    }
                )
            transitions_by_state[name] = options

        for name, branch_info in async_branch_map.items():
            state_map[name]["branch_item"] = branch_info.get("branch_item")
            state_map[name]["branch_completion_predicate"] = branch_info.get(
                "completion_predicate"
            )
            state_map[name]["branch_exit_target"] = branch_info.get("exit_target")

        design = {
            "states": state_order,
            "initial_state": state_order[0],
            "terminal_states": ["DONE"],
            "transitions_by_state": transitions_by_state,
            "fallback_policy": {
                "on_invalid_action": "retry_with_valid_action",
                "on_dead_end": "abort_episode",
            },
            "success_signals": [
                "All Robotouille goal predicates are satisfied in the simulator.",
            ],
            "risk_notes": [
                "Human-verified/stateful Robotouille ground truth compiled into RuntimeFSM form.",
                "Async branch states use runtime executable filtering plus branch return logic.",
            ],
        }
        return {
            "design": design,
            "actions": sorted(actions),
            "state_map": state_map,
            "state_order": state_order,
            "root_cluster_starts": root_cluster_starts,
            "root_cluster_end_states": root_cluster_end_states,
            "state_to_root_cluster": state_to_root_cluster,
            "async_branch_map": async_branch_map,
        }

    def _default_branch_exit_target(
        self,
        branch_item: str,
        state_map: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        candidate = f"NAVIGATE_TO_{branch_item}_FOR_SERVE"
        if candidate in state_map:
            return candidate
        return None

    @staticmethod
    def _async_completion_predicate(state_name: str) -> str:
        if state_name.endswith("_FRYING_BRANCH"):
            return "isfried"
        if state_name.endswith("_COOKING_BRANCH"):
            return "iscooked"
        if state_name == "WATER_BOILING_BRANCH":
            return "isboiling"
        return "iscooked"

    def _create_robotouille_env(self, environment_name: str, seed: int | None = None):
        robotouille_root = self.robotouille_root
        if not os.path.isdir(robotouille_root):
            raise RuntimeError(
                f"Robotouille root does not exist: {robotouille_root}. "
                "Set --robotouille-root or place the Robotouille repo next to ns-fsm-baseline."
            )
        if robotouille_root not in sys.path:
            sys.path.insert(0, robotouille_root)
        try:
            with self._robotouille_cwd():
                create_robotouille_env = self._load_create_robotouille_env()

                return create_robotouille_env(environment_name, seed=seed, noisy_randomization=False)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Robotouille dependencies are missing ({exc}). Install Robotouille first, "
                "for example: pip install gym==0.26.2 hydra-core omegaconf pygame && "
                "pip install -e Robotouille"
            ) from exc

    def _load_create_robotouille_env(self):
        """Load Robotouille env code without importing the optional LLM agents.

        The upstream package ``robotouille.__init__`` imports simulator/agent
        modules at import time. Those pull in optional LLM dependencies
        (torch/transformers/google/anthropic) that are not needed for the
        simulator environment used by this adapter.
        """

        self._install_robotouille_runtime_shims()

        package_dir = os.path.join(self.robotouille_root, "robotouille")
        package = sys.modules.get("robotouille")
        package_paths = list(getattr(package, "__path__", []) or [])
        if package is None or package_dir not in package_paths:
            package = types.ModuleType("robotouille")
            package.__file__ = os.path.join(package_dir, "__init__.py")
            package.__path__ = [package_dir]
            package.__package__ = "robotouille"
            sys.modules["robotouille"] = package

        module = importlib.import_module("robotouille.robotouille_env")
        return module.create_robotouille_env

    def _install_robotouille_runtime_shims(self) -> None:
        """Install minimal gym/pddlgym shims for headless simulator execution."""

        self._install_lightweight_gym()
        self._install_lightweight_pddlgym()

    @staticmethod
    def _install_lightweight_gym() -> None:
        if "gym" in sys.modules:
            return

        registry: dict[str, dict[str, Any]] = {}
        gym_module = types.ModuleType("gym")
        spaces_module = types.ModuleType("gym.spaces")
        envs_module = types.ModuleType("gym.envs")
        registration_module = types.ModuleType("gym.envs.registration")

        class Env:
            metadata: dict[str, Any] = {}

            def reset(self, *args, **kwargs):  # pragma: no cover - interface shim
                raise NotImplementedError

            def step(self, action):  # pragma: no cover - interface shim
                raise NotImplementedError

        class Wrapper(Env):
            def __init__(self, env):
                self.env = env
                self.action_space = getattr(env, "action_space", None)
                self.observation_space = getattr(env, "observation_space", None)

            def __getattr__(self, name):
                return getattr(self.env, name)

        class Space:
            def __init__(self, *args, seed=None, **kwargs):
                self.shape = kwargs.get("shape")
                self.dtype = kwargs.get("dtype")
                try:
                    import numpy as np

                    self.np_random = np.random.default_rng(seed)
                except Exception:
                    self.np_random = None

            def sample(self):  # pragma: no cover - interface shim
                raise NotImplementedError

        class Text(Space):
            def __init__(self, min_length=0, max_length=None, charset=None, **kwargs):
                super().__init__(**kwargs)
                self.min_length = min_length
                self.max_length = max_length
                self.charset = charset

        def register(id: str, entry_point: str, kwargs: Mapping[str, Any] | None = None, **_):
            registry[id] = {"entry_point": entry_point, "kwargs": dict(kwargs or {})}

        def make(id: str, **kwargs):
            if id not in registry:
                raise KeyError(f"Unknown gym environment id: {id}")
            spec = registry[id]
            module_name, attr_name = str(spec["entry_point"]).split(":", 1)
            module = importlib.import_module(module_name)
            constructor = getattr(module, attr_name)
            init_kwargs = dict(spec["kwargs"])
            init_kwargs.update(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"disable_env_checker", "render_mode"}
                }
            )
            return constructor(**init_kwargs)

        spaces_module.Space = Space
        spaces_module.Text = Text
        registration_module.register = register
        envs_module.registration = registration_module
        gym_module.Env = Env
        gym_module.Wrapper = Wrapper
        gym_module.Space = Space
        gym_module.spaces = spaces_module
        gym_module.envs = envs_module
        gym_module.make = make
        gym_module.register = register

        sys.modules["gym"] = gym_module
        sys.modules["gym.spaces"] = spaces_module
        sys.modules["gym.envs"] = envs_module
        sys.modules["gym.envs.registration"] = registration_module

    def _install_lightweight_pddlgym(self) -> None:
        pddlgym_module = sys.modules.get("pddlgym")
        if pddlgym_module is not None and hasattr(pddlgym_module, "register_pddl_env"):
            return

        package_dir = os.path.join(self.robotouille_root, "pddlgym")
        pddlgym_module = types.ModuleType("pddlgym")
        pddlgym_module.__file__ = os.path.join(package_dir, "__init__.py")
        pddlgym_module.__path__ = [package_dir]
        pddlgym_module.__package__ = "pddlgym"
        sys.modules["pddlgym"] = pddlgym_module

        def make(*args, **kwargs):
            gym = sys.modules["gym"]
            return gym.make(*args, disable_env_checker=True, **kwargs)

        def register_pddl_env(name: str, is_test_env: bool, other_args: Mapping[str, Any]):
            dir_path = os.path.join(os.path.dirname(os.path.realpath(pddlgym_module.__file__)), "pddl")
            domain_file = os.path.join(dir_path, f"{name.lower()}.pddl")
            gym_name = name.capitalize()
            problem_dirname = name.lower()
            if is_test_env:
                gym_name += "Test"
                problem_dirname += "_test"
            problem_dir = os.path.join(dir_path, problem_dirname)
            from gym.envs.registration import register

            register(
                id=f"PDDLEnv{gym_name}-v0",
                entry_point="pddlgym.core:PDDLEnv",
                kwargs={
                    "domain_file": domain_file,
                    "problem_dir": problem_dir,
                    **dict(other_args),
                },
            )

        pddlgym_module.make = make
        pddlgym_module.register_pddl_env = register_pddl_env
        pddlgym_module.structs = importlib.import_module("pddlgym.structs")

    @contextmanager
    def _robotouille_cwd(self):
        current = os.getcwd()
        os.chdir(self.robotouille_root)
        try:
            yield
        finally:
            os.chdir(current)

    def _refresh_valid_actions(self) -> None:
        if self.env is None:
            self.current_valid_actions = []
            self.current_valid_action_strings = []
            self._current_matches = {}
            return
        valid_actions, str_valid_actions = self.env.current_state.get_valid_actions_and_str()
        self.current_valid_actions = list(valid_actions)
        self.current_valid_action_strings = list(str_valid_actions)
        self._current_matches = {}
        self._last_match_reason = {}

        state_map = (self.task_spec or {}).get("metadata", {}).get("compiled_state_map", {})
        current = state_map.get(self.current_fsm_state or "", {})
        for transition in self._state_transition_specs(current):
            action_spec = transition.get("action", {})
            if not isinstance(action_spec, Mapping):
                continue
            canonical = canonical_action_name(action_spec)
            match = self._match_template_action(action_spec)
            if match is not None:
                self._current_matches[canonical] = match

        for action_spec in current.get("fsm_allowed_actions", []):
            canonical = canonical_action_name(action_spec)
            match = self._match_template_action(action_spec)
            if match is not None:
                self._current_matches[canonical] = match

        if current.get("kind") == "async-branch":
            wait_spec = {"template": "wait"}
            wait_match = self._match_template_action(wait_spec)
            if wait_match is not None:
                self._current_matches["wait"] = wait_match

    def _match_template_action(
        self,
        action_spec: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]] | None:
        template = str(action_spec.get("template", "")).strip()
        desired_item = self._alias_to_runtime_name(str(action_spec.get("item", "")).strip())
        desired_container = str(action_spec.get("container", "")).strip()
        desired_target = str(action_spec.get("target", "")).strip()
        desired_source_container = str(action_spec.get("source_container", "")).strip()
        desired_target_container = str(action_spec.get("target_container", "")).strip()

        for (action, param_arg_dict), _desc in zip(
            self.current_valid_actions,
            self.current_valid_action_strings,
        ):
            values = {key: getattr(obj, "name", "") for key, obj in param_arg_dict.items()}
            object_types = {key: getattr(obj, "object_type", "") for key, obj in param_arg_dict.items()}
            if not self._template_matches_runtime_action(
                template,
                str(getattr(action, "name", "")),
                values,
                desired_item,
                desired_target,
            ):
                continue

            if desired_item:
                main_item = values.get("i1") or values.get("c1") or values.get("m1")
                if main_item and main_item != desired_item:
                    continue
                if not main_item and desired_item not in values.values():
                    continue
            if desired_container and not any(
                name.startswith(desired_container) for name in values.values()
            ):
                continue
            if desired_source_container and not any(
                name.startswith(desired_source_container) for name in values.values()
            ):
                continue
            if desired_target_container and not any(
                name.startswith(desired_target_container) for name in values.values()
            ):
                continue
            desired_table_mode = str(action_spec.get("table_mode", "")).strip()
            if desired_target and not self._target_matches(
                desired_target,
                values,
                object_types,
                desired_item,
                desired_table_mode,
            ):
                continue
            return action, param_arg_dict
        return None

    @staticmethod
    def _template_matches_runtime_action(
        desired_template: str,
        runtime_template: str,
        values: Mapping[str, str],
        desired_item: str,
        desired_target: str = "",
    ) -> bool:
        if runtime_template == desired_template:
            return True
        if desired_template == "place-item" and runtime_template == "stack":
            if desired_target and desired_target not in {
                "table",
                "serving table while holding item",
                "serving table while holding bowl",
            }:
                return False
            return not desired_item or desired_item in values.values()
        if desired_template == "pick-up-item" and runtime_template == "unstack":
            return not desired_item or desired_item in values.values()
        return False

    def _target_matches(
        self,
        desired_target: str,
        values: Mapping[str, str],
        object_types: Mapping[str, str],
        desired_item: str,
        desired_table_mode: str = "",
    ) -> bool:
        station_names = [name for key, name in values.items() if object_types.get(key) == "station"]
        destination = values.get("s2") or values.get("s1") or ""
        candidate_stations = [destination] if destination else station_names
        if "while holding item" in desired_target and not self._robot_holds_item(desired_item):
            return False
        if "while holding pot" in desired_target and not self._robot_holds_container("pot"):
            return False
        if desired_target == "stove":
            return any(name.startswith("stove") for name in candidate_stations)
        if desired_target == "fryer":
            return any(name.startswith("fryer") for name in candidate_stations)
        if desired_target == "board":
            return any(name.startswith("board") for name in candidate_stations)
        if desired_target == "sink":
            return any(name.startswith("sink") for name in candidate_stations)
        if desired_target == "table":
            if desired_table_mode and self.serving_table_target:
                return any(self._matches_serving_table(name) for name in candidate_stations)
            return any(name.startswith("table") for name in candidate_stations)
        if desired_target == "occupied stove":
            return any(
                name.startswith("stove") and not self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "occupied fryer":
            return any(
                name.startswith("fryer") and not self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "empty stove":
            return any(
                name.startswith("stove") and self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "empty fryer":
            return any(
                name.startswith("fryer") and self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "empty table while holding item":
            return any(
                name.startswith("table") and self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target.startswith("station containing "):
            target_item = self._alias_to_runtime_name(desired_target.split("station containing ", 1)[1])
            return any(
                self._predicate_true("item_at", target_item, station)
                or self._predicate_true("container_at", target_item, station)
                for station in candidate_stations
            )
        if desired_target == "stove while holding item":
            return any(name.startswith("stove") for name in candidate_stations)
        if desired_target == "empty stove while holding item":
            return any(
                name.startswith("stove") and self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "empty fryer while holding item":
            return any(
                name.startswith("fryer") and self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "fryer while holding item":
            return any(name.startswith("fryer") for name in candidate_stations)
        if desired_target == "cutting board while holding item":
            return any(name.startswith("board") for name in candidate_stations)
        if desired_target == "sink while holding pot":
            return any(name.startswith("sink") for name in candidate_stations)
        if desired_target == "stove while holding water-filled pot":
            return any(name.startswith("stove") for name in candidate_stations)
        if desired_target == "empty stove while holding water-filled pot":
            return any(
                name.startswith("stove") and self._station_empty(name)
                for name in candidate_stations
            )
        if desired_target == "pot on stove while holding ingredient":
            return any(name.startswith("stove") for name in candidate_stations)
        if desired_target == "bowl while holding pot":
            return any(name.startswith("table") or name.startswith("board") or name.startswith("sink") for name in candidate_stations)
        if desired_target == "serving table while holding bowl":
            return any(self._matches_serving_table(name) for name in candidate_stations)
        if desired_target == "serving table while holding item":
            return any(self._matches_serving_table(name) for name in candidate_stations)
        if desired_target == "pot location":
            return True
        return True

    def _match_model_name(
        self,
        raw_name: str,
        legal_names: list[str],
        *,
        include_runtime_action_descriptions: bool,
        include_action_variants: bool,
    ) -> str | None:
        names = [str(name) for name in legal_names if str(name)]
        if not raw_name or not names:
            return None

        lowered = {name.lower(): name for name in names}
        candidates = [raw_name]

        if include_runtime_action_descriptions:
            runtime_aliases = self._runtime_action_description_aliases()
            for candidate in list(candidates):
                if candidate in runtime_aliases:
                    return runtime_aliases[candidate]
                lowered_candidate = candidate.lower()
                if lowered_candidate in runtime_aliases:
                    return runtime_aliases[lowered_candidate]

        alias_replaced = self._replace_runtime_aliases(raw_name)
        if alias_replaced not in candidates:
            candidates.append(alias_replaced)

        if include_action_variants:
            for candidate in list(candidates):
                candidates.extend(self._semantic_action_variants(candidate))

        for candidate in dict.fromkeys(self._clean_model_text(item) for item in candidates):
            if candidate in names:
                return candidate
            match = lowered.get(candidate.lower())
            if match:
                return match
        return None

    @staticmethod
    def _clean_model_text(text: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.lower().startswith("action:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        return cleaned.strip("`").strip().strip('"').strip("'").rstrip(".").strip()

    def _replace_runtime_aliases(self, text: str) -> str:
        replaced = str(text)
        reverse_aliases = {
            runtime_name: alias
            for alias, runtime_name in self._alias_map.items()
            if alias and runtime_name
        }
        for runtime_name, alias in sorted(
            reverse_aliases.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            replaced = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(runtime_name)}(?![A-Za-z0-9])",
                alias,
                replaced,
            )
        return replaced

    def _runtime_action_description_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for canonical, match in self._current_matches.items():
            for idx, runtime_match in enumerate(self.current_valid_actions):
                if runtime_match != match or idx >= len(self.current_valid_action_strings):
                    continue
                description = self._clean_model_text(self.current_valid_action_strings[idx])
                if description:
                    aliases[description] = canonical
                    aliases[description.lower()] = canonical
        return aliases

    def _semantic_action_variants(self, action_name: str) -> list[str]:
        if "|" not in action_name:
            return []

        parts = [part.strip() for part in action_name.split("|") if part.strip()]
        if not parts:
            return []
        template = parts[0].lower()
        fields: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()

        variants: list[str] = []
        for target in self._semantic_targets(template, fields.get("target", "")):
            variant = dict(fields)
            variant["target"] = target
            variants.append(canonical_action_name({"template": template, **variant}))
        return variants

    @staticmethod
    def _semantic_targets(template: str, target: str) -> list[str]:
        lowered = str(target).lower().strip()
        if not lowered:
            return []

        targets: list[str] = []
        holding_item = "holding" in lowered
        if "board" in lowered:
            targets.extend(["cutting board while holding item", "board"] if template == "move" else ["board"])
        if "stove" in lowered:
            targets.extend(["stove while holding item", "stove"] if template == "move" or holding_item else ["stove"])
        if "fryer" in lowered:
            targets.extend(["fryer while holding item", "fryer"] if template == "move" or holding_item else ["fryer"])
        if "sink" in lowered:
            targets.extend(["sink while holding pot", "sink"] if "pot" in lowered else ["sink"])
        if "table" in lowered:
            if "bowl" in lowered:
                targets.append("serving table while holding bowl")
            if holding_item:
                targets.append("serving table while holding item")
            targets.append("table")
        return list(dict.fromkeys(targets))

    def _matches_serving_table(self, station: str) -> bool:
        if not station.startswith("table"):
            return False
        if not self.serving_table_target:
            return True
        return station == self.serving_table_target

    def _predicate_true(self, predicate_name: str, *runtime_args: str) -> bool:
        if self.env is None:
            return False
        for predicate, value in self.env.current_state.predicates.items():
            if not value or getattr(predicate, "name", "") != predicate_name:
                continue
            params = [getattr(obj, "name", "") for obj in getattr(predicate, "params", [])]
            if list(runtime_args) == params:
                return True
        return False

    def _state_completion_satisfied(self, state: Mapping[str, Any]) -> bool:
        conditions = state.get("completion_condition", [])
        if not isinstance(conditions, list) or not conditions:
            return False
        return all(self._completion_condition_true(str(condition)) for condition in conditions)

    def _completion_condition_true(self, condition: str) -> bool:
        condition = condition.strip()
        if not condition:
            return True
        if " or " in condition:
            return any(self._completion_condition_true(part) for part in condition.split(" or "))
        if condition == "All official goal predicates are satisfied.":
            return bool(self.env and self.env.current_state.is_goal_reached())
        if condition.startswith("robot is at station containing "):
            alias = condition.split("robot is at station containing ", 1)[1].strip()
            return self._robot_at_station_containing(alias)
        if condition.startswith("robot has "):
            alias = condition.split("robot has ", 1)[1].strip()
            return self._robot_holds_item(self._alias_to_runtime_name(alias))

        predicate_match = re.fullmatch(r"(is[a-z_]+)\\(([^)]+)\\)", condition)
        if predicate_match:
            predicate, alias = predicate_match.groups()
            runtime_item = "water" if alias == "WATER" else self._alias_to_runtime_name(alias)
            return self._predicate_true(predicate, runtime_item)

        location_match = re.fullmatch(r"(.+) is on (stove|fryer|table|board|sink)", condition)
        if location_match:
            alias, station_prefix = location_match.groups()
            item = self._alias_to_runtime_name(alias.strip())
            return any(
                self._predicate_true("item_on", item, station)
                or self._predicate_true("item_at", item, station)
                or self._predicate_true("container_at", item, station)
                for station in self._stations_with_prefix(station_prefix)
            )
        location_match = re.fullmatch(r"(.+) is at (stove|fryer|table|board|sink)", condition)
        if location_match:
            alias, station_prefix = location_match.groups()
            item = self._alias_to_runtime_name(alias.strip())
            return any(
                self._predicate_true("item_at", item, station)
                or self._predicate_true("container_at", item, station)
                for station in self._stations_with_prefix(station_prefix)
            )
        return False

    def _robot_holds_item(self, runtime_item: str = "") -> bool:
        if self.env is None:
            return False
        for predicate, value in self.env.current_state.predicates.items():
            if not value or getattr(predicate, "name", "") != "has_item":
                continue
            params = [getattr(obj, "name", "") for obj in getattr(predicate, "params", [])]
            if len(params) != 2 or not params[0].startswith("robot"):
                continue
            if not runtime_item or params[1] == runtime_item:
                return True
        return False

    def _robot_holds_container(self, container_prefix: str = "") -> bool:
        if self.env is None:
            return False
        for predicate, value in self.env.current_state.predicates.items():
            if not value or getattr(predicate, "name", "") != "has_container":
                continue
            params = [getattr(obj, "name", "") for obj in getattr(predicate, "params", [])]
            if len(params) != 2 or not params[0].startswith("robot"):
                continue
            if not container_prefix or params[1].startswith(container_prefix):
                return True
        return False

    def _post_action_fsm_next_state(self, action_name: str) -> str | None:
        if self.task_spec is None:
            return None
        state_map = self.task_spec.get("metadata", {}).get("compiled_state_map", {})
        state_to_root = self.task_spec.get("metadata", {}).get("state_to_root_cluster", {})
        root_end_states = self.task_spec.get("metadata", {}).get("root_cluster_end_states", {})
        current = state_map.get(self.current_fsm_state or "", {})
        if not current:
            return None

        if current.get("kind") == "async-branch":
            if self._async_branch_done(current):
                return current.get("branch_exit_target") or self.current_fsm_state
            return self.current_fsm_state

        for transition in self._state_transition_specs(current):
            action = transition.get("action", {})
            if not isinstance(action, Mapping):
                continue
            if canonical_action_name(action) != action_name:
                continue
            if not self._guard_satisfied(str(transition.get("guard", "")).strip()):
                continue
            target = str(transition.get("next_state", "")).strip()
            if target:
                return target

        target = str(current.get("next_state_on_completion", "")).strip()
        if target:
            return target

        root_start = state_to_root.get(self.current_fsm_state or "")
        if root_start and root_end_states.get(root_start) == self.current_fsm_state:
            self.completed_root_clusters.add(root_start)

        return_branch = str(current.get("return_branch_state", "")).strip()
        if return_branch:
            return return_branch
        return None

    def _subgoal_available_from_branch(self, target_state: str) -> bool:
        if target_state in self.completed_root_clusters:
            return False
        if self.task_spec is None:
            return False
        state_map = self.task_spec.get("metadata", {}).get("compiled_state_map", {})
        target = state_map.get(target_state, {})
        transitions = self._state_transition_specs(target)
        actions = [
            transition.get("action", {})
            for transition in transitions
            if isinstance(transition.get("action"), Mapping)
            and self._guard_satisfied(str(transition.get("guard", "")).strip())
        ]
        if not actions:
            actions = target.get("fsm_allowed_actions", [])
        if not actions:
            return False
        return any(self._match_template_action(action) is not None for action in actions)

    def _skip_target(self, root_state: str) -> str | None:
        if self.task_spec is None:
            return None
        transitions = (
            self.task_spec.get("metadata", {})
            .get("compiled_fsm_design", {})
            .get("transitions_by_state", {})
            .get(root_state, [])
        )
        action_name = f"skip_completed::{root_state}"
        for option in transitions:
            if option.get("action") == action_name:
                return str(option.get("next_state", "")).strip() or None
        return None

    @staticmethod
    def _state_transition_specs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        transitions = state.get("transitions", [])
        if not isinstance(transitions, list):
            return []
        return [dict(item) for item in transitions if isinstance(item, Mapping)]

    def _cluster_members_and_ends(
        self,
        start: str,
        state_map: Mapping[str, Mapping[str, Any]],
        root_cluster_starts: set[str],
    ) -> tuple[set[str], set[str]]:
        members: set[str] = set()
        ends: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in members:
                continue
            if current != start and current in root_cluster_starts:
                continue
            state = state_map.get(current, {})
            if not state:
                continue
            members.add(current)
            next_states: list[str] = []
            for transition in self._state_transition_specs(state):
                target = str(transition.get("next_state", "")).strip()
                if target and target in state_map:
                    next_states.append(target)
            fallback_target = str(state.get("next_state_on_completion", "")).strip()
            if fallback_target and fallback_target in state_map:
                next_states.append(fallback_target)
            next_states = [
                item
                for item in dict.fromkeys(next_states)
                if item == start or item not in root_cluster_starts
            ]
            if not next_states:
                ends.add(current)
            else:
                stack.extend(next_states)
        return members, ends

    def _guard_satisfied(self, guard: str) -> bool:
        if not guard:
            return True
        if guard.startswith("stove_blocked_for:"):
            alias = guard.split(":", 1)[1]
            return self._stove_blocked_for(alias)
        if guard.startswith("stove_ready_for:"):
            alias = guard.split(":", 1)[1]
            return self._stove_ready_for(alias)
        if guard.startswith("fryer_blocked_for:"):
            alias = guard.split(":", 1)[1]
            return self._fryer_blocked_for(alias)
        if guard.startswith("fryer_ready_for:"):
            alias = guard.split(":", 1)[1]
            return self._fryer_ready_for(alias)
        if guard.startswith("robot_at_station_containing:"):
            alias = guard.split(":", 1)[1]
            return self._robot_at_station_containing(alias)
        if guard.startswith("not_robot_at_station_containing:"):
            alias = guard.split(":", 1)[1]
            return not self._robot_at_station_containing(alias)
        return True

    def _stove_ready_for(self, alias: str) -> bool:
        return self._surface_ready_for("stove", alias)

    def _stove_blocked_for(self, alias: str) -> bool:
        return self._surface_blocked_for("stove", alias)

    def _fryer_ready_for(self, alias: str) -> bool:
        return self._surface_ready_for("fryer", alias)

    def _fryer_blocked_for(self, alias: str) -> bool:
        return self._surface_blocked_for("fryer", alias)

    def _surface_ready_for(self, station_prefix: str, alias: str) -> bool:
        item = self._alias_to_runtime_name(alias)
        for station in self._stations_with_prefix(station_prefix):
            if self._predicate_true("item_on", item, station):
                return True
            if self._predicate_true("item_at", item, station):
                return True
            if self._predicate_true("container_at", item, station):
                return True
            if self._station_empty(station):
                return True
        return False

    def _surface_blocked_for(self, station_prefix: str, alias: str) -> bool:
        return not self._surface_ready_for(station_prefix, alias) and any(
            not self._station_empty(station)
            for station in self._stations_with_prefix(station_prefix)
        )

    def _async_branch_done(self, branch_state: Mapping[str, Any]) -> bool:
        branch_item = str(branch_state.get("branch_item") or "")
        if not branch_item:
            return False
        predicate = str(branch_state.get("branch_completion_predicate") or "iscooked")
        runtime_item = "water" if branch_item == "WATER" else self._alias_to_runtime_name(branch_item)
        return self._predicate_true(predicate, runtime_item)

    def _robot_at_station_containing(self, alias: str) -> bool:
        item = self._alias_to_runtime_name(alias)
        robot_station = self._robot_station()
        return bool(
            robot_station
            and (
                self._predicate_true("item_at", item, robot_station)
                or self._predicate_true("container_at", item, robot_station)
            )
        )

    def _robot_station(self) -> str | None:
        if self.env is None:
            return None
        for predicate, value in self.env.current_state.predicates.items():
            if not value or getattr(predicate, "name", "") != "loc":
                continue
            params = [getattr(obj, "name", "") for obj in getattr(predicate, "params", [])]
            if len(params) == 2 and params[0].startswith("robot"):
                return params[1]
        return None

    def _stations_with_prefix(self, prefix: str) -> list[str]:
        if self.env is None:
            return []
        return [
            getattr(obj, "name", "")
            for obj in getattr(self.env.current_state, "objects", [])
            if getattr(obj, "object_type", "") == "station"
            and getattr(obj, "name", "").startswith(prefix)
        ]

    def _station_empty(self, station: str) -> bool:
        if not station:
            return False
        if self._predicate_true("station_empty", station):
            return True
        return not any(
            value
            and getattr(predicate, "name", "") in {"item_at", "container_at"}
            and len(getattr(predicate, "params", [])) >= 2
            and getattr(getattr(predicate, "params", [])[1], "name", "") == station
            for predicate, value in (self.env.current_state.predicates.items() if self.env else [])
        )

    def _goal_serving_table_target(self) -> str | None:
        if not self.raw_task:
            return None
        counts: dict[str, int] = {}
        for goal in self.raw_task.get("goal_predicates", []):
            if not isinstance(goal, Mapping):
                continue
            args = [str(arg) for arg in goal.get("args", [])]
            ids = list(goal.get("ids", []))
            for idx, arg in enumerate(args):
                if arg != "table" or idx >= len(ids):
                    continue
                try:
                    station = f"table{int(ids[idx])}"
                except (TypeError, ValueError):
                    continue
                counts[station] = counts.get(station, 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @staticmethod
    def _action_name(action: str | Mapping[str, Any]) -> str:
        if isinstance(action, Mapping):
            return str(action.get("action", "")).strip()
        return str(action).strip()

    def _alias_to_runtime_name(self, alias: str) -> str:
        if alias in self._alias_map:
            return self._alias_map[alias]
        if "__" not in alias:
            return alias
        name, suffix = alias.split("__", 1)
        return f"{name}{suffix}"

    def _build_runtime_alias_map(self) -> dict[str, str]:
        if self.env is None or not self.raw_task:
            return {}

        aliases_by_base: dict[str, list[str]] = {}
        blob = json.dumps(self.raw_task, sort_keys=True)
        for alias in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)__([0-9]+)\b", blob):
            base, suffix = alias
            full_alias = f"{base}__{suffix}"
            bucket = aliases_by_base.setdefault(base, [])
            if full_alias not in bucket:
                bucket.append(full_alias)

        runtime_by_base: dict[str, list[str]] = {}
        for obj in getattr(self.env.current_state, "objects", []):
            object_type = getattr(obj, "object_type", "")
            if object_type not in {"item", "container", "meal"}:
                continue
            name = getattr(obj, "name", "")
            base = re.sub(r"\d+$", "", name)
            runtime_by_base.setdefault(base, []).append(name)

        def numeric_suffix(value: str) -> int:
            match = re.search(r"(\d+)$", value)
            return int(match.group(1)) if match else 0

        alias_map: dict[str, str] = {}
        for base, aliases in aliases_by_base.items():
            runtime_names = sorted(runtime_by_base.get(base, []), key=numeric_suffix)
            for alias, runtime_name in zip(
                sorted(aliases, key=lambda item: int(item.rsplit("__", 1)[1])),
                runtime_names,
            ):
                alias_map[alias] = runtime_name
        return alias_map
