"""Robotouille benchmark adapter for the NS-FSM pipeline.

This adapter consumes the human-verified/stateful Robotouille ground-truth FSM
JSON, runs the official Robotouille simulator, and exposes current executable
actions as canonical NS-FSM action names.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
import sys
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
        environment_name = str(self.task_spec.get("metadata", {}).get("environment_name", ""))
        seed = self.task_spec.get("metadata", {}).get("task_seed")
        env = self._create_robotouille_env(environment_name, seed=seed)
        obs, _info = env.reset()
        self.env = env
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
        return deepcopy(self.state)

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
        if current.get("kind") == "async-branch":
            branch_item = current.get("branch_item")
            if branch_item and self._predicate_true("iscooked", branch_item):
                exit_target = current.get("branch_exit_target")
                if exit_target:
                    self.branch_exit_target = exit_target
                    tools.append("wait")
            else:
                if self._current_matches.get("wait") is not None:
                    tools.append("wait")
                for target_state in current.get("parallel_subgoals", []):
                    if self._subgoal_available_from_branch(target_state):
                        tools.append(f"advance_pending_subgoal::{target_state}")
            return tools

        for action in current.get("fsm_allowed_actions", []):
            canonical = canonical_action_name(action)
            if canonical in self._current_matches:
                tools.append(canonical)
        return tools

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
        lowered = {name.lower(): name for name in legal_names}
        return lowered.get(action_name.lower())

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
        root_cluster_starts = [
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
        for start in root_cluster_starts:
            current = start
            seen: set[str] = set()
            while True:
                state_to_root_cluster[current] = start
                if current in seen:
                    root_cluster_end_states[start] = current
                    break
                seen.add(current)
                next_state = str(state_map.get(current, {}).get("next_state_on_completion", "")).strip()
                if not next_state:
                    root_cluster_end_states[start] = current
                    break
                current = next_state

        async_branch_map: dict[str, dict[str, Any]] = {}
        for state in states:
            name = str(state["name"])
            if state.get("kind") != "async-branch":
                continue
            branch_item = name.rsplit("_", 2)[0]
            exit_target = self._default_branch_exit_target(branch_item, state_map)
            async_branch_map[name] = {
                "branch_item": branch_item,
                "exit_target": exit_target,
                "parallel_subgoals": list(state.get("parallel_subgoals", [])),
            }

        actions: set[str] = set()
        transitions_by_state: dict[str, list[dict[str, str]]] = {}
        next_root_by_root: dict[str, str] = {}
        for idx, root in enumerate(root_cluster_starts):
            if idx + 1 < len(root_cluster_starts):
                next_root_by_root[root] = root_cluster_starts[idx + 1]
        for state in states:
            name = str(state["name"])
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
            transitions_by_state[name] = options

        for branch_name, branch_info in async_branch_map.items():
            for start_state in branch_info.get("parallel_subgoals", []):
                end_state = root_cluster_end_states.get(start_state)
                if end_state and end_state in state_map:
                    state_map[end_state]["return_branch_state"] = branch_name

        for name, branch_info in async_branch_map.items():
            state_map[name]["branch_item"] = branch_info.get("branch_item")
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
                from robotouille.robotouille_env import create_robotouille_env

                return create_robotouille_env(environment_name, seed=seed, noisy_randomization=False)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Robotouille dependencies are missing ({exc}). Install Robotouille first, "
                "for example: pip install gym==0.26.2 hydra-core omegaconf pygame && "
                "pip install -e Robotouille"
            ) from exc

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
            if str(getattr(action, "name", "")) != template:
                continue
            values = {key: getattr(obj, "name", "") for key, obj in param_arg_dict.items()}
            object_types = {key: getattr(obj, "object_type", "") for key, obj in param_arg_dict.items()}

            if desired_item and desired_item not in values.values():
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
            if desired_target and not self._target_matches(desired_target, values, object_types, desired_item):
                continue
            return action, param_arg_dict
        return None

    def _target_matches(
        self,
        desired_target: str,
        values: Mapping[str, str],
        object_types: Mapping[str, str],
        desired_item: str,
    ) -> bool:
        station_names = [name for key, name in values.items() if object_types.get(key) == "station"]
        if desired_target == "stove":
            return any(name.startswith("stove") for name in station_names)
        if desired_target == "fryer":
            return any(name.startswith("fryer") for name in station_names)
        if desired_target == "board":
            return any(name.startswith("board") for name in station_names)
        if desired_target == "sink":
            return any(name.startswith("sink") for name in station_names)
        if desired_target == "table":
            return any(name.startswith("table") for name in station_names)
        if desired_target.startswith("station containing "):
            target_item = self._alias_to_runtime_name(desired_target.split("station containing ", 1)[1])
            return any(self._predicate_true("item_at", target_item, station) for station in station_names)
        if desired_target == "stove while holding item":
            return any(name.startswith("stove") for name in station_names)
        if desired_target == "fryer while holding item":
            return any(name.startswith("fryer") for name in station_names)
        if desired_target == "cutting board while holding item":
            return any(name.startswith("board") for name in station_names)
        if desired_target == "sink while holding pot":
            return any(name.startswith("sink") for name in station_names)
        if desired_target == "stove while holding water-filled pot":
            return any(name.startswith("stove") for name in station_names)
        if desired_target == "pot on stove while holding ingredient":
            return any(name.startswith("stove") for name in station_names)
        if desired_target == "bowl while holding pot":
            return any(name.startswith("table") or name.startswith("board") or name.startswith("sink") for name in station_names)
        if desired_target == "serving table while holding bowl":
            return any(name.startswith("table") for name in station_names)
        if desired_target == "serving table while holding item":
            return any(name.startswith("table") for name in station_names)
        if desired_target == "pot location":
            return True
        return True

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
            branch_item = current.get("branch_item")
            if branch_item and self._predicate_true("iscooked", self._alias_to_runtime_name(branch_item)):
                return current.get("branch_exit_target")
            return self.current_fsm_state

        target = str(current.get("next_state_on_completion", "")).strip()
        if target:
            return target

        root_start = state_to_root.get(self.current_fsm_state or "")
        if root_start and root_end_states.get(root_start) == self.current_fsm_state:
            self.completed_root_clusters.add(root_start)

        return_branch = str(current.get("return_branch_state", "")).strip()
        if return_branch:
            branch_info = state_map.get(return_branch, {})
            branch_item = branch_info.get("branch_item")
            if branch_item and self._predicate_true("iscooked", self._alias_to_runtime_name(branch_item)):
                return branch_info.get("branch_exit_target")
            return return_branch
        return None

    def _subgoal_available_from_branch(self, target_state: str) -> bool:
        if target_state in self.completed_root_clusters:
            return False
        if self.task_spec is None:
            return False
        state_map = self.task_spec.get("metadata", {}).get("compiled_state_map", {})
        target = state_map.get(target_state, {})
        actions = target.get("fsm_allowed_actions", [])
        if not actions:
            return False
        canonical = canonical_action_name(actions[0])
        return canonical in self._current_matches

    @staticmethod
    def _action_name(action: str | Mapping[str, Any]) -> str:
        if isinstance(action, Mapping):
            return str(action.get("action", "")).strip()
        return str(action).strip()

    @staticmethod
    def _alias_to_runtime_name(alias: str) -> str:
        if "__" not in alias:
            return alias
        name, suffix = alias.split("__", 1)
        return f"{name}{suffix}"
