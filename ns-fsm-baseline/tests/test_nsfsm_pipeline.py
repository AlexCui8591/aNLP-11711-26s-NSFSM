"""Smoke and unit tests for the NS-FSM MVP pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SRC)
sys.path.insert(0, SCRIPTS)

from context_manager import ContextManager
from datalog_verifier import DatalogVerifier
from datasets.generic import GenericScenarioAdapter
from datasets.minecraft import MinecraftAdapter
from datasets.registry import get_adapter, list_datasets
from datasets.robotouille import RobotouilleAdapter
from datasets.structured import StructuredScenarioAdapter
from datasets.swe_bench import SWEBenchAdapter
from action_parser import ActionParser
from env_wrapper import MCTextWorldWrapper
from fsm_builder import FSMBuilder
from fsm_designer import LLMFSMDesigner
from fsm_validator import FSMDesignValidator, build_generic_tool_use_fsm_design
from nsfsm_agent import NSFSMAgent
from planner import Planner
from rule_checker import RuleChecker


class MockLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


class SequenceLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.responses:
            raise RuntimeError("No mocked LLM responses left.")
        return self.responses.pop(0)


class FailingLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("mock LLM outage")


class NSFSMTests(unittest.TestCase):
    def test_generic_adapter_and_agent(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec({"instruction": "Create a short evaluation plan."}).to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        result = NSFSMAgent(spec, adapter, fsm, planner_only=True).run_episode()
        self.assertTrue(result["success"])
        self.assertEqual(result["termination"], "success")
        self.assertGreater(result["total_steps"], 0)

    def test_require_llm_rejects_planner_paths(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec({"instruction": "Create a short evaluation plan."}).to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        with self.assertRaisesRegex(ValueError, "requires an LLM client"):
            NSFSMAgent(spec, adapter, fsm, require_llm=True)

        fsm = FSMBuilder().from_template(spec, adapter)
        with self.assertRaisesRegex(ValueError, "planner_only=False"):
            NSFSMAgent(spec, adapter, fsm, llm=MockLLM("{}"), planner_only=True, require_llm=True)

    def test_require_llm_does_not_fallback_on_llm_failure(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec({"instruction": "Create a short evaluation plan."}).to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        with self.assertRaisesRegex(RuntimeError, "LLM inference is required"):
            NSFSMAgent(spec, adapter, fsm, llm=FailingLLM(), require_llm=True).run_episode()

    def test_llm_config_can_ignore_runtime_env_overrides(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed in this Python environment.")

        from llm_interface import LLMInterface

        env_keys = ["NSFSM_LLM_API_BASE", "NSFSM_LLM_MODEL_NAME", "NSFSM_LLM_API_KEY", "OPENAI_API_KEY"]
        old_env = {key: os.environ.get(key) for key in env_keys}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(
                "\n".join(
                    [
                        "llm:",
                        '  model_name: "gpt-4o"',
                        '  api_base: "https://api.openai.com/v1"',
                        '  api_key_env: "OPENAI_API_KEY"',
                        "  ignore_env_overrides: true",
                    ]
                )
            )
            config_path = f.name
        try:
            os.environ["NSFSM_LLM_API_BASE"] = "http://localhost:8000/v1"
            os.environ["NSFSM_LLM_MODEL_NAME"] = "Qwen/Qwen2.5-7B-Instruct"
            os.environ["NSFSM_LLM_API_KEY"] = "not-needed"
            os.environ["OPENAI_API_KEY"] = "unit-test-openai-key"
            llm = LLMInterface(config_path)
            self.assertEqual(llm.api_base, "https://api.openai.com/v1")
            self.assertEqual(llm.model, "gpt-4o")
            self.assertEqual(llm.api_key, "unit-test-openai-key")
        finally:
            os.unlink(config_path)
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_structured_adapter_builds_linear_fsm(self):
        adapter = StructuredScenarioAdapter()
        spec = adapter.to_task_spec(
            {
                "task": "Evaluate QA answers.",
                "actions": ["load_dataset", "run_model", "score_answers"],
                "success_condition": "Report exists.",
            }
        ).to_dict()
        design = adapter.build_fsm_design(spec)
        result = FSMDesignValidator().validate(design, spec, adapter)
        self.assertTrue(result["valid"], result)
        fsm = FSMBuilder().from_design(design, spec, adapter)
        self.assertEqual(fsm.get_valid_actions(), ["load_dataset"])

    def test_llm_fsm_designer_with_mocked_output(self):
        response = json.dumps(build_generic_tool_use_fsm_design())
        with tempfile.TemporaryDirectory() as tmp:
            designer = LLMFSMDesigner(llm=MockLLM(response), cache_dir=tmp, force_refresh=True)
            spec = GenericScenarioAdapter().to_task_spec("Draft a plan.").to_dict()
            design = designer.design(spec)
        self.assertIn("transitions_by_state", design)
        self.assertNotIn("legal_actions_by_state", response)

    def test_validator_derives_actions_and_fallback(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec("Draft a plan.").to_dict()
        design = build_generic_tool_use_fsm_design(spec)
        result = FSMDesignValidator().validate(design, spec, adapter)
        self.assertTrue(result["valid"], result)
        self.assertIn("actions", result["fsm_design"])
        self.assertIn("legal_actions_by_state", result["fsm_design"])
        bad = {"states": ["START"], "initial_state": "START"}
        fallback = FSMDesignValidator().validate(bad, spec, adapter)
        self.assertTrue(fallback["valid"])
        self.assertTrue(fallback["fallback_used"])

    def test_datalog_reachability_and_transition_checks(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec("Draft a plan.").to_dict()
        design = FSMDesignValidator().validate(
            build_generic_tool_use_fsm_design(spec),
            spec,
            adapter,
        )["fsm_design"]
        verifier = DatalogVerifier()
        result = verifier.verify(design, spec, adapter)
        self.assertTrue(result["ok"], result)
        self.assertTrue(verifier.verify_transition("START", "read_task", "READ_TASK")["valid"])
        self.assertFalse(verifier.verify_transition("START", "finalize", "DONE")["valid"])

    def test_registry_templates_rule_checker_and_planner(self):
        self.assertIn("generic", list_datasets())
        adapter = get_adapter("unknown_dataset")
        self.assertEqual(adapter.dataset_name, "generic")
        spec = adapter.to_task_spec("Draft a plan.").to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        state = adapter.reset(spec)
        action = Planner().next_action(spec, state, fsm.get_valid_actions(), [], fsm.get_valid_transitions())
        self.assertEqual(action["action"], "read_task")
        check = RuleChecker().check(
            "read_task",
            fsm.get_valid_actions(),
            state,
            spec,
            adapter,
            fsm.verify_transition("read_task", "READ_TASK"),
        )
        self.assertTrue(check["legal"], check)

    def test_context_manager_preserves_current_state_and_transitions(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec("Draft a plan.").to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        state = adapter.reset(spec)
        history = [{"step": i, "action": "x", "success": True} for i in range(20)]
        packet = ContextManager(max_chars=1200, recent_k=4).build_packet(
            spec,
            fsm,
            adapter,
            state,
            history,
            [],
            [],
        )
        self.assertEqual(packet["fsm_state"], "START")
        self.assertTrue(packet["transition_options"])
        self.assertLessEqual(len(packet["recent_history"]), 4)

    def test_minecraft_adapter_fallback_simulator(self):
        adapter = MinecraftAdapter(max_steps=5)
        spec = adapter.to_task_spec("minecraft/stick").to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        self.assertEqual(fsm.get_valid_transitions()[0]["action"], "mine_oak_log")
        adapter.reset(spec)
        result = None
        for action in ["mine_oak_log", "craft_oak_planks", "craft_stick"]:
            result = adapter.step(action)
            self.assertTrue(result.info["success"], result.info)
        self.assertTrue(result.done)

    def test_baseline_env_and_action_parser_fallbacks(self):
        parser = ActionParser()
        self.assertIn("mine_oak_log", parser.get_candidate_actions({}))

        env = MCTextWorldWrapper(max_steps=5)
        env.reset("stick")
        for action in ["mine_oak_log", "craft_oak_planks", "craft_stick"]:
            _obs, done, info = env.step(action)
            self.assertTrue(info["success"], info)
        self.assertTrue(done)
        self.assertGreaterEqual(env.get_inventory().get("stick", 0), 1)

    def test_swe_bench_adapter_and_agent(self):
        adapter = SWEBenchAdapter()
        spec = adapter.to_task_spec(
            {
                "problem_statement": "Fix a small bug.",
                "candidate_files": ["buggy.py"],
            }
        ).to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        result = NSFSMAgent(spec, adapter, fsm, planner_only=True).run_episode()
        self.assertTrue(result["success"], result)

    def test_robotouille_ground_truth_builder_and_llm_fallback(self):
        import build_robotouille_ground_truth_fsm as gt_builder

        task_path = os.path.join(ROOT, "config", "robotouille_ground_truth_asynchronous_tasks.json")
        with open(task_path, "r", encoding="utf-8") as f:
            task = json.load(f)["tasks"][0]

        seed = gt_builder.build_task_fsm(task, mode="branching")
        self.assertFalse(gt_builder.ground_truth_validation_issues(seed))

        generated_json = json.dumps(
            {
                "task_id": task["task_id"],
                "verified_plan_status": "stateful_ground_truth_llm_candidate",
                "fsm_generation_mode": "llm_generated",
                "state_list": seed["state_list"],
            }
        )
        generated, event = gt_builder.generate_llm_ground_truth_candidate(
            llm_generator=SequenceLLM([generated_json]),
            task=task,
            seed_candidate=seed,
            attempts=1,
        )
        self.assertTrue(event["accepted"], event)
        self.assertEqual(generated["fsm_generation_mode"], "llm_generated")

        judged = gt_builder.judge_and_fallback_task_fsm(
            task=task,
            candidate=generated,
            llm_judge=SequenceLLM(
                [
                    json.dumps(
                        {
                            "verdict": "fail",
                            "score": 0.1,
                            "issues": ["unit-test rejection"],
                            "repair_hints": [],
                            "patched_state_list": None,
                        }
                    )
                ]
            ),
            llm_required=False,
            judge_attempts=1,
            fallback_candidate=seed,
            fallback_source="last_good_ground_truth",
            candidate_event=event,
        )
        metadata = judged["ground_truth_generation"]
        self.assertTrue(metadata["fallback_used"], metadata)
        self.assertEqual(metadata["status"], "fallback_to_last_good_ground_truth")

    def test_agent_accepts_fsm_action_before_adapter_execution_check(self):
        class PosthocAdapter:
            dataset_name = "posthoc"

            def reset(self, task_spec):
                self.state = {"step_count": 0, "done": False}
                return dict(self.state)

            def is_done(self, state, task_spec):
                return bool(state.get("done"))

            def get_available_tools(self, task_spec, state):
                return ["safe"]

            def normalize_action(self, raw_action, legal_actions):
                action = raw_action.get("action") if isinstance(raw_action, dict) else raw_action
                names = [item.get("action") if isinstance(item, dict) else item for item in legal_actions]
                return action if action in names else None

            def step(self, action):
                action_name = action.get("action") if isinstance(action, dict) else action
                self.state = {"step_count": 1, "done": action_name == "safe"}
                return type(
                    "StepResult",
                    (),
                    {"state": dict(self.state), "done": self.state["done"], "info": {"success": self.state["done"]}},
                )()

            def format_state_for_prompt(self, state):
                return str(state)

            def summarize_result(self, state):
                return {"success": bool(state.get("done")), "total_steps": state.get("step_count", 0)}

        spec = {
            "dataset": "posthoc",
            "task_id": "posthoc/demo",
            "task_type": "symbolic_planning",
            "instruction": "Demonstrate post-hoc verification.",
            "initial_state": {},
            "goal_condition": {"done": True},
            "available_tools": ["unsafe", "safe"],
            "max_steps": 1,
            "success_criteria": ["done"],
            "metadata": {},
        }
        design = {
            "states": ["START", "DONE"],
            "initial_state": "START",
            "terminal_states": ["DONE"],
            "transitions_by_state": {
                "START": [
                    {"action": "unsafe", "next_state": "DONE", "condition": "FSM permits this but env does not."},
                    {"action": "safe", "next_state": "DONE", "condition": "FSM and env both permit this."},
                ],
                "DONE": [],
            },
            "fallback_policy": {"on_invalid_action": "fallback_to_verified_intersection"},
            "success_signals": ["done"],
            "risk_notes": ["unit test"],
        }
        adapter = PosthocAdapter()
        fsm = FSMBuilder().from_design(design, spec, adapter)
        result = NSFSMAgent(
            spec,
            adapter,
            fsm,
            llm=SequenceLLM(
                [
                    json.dumps(
                        {
                            "thought": "unsafe is still in the current FSM state",
                            "action": "unsafe",
                            "next_state": "DONE",
                        }
                    )
                ]
            ),
            max_llm_retries=1,
        ).run_episode()
        self.assertFalse(result["success"], result)
        self.assertEqual(result["termination"], "max_steps")
        self.assertEqual(result["trajectory"][0]["proposal"]["action"], "unsafe")
        self.assertEqual(result["trajectory"][0]["action"], "unsafe")
        self.assertEqual(result["trajectory"][0]["decision_source"], "llm")
        self.assertFalse(result["trajectory"][0]["forced_choice"])
        self.assertFalse(result["trajectory"][0]["success"])
        self.assertEqual(result["blocked_action_count"], 0)
        self.assertEqual(result["fallback_action_count"], 0)

        retry_llm = SequenceLLM(
            [
                json.dumps({"thought": "first action is outside the FSM", "action": "outside_fsm", "next_state": "DONE"}),
                json.dumps({"thought": "use the verified option", "action": "safe", "next_state": "DONE"}),
            ]
        )
        adapter = PosthocAdapter()
        fsm = FSMBuilder().from_design(design, spec, adapter)
        result = NSFSMAgent(spec, adapter, fsm, llm=retry_llm, max_llm_retries=1).run_episode()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["trajectory"][0]["proposal"]["action"], "safe")
        self.assertEqual(result["trajectory"][0]["action"], "safe")
        self.assertEqual(result["trajectory"][0]["decision_source"], "llm")
        self.assertFalse(result["trajectory"][0]["forced_choice"])
        self.assertEqual(result["blocked_action_count"], 1)
        self.assertEqual(result["fallback_action_count"], 0)

        invalid_twice_llm = SequenceLLM(
            [
                json.dumps({"thought": "first invalid option", "action": "outside_fsm", "next_state": "DONE"}),
                json.dumps({"thought": "still invalid", "action": "still_outside_fsm", "next_state": "DONE"}),
            ]
        )
        adapter = PosthocAdapter()
        fsm = FSMBuilder().from_design(design, spec, adapter)
        result = NSFSMAgent(spec, adapter, fsm, llm=invalid_twice_llm, max_llm_retries=1).run_episode()
        self.assertFalse(result["success"], result)
        self.assertEqual(result["trajectory"][0]["proposal"]["action"], "still_outside_fsm")
        self.assertEqual(result["trajectory"][0]["action"], "unsafe")
        self.assertEqual(result["trajectory"][0]["decision_source"], "forced_choice")
        self.assertTrue(result["trajectory"][0]["forced_choice"])
        self.assertEqual(result["blocked_action_count"], 2)
        self.assertEqual(result["fallback_action_count"], 1)

    def test_robotouille_normalizes_runtime_aliases_and_prose(self):
        class Obj:
            def __init__(self, name, object_type="item"):
                self.name = name
                self.object_type = object_type

        adapter = RobotouilleAdapter()
        adapter._alias_map = {"chicken__3": "chicken1", "tomato__4": "tomato1"}
        move_match = (
            Obj("move", "action"),
            {"s1": Obj("sink1", "station"), "s2": Obj("board1", "station")},
        )
        cut_match = (
            Obj("cut", "action"),
            {"i1": Obj("tomato1"), "s1": Obj("board1", "station")},
        )
        adapter.current_valid_actions = [move_match, cut_match]
        adapter.current_valid_action_strings = [
            "Move robot1 from sink1 to board1",
            "Cut tomato1 on board1 using robot1",
        ]
        adapter._current_matches = {
            "move|target=station containing pot": move_match,
            "cut|item=tomato__4|target=board": cut_match,
        }

        self.assertEqual(
            adapter.normalize_action(
                "Move robot1 from sink1 to board1",
                ["move|target=station containing pot"],
            ),
            "move|target=station containing pot",
        )
        self.assertEqual(
            adapter.normalize_action(
                "pick-up-item|item=chicken1",
                ["pick-up-item|item=chicken__3"],
            ),
            "pick-up-item|item=chicken__3",
        )
        self.assertEqual(
            adapter.normalize_action(
                "move|target=board1 while holding tomato1",
                ["move|target=cutting board while holding item"],
            ),
            "move|target=cutting board while holding item",
        )

        adapter.task_spec = {
            "metadata": {
                "compiled_state_map": {
                    "NAVIGATE_TO_STOVE_WITH_chicken__3": {},
                }
            }
        }
        self.assertEqual(
            adapter.normalize_state_name("NAVIGATE_TO_STOVE_WITH_chicken1"),
            "NAVIGATE_TO_STOVE_WITH_chicken__3",
        )

        adapter._predicate_true = (
            lambda name, *args: name == "container_at" and args == ("pot", "board1")
        )
        self.assertTrue(
            adapter._target_matches(
                "station containing pot",
                {"s2": "board1"},
                {"s2": "station"},
                "",
            )
        )

    def test_robotouille_primitive_pickup_macro_does_not_advance_on_unstack(self):
        class Obj:
            def __init__(self, name, object_type="item"):
                self.name = name
                self.object_type = object_type

        class Pred:
            def __init__(self, name, params):
                self.name = name
                self.params = params

        class FakeState:
            def __init__(self, valid_actions, valid_strings, predicates=None):
                self.valid_actions = valid_actions
                self.valid_strings = valid_strings
                self.predicates = predicates or {}

            def get_valid_actions_and_str(self):
                return self.valid_actions, self.valid_strings

            def is_goal_reached(self):
                return False

        unstack = (
            Obj("unstack", "action"),
            {
                "i1": Obj("chicken1"),
                "i2": Obj("lettuce1"),
                "s1": Obj("table1", "station"),
            },
        )
        pickup = (
            Obj("pick-up", "action"),
            {
                "i1": Obj("chicken1"),
                "s1": Obj("table1", "station"),
            },
        )

        class FakeEnv:
            def __init__(self):
                self.current_state = FakeState(
                    [unstack],
                    ["unstack(robot1,chicken1,lettuce1,table1)"],
                )
                self.calls = 0

            def step(self, actions):
                self.calls += 1
                if self.calls == 1:
                    self.current_state = FakeState(
                        [pickup],
                        ["pick-up(robot1,chicken1,table1)"],
                    )
                else:
                    has_chicken = Pred("has_item", [Obj("robot1", "robot"), Obj("chicken1")])
                    self.current_state = FakeState(
                        [],
                        [],
                        {has_chicken: True},
                    )
                return "obs", 0.0, False, {}

        adapter = RobotouilleAdapter()
        adapter.env = FakeEnv()
        adapter.task_spec = {
            "metadata": {
                "compiled_state_map": {
                    "PICK": {
                        "kind": "pick-up-item",
                        "completion_condition": ["robot has chicken__3"],
                        "fsm_allowed_actions": [
                            {"template": "pick-up-item", "item": "chicken__3"}
                        ],
                        "next_state_on_completion": "NEXT",
                    },
                    "NEXT": {"kind": "terminal"},
                },
                "compiled_fsm_design": {
                    "transitions_by_state": {
                        "PICK": [
                            {
                                "action": "pick-up-item|item=chicken__3",
                                "next_state": "NEXT",
                            }
                        ]
                    }
                },
                "state_to_root_cluster": {},
                "root_cluster_end_states": {},
            }
        }
        adapter._alias_map = {"chicken__3": "chicken1"}
        adapter.current_fsm_state = "PICK"
        adapter.state = {"step_count": 0, "goal_satisfied": False, "fsm_state": "PICK"}
        adapter._refresh_valid_actions()

        first = adapter.step("unstack(robot1,chicken1,lettuce1,table1)")
        self.assertNotIn("fsm_next_state", first.info)
        self.assertEqual(adapter.current_fsm_state, "PICK")

        second = adapter.step("pick-up(robot1,chicken1,table1)")
        self.assertEqual(second.info["fsm_next_state"], "NEXT")
        self.assertEqual(second.info["fsm_transition_action"], "pick-up-item|item=chicken__3")

    def test_robotouille_runtime_fallback_uses_primitive_not_fsm_action(self):
        class NoIntersectionAdapter:
            dataset_name = "robotouille"

            def reset(self, task_spec):
                self.state = {"step_count": 0, "done": False}
                return dict(self.state)

            def is_done(self, state, task_spec):
                return bool(state.get("done"))

            def get_available_tools(self, task_spec, state):
                return []

            def get_runtime_actions(self, task_spec, state):
                return ["move(robot1,table1,table2)"]

            def normalize_action(self, raw_action, legal_actions):
                action = raw_action.get("action") if isinstance(raw_action, dict) else raw_action
                names = [item.get("action") if isinstance(item, dict) else item for item in legal_actions]
                return action if action in names else None

            def step(self, action):
                action_name = action.get("action") if isinstance(action, dict) else action
                self.state = {"step_count": 1, "done": action_name == "move(robot1,table1,table2)"}
                return type(
                    "StepResult",
                    (),
                    {"state": dict(self.state), "done": self.state["done"], "info": {"success": self.state["done"]}},
                )()

            def format_state_for_prompt(self, state):
                return str(state)

            def summarize_result(self, state):
                return {"success": bool(state.get("done")), "total_steps": state.get("step_count", 0)}

        spec = {
            "dataset": "robotouille",
            "task_id": "robotouille/unit",
            "task_type": "symbolic_planning",
            "instruction": "Keep running from FSM fallback.",
            "initial_state": {},
            "goal_condition": {"done": True},
            "available_tools": ["safe"],
            "max_steps": 3,
            "success_criteria": ["done"],
            "metadata": {},
        }
        design = {
            "states": ["START", "DONE"],
            "initial_state": "START",
            "terminal_states": ["DONE"],
            "transitions_by_state": {
                "START": [{"action": "safe", "next_state": "DONE", "condition": "unit"}],
                "DONE": [],
            },
            "fallback_policy": {"on_invalid_action": "runtime_primitive_fallback"},
            "success_signals": ["done"],
            "risk_notes": ["unit test"],
        }
        adapter = NoIntersectionAdapter()
        fsm = FSMBuilder().from_design(design, spec, adapter)
        llm = SequenceLLM(
            [json.dumps({"thought": "bad", "action": "invalid", "next_state": "NOWHERE"})]
        )
        result = NSFSMAgent(spec, adapter, fsm, llm=llm, max_llm_retries=0).run_episode()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["trajectory"][0]["action"], "move(robot1,table1,table2)")
        self.assertEqual(result["trajectory"][0]["decision_source"], "runtime_fallback")
        self.assertFalse(result["trajectory"][0]["forced_choice"])
        self.assertEqual(
            result["trajectory"][0]["rule_check"]["reason_type"],
            "runtime_primitive_fallback",
        )

    def test_runner_smoke_and_analysis(self):
        tag = "unit_smoke_nsfsm"
        scenario_cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "run_scenario.py"),
            "--instruction",
            "Create a short plan for evaluating a model on a QA dataset.",
            "--tag",
            tag,
            "--use-fixed-generic-fsm",
            "--planner-only",
            "--quiet",
        ]
        subprocess.run(scenario_cmd, cwd=ROOT, check=True, capture_output=True, text=True)

        exp_cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "run_nsfsm_experiment.py"),
            "--dataset",
            "minecraft",
            "--task-ids",
            "minecraft/stick",
            "--runs",
            "1",
            "--tag",
            tag,
            "--use-fixed-generic-fsm",
            "--planner-only",
            "--quiet",
            "--save-fsm-design",
        ]
        subprocess.run(exp_cmd, cwd=ROOT, check=True, capture_output=True, text=True)

        analyze_cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "analyze_nsfsm.py"),
            "--tag",
            tag,
        ]
        subprocess.run(analyze_cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertTrue(
            os.path.exists(os.path.join(ROOT, "results", "analysis", tag, "combined_report.md"))
        )


if __name__ == "__main__":
    unittest.main()
