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
sys.path.insert(0, SRC)

from context_manager import ContextManager
from datalog_verifier import DatalogVerifier
from datasets.generic import GenericScenarioAdapter
from datasets.minecraft import MinecraftAdapter
from datasets.registry import get_adapter, list_datasets
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


class NSFSMTests(unittest.TestCase):
    def test_generic_adapter_and_agent(self):
        adapter = GenericScenarioAdapter()
        spec = adapter.to_task_spec({"instruction": "Create a short evaluation plan."}).to_dict()
        fsm = FSMBuilder().from_template(spec, adapter)
        result = NSFSMAgent(spec, adapter, fsm, planner_only=True).run_episode()
        self.assertTrue(result["success"])
        self.assertEqual(result["termination"], "success")
        self.assertGreater(result["total_steps"], 0)

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
