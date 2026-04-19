#!/usr/bin/env python3
"""Run one NS-FSM scenario."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Mapping

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from datasets.registry import get_adapter, registry_warning
from fsm_builder import FSMBuilder
from fsm_designer import LLMFSMDesigner
from nsfsm_agent import NSFSMAgent


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.dataset)
    warning = registry_warning(args.dataset, adapter)
    raw_task = load_raw_task(args)
    task_spec = adapter.to_task_spec(adapter.load_or_wrap(raw_task)).to_dict()
    if args.max_steps is not None:
        task_spec["max_steps"] = args.max_steps

    fsm, fsm_metadata = build_runtime_fsm(
        task_spec=task_spec,
        adapter=adapter,
        use_fixed_generic_fsm=args.use_fixed_generic_fsm,
        config_path=args.config,
    )
    agent = NSFSMAgent(
        task_spec=task_spec,
        adapter=adapter,
        fsm=fsm,
        llm=None if args.planner_only else load_runtime_llm(args.config),
        planner_only=args.planner_only,
        verbose=not args.quiet,
    )
    result = agent.run_episode()
    result["metadata"]["fsm_build"] = fsm_metadata
    if warning:
        result["metadata"]["dataset_warning"] = warning

    output_dir = os.path.join(ROOT, "results", "scenarios", args.tag)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_name(task_spec['task_id'])}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if not args.quiet:
        print(
            f"[run_scenario] success={result['success']} "
            f"termination={result['termination']} steps={result['total_steps']}"
        )
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", help="Free-form user scenario instruction.")
    parser.add_argument("--scenario-file", help="JSON scenario file.")
    parser.add_argument("--dataset", default="generic", help="Dataset adapter name.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--config",
        default=None,
        help="Path to hyperparams YAML for LLM FSM design/runtime calls.",
    )
    parser.add_argument(
        "--use-fixed-generic-fsm",
        action="store_true",
        help="Skip LLM FSM design and use template/fallback FSM.",
    )
    parser.add_argument(
        "--planner-only",
        action="store_true",
        help="Skip LLM action selection and use deterministic planner fallback.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_raw_task(args: argparse.Namespace) -> str | dict[str, Any]:
    if args.scenario_file:
        with open(args.scenario_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.instruction:
        return args.instruction
    raise SystemExit("Provide --instruction or --scenario-file.")


def build_runtime_fsm(
    task_spec: Mapping[str, Any],
    adapter: Any,
    use_fixed_generic_fsm: bool,
    config_path: str | None,
):
    builder = FSMBuilder()
    metadata: dict[str, Any] = {
        "source": "template" if use_fixed_generic_fsm else "llm_designer",
        "llm_fsm_error": None,
    }
    if use_fixed_generic_fsm:
        return builder.from_template(task_spec, adapter), metadata

    try:
        proposal = LLMFSMDesigner(config_path=config_path).design_with_metadata(task_spec, adapter)
        fsm = builder.from_design(proposal.fsm_design, task_spec, adapter, allow_fallback=True)
        metadata.update(
            {
                "task_hash": proposal.task_hash,
                "cache_path": proposal.cache_path,
                "fallback_used": getattr(fsm, "validation", {}).get("fallback_used"),
            }
        )
        return fsm, metadata
    except Exception as exc:
        metadata["source"] = "template_after_llm_error"
        metadata["llm_fsm_error"] = str(exc)
        return builder.from_template(task_spec, adapter), metadata


def load_runtime_llm(config_path: str | None):
    try:
        from llm_interface import LLMInterface
    except Exception as exc:
        raise RuntimeError(
            "LLMInterface could not be imported. Install optional LLM "
            "dependencies or run with --planner-only."
        ) from exc
    return LLMInterface(config_path)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


if __name__ == "__main__":
    main()
