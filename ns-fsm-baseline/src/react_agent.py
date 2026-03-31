"""
ReAct Agent for MC-TextWorld crafting tasks.

Loop:
  observe → build prompt → LLM generate → parse thought+action
  → env.step → check termination → repeat

Termination conditions:
  1. goal_achieved  — env signals success
  2. max_steps      — episode budget exhausted
  3. dead_loop      — same action repeated ≥ dead_loop_window times in a row
                      OR last dead_loop_window actions all failed
"""

import os
import sys
import yaml

# Make sure sibling src modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from env_wrapper import MCTextWorldWrapper
from llm_interface import LLMInterface
from action_parser import ActionParser
from prompts import (
    build_react_prompt,
    format_inventory,
    format_history,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "hyperparams.yaml"
)


# ─────────────────────────────────────────────────────────────────────────────
# Dead-loop detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_same_failed_action_loop(trajectory: list, window: int) -> bool:
    """True if the last `window` steps are the SAME action AND all FAILED.

    Repeating a successful action (e.g. mine_oak_log ×5) is perfectly valid,
    so we only trigger when the action keeps failing.
    """
    if len(trajectory) < window:
        return False
    recent = trajectory[-window:]
    actions = {s["action"] for s in recent}
    all_failed = all(not s.get("success", False) for s in recent)
    return len(actions) == 1 and all_failed


def _is_all_failed_loop(trajectory: list, window: int) -> bool:
    """True if every one of the last `window` steps failed (any mix of actions)."""
    if len(trajectory) < window:
        return False
    return all(not s.get("success", False) for s in trajectory[-window:])


def detect_dead_loop(trajectory: list, window: int = 5) -> tuple:
    """
    Returns (is_loop: bool, reason: str).

    Two checks:
      1. Same action repeated `window` times AND all failed  (tight loop)
      2. ANY `window * 2` consecutive failures               (thrashing)

    Check #2 uses a wider window to give the agent room to try alternatives.
    """
    if _is_same_failed_action_loop(trajectory, window):
        repeated = trajectory[-1]["action"]
        return True, f"same action repeated {window}× and all failed: '{repeated}'"

    thrash_window = window * 2
    if _is_all_failed_loop(trajectory, thrash_window):
        return True, f"last {thrash_window} actions all failed (thrashing)"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# ReactAgent
# ─────────────────────────────────────────────────────────────────────────────

class ReactAgent:
    """
    Single-episode ReAct agent.

    Usage:
        agent = ReactAgent()
        result = agent.run_episode("iron_pickaxe")
        # result is a dict with trajectory + metadata
    """

    def __init__(
        self,
        config_path: str = None,
        history_window: int = 10,
        dead_loop_window: int = 5,
        verbose: bool = None,
    ):
        """
        Args:
            config_path:      Path to hyperparams.yaml. Defaults to config/hyperparams.yaml.
            history_window:   Number of recent steps fed into the prompt.
            dead_loop_window: Number of recent steps used for dead-loop detection.
            verbose:          Print step-by-step progress. Reads from config if None.
        """
        config_path = config_path or _DEFAULT_CONFIG
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        max_steps = (
            self._config.get("react", {}).get("max_steps_per_episode")
            or self._config.get("env", {}).get("max_steps_per_episode", 100)
        )

        if verbose is None:
            verbose = self._config.get("logging", {}).get("verbose", True)

        self.max_steps = max_steps
        self.history_window = history_window
        self.dead_loop_window = dead_loop_window
        self.verbose = verbose

        self.llm = LLMInterface(config_path)
        self.env = MCTextWorldWrapper(max_steps=max_steps)
        self.parser = ActionParser()

    # ── public API ────────────────────────────────────────────────────────────

    def run_episode(self, goal_item: str, reflection: str = "") -> dict:
        """
        Run one full episode for `goal_item`.

        Args:
            goal_item:  Item name to craft/obtain (e.g. "iron_pickaxe").
            reflection: Optional reflection string from a prior failed attempt
                        (used by the Reflexion wrapper; ignored in plain ReAct).

        Returns:
            Episode result dict:
            {
                "goal":         str,
                "success":      bool,
                "total_steps":  int,
                "termination":  "success" | "max_steps" | "dead_loop" | "llm_error",
                "dead_loop_reason": str,       # only when termination == "dead_loop"
                "trajectory":   list[dict],    # full step records from env_wrapper
            }
        """
        obs = self.env.reset(goal_item)
        termination = "max_steps"
        dead_loop_reason = ""

        self._log(f"\n{'='*60}")
        self._log(f"[ReAct] Goal: {goal_item}  |  max_steps={self.max_steps}")
        self._log(f"{'='*60}")

        while True:
            step = self.env.step_count
            trajectory = self.env.trajectory          # live reference
            inventory = obs["inventory"]

            # ── 1. Build prompt ──────────────────────────────────────────────
            system_prompt, user_prompt = build_react_prompt(
                goal_item=goal_item,
                inventory=inventory,
                trajectory=trajectory,
                step=step,
                max_steps=self.max_steps,
                reflection=reflection,
                last_k=self.history_window,
            )

            # ── 2. LLM call ──────────────────────────────────────────────────
            try:
                response = self.llm.generate(system_prompt, user_prompt)
            except Exception as exc:
                self._log(f"[ReAct] LLM error at step {step}: {exc}")
                termination = "llm_error"
                break

            thought, _ = self.llm.parse_react_response(response)
            self._log(f"\n── Step {step + 1} ──────────────────────────────")
            if thought:
                self._log(f"Thought: {thought[:200]}")

            # ── 3. Parse action ──────────────────────────────────────────────
            action, parse_reason = self.parser.parse(response)

            if action is None:
                # Fallback: use whatever raw token the LLM emitted so the env
                # records a failed step (with a message) rather than crashing.
                _, raw_fallback = self.llm.parse_react_response(response)
                action = raw_fallback or "__invalid__"
                self._log(f"[Parser] WARN: {parse_reason} → using '{action}' as fallback")
            else:
                if parse_reason != "ok":
                    self._log(f"[Parser] {parse_reason}")
                self._log(f"Action: {action}")

            # ── 4. Env step ──────────────────────────────────────────────────
            obs, done, info = self.env.step(action)

            status_tag = "✓" if info["success"] else "✗"
            self._log(f"Result: [{status_tag}] {info['message']}")
            self._log(f"Inventory: {format_inventory(obs['inventory'])}")

            # ── 5. Check termination ─────────────────────────────────────────
            if info["goal_achieved"]:
                termination = "success"
                self._log(f"\n[ReAct] SUCCESS — obtained {goal_item} in {self.env.step_count} steps")
                break

            if info["timeout"]:
                termination = "max_steps"
                self._log(f"\n[ReAct] TIMEOUT — max_steps={self.max_steps} reached")
                break

            is_loop, loop_reason = detect_dead_loop(self.env.trajectory, self.dead_loop_window)
            if is_loop:
                termination = "dead_loop"
                dead_loop_reason = loop_reason
                self._log(f"\n[ReAct] DEAD LOOP — {loop_reason}")
                break

        # ── Build result ─────────────────────────────────────────────────────
        traj = self.env.get_trajectory()
        result = {
            "goal": goal_item,
            "success": termination == "success",
            "total_steps": traj["total_steps"],
            "termination": termination,
            "trajectory": traj["steps"],
        }
        if termination == "dead_loop":
            result["dead_loop_reason"] = dead_loop_reason

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python react_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    goal = sys.argv[1] if len(sys.argv) > 1 else "wooden_pickaxe"
    agent = ReactAgent()
    result = agent.run_episode(goal)

    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"Goal:        {result['goal']}")
    print(f"Success:     {result['success']}")
    print(f"Steps:       {result['total_steps']}")
    print(f"Termination: {result['termination']}")
    if "dead_loop_reason" in result:
        print(f"Loop reason: {result['dead_loop_reason']}")
    print(f"Trajectory:  {len(result['trajectory'])} steps recorded")
