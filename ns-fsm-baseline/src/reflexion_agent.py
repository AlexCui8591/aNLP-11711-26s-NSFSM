"""
Reflexion Agent for MC-TextWorld crafting tasks.

Architecture:
  Outer loop  — run up to `max_attempts` episodes for the same goal.
                After each failed attempt, call the LLM to generate a
                reflection (diagnosis of what went wrong) and pass it
                into the next attempt.

  Inner loop  — identical to ReactAgent but uses REFLEXION_SYSTEM_PROMPT
                and REFLEXION_STEP_PROMPT so the model sees its own
                reflection at every step of the next attempt.

Termination (outer):
  • goal achieved in any attempt  → stop, record success
  • max_attempts exhausted        → record failure

Termination (inner, per attempt):
  • success / max_steps / dead_loop  — same as ReactAgent
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from env_wrapper import MCTextWorldWrapper
from llm_interface import LLMInterface
from action_parser import ActionParser
from react_agent import detect_dead_loop, _DEFAULT_CONFIG
from prompts import (
    build_reflexion_step_prompt,
    build_reflexion_analysis_prompt,
    format_inventory,
)

_REFLECTION_SYSTEM = (
    "You are analyzing a failed Minecraft crafting attempt. "
    "Be concise and specific. Name exact items and correct ordering."
)


# ─────────────────────────────────────────────────────────────────────────────
# ReflexionAgent
# ─────────────────────────────────────────────────────────────────────────────

class ReflexionAgent:
    """
    Multi-attempt Reflexion agent.

    Usage:
        agent = ReflexionAgent()
        result = agent.run("iron_pickaxe")
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
            config_path:      Path to hyperparams.yaml.
            history_window:   Recent steps shown in prompt per step.
            dead_loop_window: Window for dead-loop detection inside each attempt.
            verbose:          Print progress. Reads from config if None.
        """
        config_path = config_path or _DEFAULT_CONFIG
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        max_steps = (
            self._config.get("reflexion", {}).get("max_steps_per_episode")
            or self._config.get("env", {}).get("max_steps_per_episode", 100)
        )
        self.max_attempts = self._config.get("reflexion", {}).get("max_attempts", 3)

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

    def run(self, goal_item: str) -> dict:
        """
        Run up to `max_attempts` episodes for `goal_item`.

        Reflection strategy: only the most recent reflection is injected
        (not accumulated), to avoid context bloat on 7B models.

        Returns:
            {
                "goal":            str,
                "success":         bool,
                "winning_attempt": int | None,   # 1-indexed; None if all failed
                "total_attempts":  int,
                "attempts": [
                    {
                        "attempt":    int,           # 1-indexed
                        "reflection": str,           # reflection fed IN to this attempt
                        "result":     dict,          # episode result (see _run_attempt)
                    },
                    ...
                ],
            }
        """
        self._log(f"\n{'#'*60}")
        self._log(f"[Reflexion] Goal: {goal_item}  |  max_attempts={self.max_attempts}")
        self._log(f"{'#'*60}")

        attempts = []
        reflection = ""          # empty on first attempt
        winning_attempt = None

        for attempt_num in range(1, self.max_attempts + 1):
            self._log(f"\n{'─'*60}")
            self._log(f"[Reflexion] Attempt {attempt_num}/{self.max_attempts}")
            if reflection:
                self._log(f"[Reflexion] Using reflection: {reflection[:200]}")
            self._log(f"{'─'*60}")

            result = self._run_attempt(goal_item, reflection, attempt_num)

            attempts.append({
                "attempt":    attempt_num,
                "reflection": reflection,
                "result":     result,
            })

            if result["success"]:
                winning_attempt = attempt_num
                self._log(
                    f"\n[Reflexion] SUCCESS on attempt {attempt_num} "
                    f"({result['total_steps']} steps)"
                )
                break

            # Failed — generate reflection for next attempt (skip if last)
            if attempt_num < self.max_attempts:
                self._log(f"\n[Reflexion] Attempt {attempt_num} failed "
                          f"({result['termination']}). Generating reflection...")
                reflection = self._generate_reflection(
                    goal_item=goal_item,
                    inventory=self.env.get_inventory(),
                    trajectory=result["trajectory"],
                )
                self._log(f"[Reflexion] Reflection: {reflection[:300]}")
            else:
                self._log(f"\n[Reflexion] All {self.max_attempts} attempts failed.")

        return {
            "goal":            goal_item,
            "success":         winning_attempt is not None,
            "winning_attempt": winning_attempt,
            "total_attempts":  len(attempts),
            "attempts":        attempts,
        }

    # ── inner episode loop ────────────────────────────────────────────────────

    def _run_attempt(self, goal_item: str, reflection: str, attempt_num: int) -> dict:
        """
        Run one episode using reflexion prompts.

        Returns the same schema as ReactAgent.run_episode:
            {
                "goal", "success", "total_steps",
                "termination",     # "success" | "max_steps" | "dead_loop" | "llm_error"
                "dead_loop_reason",  # only present when termination == "dead_loop"
                "trajectory",      # list of step dicts from env_wrapper
            }
        """
        obs = self.env.reset(goal_item)
        termination = "max_steps"
        dead_loop_reason = ""

        while True:
            step = self.env.step_count
            trajectory = self.env.trajectory   # live reference
            inventory = obs["inventory"]

            # ── 1. Build reflexion step prompt ───────────────────────────────
            system_prompt, user_prompt = build_reflexion_step_prompt(
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
                self._log(f"[Reflexion] LLM error at step {step}: {exc}")
                termination = "llm_error"
                break

            thought, _ = self.llm.parse_react_response(response)
            self._log(f"\n  ── Step {step + 1} (attempt {attempt_num}) ──")
            if thought:
                self._log(f"  Thought: {thought[:180]}")

            # ── 3. Parse action ──────────────────────────────────────────────
            action, parse_reason = self.parser.parse(response)

            if action is None:
                _, raw_fallback = self.llm.parse_react_response(response)
                action = raw_fallback or "__invalid__"
                self._log(f"  [Parser] WARN: {parse_reason} → fallback '{action}'")
            else:
                if parse_reason != "ok":
                    self._log(f"  [Parser] {parse_reason}")
                self._log(f"  Action: {action}")

            # ── 4. Env step ──────────────────────────────────────────────────
            obs, done, info = self.env.step(action)
            status_tag = "✓" if info["success"] else "✗"
            self._log(f"  Result: [{status_tag}] {info['message']}")

            # ── 5. Termination checks ────────────────────────────────────────
            if info["goal_achieved"]:
                termination = "success"
                break

            if info["timeout"]:
                termination = "max_steps"
                break

            is_loop, loop_reason = detect_dead_loop(
                self.env.trajectory, self.dead_loop_window
            )
            if is_loop:
                termination = "dead_loop"
                dead_loop_reason = loop_reason
                self._log(f"  [Reflexion] Dead loop: {loop_reason}")
                break

        traj = self.env.get_trajectory()
        result = {
            "goal":        goal_item,
            "success":     termination == "success",
            "total_steps": traj["total_steps"],
            "termination": termination,
            "trajectory":  traj["steps"],
        }
        if termination == "dead_loop":
            result["dead_loop_reason"] = dead_loop_reason

        return result

    # ── reflection generation ─────────────────────────────────────────────────

    def _generate_reflection(
        self,
        goal_item: str,
        inventory: dict,
        trajectory: list,
    ) -> str:
        """
        Ask the LLM to diagnose a failed trajectory and return a reflection string.
        Uses the last 20 steps to keep the prompt within context limits.
        """
        analysis_prompt = build_reflexion_analysis_prompt(
            goal_item=goal_item,
            inventory=inventory,
            trajectory=trajectory,
            last_k=20,
        )
        try:
            raw = self.llm.generate(
                system_prompt=_REFLECTION_SYSTEM,
                user_prompt=analysis_prompt,
            )
            return self.llm.parse_reflection(raw)
        except Exception as exc:
            self._log(f"[Reflexion] Reflection generation failed: {exc}")
            return ""

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test (python reflexion_agent.py [goal])
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    goal = sys.argv[1] if len(sys.argv) > 1 else "wooden_pickaxe"
    agent = ReflexionAgent()
    result = agent.run(goal)

    print(f"\n{'='*60}")
    print("REFLEXION SUMMARY")
    print(f"{'='*60}")
    print(f"Goal:            {result['goal']}")
    print(f"Success:         {result['success']}")
    print(f"Winning attempt: {result['winning_attempt']}")
    print(f"Total attempts:  {result['total_attempts']}")
    for a in result["attempts"]:
        r = a["result"]
        print(
            f"  Attempt {a['attempt']}: "
            f"{'✓' if r['success'] else '✗'}  "
            f"{r['total_steps']} steps  [{r['termination']}]"
        )
