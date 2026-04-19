import os
import sys
import copy

try:
    from .minecraft_fallback import MinecraftFallbackSimulator
except ImportError:  # pragma: no cover - supports direct src execution
    from minecraft_fallback import MinecraftFallbackSimulator

# Add MC-TextWorld to path
mc_path = os.path.join(os.path.dirname(__file__), '..', '..', 'MC-TextWorld')
sys.path.insert(0, mc_path)

try:
    from mctextworld.simulator import Env
except ImportError:  # pragma: no cover - depends on optional external package
    Env = None


class MCTextWorldWrapper:
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.env = None
        self.goal = None
        self.step_count = 0
        self.trajectory = []
        self._action_lib = None          # lazily cached reference
        self._fallback_simulator = None
        self._fallback_env = False
        self._fallback_obs = None

    def reset(self, goal: str) -> dict:
        """
        Initialize the environment for a specific goal item.
        Returns the initial observation dict.
        """
        self.goal = goal
        self.step_count = 0
        self.trajectory = []
        self._fallback_env = Env is None

        if self._fallback_env:
            self.env = None
            self._fallback_obs = {
                "inventory": {},
                "position": [0, 0, 0],
                "biome": "plains",
            }
            return self._get_obs(self._fallback_obs)

        self.env = Env(
            task_name=f"Obtain 1 {goal}",
            init_inv={},
            task_obj={goal: 1},
        )
        self.env.maximum_step = self.max_steps
        obs, _, _, _ = self.env.reset()
        return self._get_obs(obs)

    def step(self, action: str) -> tuple:
        """
        Execute an action string (e.g. 'craft_iron_pickaxe').
        Returns (obs, done, info).
          - obs:  dict with inventory, position, biome, step, max_steps, goal
          - done: True when goal achieved or max_steps reached
          - info: {'success': bool, 'message': str, 'goal_achieved': bool, 'timeout': bool}
        """
        if self._fallback_env:
            return self._fallback_step(action)

        inv_before = copy.deepcopy(self.env.obs['inventory'])
        obs, _reward, goal_achieved, raw_info = self.env.step(action)
        self.step_count += 1

        success = raw_info.get('action success', False)
        message = raw_info.get(
            'message',
            'Action executed successfully' if success else 'Action failed: preconditions not met'
        )

        # Replace MC-TextWorld's unhelpful "unvalid action!" with a diagnostic
        # that tells the LLM exactly what materials / tools are missing.
        if not success and message in ('unvalid action!', 'Action failed: preconditions not met'):
            message = self._diagnose_failure(action, inv_before)
        timeout = self.step_count >= self.max_steps

        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "success": success,
            "inventory_before": inv_before,
            "inventory_after": copy.deepcopy(obs['inventory']),
            "message": message,
            "done": goal_achieved,
        })

        info = {
            "success": success,
            "message": message,
            "goal_achieved": goal_achieved,
            "timeout": timeout,
        }
        return self._get_obs(obs), goal_achieved or timeout, info

    def get_inventory(self) -> dict:
        """Return current inventory as {item: qty}, excluding zero-quantity entries."""
        if self._fallback_env and self._fallback_obs is not None:
            return {
                k: v for k, v in self._fallback_obs.get('inventory', {}).items() if v > 0
            }
        if self.env is None:
            return {}
        return {k: v for k, v in self.env.obs['inventory'].items() if v > 0}

    def get_trajectory(self) -> dict:
        """Return the full trajectory for the current episode."""
        return {
            "goal": self.goal,
            "max_steps": self.max_steps,
            "total_steps": self.step_count,
            "final_success": any(s["done"] for s in self.trajectory),
            "steps": self.trajectory,
        }

    # ── failure diagnosis ─────────────────────────────────────────────────────

    def _diagnose_failure(self, action: str, inventory: dict) -> str:
        """
        Explain *why* an action failed by comparing its requirements against
        the current inventory.  Returns a human-readable message.
        """
        alib = self._get_action_lib()

        # Case 1: action not in library at all
        if action not in alib:
            return f"Unknown action '{action}' — not in the action library"

        # Case 2: action exists but no variant's requirements are met.
        # Pick the variant with the fewest missing items to give the best hint.
        best_msg = None
        best_missing_count = float('inf')

        for variant in alib[action]:
            missing_mat = {}
            missing_tool = {}

            for item, need in variant.get('precondition', {}).items():
                have = inventory.get(item, 0)
                if have < need:
                    missing_mat[item] = f"need {need}, have {have}"

            for item, need in variant.get('tool', {}).items():
                have = inventory.get(item, 0)
                if have < need:
                    missing_tool[item] = f"need {need}, have {have}"

            n_missing = len(missing_mat) + len(missing_tool)
            if n_missing == 0:
                # Shouldn't reach here (env said it failed) — fall back.
                continue
            if n_missing < best_missing_count:
                best_missing_count = n_missing
                parts = []
                if missing_tool:
                    parts.append("missing tool: " + ", ".join(
                        f"{k} ({v})" for k, v in missing_tool.items()))
                if missing_mat:
                    parts.append("missing materials: " + ", ".join(
                        f"{k} ({v})" for k, v in missing_mat.items()))
                best_msg = f"Failed {action}: " + "; ".join(parts)

        return best_msg or f"Action '{action}' failed (preconditions not met)"

    def _get_action_lib(self) -> dict:
        """Lazily cache a reference to the MC-TextWorld action library dict."""
        if self._action_lib is None:
            if self.env is not None:
                self._action_lib = self.env.action_lib.action_lib
            else:
                self._action_lib = self._get_fallback_simulator().action_library
        return self._action_lib

    def _get_obs(self, raw_obs: dict) -> dict:
        return {
            "inventory": {k: v for k, v in raw_obs['inventory'].items() if v > 0},
            "position": raw_obs.get('position', [0, 0, 0]),
            "biome": raw_obs.get('biome', 'plains'),
            "step": self.step_count,
            "max_steps": self.max_steps,
            "goal": self.goal,
        }

    # ── fallback simulator ───────────────────────────────────────────────────

    def _fallback_step(self, action: str) -> tuple:
        """Execute an action with a lightweight inventory simulator.

        This is used when the optional MC-TextWorld package is not installed,
        for example on an HPC login node where only the repository files were
        pulled. It preserves the high-level action semantics needed for ReAct,
        Reflexion, and NS-FSM comparison smoke tests.
        """
        if self._fallback_obs is None:
            raise RuntimeError("Call reset(goal) before step(action).")

        inv_before = copy.deepcopy(self._fallback_obs["inventory"])
        success, message, inv_after = self._get_fallback_simulator().step(action, inv_before)
        self._fallback_obs["inventory"] = inv_after
        self.step_count += 1

        goal_achieved = inv_after.get(self.goal, 0) >= 1
        timeout = self.step_count >= self.max_steps
        done = goal_achieved or timeout

        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "success": success,
            "inventory_before": inv_before,
            "inventory_after": copy.deepcopy(inv_after),
            "message": message,
            "done": goal_achieved,
        })

        info = {
            "success": success,
            "message": message,
            "goal_achieved": goal_achieved,
            "timeout": timeout,
            "fallback_env": True,
        }
        return self._get_obs(self._fallback_obs), done, info

    def _get_fallback_simulator(self) -> MinecraftFallbackSimulator:
        if self._fallback_simulator is None:
            self._fallback_simulator = MinecraftFallbackSimulator()
        return self._fallback_simulator
