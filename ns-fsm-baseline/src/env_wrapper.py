import os
import sys
import copy

# Add MC-TextWorld to path
mc_path = os.path.join(os.path.dirname(__file__), '..', '..', 'MC-TextWorld')
sys.path.insert(0, mc_path)

from mctextworld.simulator import Env


class MCTextWorldWrapper:
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.env = None
        self.goal = None
        self.step_count = 0
        self.trajectory = []

    def reset(self, goal: str) -> dict:
        """
        Initialize the environment for a specific goal item.
        Returns the initial observation dict.
        """
        self.goal = goal
        self.step_count = 0
        self.trajectory = []
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
        inv_before = copy.deepcopy(self.env.obs['inventory'])
        obs, _reward, goal_achieved, raw_info = self.env.step(action)
        self.step_count += 1

        success = raw_info.get('action success', False)
        message = raw_info.get(
            'message',
            'Action executed successfully' if success else 'Action failed: preconditions not met'
        )
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

    def _get_obs(self, raw_obs: dict) -> dict:
        return {
            "inventory": {k: v for k, v in raw_obs['inventory'].items() if v > 0},
            "position": raw_obs.get('position', [0, 0, 0]),
            "biome": raw_obs.get('biome', 'plains'),
            "step": self.step_count,
            "max_steps": self.max_steps,
            "goal": self.goal,
        }
