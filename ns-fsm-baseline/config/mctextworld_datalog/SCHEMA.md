# MC-TextWorld Datalog Export

All `.facts` files are UTF-8, tab-separated, and headerless.

## Action domain facts

- `action.facts`: `aid`
- `action_type.facts`: `aid`, `type`
- `recipe.facts`: `rid`, `aid`
- `recipe_order.facts`: `rid`, `index`
- `precondition.facts`: `rid`, `item`, `count`
- `tool_need.facts`: `rid`, `tool`, `count`
- `output.facts`: `rid`, `item`, `count`
- `item.facts`: `item`
- `produces.facts`: `aid`, `item`, `count`

`rid` is generated as `<action_id>__<variant_index>`.

## Task and plan facts

- `task.facts`: `task_id`
- `task_group.facts`: `task_id`, `group`
- `task_biome.facts`: `task_id`, `biome`
- `goal.facts`: `task_id`, `item`, `count`
- `init_has.facts`: `task_id`, `item`, `count`
- `plan_task.facts`: `task_id`
- `plan_step.facts`: `task_id`, `step_index`, `action_id`
- `plan_subgoal.facts`: `task_id`, `step_index`, `item`, `count`

`plans.json` steps are high-level controller steps.  The optional replay
validator follows the plan-pointer behavior in `mctextworld/run.py`: before
each action, it checks the current step subgoal and advances the pointer by one
when that subgoal is already satisfied.
