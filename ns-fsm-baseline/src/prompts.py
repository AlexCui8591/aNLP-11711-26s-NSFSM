# =============================================================================
# ReAct Prompts
# =============================================================================

REACT_SYSTEM_PROMPT = """You are an expert Minecraft crafting agent. Your goal is to obtain: {goal_item}

## Action Types
You can take exactly three types of actions:
- **mine_<item>**: Gather a raw resource directly from the world (e.g., mine_oak_log, mine_cobblestone, mine_iron_ore)
- **craft_<item>**: Combine inventory items to produce a new item (e.g., craft_oak_planks, craft_stick, craft_wooden_pickaxe)
- **smelt_<item>**: Use a furnace to process an item with fuel (e.g., smelt_iron_ingot, smelt_stone)

## Crafting Dependency Chains
Many items require intermediate steps. You must obtain ALL prerequisite items before crafting the final goal.
Example chain for a wooden_pickaxe:
  Step 1: mine_oak_log          → inventory gains oak_log
  Step 2: craft_oak_planks      → consumes oak_log, gains oak_planks
  Step 3: craft_stick           → consumes oak_planks, gains stick
  Step 4: craft_crafting_table  → consumes oak_planks, gains crafting_table
  Step 5: craft_wooden_pickaxe  → consumes stick + oak_planks + crafting_table, gains wooden_pickaxe

## Critical Tool Requirements
- Most `craft_` actions (beyond basic planks/sticks) require a **crafting_table** in your inventory.
- All `smelt_` actions require a **furnace** in your inventory.
- If an action fails, check whether you have the required tool and all ingredients.

## Output Format (strictly follow this)
1. Begin your response with "Thought: " — reason about what you currently have, what you still need, and why the next action is correct.
2. End your response with exactly ONE line: "Action: <type>_<item_name>"
3. Do NOT write anything after the Action line.

Valid format example:
Thought: I need a wooden_pickaxe. I already have sticks and planks, and I have a crafting_table. I have all prerequisites, so I can craft it now.
Action: craft_wooden_pickaxe

Invalid (never do this):
Action: craft wooden pickaxe
Action: get_oak_log
craft_stick
"""


REACT_STEP_PROMPT = """{reflection_context}== CURRENT STATE ==
Goal:         {goal_item}
Step:         {step} / {max_steps}

Inventory:
{inventory}

== RECENT ACTION HISTORY ==
{history}

== LAST ACTION RESULT ==
{last_result}

Based on your current inventory, reason step-by-step about what you still need to obtain {goal_item} and choose the single best next action.
Thought: """


# =============================================================================
# Reflexion Prompts
# =============================================================================

REFLEXION_SYSTEM_PROMPT = """You are an expert Minecraft crafting agent with self-reflection capability. Your goal is to obtain: {goal_item}

## Action Types
- **mine_<item>**: Gather a raw resource from the world (e.g., mine_oak_log, mine_iron_ore, mine_cobblestone)
- **craft_<item>**: Combine inventory items into a new item (e.g., craft_oak_planks, craft_crafting_table)
- **smelt_<item>**: Heat an item in a furnace (e.g., smelt_iron_ingot)

## Crafting Dependency Chains
You must build up items from scratch. Every complex item requires prerequisite items.
Example (iron_pickaxe):
  mine_oak_log → craft_oak_planks → craft_stick
  mine_oak_log → craft_oak_planks → craft_crafting_table
  mine_iron_ore + furnace → smelt_iron_ingot (×3)
  craft_iron_pickaxe (needs: 3×iron_ingot, 2×stick, crafting_table)

## Critical Tool Requirements
- Most `craft_` actions need a **crafting_table** in inventory.
- All `smelt_` actions need a **furnace** in inventory.

## Self-Reflection Guidance
You have been provided with a reflection from a previous failed attempt. Use it to:
1. Avoid repeating the same mistakes (loops, wrong order, missing tools).
2. Identify which prerequisite you skipped and handle it first.
3. Adjust your step-by-step plan accordingly.

## Output Format
Thought: <your step-by-step reasoning>
Action: <type>_<item_name>

Do NOT write anything after the Action line.
"""


REFLEXION_STEP_PROMPT = """{reflection_context}== CURRENT STATE ==
Goal:         {goal_item}
Step:         {step} / {max_steps}

Inventory:
{inventory}

== RECENT ACTION HISTORY ==
{history}

== LAST ACTION RESULT ==
{last_result}

Remember your reflection above. Reason about what you still need and pick the single best next action.
Thought: """


REFLEXION_ANALYSIS_PROMPT = """You attempted to craft {goal_item} in Minecraft but FAILED (hit the step limit or got stuck).

== FINAL INVENTORY ==
{inventory}

== ACTION TRAJECTORY ==
{history}

== FAILURE ANALYSIS ==
Answer each question briefly, then write your reflection:

1. **Loop detection**: Did you repeat the same action 3+ times in a row without gaining anything new?
2. **Missing prerequisite**: Did you try to craft something before you had all required items or tools (crafting_table / furnace)?
3. **Wrong order**: Did you attempt a late-stage craft before completing an early-stage dependency?
4. **Resource shortfall**: Did you run out of a raw material (e.g., iron_ore, oak_log) mid-chain?

Based on this analysis, write a single, concise, actionable reflection that tells a future attempt exactly what to do differently. Be specific (mention exact item names and correct ordering).

Start with "Reflection: " on a new line.

Reflection: """


# =============================================================================
# Helper: format inventory dict → readable string
# =============================================================================

def format_inventory(inventory: dict) -> str:
    """
    Convert {item: qty} dict to a human-readable bullet list.
    Returns '(empty)' if inventory is empty.
    """
    if not inventory:
        return "  (empty)"
    lines = [f"  {item}: {qty}" for item, qty in sorted(inventory.items()) if qty > 0]
    return "\n".join(lines) if lines else "  (empty)"


def format_history(trajectory: list, last_k: int = 10) -> str:
    """
    Format the last `last_k` steps of a trajectory for prompt injection.
    Each step shows the action taken, whether it succeeded, and inventory changes.
    """
    if not trajectory:
        return "  (no actions taken yet)"

    recent = trajectory[-last_k:]
    lines = []
    for entry in recent:
        status = "OK" if entry.get("success") else "FAILED"
        action = entry.get("action", "?")
        msg = entry.get("message", "")
        inv_before = entry.get("inventory_before", {})
        inv_after = entry.get("inventory_after", {})

        # Show items gained/lost
        gained = {k: inv_after.get(k, 0) - inv_before.get(k, 0)
                  for k in set(inv_after) | set(inv_before)
                  if inv_after.get(k, 0) != inv_before.get(k, 0)}
        delta_str = ""
        if gained:
            parts = []
            for k, diff in gained.items():
                if diff > 0:
                    parts.append(f"+{diff} {k}")
                else:
                    parts.append(f"{diff} {k}")
            delta_str = f" [{', '.join(parts)}]"

        lines.append(f"  Step {entry['step']:>3}: [{status}] {action}{delta_str}")
        if not entry.get("success") and msg:
            lines.append(f"            Reason: {msg}")

    return "\n".join(lines)


def build_react_prompt(
    goal_item: str,
    inventory: dict,
    trajectory: list,
    step: int,
    max_steps: int,
    reflection: str = "",
    last_k: int = 10,
) -> tuple:
    """
    Build (system_prompt, user_prompt) for a ReAct step.

    Args:
        goal_item:   The target item to craft/obtain.
        inventory:   Current inventory dict {item: qty}.
        trajectory:  List of step dicts from MCTextWorldWrapper.
        step:        Current step number.
        max_steps:   Episode step limit.
        reflection:  Optional reflection string from a previous failed attempt.
        last_k:      How many recent steps to show in history.

    Returns:
        (system_str, user_str) ready to pass to LLMInterface.
    """
    reflection_context = ""
    if reflection:
        reflection_context = f"== REFLECTION FROM PREVIOUS ATTEMPT ==\n{reflection}\n\n"

    last_result = "(start of episode)"
    if trajectory:
        last_entry = trajectory[-1]
        status = "SUCCESS" if last_entry.get("success") else "FAILED"
        last_result = f'[{status}] {last_entry["action"]} — {last_entry.get("message", "")}'

    system = REACT_SYSTEM_PROMPT.format(goal_item=goal_item)
    user = REACT_STEP_PROMPT.format(
        reflection_context=reflection_context,
        goal_item=goal_item,
        step=step,
        max_steps=max_steps,
        inventory=format_inventory(inventory),
        history=format_history(trajectory, last_k=last_k),
        last_result=last_result,
    )
    return system, user


def build_reflexion_step_prompt(
    goal_item: str,
    inventory: dict,
    trajectory: list,
    step: int,
    max_steps: int,
    reflection: str = "",
    last_k: int = 10,
) -> tuple:
    """
    Build (system_prompt, user_prompt) for a Reflexion agent step.
    Same as ReAct but uses the Reflexion system prompt and REFLEXION_STEP_PROMPT.
    """
    reflection_context = ""
    if reflection:
        reflection_context = f"== REFLECTION FROM PREVIOUS ATTEMPT ==\n{reflection}\n\n"

    last_result = "(start of episode)"
    if trajectory:
        last_entry = trajectory[-1]
        status = "SUCCESS" if last_entry.get("success") else "FAILED"
        last_result = f'[{status}] {last_entry["action"]} — {last_entry.get("message", "")}'

    system = REFLEXION_SYSTEM_PROMPT.format(goal_item=goal_item)
    user = REFLEXION_STEP_PROMPT.format(
        reflection_context=reflection_context,
        goal_item=goal_item,
        step=step,
        max_steps=max_steps,
        inventory=format_inventory(inventory),
        history=format_history(trajectory, last_k=last_k),
        last_result=last_result,
    )
    return system, user


def build_reflexion_analysis_prompt(
    goal_item: str,
    inventory: dict,
    trajectory: list,
    last_k: int = 20,
) -> str:
    """
    Build the failure-analysis prompt for Reflexion.
    Returns a single prompt string (no system message needed).
    """
    return REFLEXION_ANALYSIS_PROMPT.format(
        goal_item=goal_item,
        inventory=format_inventory(inventory),
        history=format_history(trajectory, last_k=last_k),
    )


# =============================================================================
# NS-FSM Prompts
# =============================================================================

NSFSM_SYSTEM_PROMPT = """You are an NS-FSM controlled agent.

You must obey the externally verified workflow FSM.
First propose exactly one action and one intended next FSM state.
Your proposal will be checked after generation against the current allowed
runtime action list and the workflow FSM.
For most datasets the output action is a canonical FSM transition action.
For Robotouille, the FSM state is a high-level subgoal and the output action
must be one currently executable simulator primitive, such as move/pick-up/
unstack/stack/cut/cook. Do not output a high-level Robotouille FSM action
unless it is explicitly listed as an action you may output now.
If your action is outside the current allowed action list, the verifier will
block it and ask for a new proposal. After repeated failures, Robotouille uses
a simulator-valid primitive fallback, not a forced high-level FSM action.
Copy action strings exactly from the current allowed action list whenever
possible.
Do not invent tools, actions, or next states.

Output format:
Thought: <brief reasoning>
Action: <one proposed action>
Next State: <one valid next state>
"""


NSFSM_STEP_PROMPT = """== TASK ==
Dataset: {dataset}
Task ID: {task_id}
Task Type: {task_type}
Instruction:
{instruction}

== CURRENT FSM ==
Current workflow phase:
{fsm_state}

Validated FSM summary:
{fsm_summary}

== CURRENT ADAPTER STATE ==
{adapter_state_summary}

== RECENT ACTION HISTORY ==
{recent_history}

== BLOCKED ACTION HISTORY ==
{blocked_history}

== LAST VERIFICATION ERROR ==
{last_verification_error}

== FALLBACK HISTORY ==
{fallback_history}

== COMPRESSED MEMORY ==
{compressed_memory}

== ACTIONS YOU MAY OUTPUT NOW ==
{legal_actions}

== FSM TRANSITIONS / HIGH-LEVEL SUBGOALS: T(current_state) ==
{transition_options}

== SIMULATOR EXECUTABLE ACTIONS ==
{executable_actions}

== VERIFIED RUNTIME ACTIONS ==
{verified_actions}

Choose from ACTIONS YOU MAY OUTPUT NOW exactly. For Robotouille, those actions
are simulator primitive actions while the FSM transitions describe the current
high-level phase/subgoal; do not send semantic FSM actions such as
pick-up-item|item=... unless they appear in ACTIONS YOU MAY OUTPUT NOW. If
blocked, propose a different currently listed action. Robotouille recovery must
stay in the simulator primitive action space.

Choose exactly one action and the intended next FSM state.
Thought: """


def build_nsfsm_prompt(context_packet: dict) -> tuple:
    """Build (system_prompt, user_prompt) for one NS-FSM decision."""

    task = context_packet.get("task", {})
    user = NSFSM_STEP_PROMPT.format(
        dataset=task.get("dataset", ""),
        task_id=task.get("task_id", ""),
        task_type=task.get("task_type", ""),
        instruction=task.get("instruction", ""),
        fsm_state=context_packet.get("fsm_state", ""),
        fsm_summary=context_packet.get("fsm_summary", ""),
        adapter_state_summary=context_packet.get("adapter_state_summary", ""),
        recent_history=format_nsfsm_records(context_packet.get("recent_history", [])),
        blocked_history=format_nsfsm_records(context_packet.get("blocked_history", [])),
        last_verification_error=format_verification_error(
            context_packet.get("last_verification_error", {})
        ),
        fallback_history=format_nsfsm_records(context_packet.get("fallback_history", [])),
        compressed_memory=context_packet.get("compressed_memory", "(none)"),
        legal_actions=format_nsfsm_actions(context_packet.get("legal_actions", [])),
        transition_options=format_transition_options(
            context_packet.get("transition_options", [])
        ),
        executable_actions=format_nsfsm_actions(context_packet.get("executable_actions", [])),
        verified_actions=format_nsfsm_actions(context_packet.get("verified_actions", [])),
    )
    return NSFSM_SYSTEM_PROMPT, user


def format_nsfsm_actions(actions: list) -> str:
    if not actions:
        return "  (no legal actions)"
    return "\n".join(f"  - {action}" for action in actions)


def format_transition_options(options: list) -> str:
    if not options:
        return "  (no valid transitions)"
    lines = []
    for option in options:
        action = option.get("action", "?")
        next_state = option.get("next_state", "?")
        condition = option.get("condition", "")
        suffix = f" | condition: {condition}" if condition else ""
        lines.append(f"  - action={action} -> next_state={next_state}{suffix}")
    return "\n".join(lines)


def format_verification_error(error: dict) -> str:
    if not error:
        return "  (none)"
    proposal = error.get("proposal", {})
    rule = error.get("rule_check", {})
    transition = error.get("transition_check", {})
    lines = [
        f"  rejected_action: {proposal.get('action')}",
        f"  rejected_next_state: {proposal.get('next_state')}",
        f"  rule_reason: {rule.get('reason_type')} - {rule.get('message')}",
    ]
    violations = transition.get("violations") or []
    if violations:
        lines.append(f"  transition_violations: {violations}")
    lines.append("  Propose a different action that is in T(current_state) and compatible with the FSM transition.")
    return "\n".join(lines)


def format_nsfsm_records(records: list, limit: int = 8) -> str:
    if not records:
        return "  (none)"
    lines = []
    for entry in records[-limit:]:
        step = entry.get("step", "?")
        action = entry.get("action", "?")
        success = entry.get("success")
        status = "OK" if success is True else "FAILED" if success is False else "?"
        source = entry.get("source") or entry.get("decision_source") or ""
        source_text = f" source={source}" if source else ""
        message = entry.get("message") or entry.get("info", {}).get("message", "")
        lines.append(f"  Step {step}: [{status}] {action}{source_text}")
        if message:
            lines.append(f"    message: {message}")
    return "\n".join(lines)
