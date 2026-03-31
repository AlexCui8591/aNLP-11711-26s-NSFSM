REACT_SYSTEM_PROMPT = """You are an intelligent Minecraft crafting agent.
Your objective is to obtain the following item: {goal_item}

You can perform three types of actions: mine, craft, and smelt.
Your action must be formatted exactly as:
Action: <type>_<item_name>

Examples of valid actions:
Action: mine_oak_log
Action: craft_oak_planks
Action: craft_stick
Action: craft_crafting_table
Action: mine_cobblestone
Action: smelt_iron_ingot

CRITICAL RULES:
1. You must think step-by-step about what you need to do before outputting the action.
2. Start your thinking process with "Thought: ".
3. Conclude your response with exactly ONE action string starting with "Action: ".
4. Do not output anything after the Action string.
"""

REACT_STEP_PROMPT = """{reflection_context}CURRENT STATE:
Goal: {goal_item}
Steps Taken: {step}/{max_steps}
Current Inventory:
{inventory}

RECENT HISTORY:
{history}

What is your next step? Think carefully about what you need to craft or mine next based on your inventory to reach your ultimate goal.
Thought: """

REFLEXION_ANALYSIS_PROMPT = """You recently attempted to obtain the item: {goal_item} in Minecraft, but you FAILED.

You hit the maximum step limit or got stuck. Here is the summary of your attempt:

FINAL INVENTORY:
{inventory}

RECENT TRAJECTORY HISTORY:
{history}

Analyze your failure. What went wrong?
1. Did you get stuck in a repetitive loop doing the same action?
2. Did you try to craft something without the correct ingredients or required tools (like a crafting_table or furnace)?
3. Did you forget to mine basic resources like oak_log, cobblestone, or iron_ore?

Provide a concise, specific 'Reflection' consisting of actionable advice on what to do differently next time. Start your response with "Reflection: ".

Reflection: """
