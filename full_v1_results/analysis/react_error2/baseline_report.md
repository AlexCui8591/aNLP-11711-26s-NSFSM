# Baseline Reproduction

- Generated at: `2026-04-01 23:01:27`
- Inputs: `full_v1_results/full_v1/react`
- ReAct episodes analyzed: `896`
- Reflexion runs analyzed: `0`

# Error Analysis

## React

- Success rate: `114/896` (12.7%)
- Average steps on success: `20.5`
- Invalid action rate: `5.7%`
- Repeated action rate: `17.6%`
- Dead loop rate: `13.2%`
- Cascade failure rate: `17.4%`
- Plan knowledge error rate: `90.2%`
- Sequencing error rate: `71.7%`

- Performance decay slope: `-0.3044` success-rate points per dependency-depth unit

# Reflection

- The best evidence for structural weakness is the combination of declining success rate with rising dependency depth.
- Plan-knowledge errors indicate recipe confusion or unsupported actions.
- Sequencing errors indicate the agent knows the right recipe family but attempts late-stage actions before satisfying prerequisites.
- Dead loops and cascade failures isolate where plain prompt-based control keeps wasting budget after the first mistake.

# Representative Failures

## Case 1: React on `blast_furnace`
- Group: `Iron`
- Termination: `max_steps`
- Steps: `100`
- Dominant error: `dead_loop`
- First failure step: `1`
- First failure: `Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)`
- Failure trace:
  - Step 1: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
  - Step 4: `craft_crafting_table` -> Failed craft_crafting_table: missing materials: oak_planks (need 4, have 0)
  - Step 9: `craft_stone_pickaxe` -> Failed craft_stone_pickaxe: missing materials: stick (need 2, have 0), cobblestone (need 3, have 0)

## Case 2: React on `blast_furnace`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `39`
- Dominant error: `sequencing_error`
- First failure step: `1`
- First failure: `Failed mine_iron_ore: missing tool: stone_pickaxe (need 1, have 0)`
- Failure trace:
  - Step 1: `mine_iron_ore` -> Failed mine_iron_ore: missing tool: stone_pickaxe (need 1, have 0)
  - Step 2: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
  - Step 4: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)

## Case 3: React on `stonecutter`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `62`
- Dominant error: `plan_knowledge_error`
- First failure step: `1`
- First failure: `Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)`
- Failure trace:
  - Step 1: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
  - Step 6: `craft_stone_pickaxe` -> Failed craft_stone_pickaxe: missing materials: stick (need 2, have 0), cobblestone (need 3, have 0)
  - Step 7: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)

## Case 4: React on `wooden_shovel`
- Group: `Wooden`
- Termination: `llm_error`
- Steps: `0`
- Dominant error: `none`

## Case 5: React on `blast_furnace`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `36`
- Dominant error: `sequencing_error`
- First failure step: `1`
- First failure: `Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)`
- Failure trace:
  - Step 1: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
  - Step 5: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)
  - Step 6: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)
