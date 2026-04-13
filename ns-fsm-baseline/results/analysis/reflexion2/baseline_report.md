# Baseline Reproduction

- Generated at: `2026-04-02 21:52:21`
- Inputs: `full/full_v1/reflexion`
- ReAct episodes analyzed: `0`
- Reflexion runs analyzed: `1005`

# Error Analysis

## Reflexion

- Success rate: `266/1005` (26.5%)
- Average steps on success: `31.7`
- Invalid action rate: `2.6%`
- Repeated action rate: `15.3%`
- Dead loop rate: `12.4%`
- Cascade failure rate: `9.1%`
- Plan knowledge error rate: `82.7%`
- Sequencing error rate: `88.2%`

- Performance decay slope: `-0.3011` success-rate points per dependency-depth unit

## Reflexion Fix Rate

- Eligible runs: `771`
- Fixed runs: `32`
- Fix rate: `4.2%`

# Reflection

- The best evidence for structural weakness is the combination of declining success rate with rising dependency depth.
- Plan-knowledge errors indicate recipe confusion or unsupported actions.
- Sequencing errors indicate the agent knows the right recipe family but attempts late-stage actions before satisfying prerequisites.
- Dead loops and cascade failures isolate where plain prompt-based control keeps wasting budget after the first mistake.

# Representative Failures

## Case 1: Reflexion on `blast_furnace`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `65`
- Dominant error: `sequencing_error`
- First failure step: `2`
- First failure: `Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: stick (need 2, have 0), oak_planks (need 3, have 0)`
- Failure trace:
  - Step 2: `craft_wooden_pickaxe` -> Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: stick (need 2, have 0), oak_planks (need 3, have 0)
  - Step 3: `craft_crafting_table` -> Failed craft_crafting_table: missing materials: oak_planks (need 4, have 0)
  - Step 7: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)

## Case 2: Reflexion on `stonecutter`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `8`
- Dominant error: `dead_loop`
- First failure step: `4`
- First failure: `Failed craft_stick: missing materials: oak_planks (need 2, have 0)`
- Failure trace:
  - Step 4: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)
  - Step 5: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)
  - Step 6: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)

## Case 3: Reflexion on `activator_rail`
- Group: `Redstone`
- Termination: `max_steps`
- Steps: `100`
- Dominant error: `plan_knowledge_error`
- First failure step: `4`
- First failure: `Failed craft_crafting_table: missing materials: oak_planks (need 4, have 2)`
- Failure trace:
  - Step 4: `craft_crafting_table` -> Failed craft_crafting_table: missing materials: oak_planks (need 4, have 2)
  - Step 5: `craft_oak_planks` -> Failed craft_oak_planks: missing materials: oak_log (need 1, have 0)
  - Step 6: `craft_crafting_table` -> Failed craft_crafting_table: missing materials: oak_planks (need 4, have 2)

## Case 4: Reflexion on `blast_furnace`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `49`
- Dominant error: `sequencing_error`
- First failure step: `3`
- First failure: `Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: stick (need 2, have 0), oak_planks (need 3, have 0)`
- Failure trace:
  - Step 3: `craft_wooden_pickaxe` -> Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: stick (need 2, have 0), oak_planks (need 3, have 0)
  - Step 6: `craft_wooden_pickaxe` -> Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: oak_planks (need 3, have 2)
  - Step 7: `craft_crafting_table` -> Failed craft_crafting_table: missing materials: oak_planks (need 4, have 2)

## Case 5: Reflexion on `blast_furnace`
- Group: `Iron`
- Termination: `dead_loop`
- Steps: `45`
- Dominant error: `sequencing_error`
- First failure step: `3`
- First failure: `Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: stick (need 2, have 0)`
- Failure trace:
  - Step 3: `craft_wooden_pickaxe` -> Failed craft_wooden_pickaxe: missing tool: crafting_table (need 1, have 0); missing materials: stick (need 2, have 0)
  - Step 5: `craft_wooden_pickaxe` -> Failed craft_wooden_pickaxe: missing materials: stick (need 2, have 0), oak_planks (need 3, have 0)
  - Step 6: `craft_stick` -> Failed craft_stick: missing materials: oak_planks (need 2, have 0)
