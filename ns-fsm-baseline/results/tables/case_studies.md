## Case 1: React on `blast_furnace`
- Group: `Iron`
- Termination: `max_steps`
- Steps: `100`
- Dominant error: `plan_knowledge_error`
- First failure step: `1`
- First failure: `Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)`
- Failure trace:
  - Step 1: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
  - Step 2: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
  - Step 3: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)

## Case 2: React on `bowl`
- Group: `Wooden`
- Termination: `dead_loop`
- Steps: `5`
- Dominant error: `dead_loop`
- First failure step: `1`
- First failure: `action mine_clay NOT exist!'`
- Failure trace:
  - Step 1: `mine_clay` -> action mine_clay NOT exist!'
  - Step 2: `mine_clay` -> action mine_clay NOT exist!'
  - Step 3: `mine_clay` -> action mine_clay NOT exist!'

## Case 3: React on `torch`
- Group: `Stone`
- Termination: `dead_loop`
- Steps: `48`
- Dominant error: `plan_knowledge_error`
- First failure step: `4`
- First failure: `Failed craft_torch: missing materials: coal (need 1, have 0)`
- Failure trace:
  - Step 4: `craft_torch` -> Failed craft_torch: missing materials: coal (need 1, have 0)
  - Step 5: `mine_coal_ore` -> action mine_coal_ore NOT exist!'
  - Step 6: `mine_cobblestone` -> Failed mine_cobblestone: missing tool: wooden_pickaxe (need 1, have 0)
