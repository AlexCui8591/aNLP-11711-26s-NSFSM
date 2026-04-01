# Baseline Reproduction

- Generated at: `2026-03-31 16:42:52`
- Inputs: `ns-fsm-baseline/results/trajectories`
- ReAct episodes analyzed: `16`
- Reflexion runs analyzed: `0`

# Error Analysis

## React

- Success rate: `13/16` (81.2%)
- Average steps on success: `24.8`
- Invalid action rate: `12.6%`
- Repeated action rate: `22.7%`
- Dead loop rate: `31.2%`
- Cascade failure rate: `33.3%`
- Plan knowledge error rate: `100.0%`
- Sequencing error rate: `66.7%`

- Performance decay slope: `-0.1663` success-rate points per dependency-depth unit

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
