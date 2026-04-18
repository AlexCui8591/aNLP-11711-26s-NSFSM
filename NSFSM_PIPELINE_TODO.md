# NS-FSM Multi-Benchmark Pipeline - Implementation Plan (v1)

## Background

The current repository already supports MC-TextWorld baseline experiments with ReAct
and Reflexion. The next step is to build a basic NS-FSM pipeline that can run in
two situations:

```text
1. A user gives an arbitrary natural-language scenario.
2. A benchmark provides structured tasks and an evaluator.
```

This means the pipeline must be generic first. Benchmark-specific adapters are
still useful, but they should be optional extensions for reproducible evaluation,
not required for the system to run.

The key requirement is:

```text
The task is variable.
The benchmark may change.
The user may provide only a natural-language scenario.
The pipeline should create the task-specific FSM/action list at runtime.
```

Therefore, we should not hand-write one FSM per task or require a custom adapter
for every new scenario. Instead, we should build:

```text
User scenario or raw benchmark task
-> GenericScenarioAdapter or benchmark adapter converts it into TaskSpec
-> LLM FSM designer proposes task-specific states, legal actions, and transitions
-> FSM validator checks the proposed FSM schema and safety constraints
-> FSMBuilder uses the validated FSM, or falls back to GenericToolUseFSM
-> RuleChecker creates dynamic legal action list
-> NSFSMAgent chooses, blocks, falls back, and executes actions
-> Runner saves results in a common format
-> Analysis compares scenarios, datasets, and agents
```

## Confirmed Design Choices

| Item | Decision |
| --- | --- |
| Core idea | Use reusable FSM templates, not per-task hand-written FSMs |
| Default mode | Generic scenario mode for arbitrary user-provided tasks |
| Benchmark mode | Dataset adapters for reproducible benchmark execution |
| FSM type | Workflow-control FSM plus dataset-specific rule checker |
| FSM creation | LLM-assisted FSM proposal, then deterministic validation |
| Formal verification | Datalog validates both whole FSM and per-state transitions |
| Legal actions | `T(s)` generated for the current state and verified before execution |
| Adapter priority | Build `generic` first, then benchmark adapters |
| Safety fallback | Use fixed `GenericToolUseFSM` if LLM-generated FSM is invalid |
| Minecraft support | Optional benchmark adapter, not hard-coded in core |
| SWE-bench support | Optional code-repair adapter for formal evaluation |
| Training | Out of scope for basic pipeline; add after inference pipeline works |
| Output schema | Keep compatible with existing ReAct/Reflexion result JSONs |

## Support Levels

The pipeline should support three levels of task structure.

| Level | Input | Adapter | Reliability | Use case |
| --- | --- | --- | --- | --- |
| Level 0 | Natural-language task only | `GenericScenarioAdapter` | Flexible, less benchmark-rigorous | User says "run this scenario" |
| Level 1 | Task + action list + success condition | `StructuredScenarioAdapter` | More controllable | User provides a lightweight schema |
| Level 2 | Official benchmark record | Benchmark adapter | Reproducible and evaluable | SWE-bench, Minecraft, WebShop, other tool-use benchmarks |

The basic MVP must complete Level 0 first.

## Target Architecture

```text
ns-fsm-baseline/
├── config/
│   ├── datasets.yaml
│   ├── scenarios/
│   │   └── examples.json
│   └── datasets/
│       ├── minecraft/
│       │   ├── tasks.json
│       │   └── transitions.json
│       └── swe_bench/
│           ├── tasks.json
│           └── workflow.json
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── generic.py
│   │   ├── structured.py
│   │   ├── minecraft.py
│   │   └── swe_bench.py
│   ├── fsm_templates/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── code_repair.py
│   │   ├── symbolic_planning.py
│   │   └── generic_tool_use.py
│   ├── fsm.py
│   ├── fsm_designer.py
│   ├── fsm_builder.py
│   ├── fsm_validator.py
│   ├── datalog_verifier.py
│   ├── context_manager.py
│   ├── rule_checker.py
│   ├── planner.py
│   ├── nsfsm_agent.py
│   └── prompts.py
├── scripts/
│   ├── run_scenario.py
│   ├── build_task_specs.py
│   ├── run_nsfsm_experiment.py
│   └── analyze_nsfsm.py
└── results/
    ├── full/
    └── analysis/
```

## Core Abstractions

### 1. TaskSpec

Every user scenario or benchmark instance should be converted into this common
format.

```python
TaskSpec = {
    "dataset": str,
    "task_id": str,
    "task_type": str,
    "instruction": str,
    "initial_state": dict,
    "goal_condition": dict,
    "available_tools": list[str],
    "max_steps": int,
    "success_criteria": list[str],
    "metadata": dict,
}
```

Examples of `task_type`:

```text
symbolic_planning
code_repair
web_navigation
generic_tool_use
```

Level 0 generic scenario example:

```python
{
    "dataset": "generic",
    "task_id": "scenario/user_task_001",
    "task_type": "generic_tool_use",
    "instruction": "Analyze this repository and propose an experiment pipeline.",
    "initial_state": {
        "task_read": False,
        "context_gathered": False,
        "plan_created": False,
        "steps_completed": [],
        "checked": False,
        "finalized": False,
    },
    "goal_condition": {"finalized": True},
    "available_tools": [
        "read_task",
        "inspect_context",
        "ask_clarifying_question",
        "write_plan",
        "execute_step",
        "check_progress",
        "revise",
        "finalize",
    ],
    "max_steps": 30,
    "success_criteria": ["A final answer or artifact is produced."],
    "metadata": {"source": "user_instruction"},
}
```

### 2. FSMDesign

For arbitrary user scenarios, the system should call the LLM API once at task
setup time to propose a task-specific FSM. The LLM output must be JSON and must
match this schema.

```python
FSMDesign = {
    "states": list[str],
    "initial_state": str,
    "terminal_states": list[str],
    "transitions_by_state": dict[str, list[dict]],
    "fallback_policy": dict,
    "success_signals": list[str],
    "risk_notes": list[str],
}
```

`transitions_by_state` is the canonical source. The code should derive
`actions`, `legal_actions_by_state`, flat `transition(...)` facts, and `T(s)`
from it.

Transition option schema:

```python
{
    "action": str,
    "next_state": str,
    "condition": str,
}
```

Example LLM-generated FSM for a generic planning task:

```python
{
    "states": [
        "START",
        "UNDERSTAND_TASK",
        "GATHER_CONTEXT",
        "DRAFT_PLAN",
        "CHECK_PLAN",
        "REVISE_PLAN",
        "FINALIZE",
        "DONE"
    ],
    "initial_state": "START",
    "terminal_states": ["DONE"],
    "transitions_by_state": {
        "START": [
            {"action": "read_task", "next_state": "UNDERSTAND_TASK", "condition": "task is read"}
        ],
        "UNDERSTAND_TASK": [
            {"action": "inspect_context", "next_state": "GATHER_CONTEXT", "condition": "more context needed"},
            {"action": "write_plan", "next_state": "DRAFT_PLAN", "condition": "enough context exists"}
        ],
        "GATHER_CONTEXT": [
            {"action": "inspect_context", "next_state": "GATHER_CONTEXT", "condition": "more context is still needed"},
            {"action": "write_plan", "next_state": "DRAFT_PLAN", "condition": "context gathered"}
        ],
        "DRAFT_PLAN": [
            {"action": "check_progress", "next_state": "CHECK_PLAN", "condition": "draft exists"}
        ],
        "CHECK_PLAN": [
            {"action": "revise", "next_state": "REVISE_PLAN", "condition": "draft is incomplete"},
            {"action": "finalize", "next_state": "DONE", "condition": "draft satisfies task"}
        ],
        "REVISE_PLAN": [
            {"action": "write_plan", "next_state": "DRAFT_PLAN", "condition": "revision should update the draft"}
        ],
        "FINALIZE": [
            {"action": "finalize", "next_state": "DONE", "condition": "final output produced"}
        ],
        "DONE": []
    },
    "fallback_policy": {
        "on_invalid_action": "choose first legal action for current state",
        "on_dead_end": "fall back to GenericToolUseFSM"
    },
    "success_signals": ["final output produced", "user-facing artifact exists"],
    "risk_notes": ["FSM is workflow-level, not benchmark-grade evaluator"]
}
```

Derived structures:

```python
actions = unique transition["action"] across transitions_by_state
legal_actions_by_state[state] = [transition["action"] for transition in transitions_by_state[state]]
transition_facts = [
    (state, transition["action"], transition["next_state"])
    for state, transitions in transitions_by_state.items()
    for transition in transitions
]
T(state) = [
    (transition["action"], transition["next_state"])
    for transition in transitions_by_state[state]
]
```

### 3. Datalog Verification Model

The LLM-generated `FSMDesign` should be translated into Datalog facts. Datalog
then derives reachability, semantic consistency, per-state action lists `T(s)`,
and valid next-state transitions.

Core facts:

```prolog
state(S).
action(A).
initial(S).
terminal(S).
legal(S, A).
transition(S1, A, S2).
current(S).
proposed_action(A).
proposed_next_state(S).
adapter_action(A).
requires(A, P).
provides_state_var(S, V).
updates(A, V).
must_happen_before(A1, A2).
```

Derived predicates:

```prolog
reachable(S) :- initial(S).
reachable(S2) :- reachable(S1), transition(S1, A, S2).

can_reach_terminal(S) :- terminal(S).
can_reach_terminal(S1) :- transition(S1, A, S2), can_reach_terminal(S2).

valid_transition(S1, A, S2) :-
    state(S1), state(S2), action(A), legal(S1, A), transition(S1, A, S2).

valid_current_action(A) :-
    current(S), legal(S, A).

valid_current_transition(S2) :-
    current(S1), proposed_action(A), proposed_next_state(S2),
    transition(S1, A, S2), legal(S1, A).
```

Violation predicates:

```prolog
violation(unknown_source_state, S1) :-
    transition(S1, A, S2), not state(S1).

violation(unknown_target_state, S2) :-
    transition(S1, A, S2), not state(S2).

violation(unknown_action, A) :-
    legal(S, A), not action(A).

violation(action_not_allowed_by_adapter, A) :-
    action(A), not adapter_action(A).

violation(illegal_transition_action, S, A) :-
    transition(S, A, S2), not legal(S, A).

violation(invalid_current_action, A) :-
    proposed_action(A), not valid_current_action(A).

violation(invalid_next_state, S2) :-
    proposed_next_state(S2), not valid_current_transition(S2).

violation(unreachable_state, S) :-
    state(S), not reachable(S).

violation(dead_end_state, S) :-
    state(S), not terminal(S), not can_reach_terminal(S).

violation(no_legal_action, S) :-
    state(S), not terminal(S), not legal(S, A).

violation(premature_finalize, S) :-
    legal(S, finalize), not can_reach_terminal(S).
```

The validator should reject an FSM if any hard violation is derived. Warnings can
be allowed for non-critical issues such as duplicate paths or optional missing
metadata.

At runtime, Datalog should be used as `T(s)` post-hoc verification:

```text
current_state = s_t
valid actions = T(s_t) = {a | legal(s_t, a)}
LLM proposes action a_t and optional next state s_{t+1}
Datalog verifies transition(s_t, a_t, s_{t+1})
If valid: execute action and update to s_{t+1}
If invalid: retry with feedback or constrained fallback
```

Runtime facts example:

```prolog
current("GATHER_CONTEXT").
proposed_action("write_plan").
proposed_next_state("DRAFT_PLAN").
```

Runtime acceptance rule:

```prolog
accept(A, S2) :-
    current(S1),
    proposed_action(A),
    proposed_next_state(S2),
    legal(S1, A),
    transition(S1, A, S2).
```

### 4. Runtime State

The FSM should track control-relevant state only. It should not try to encode
the full environment or full codebase.

```python
RuntimeState = {
    "phase": str,
    "task_done": bool,
    "step_count": int,
    "last_action": str | None,
    "last_action_status": str,
    "last_error": str | None,
    "metadata": dict,
}
```

Dataset adapters can add dataset-specific fields. For example:

```text
Generic: task_read, context_gathered, plan_created, checked, finalized
Minecraft: inventory, goal item, recent environment messages
SWE-bench: repo_ready, candidate_files, modified_files, last_test_status
```

### 5. FSM Template

The FSM template defines high-level workflow phases and allowed transitions.

Example for generic tool use:

```text
START
-> READ_TASK
-> GATHER_CONTEXT
-> ACT
-> CHECK_PROGRESS
-> REVISE
-> FINALIZE
-> DONE
```

Example for code repair:

```text
START
-> SETUP
-> UNDERSTAND_TASK
-> LOCALIZE
-> INSPECT
-> PATCH
-> TEST
-> REVISE
-> FINALIZE
-> DONE
```

### 6. Dynamic Legal Action List

The legal action list is generated at every step from either:

```text
1. The validated LLM-generated FSMDesign.
2. The fixed GenericToolUseFSM fallback.
3. A benchmark-specific FSM template.
```

```python
legal_actions = fsm.get_legal_actions(task_spec, runtime_state)
legal_actions = rule_checker.filter(legal_actions, task_spec, runtime_state)
```

The transition list should be treated as `T(s_t)`:

```text
T(s_t) = all valid (action, next_state) pairs from the current FSM state.
Actions(s_t) = projection of T(s_t) onto action names.
```

The LLM should choose from this list and should also predict the intended next
FSM state. If it proposes an illegal action or an invalid next state, NS-FSM
blocks it and uses retry or planner fallback.

Expected runtime LLM output:

```python
{
    "thought": "I have enough context and should draft the plan.",
    "action": "write_plan",
    "next_state": "DRAFT_PLAN",
    "payload": {"summary": "..."}
}
```

If the LLM-generated FSM is invalid or unavailable, arbitrary user scenarios fall
back to this coarse generic legal-action map:

```text
START: read_task
READ_TASK: inspect_context, ask_clarifying_question, write_plan
GATHER_CONTEXT: write_plan, execute_step
MAKE_PLAN: execute_step, inspect_context
EXECUTE_STEP: check_progress, revise, finalize
CHECK_PROGRESS: revise, execute_step, finalize
FINALIZE: done
```

This lets the pipeline run even when no benchmark-specific adapter exists and
even when the LLM's proposed FSM fails validation.

### 7. Unified Context / Model Memory

This pipeline should not include a retrieval or RAG module. All model memory
should be provided through one unified context window.

The context should contain:

```text
Task instruction
Validated FSM summary
Current FSM state
Current transition list T(s_t)
Recent action history
Blocked action history
Fallback history
Adapter state summary
Any user-provided notes or constraints
```

Context rules:

```text
Do not add a retriever.
Do not query an external knowledge base.
Do not maintain separate RAG memory.
Keep one bounded context packet for each LLM call.
Summarize old history when the context becomes too long.
Always include current state and T(s_t), even after summarization.
```

## Implementation TODO

### Phase 0 - Repository Cleanup

- [ ] Keep `ns-fsm-baseline/` as the main implementation root.
- [ ] Ignore generated files such as `.DS_Store`, `__pycache__`, `.swp`.
- [ ] Do not put benchmark-specific logic in `nsfsm_agent.py`.
- [ ] Keep existing ReAct and Reflexion code unchanged unless needed for shared utilities.

### Phase 1 - Dataset Adapter Contract

- [ ] Create `src/datasets/base.py`.
- [ ] Define `TaskSpec` and `StepResult` data structures.
- [ ] Define `DatasetAdapter` base class.
- [ ] Make the base interface support both user scenarios and benchmark records.
- [ ] Required method: `list_tasks()`.
- [ ] Required method: `load_or_wrap(raw_input)`.
- [ ] Required method: `to_task_spec(raw_task)`.
- [ ] Required method: `reset(task_spec)`.
- [ ] Required method: `step(action)`.
- [ ] Required method: `get_observation()`.
- [ ] Required method: `is_done(state, task_spec)`.
- [ ] Required method: `get_available_tools(task_spec, state)`.
- [ ] Required method: `normalize_action(raw_action, legal_actions)`.
- [ ] Required method: `format_state_for_prompt(state)`.
- [ ] Required method: `summarize_result(state)`.

Adapter interface sketch:

```python
class DatasetAdapter:
    dataset_name: str

    def list_tasks(self) -> list[dict]:
        raise NotImplementedError

    def load_or_wrap(self, raw_input: str | dict) -> dict:
        raise NotImplementedError

    def to_task_spec(self, raw_task: dict) -> dict:
        raise NotImplementedError

    def reset(self, task_spec: dict) -> dict:
        raise NotImplementedError

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        raise NotImplementedError

    def get_available_tools(self, task_spec: dict, state: dict) -> list[dict]:
        raise NotImplementedError

    def normalize_action(self, raw_action: str, legal_actions: list[dict]) -> dict | None:
        raise NotImplementedError

    def format_state_for_prompt(self, state: dict) -> str:
        raise NotImplementedError
```

### Phase 2 - Generic Scenario Adapter

- [ ] Create `src/datasets/generic.py`.
- [ ] Implement `GenericScenarioAdapter`.
- [ ] Accept a raw natural-language instruction.
- [ ] Convert the instruction into a Level 0 `TaskSpec`.
- [ ] Initialize generic state fields: `task_read`, `context_gathered`, `plan_created`, `steps_completed`, `checked`, `finalized`.
- [ ] Expose generic high-level tools.
- [ ] Implement `read_task`.
- [ ] Implement `inspect_context`.
- [ ] Implement `ask_clarifying_question`.
- [ ] Implement `write_plan`.
- [ ] Implement `execute_step`.
- [ ] Implement `check_progress`.
- [ ] Implement `revise`.
- [ ] Implement `finalize`.
- [ ] Keep execution conservative when no real tool backend exists.
- [ ] Allow optional tool binding when running inside a repository.
- [ ] Mark success when the adapter reaches `finalized = True`.

Generic tool list:

```text
read_task
inspect_context
ask_clarifying_question
write_plan
execute_step
check_progress
revise
finalize
```

Generic adapter behavior:

```text
If no external environment is provided, actions update workflow state only.
If a local repository is available, inspect/execute actions can call safe local tools.
If a benchmark adapter is available, use that instead of generic mode.
```

### Phase 3 - LLM-Assisted FSM Designer

- [ ] Create `src/fsm_designer.py`.
- [ ] Implement `LLMFSMDesigner`.
- [ ] Call the existing `LLMInterface` once during task setup.
- [ ] Input to the API: task instruction, optional user-provided tools, optional success criteria, optional benchmark metadata.
- [ ] Output from the API: strict JSON matching `FSMDesign`.
- [ ] Ask the LLM to propose workflow states.
- [ ] Ask the LLM to propose `transitions_by_state`.
- [ ] Each state should list valid transition options: action, next_state, condition.
- [ ] Do not ask the LLM to separately generate `actions`, `legal_actions_by_state`, or flat `transitions`.
- [ ] Derive action list and legal action list in code from `transitions_by_state`.
- [ ] Ask the LLM to mark terminal states.
- [ ] Ask the LLM to include fallback policy and success signals.
- [ ] Save the raw LLM FSM proposal in result metadata for debugging.
- [ ] Cache generated FSM designs by task hash to avoid repeated API calls.
- [ ] Never execute an LLM-generated FSM before validation.

FSM designer prompt requirements:

```text
You are designing a workflow-level FSM for an NS-FSM agent.
Return JSON only.
Do not include code fences.
Do not invent external tools unless provided.
Every state must have legal actions.
Every non-terminal state must have at least one outgoing transition.
Use `transitions_by_state` as the only source of transition information.
Every transition option must include `action`, `next_state`, and `condition`.
Every `transitions_by_state` key must be a known state.
Every transition `next_state` must be a known state.
Do not output separate `actions`, `legal_actions_by_state`, or `transitions`.
Include a safe fallback policy.
```

Expected output file:

```text
results/fsm_designs/<task_hash>.json
```

### Phase 4 - FSM Validator

- [ ] Create `src/fsm_validator.py`.
- [ ] Implement `FSMDesignValidator`.
- [ ] Validate JSON parse success.
- [ ] Validate required fields exist.
- [ ] Validate state names are unique.
- [ ] Validate `initial_state` exists.
- [ ] Validate every terminal state exists.
- [ ] Validate every action is known or allowed by the adapter.
- [ ] Validate `transitions_by_state` exists.
- [ ] Validate every `transitions_by_state` key is a known state.
- [ ] Validate every non-terminal state has at least one transition option.
- [ ] Validate every transition option has `action`, `next_state`, and `condition`.
- [ ] Validate every transition `next_state` is a known state.
- [ ] Derive action set from `transitions_by_state`.
- [ ] Derive legal actions by state from `transitions_by_state`.
- [ ] Derive flat transition facts from `transitions_by_state`.
- [ ] Validate every non-terminal state can reach a terminal state.
- [ ] Reject empty FSMs.
- [ ] Reject FSMs with no path from initial to terminal state.
- [ ] Reject FSMs that allow `finalize` before task is read.
- [ ] Return structured validation errors.
- [ ] If validation fails, fall back to `GenericToolUseFSM`.
- [ ] Call `DatalogVerifier` after basic schema validation.
- [ ] Reject FSMs with hard Datalog violations.
- [ ] Store Datalog-derived violations in validation result metadata.

Validation result schema:

```python
{
    "valid": bool,
    "errors": list[str],
    "warnings": list[str],
    "fsm_design": dict | None,
    "fallback_used": bool,
}
```

### Phase 5 - Datalog Formal Verifier

- [ ] Create `src/datalog_verifier.py`.
- [ ] Implement `DatalogVerifier`.
- [ ] Convert `FSMDesign` into Datalog-style facts.
- [ ] Treat `transitions_by_state` as the canonical FSM source.
- [ ] Derive `action(A)` facts from transition options.
- [ ] Derive `legal(S, A)` facts from transition options.
- [ ] Derive `transition(S, A, S2)` facts from transition options.
- [ ] Convert adapter tool/action constraints into facts.
- [ ] Convert task success requirements into facts when available.
- [ ] Implement `get_valid_actions(current_state)` to return `T(s)`.
- [ ] Implement `verify_transition(current_state, action, proposed_next_state)`.
- [ ] Implement reachability closure.
- [ ] Implement terminal reachability closure.
- [ ] Implement hard violation predicates.
- [ ] Implement warning predicates.
- [ ] Return derived facts, hard violations, and warnings.
- [ ] Keep MVP implementation dependency-light with a small internal Datalog/fixpoint evaluator.
- [ ] Optionally add a backend flag later for Souffle, pyDatalog, or Z3.

MVP verifier API:

```python
class DatalogVerifier:
    def verify(self, fsm_design: dict, task_spec: dict, adapter) -> dict:
        return {
            "ok": bool,
            "facts": list[tuple],
            "derived": list[tuple],
            "violations": list[dict],
            "warnings": list[dict],
        }

    def get_valid_actions(self, current_state: str) -> list[str]:
        ...

    def get_valid_transitions(self, current_state: str) -> list[dict]:
        return [
            {"action": str, "next_state": str, "condition": str},
            ...
        ]

    def verify_transition(
        self,
        current_state: str,
        action: str,
        proposed_next_state: str,
    ) -> dict:
        return {
            "valid": bool,
            "violations": list[dict],
            "expected_next_states": list[str],
        }
```

Required hard checks:

```text
All transition states exist.
All legal actions exist.
All transition actions are legal in their source state.
All adapter actions are respected.
Initial state can reach at least one terminal state.
Every non-terminal reachable state can reach a terminal state.
No non-terminal state is actionless.
Finalize is not legal before the task has a path to completion.
```

Required runtime checks:

```text
Current state exists.
Action is in T(current_state).
Proposed next state exists.
Transition(current_state, action, proposed_next_state) exists.
If multiple next states are possible, proposed next state must be one of them.
If action is valid but next state is missing, infer next state only when transition is deterministic.
```

### Phase 6 - Structured Scenario Adapter

- [ ] Create `src/datasets/structured.py`.
- [ ] Implement `StructuredScenarioAdapter`.
- [ ] Accept user-provided fields: task, actions, success condition.
- [ ] Generate `TaskSpec` from the provided schema.
- [ ] Build a simple linear or partially ordered FSM from the action list.
- [ ] Use user-provided success condition when available.
- [ ] Fall back to generic success criteria when unavailable.

Structured scenario input example:

```python
{
    "task": "Evaluate model answers on my QA dataset.",
    "actions": ["load_dataset", "run_model", "score_answers", "write_report"],
    "success_condition": "A report is generated."
}
```

### Phase 7 - Dataset Registry

- [ ] Create `src/datasets/registry.py`.
- [ ] Add `register_dataset(name, adapter_cls)`.
- [ ] Add `get_adapter(name)`.
- [ ] Register `generic`.
- [ ] Register `structured`.
- [ ] Register `minecraft`.
- [ ] Register `swe_bench` later.
- [ ] If dataset is unknown, fall back to `generic` instead of failing.
- [ ] Add a warning when falling back to `generic`.

Registry sketch:

```python
DATASET_REGISTRY = {
    "generic": GenericScenarioAdapter,
    "structured": StructuredScenarioAdapter,
    "minecraft": MinecraftAdapter,
    "swe_bench": SWEBenchAdapter,
}
```

### Phase 8 - Generic Tool-Use FSM Template

- [ ] Create `src/fsm_templates/generic_tool_use.py`.
- [ ] Define phases: `START`, `READ_TASK`, `GATHER_CONTEXT`, `MAKE_PLAN`, `EXECUTE_STEP`, `CHECK_PROGRESS`, `REVISE`, `FINALIZE`, `DONE`.
- [ ] Generate legal actions from current phase.
- [ ] Allow loops between `EXECUTE_STEP`, `CHECK_PROGRESS`, and `REVISE`.
- [ ] Allow early finalization only after a check or explicit success signal.
- [ ] Use this template as the default when task type is unknown.

Generic phase transitions:

```text
START -> READ_TASK
READ_TASK -> GATHER_CONTEXT
GATHER_CONTEXT -> MAKE_PLAN
MAKE_PLAN -> EXECUTE_STEP
EXECUTE_STEP -> CHECK_PROGRESS
CHECK_PROGRESS -> REVISE
CHECK_PROGRESS -> FINALIZE
REVISE -> EXECUTE_STEP
FINALIZE -> DONE
```

### Phase 9 - Minecraft Adapter

- [ ] Create `src/datasets/minecraft.py`.
- [ ] Wrap existing `MCTextWorldWrapper`.
- [ ] Reuse existing `ActionParser`.
- [ ] Reuse existing `GroundTruth`.
- [ ] Convert `config/goals_67.json` into `TaskSpec` objects.
- [ ] Use inventory as dataset-specific state.
- [ ] Expose legal actions from MC-TextWorld action library.
- [ ] Normalize actions such as `mine_log` -> `mine_oak_log`.
- [ ] Keep Minecraft-specific action names inside this adapter.
- [ ] Add smoke task: `stick`.
- [ ] Add smoke task: `wooden_pickaxe`.

Minecraft `TaskSpec` example:

```python
{
    "dataset": "minecraft",
    "task_id": "minecraft/iron_pickaxe",
    "task_type": "symbolic_planning",
    "instruction": "Obtain 1 iron_pickaxe.",
    "initial_state": {"inventory": {}},
    "goal_condition": {"inventory.iron_pickaxe": ">=1"},
    "available_tools": ["mine", "craft", "smelt"],
    "max_steps": 100,
    "metadata": {"goal_item": "iron_pickaxe", "group": "Iron"},
}
```

### Phase 10 - FSM Template System

- [ ] Create `src/fsm_templates/base.py`.
- [ ] Define `FSMTemplate` base class.
- [ ] Add method `initial_phase(task_spec)`.
- [ ] Add method `get_legal_actions(task_spec, state, adapter)`.
- [ ] Add method `next_phase(task_spec, state, action, info)`.
- [ ] Add method `is_terminal(task_spec, state)`.
- [ ] Ensure `GenericToolUseFSM` is the default template.
- [ ] Create `src/fsm_templates/symbolic_planning.py`.
- [ ] Create `src/fsm_templates/code_repair.py`.
- [ ] Add template registry by `task_type`.

Template registry sketch:

```python
TASK_TYPE_TO_TEMPLATE = {
    "generic_tool_use": GenericToolUseFSM,
    "symbolic_planning": SymbolicPlanningFSM,
    "code_repair": CodeRepairFSM,
}
```

### Phase 11 - Generic FSM Builder

- [ ] Create `src/fsm.py`.
- [ ] Define `FSMState`.
- [ ] Define `FSMAction`.
- [ ] Define `FSMTransition`.
- [ ] Create `src/fsm_builder.py`.
- [ ] Accept validated `FSMDesign` as the preferred FSM source.
- [ ] Fall back to `GenericToolUseFSM` if no valid `FSMDesign` exists.
- [ ] Build FSM from `TaskSpec.task_type`.
- [ ] Attach selected template.
- [ ] Attach dataset adapter.
- [ ] Initialize runtime phase.
- [ ] Store visited phases/actions.
- [ ] Store blocked action history.
- [ ] Store fallback action history.

Runtime build flow:

```python
adapter = get_adapter(task_spec["dataset"])
template = get_template(task_spec["task_type"])
fsm = FSMBuilder.build(task_spec, adapter, template)
```

### Phase 12 - Rule Checker

- [ ] Create `src/rule_checker.py`.
- [ ] Implement generic legality result schema.
- [ ] Check whether proposed action is in current legal action list.
- [ ] Check dataset adapter constraints.
- [ ] Check workflow phase constraints.
- [ ] Check max step budget.
- [ ] Return structured block reasons.

Legality result schema:

```python
{
    "legal": bool,
    "reason_type": str,
    "message": str,
    "matched_action": dict | None,
}
```

Generic reason types:

```text
ok
unknown_action
not_in_legal_actions
missing_precondition
invalid_state_transition
unsafe_action
budget_exceeded
dataset_rule_violation
parse_error
```

### Phase 13 - Planner Fallback

- [ ] Create `src/planner.py`.
- [ ] Implement `Planner.next_action(task_spec, state, legal_actions, history)`.
- [ ] Make generic fallback the first implemented policy.
- [ ] For symbolic planning, choose first legal action on dependency path.
- [ ] For code repair, follow repair loop policy.
- [ ] For generic tool use, choose the safest progress action.
- [ ] Record when fallback is used.
- [ ] Record why fallback was used.

Generic fallback policy:

```text
If no task context has been gathered: gather context.
If context exists but no action has been taken: act.
If action has been taken but not checked: check progress.
If check failed: revise.
If check passed: finalize.
```

Code-repair fallback policy:

```text
If repo is not ready: setup_repo.
If issue is not read: read_problem_statement.
If no candidate files: search_code.
If candidate files exist but none opened: open_file.
If no patch exists: edit_file.
If patch exists and tests not run: run_tests.
If tests failed: analyze_failure.
If tests passed: finalize_patch.
```

### Phase 14 - NS-FSM Prompting

- [ ] Add generic NS-FSM prompts to `src/prompts.py`.
- [ ] Prompt should include dataset name.
- [ ] Prompt should include task instruction.
- [ ] Prompt should include current FSM phase.
- [ ] Prompt should include adapter-formatted state.
- [ ] Prompt should include recent action history.
- [ ] Prompt should include last blocked action and reason.
- [ ] Prompt should include legal actions only.
- [ ] Prompt should force one action from the legal list.

Prompt skeleton:

```text
You are an NS-FSM controlled agent.

Task:
{instruction}

Current workflow phase:
{phase}

Current state:
{state}

Recent history:
{history}

Last blocked action:
{last_blocked_action}

Legal next actions:
{legal_actions}

Valid next states for each action:
{transition_options}

Choose exactly one legal action and the intended next FSM state.

Thought: ...
Action: ...
Next State: ...
```

### Phase 15 - Unified Context Manager

- [ ] Create `src/context_manager.py`.
- [ ] Implement `ContextManager`.
- [ ] Build one bounded context packet for each LLM call.
- [ ] Include task instruction.
- [ ] Include current FSM state.
- [ ] Include current transition list `T(s_t)`.
- [ ] Include compact validated FSM summary.
- [ ] Include recent action history.
- [ ] Include blocked action history.
- [ ] Include fallback history.
- [ ] Include adapter state summary.
- [ ] Summarize older history when context is too long.
- [ ] Never call a retriever or external knowledge base.
- [ ] Never maintain separate RAG memory.

Context packet schema:

```python
{
    "instruction": str,
    "fsm_state": str,
    "transition_options": list[dict],
    "recent_history": list[dict],
    "blocked_history": list[dict],
    "fallback_history": list[dict],
    "adapter_state_summary": str,
    "compressed_memory": str,
}
```

### Phase 16 - Generic NSFSMAgent

- [ ] Create `src/nsfsm_agent.py`.
- [ ] Accept `task_spec`, `adapter`, `fsm`, `llm`, `rule_checker`, `planner`.
- [ ] Reset adapter at episode start.
- [ ] Build legal action list each step.
- [ ] Build valid transition list `T(current_state)` each step.
- [ ] Derive legal action names from `T(current_state)`.
- [ ] Include valid next-state options for each action in the prompt.
- [ ] Build prompt from the unified `ContextManager` packet.
- [ ] Ask LLM for proposed action and proposed next FSM state.
- [ ] Normalize proposed action through adapter.
- [ ] Check action through `RuleChecker`.
- [ ] Check `(current_state, action, proposed_next_state)` through `DatalogVerifier`.
- [ ] If legal, execute proposed action.
- [ ] If illegal, block proposed action and record reason.
- [ ] If illegal, ask `Planner` for fallback action.
- [ ] Execute fallback action if available.
- [ ] Terminate on goal success.
- [ ] Terminate on max steps.
- [ ] Terminate on no valid action.
- [ ] Terminate on repeated blocked actions or dead loop.
- [ ] Return JSON-compatible result dict.

Result schema:

```python
{
    "dataset": str,
    "task_id": str,
    "task_type": str,
    "success": bool,
    "total_steps": int,
    "termination": str,
    "blocked_action_count": int,
    "fallback_action_count": int,
    "trajectory": list[dict],
    "metadata": dict,
}
```

### Phase 17 - Scenario Runner

- [ ] Create `scripts/run_scenario.py`.
- [ ] Accept `--instruction`.
- [ ] Accept `--scenario-file`.
- [ ] Accept `--dataset`, defaulting to `generic`.
- [ ] Accept `--max-steps`.
- [ ] Accept `--tag`.
- [ ] Build a `TaskSpec` through `GenericScenarioAdapter` or `StructuredScenarioAdapter`.
- [ ] Call `LLMFSMDesigner` unless `--use-fixed-generic-fsm` is passed.
- [ ] Validate the LLM-generated FSM before use.
- [ ] Save the generated FSM design alongside the scenario result.
- [ ] Run the generic NS-FSM loop.
- [ ] Save one result JSON under `results/scenarios/<tag>/`.

Example free-form scenario command:

```bash
python scripts/run_scenario.py \
  --instruction "Analyze this repository and propose an experiment pipeline." \
  --tag repo_pipeline_plan
```

Example structured scenario command:

```bash
python scripts/run_scenario.py \
  --dataset structured \
  --scenario-file config/scenarios/qa_eval_pipeline.json \
  --tag qa_eval_pipeline
```

### Phase 18 - Experiment Runner

- [ ] Create `scripts/run_nsfsm_experiment.py`.
- [ ] Support `--dataset`.
- [ ] Support `--task-type`.
- [ ] Support `--task-ids`.
- [ ] Support `--groups` when adapter supports groups.
- [ ] Support `--runs`.
- [ ] Support `--tag`.
- [ ] Support `--resume`.
- [ ] Support `--quiet`.
- [ ] Support `--summary-only`.
- [ ] Support `--use-fixed-generic-fsm`.
- [ ] Support `--save-fsm-design`.
- [ ] Save one result JSON per `(dataset, task_id, run_id)`.
- [ ] Save combined summary.
- [ ] Skip completed files when `--resume` is used.

Output layout:

```text
results/full/<tag>/<dataset>/nsfsm/<task_id>_run01.json
results/full/<tag>/<dataset>/nsfsm_summary.json
results/full/<tag>/combined_summary.json
```

Example commands:

```bash
cd ns-fsm-baseline

python scripts/run_nsfsm_experiment.py \
  --dataset generic \
  --task-ids scenario/user_task_001 \
  --runs 1 \
  --tag generic_smoke \
  --quiet
```

```bash
python scripts/run_nsfsm_experiment.py \
  --dataset minecraft \
  --task-ids minecraft/stick \
  --runs 1 \
  --tag nsfsm_smoke \
  --quiet
```

```bash
python scripts/run_nsfsm_experiment.py \
  --dataset minecraft \
  --groups Wooden \
  --runs 1 \
  --tag nsfsm_wooden_smoke \
  --quiet
```

### Phase 19 - SWE-Bench Adapter

- [ ] Create `src/datasets/swe_bench.py`.
- [ ] Convert SWE-bench instances into `TaskSpec`.
- [ ] Use `task_type = "code_repair"`.
- [ ] Track repo setup state.
- [ ] Track issue text.
- [ ] Track candidate files.
- [ ] Track opened files.
- [ ] Track modified files.
- [ ] Track test commands run.
- [ ] Track last test status.
- [ ] Track final patch readiness.
- [ ] Implement high-level tool actions.
- [ ] Implement `search_code`.
- [ ] Implement `open_file`.
- [ ] Implement `edit_file`.
- [ ] Implement `run_tests`.
- [ ] Implement `finalize_patch`.
- [ ] Block unsafe shell commands.
- [ ] Block file edits outside repository.
- [ ] Block finalization if no patch exists.

SWE-bench `TaskSpec` example:

```python
{
    "dataset": "swe_bench",
    "task_id": "django__django-12345",
    "task_type": "code_repair",
    "instruction": "Fix the bug described in the issue.",
    "initial_state": {
        "repo_ready": False,
        "issue_read": False,
        "candidate_files": [],
        "opened_files": [],
        "modified_files": [],
        "patch_exists": False,
        "tests_run": [],
        "last_test_status": "not_run",
        "final_patch_ready": False
    },
    "goal_condition": {"final_patch_ready": True},
    "available_tools": [
        "setup_repo",
        "read_problem_statement",
        "search_code",
        "open_file",
        "edit_file",
        "run_tests",
        "analyze_failure",
        "finalize_patch"
    ],
    "max_steps": 80,
    "metadata": {
        "repo": "django/django",
        "base_commit": "...",
        "problem_statement": "..."
    }
}
```

### Phase 20 - Analysis

- [ ] Create `scripts/analyze_nsfsm.py`.
- [ ] Report success rate.
- [ ] Report average steps.
- [ ] Report blocked action rate.
- [ ] Report fallback usage rate.
- [ ] Report LLM-generated FSM validation failure rate.
- [ ] Report Datalog hard violation rate.
- [ ] Report fixed-generic-FSM fallback rate.
- [ ] Report termination distribution.
- [ ] Report per-dataset summary.
- [ ] Report per-task-type summary.
- [ ] Report per-group summary when groups exist.
- [ ] Compare `react`, `reflexion`, and `nsfsm` when available.

Required output files:

```text
results/analysis/<tag>/summary_by_dataset.csv
results/analysis/<tag>/summary_by_task_type.csv
results/analysis/<tag>/summary_by_task.csv
results/analysis/<tag>/blocked_actions.csv
results/analysis/<tag>/fallback_actions.csv
results/analysis/<tag>/fsm_validation.csv
results/analysis/<tag>/datalog_violations.csv
results/analysis/<tag>/combined_report.md
```

### Phase 21 - Tests

- [ ] Add tests for `DatasetAdapter` interface.
- [ ] Add tests for `GenericScenarioAdapter`.
- [ ] Add tests for `StructuredScenarioAdapter`.
- [ ] Add tests for `LLMFSMDesigner` with mocked LLM output.
- [ ] Add tests for `FSMDesignValidator`.
- [ ] Add tests for `DatalogVerifier`.
- [ ] Add tests for deriving `actions` from `transitions_by_state`.
- [ ] Add tests for deriving `legal_actions_by_state` from `transitions_by_state`.
- [ ] Add tests for deriving flat Datalog transition facts from `transitions_by_state`.
- [ ] Add tests for reachability violation.
- [ ] Add tests for dead-end state violation.
- [ ] Add tests for illegal transition action violation.
- [ ] Add tests for `T(current_state)` valid action extraction.
- [ ] Add tests for valid `(state, action, next_state)` verification.
- [ ] Add tests for invalid proposed next-state rejection.
- [ ] Add tests for invalid LLM-generated FSM fallback.
- [ ] Add tests for `MinecraftAdapter`.
- [ ] Add tests for dataset registry.
- [ ] Add tests for FSM template selection.
- [ ] Add tests for `RuleChecker`.
- [ ] Add tests for `Planner`.
- [ ] Add tests for `NSFSMAgent` with mocked LLM output.
- [ ] Add tests for `ContextManager`.
- [ ] Add tests that old history is summarized but current state and `T(s_t)` remain present.
- [ ] Add tests for blocked illegal action.
- [ ] Add tests for fallback after blocked action.
- [ ] Add tests for runner resume logic.
- [ ] Add tests for result schema.

Smoke tests:

```bash
python scripts/run_scenario.py \
  --instruction "Create a short plan for evaluating a model on a QA dataset." \
  --tag smoke_generic_plan
```

```bash
python scripts/run_nsfsm_experiment.py \
  --dataset minecraft \
  --task-ids minecraft/stick \
  --runs 1 \
  --tag smoke_stick \
  --quiet
```

```bash
python scripts/run_nsfsm_experiment.py \
  --dataset minecraft \
  --groups Wooden \
  --runs 1 \
  --tag smoke_wooden \
  --quiet
```

## Runtime Algorithm

```text
for raw_task_or_instruction in input_tasks:
    adapter = DatasetRegistry.get_adapter(dataset_name or "generic")

    # Generic mode accepts plain natural language.
    # Benchmark mode accepts official benchmark records.
    raw_task = adapter.load_or_wrap(raw_task_or_instruction)
    task_spec = adapter.to_task_spec(raw_task)

    if use_llm_fsm_designer:
        proposed_design = LLMFSMDesigner.generate(task_spec, adapter)
        validation = FSMDesignValidator.validate_schema(proposed_design, task_spec, adapter)
        if validation.valid:
            datalog_result = DatalogVerifier.verify(proposed_design, task_spec, adapter)
            validation = FSMDesignValidator.merge_datalog_result(validation, datalog_result)
    else:
        validation = {"valid": False, "fallback_used": True}

    if validation.valid:
        fsm = FSMBuilder.from_design(task_spec, adapter, validation.fsm_design)
    else:
        fsm_template = FSMTemplateRegistry.get("generic_tool_use")
        fsm = FSMBuilder.build(task_spec, adapter, fsm_template)

    state = adapter.reset(task_spec)

    while not done:
        current_state = fsm.current_state
        transition_options = datalog.get_valid_transitions(current_state)   # T(s_t)
        legal_actions = datalog.get_valid_actions(current_state)
        prompt = build_nsfsm_prompt(
            task_spec,
            state,
            legal_actions,
            transition_options,
            history,
        )
        raw_action = llm.generate(prompt)
        proposed_action, proposed_next_state = adapter.normalize_action(
            raw_action,
            legal_actions,
            transition_options,
        )

        check = rule_checker.check(proposed_action, legal_actions, state)
        transition_check = datalog.verify_transition(
            current_state,
            proposed_action,
            proposed_next_state,
        )

        if check.legal and transition_check.valid:
            action_to_execute = proposed_action
            next_fsm_state = proposed_next_state
            blocked = False
        else:
            record_blocked_action(proposed_action, check, transition_check)
            action_to_execute, next_fsm_state = planner.next_action(
                task_spec,
                state,
                legal_actions,
                transition_options,
                history,
            )
            blocked = True

        if action_to_execute is None:
            terminate("no_valid_action")

        next_state, done, info = adapter.step(action_to_execute)
        fsm.update(task_spec, state, action_to_execute, next_fsm_state, info)
        record_step(...)
        state = next_state
```

## MVP Build Order

1. Build `DatasetAdapter` base interface.
2. Build `GenericScenarioAdapter`.
3. Build `LLMFSMDesigner` with mocked API output first.
4. Build `FSMDesignValidator`.
5. Build `DatalogVerifier`.
6. Connect `FSMDesignValidator` to `DatalogVerifier`.
7. Build `GenericToolUseFSM` as fallback.
8. Build dataset registry with `generic` fallback.
9. Build `RuleChecker`.
10. Build generic `Planner` fallback.
11. Build `ContextManager`.
12. Build `NSFSMAgent`.
13. Build `run_scenario.py`.
14. Run free-form scenario smoke test with mocked FSM design.
15. Run free-form scenario smoke test with real LLM API FSM design.
16. Build `StructuredScenarioAdapter`.
17. Build `run_nsfsm_experiment.py`.
18. Build `MinecraftAdapter`.
19. Run Minecraft smoke test on `stick`.
20. Run Minecraft smoke test on `Wooden`.
21. Add basic `analyze_nsfsm.py`.
22. Add `CodeRepairFSM`.
23. Add `SWEBenchAdapter`.
24. Run one SWE-bench-style smoke task.
25. Generalize reports across datasets.

## Definition of Done

The basic multi-benchmark NS-FSM pipeline is complete when:

- [ ] A user can provide an arbitrary natural-language task and run it through `generic` mode.
- [ ] The system can call an LLM API to propose task-specific FSM states, legal actions, and transitions.
- [ ] The LLM-generated FSM uses `transitions_by_state` as the canonical source.
- [ ] The system derives action lists, legal action lists, and Datalog transition facts from `transitions_by_state`.
- [ ] The system validates every LLM-generated FSM before execution.
- [ ] Datalog verification derives reachability, dead-end, and semantic violation predicates.
- [ ] At runtime, Datalog returns `T(current_state)` for each current FSM state.
- [ ] At runtime, every LLM-proposed `(action, next_state)` is verified against `transition(current_state, action, next_state)`.
- [ ] All model memory is passed through one unified bounded context packet.
- [ ] No retrieval or RAG module is required for the basic pipeline.
- [ ] Any FSM with hard Datalog violations is rejected before execution.
- [ ] The system falls back to fixed `GenericToolUseFSM` when the generated FSM is invalid.
- [ ] The same `NSFSMAgent` can run on Minecraft and at least one non-Minecraft task type.
- [ ] Core NS-FSM code contains no Minecraft-only assumptions.
- [ ] Dataset-specific logic is isolated inside adapters.
- [ ] Dedicated benchmark adapters are optional, not required for arbitrary scenarios.
- [ ] The runner accepts `--dataset`.
- [ ] The FSM/action list is created dynamically at runtime.
- [ ] Illegal LLM actions are blocked before execution.
- [ ] Planner fallback is used when the LLM proposes an illegal action.
- [ ] Results are saved in a consistent JSON schema.
- [ ] Analysis reports success rate, blocked action rate, and fallback usage rate.
- [ ] Existing ReAct/Reflexion baselines remain runnable.

## Non-Goals For Basic Pipeline

- [ ] Do not implement SFT training yet.
- [ ] Do not implement DPO training yet.
- [ ] Do not implement RAG or retrieval for the basic pipeline.
- [ ] Do not maintain separate retrieval memory.
- [ ] Do not build a complete code-state FSM for SWE-bench.
- [ ] Do not hand-write one FSM per task.
- [ ] Do not require a dedicated adapter for every arbitrary user scenario.
- [ ] Do not trust or execute an LLM-generated FSM without schema validation.
- [ ] Do not accept an LLM-generated FSM that fails Datalog verification.
- [ ] Do not expose benchmark gold solutions to the agent.
- [ ] Do not make Minecraft the only supported abstraction.

## Key Principle

```text
Generic mode makes the system runnable for arbitrary scenarios.
LLM FSM designer proposes task-specific workflow states and legal actions.
FSM validator decides whether the proposal is safe to use.
Datalog verifier derives formal consistency, reachability, and violation facts.
ContextManager provides one unified bounded model-memory context.
Adapters understand benchmarks.
FSM templates understand workflows.
RuleChecker understands legality.
Planner understands fallback progress.
NSFSMAgent coordinates the loop.
```

This separation is what allows the pipeline to receive a new benchmark task,
create the task-specific legal-action list at runtime, and continue execution
without redesigning the whole system.
