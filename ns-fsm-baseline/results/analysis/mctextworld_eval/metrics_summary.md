# MC-TextWorld Evaluation Summary

## Overall Benchmark Comparison

| Method | Status | Success Rate | Runs | Successes |
| --- | --- | ---: | ---: | ---: |
| Qwen ReAct | measured | 14.3% | 1005 | 144 |
| Qwen Reflexion | measured | 26.5% | 1005 | 266 |
| Qwen NS-FSM (projected) | projected | 39.7% | 1005 |  |
| GPT-5-mini Reflexion | measured | 99.0% | 201 |  |
| GPT-5-mini NS-FSM (projected) | projected | 97.5% | 164 |  |

## Success Rate by Dependency Depth

| Method | Group | Depth | Success Rate |
| --- | --- | ---: | ---: |
| Qwen ReAct | Wooden | 3 | 80.0% |
| Qwen ReAct | Stone | 7 | 15.6% |
| Qwen ReAct | Iron | 12 | 0.8% |
| Qwen ReAct | Redstone | 14 | 1.1% |
| Qwen ReAct | Golden | 15 | 0.0% |
| Qwen ReAct | Armor | 15 | 0.0% |
| Qwen ReAct | Diamond | 18 | 0.0% |
| Qwen Reflexion | Wooden | 3 | 91.3% |
| Qwen Reflexion | Stone | 7 | 64.4% |
| Qwen Reflexion | Iron | 12 | 7.1% |
| Qwen Reflexion | Redstone | 14 | 0.0% |
| Qwen Reflexion | Golden | 15 | 15.6% |
| Qwen Reflexion | Armor | 15 | 5.6% |
| Qwen Reflexion | Diamond | 18 | 0.0% |
| Qwen NS-FSM (projected) | Wooden | 3 | 91.0% |
| Qwen NS-FSM (projected) | Stone | 7 | 78.4% |
| Qwen NS-FSM (projected) | Iron | 12 | 22.8% |
| Qwen NS-FSM (projected) | Redstone | 14 | 18.6% |
| Qwen NS-FSM (projected) | Golden | 15 | 23.8% |
| Qwen NS-FSM (projected) | Armor | 15 | 24.6% |
| Qwen NS-FSM (projected) | Diamond | 18 | 15.0% |

## ReAct Failure Attribution

| Error Type | Count | Percent of Failures |
| --- | ---: | ---: |
| tool_dependency_error | 459 | 53.3% |
| plan_knowledge_error | 283 | 32.9% |
| resource_shortage | 116 | 13.5% |
| none | 3 | 0.3% |

## ReAct Termination Reasons

| Termination Reason | Count | Percent |
| --- | ---: | ---: |
| dead_loop | 856 | 85.2% |
| success | 144 | 14.3% |
| llm_error | 3 | 0.3% |
| max_steps | 2 | 0.2% |
