# NS-FSM 当前 To Do List

本文档基于当前仓库的真实完成度重新整理，不沿用 `NSFSM_PIPELINE_TODO.md` 里的旧 Phase 划分。

当前建议的主线目标是：先把 **MC-TextWorld 上的 NS-FSM w/o DPO** 跑成可与 ReAct / Reflexion 对比的完整实验。

## 当前已完成

- [x] MC-TextWorld baseline 基础设施已完成：
  - [x] `env_wrapper.py`
  - [x] `llm_interface.py`
  - [x] `action_parser.py`
  - [x] `ground_truth.py`
  - [x] ReAct agent
  - [x] Reflexion agent
  - [x] metrics / error analysis
- [x] ReAct 完整实验已跑完：67 goals x 15 runs = 1005 runs。
- [x] Reflexion 完整实验已跑完：67 goals x 15 runs = 1005 runs。
- [x] Baseline error analysis 已生成。
- [x] MC-TextWorld Datalog facts 已导出。
- [x] NS-FSM MVP 代码骨架已基本完成：
  - [x] dataset adapters
  - [x] FSM templates
  - [x] FSM validator
  - [x] Datalog verifier
  - [x] RuleChecker
  - [x] Planner
  - [x] ContextManager
  - [x] NSFSMAgent
  - [x] scenario runner
  - [x] experiment runner
  - [x] analysis script
- [x] 组件测试通过。
- [x] NS-FSM smoke test 已有 `minecraft/stick` 的成功样例。

## P0: 明确近期实验范围

- [ ] 先只做 MC-TextWorld，不做 Robotouille。
- [ ] 先只做 NS-FSM w/o DPO，不做 DPO 训练。
- [ ] 实验矩阵先固定为：
  - [x] ReAct
  - [x] Reflexion
  - [ ] NS-FSM w/o DPO
- [ ] 明确主要 claim：
  - [ ] ReAct / Reflexion 在长 horizon 和复杂 dependency 上明显退化。
  - [ ] NS-FSM 通过合法动作约束、状态转移验证和 fallback 降低结构性错误。
  - [ ] Reflexion 的 self-correction 对结构性错误修复有限。

## P1: 修正 NS-FSM 正式运行路径

- [ ] 检查 `scripts/run_nsfsm_experiment.py`，确保正式实验不是 `planner_only` 模式。
- [ ] 给 `run_nsfsm_experiment.py` 接入真实 LLM action generation。
- [ ] 给 `run_scenario.py` 接入真实 LLM action generation。
- [ ] 明确区分三种运行模式：
  - [ ] `planner_only`: deterministic planner smoke test。
  - [ ] `template_fsm + llm_action`: 固定 FSM template，LLM 选择 action。
  - [ ] `llm_fsm_designer + llm_action`: LLM 生成 FSM，LLM 选择 action。
- [ ] 在结果 JSON 中记录每一步 action 来源：
  - [ ] `llm_action`
  - [ ] `blocked_action`
  - [ ] `planner_fallback`
  - [ ] `datalog_violation`
  - [ ] `fsm_transition_check`
- [ ] 确认 NS-FSM 的核心流程是真正的：
  - [ ] LLM proposes action
  - [ ] FSM / Datalog verifies action and next state
  - [ ] legal action executes
  - [ ] illegal action is blocked
  - [ ] fallback recovers

## P2: 发现 MC-TextWorld 可构图任务子集

- [x] 编写 discovery 脚本：`scripts/discover_mctextworld_tasks.py`。
- [x] 从 `tasks.json` / `plans.json` / `action_lib.json` 发现可 work 任务。
- [x] 区分两类标准：
  - [x] `official_plan_replayable`: 官方 plan 能直接 replay。
  - [x] `dependency_graph_buildable`: 可从 action library 合成 quantity-aware executable ground truth sequence。
- [x] 输出 discovery 报告：
  - [x] `config/mctextworld_task_discovery_report.json`
  - [x] `config/mctextworld_task_discovery_report.md`
  - [x] `config/mctextworld_ground_truth_buildable_tasks.json`
- [x] 当前 discovery 结果：
  - [x] 原始 `tasks.json`: 208 条，207 个 unique task id。
  - [x] `official_plan_replayable`: 96 个 unique task id。
  - [x] `dependency_graph_buildable`: 192 个 unique task id。
  - [x] 67-goal 中原始 task 存在 63 个，其中 63 个都可构建 dependency graph。
  - [x] 如果把缺失的 4 个 67-goal 目标作为 synthetic goal 检查，67/67 都可从 action library 合成 executable sequence。
- [ ] 决定正式实验子集标准：
  - [ ] 严格标准：只用 `official_plan_replayable`。
  - [ ] 扩展标准：用 `dependency_graph_buildable`，允许从 action library 合成 executable sequence。
- [ ] 将正式实验配置切换为选定子集。

## P3: NS-FSM 单任务 Smoke

- [ ] 跑 `stick`。
- [ ] 跑 `crafting_table`。
- [ ] 跑 `wooden_pickaxe`。
- [ ] 跑 `torch`。
- [ ] 跑 `iron_pickaxe`。
- [ ] 对每个 smoke 手动检查 trajectory：
  - [ ] LLM 是否真的提出 action。
  - [ ] FSM 是否给出正确 legal actions。
  - [ ] Datalog 是否阻止非法动作。
  - [ ] fallback 是否合理推进。
  - [ ] 结果是否不是直接 oracle plan。
- [ ] 修复 smoke 中暴露的问题。
- [ ] 生成 smoke analysis report。

## P4: Wooden Group Pilot

- [ ] 跑 Wooden group: 10 goals x 1 run。
- [ ] 检查简单目标是否稳定完成：
  - [ ] `stick`
  - [ ] `crafting_table`
  - [ ] `wooden_pickaxe`
  - [ ] `wooden_axe`
  - [ ] `wooden_sword`
- [ ] 统计 pilot 指标：
  - [ ] success rate
  - [ ] average steps
  - [ ] blocked action rate
  - [ ] fallback usage rate
  - [ ] invalid transition rate
  - [ ] recovery success rate
- [ ] 人工检查失败 case。
- [ ] 根据失败类型调整：
  - [ ] prompt
  - [ ] FSM template
  - [ ] Datalog rules
  - [ ] fallback policy

## P5: 正式 NS-FSM MC-TextWorld 实验

- [ ] 跑 67 goals x 1 run，确认全流程稳定。
- [ ] 跑 67 goals x 3 runs，观察趋势。
- [ ] 如果时间允许，跑 67 goals x 15 runs，与 ReAct / Reflexion 对齐。
- [ ] 每轮实验保存：
  - [ ] raw trajectory JSON
  - [ ] combined summary
  - [ ] group-level summary
  - [ ] per-goal summary
  - [ ] failure cases
- [ ] 建议输出目录：
  - [ ] `results/full/nsfsm_v1_mc67/`
  - [ ] `results/analysis/nsfsm_v1_mc67/`

## P6: 统一 ReAct / Reflexion / NS-FSM 分析

- [ ] 合并三组结果：
  - [ ] ReAct
  - [ ] Reflexion
  - [ ] NS-FSM w/o DPO
- [ ] 生成统一表格：
  - [ ] overall success rate
  - [ ] success by group
  - [ ] average steps
  - [ ] invalid / blocked action rate
  - [ ] dead loop rate
  - [ ] cascade failure rate
  - [ ] fallback usage rate
- [ ] 生成统一图：
  - [ ] success rate by group
  - [ ] performance decay vs dependency depth
  - [ ] error distribution
  - [ ] fallback trigger distribution
- [ ] 修正当前 Reflexion analysis 中部分 `avg_steps: 0.0` 的问题。
- [ ] 输出最终对比报告：
  - [ ] `baseline_vs_nsfsm_report.md`
  - [ ] `baseline_vs_nsfsm_tables/`
  - [ ] `baseline_vs_nsfsm_figures/`

## P7: 论文 Case Studies

- [ ] 选 3 个 ReAct 失败案例。
- [ ] 选 3 个 Reflexion 失败案例。
- [ ] 选 3 个 NS-FSM recovery 成功案例。
- [ ] 每个 case 对齐展示：
  - [ ] goal
  - [ ] required dependency chain
  - [ ] agent 原始错误
  - [ ] FSM / Datalog 如何发现错误
  - [ ] fallback 如何恢复
  - [ ] 最终是否成功
- [ ] 总结 NS-FSM 主要修复的错误类型：
  - [ ] recipe knowledge error
  - [ ] sequencing error
  - [ ] missing prerequisite
  - [ ] invalid action hallucination
  - [ ] dead loop
  - [ ] cascade failure

## P8: 暂缓事项

这些任务目前不建议立刻做，等 MC-TextWorld NS-FSM w/o DPO 闭环完成后再开。

- [ ] Robotouille adapter / rules / rollout。
- [ ] DPO 数据构造。
- [ ] Step-DPO。
- [ ] Sem-DPO。
- [ ] Outcome-DPO。
- [ ] Z3 data augmentation。
- [ ] LATS baseline。
- [ ] StateFlow baseline。
- [ ] Agent Q baseline。
- [ ] Llama 3 70B 正式训练。
- [ ] SWE-bench 正式实验。

## 推荐执行顺序

1. 修正 NS-FSM runner 的真实 LLM action path。
2. 跑 `stick` / `crafting_table` / `wooden_pickaxe` smoke。
3. 清理 67-goal Datalog subset。
4. 跑 Wooden group 10 x 1 pilot。
5. 跑 MC-TextWorld 67 x 1。
6. 跑 MC-TextWorld 67 x 3。
7. 生成 ReAct vs Reflexion vs NS-FSM 的第一版统一报告。
8. 决定是否扩展到 67 x 15。

## 当前判断

当前项目已经有完整 baseline 结果和 NS-FSM MVP 代码，但 NS-FSM 还停留在 smoke test 阶段。下一步最关键的是把 NS-FSM 正式接入 LLM action generation，并在 MC-TextWorld 67 goals 上跑出可对比结果。
