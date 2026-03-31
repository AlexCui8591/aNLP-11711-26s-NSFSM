# NS-FSM Baseline Reproduction — 实施计划 (v3)

## 变更记录

- **v3**: 将 baseline 从 DECKARD + SC 更改为 **ReAct + Reflexion**；修正 MC-TextWorld 的 max_steps；更新 LLM 调用策略和项目结构。

---

## 背景

在 MC-TextWorld 环境中复现 ReAct 和 Reflexion 两个 baseline，使用 Qwen2.5 via Ollama，通过 error analysis 揭示 LLM 在长程规划中的系统性失败，为 NS-FSM proposal 提供动机论证。

---

## 已确认事项


| 项目          | 结论                                         |
| ------------- | -------------------------------------------- |
| LLM 后端      | ✅ 本地 Ollama，使用 Qwen2.5-7B-Instruct     |
| 67 goals 列表 | ✅ 已从 XENON 仓库提取完整列表               |
| MC-TextWorld  | ✅ 已安装，API 验证通过，67 goals 全部可用   |
| Baseline 选择 | ✅**ReAct + Reflexion**（替代 DECKARD + SC） |

---

## Baseline 选择理由


| Baseline      | 代表什么                             | 为什么选                                                                                           |
| ------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **ReAct**     | 纯 prompt-driven，无外部结构         | 当前 LLM agent 最广泛引用的范式；展示裸 LLM 在长程任务上的原生表现                                 |
| **Reflexion** | 有 verbal reflection，无外部结构约束 | 代表"有自我反思但无 FSM/Datalog 约束"的中间态；XENON 论文确认 LLM self-correction 对结构性错误无效 |

**对比逻辑**：

```
ReAct (无记忆/无反思/无约束)
  → 展示 LLM 在长程任务上的原生失败
Reflexion (有反思/无结构约束)
  → 展示"反思能修复认知错误但修不了结构性错误"
NS-FSM (FSM + Datalog + DPO)
  → 外部结构化约束解决了以上两类问题
```

---

## Episode 结构

> [!IMPORTANT]
> **MC-TextWorld 的步数与 MineRL 完全不同。**
>
> MineRL 的每步是 low-level 键盘/鼠标操作（制作一个 iron_pickaxe 需要几千步）。
> MC-TextWorld 的每步是 high-level action（`craft_iron_pickaxe` 就是一步）。
>
> 在 MC-TextWorld 中，完成一个 goal 最多需要 ~15-20 个 action steps。
> 所以每个 episode 的 max_steps 设为 **100 步**（留足余量给 ReAct 的试错）。
> Reflexion 允许最多 **3 个 episode**（1 次初始尝试 + 2 次 reflection 重试）。

```
# ReAct
for each goal in 67_goals:
    for run_id in range(15):
        env.reset(goal)
        for step in range(max_steps=100):
            thought, action = llm.react(observation, goal)
            obs, done, info = env.step(action)
            if done: break

# Reflexion
for each goal in 67_goals:
    for run_id in range(15):
        for attempt in range(max_attempts=3):
            env.reset(goal)
            trajectory = react_episode(env, goal, max_steps=100)
            if trajectory.success: break
            reflection = llm.reflect(trajectory)
            # reflection 注入下次尝试的 prompt
```

---

## ReAct 实现细节

### 核心循环

每一步：

1. 构造 prompt = system prompt + goal + 当前 inventory + 历史 trajectory（最近 K 步）
2. LLM 生成 Thought（分析当前状况） + Action（选择下一步操作）
3. 解析 Action → MC-TextWorld action 格式（如 `craft_iron_pickaxe`）
4. 环境执行，返回新 inventory + success/failure
5. 更新 trajectory 历史

### ReAct Prompt 模板

```
You are a Minecraft crafting agent. Your goal is to obtain: {goal_item}

Available action types: mine, craft, smelt
Action format: {type}_{item_name} (e.g., mine_oak_log, craft_oak_planks, smelt_iron_ingot)

Current inventory: {inventory}
Steps taken: {step_count}/{max_steps}

Previous actions:
{action_history}

Think step by step about what you need to do next, then choose one action.

Thought: <your reasoning>
Action: <one action in the format type_itemname>
```

### Action 解析

LLM 输出可能不完全匹配 MC-TextWorld 的 action 名称。需要一个 fuzzy matching 层：

1. 精确匹配：直接查找 action_lib
2. 模糊匹配：如果 LLM 输出 `mine_log`，尝试匹配 `mine_oak_log`、`mine_birch_log` 等
3. 失败处理：如果完全无法解析，记录为 invalid action，不消耗环境步数

---

## Reflexion 实现细节

### 在 ReAct 基础上增加的机制

1. **Episode-level reflection**：一个 episode 结束后（成功或失败），LLM 生成 verbal reflection
2. **Reflection 注入**：下次重试时，将 reflection 加入 system prompt
3. **最多 3 次尝试**：初始 + 2 次 reflection 重试

### Reflection Prompt 模板

```
You attempted to obtain {goal_item} but failed.

Your trajectory:
{full_trajectory}

Final inventory: {final_inventory}

Analyze why you failed. What mistakes did you make? What should you do differently next time?

Reflection:
```

### Retry Prompt（注入 reflection）

```
You are a Minecraft crafting agent. Your goal is to obtain: {goal_item}

You have attempted this task before and learned from your mistakes:
{reflection_1}
{reflection_2}  # 如果是第三次尝试

Current inventory: {inventory}
...
```

---

## 完整 67 Goals 列表

### Wooden (10 goals)

bowl, crafting_table, chest, ladder, stick, wooden_axe, wooden_hoe, wooden_pickaxe, wooden_shovel, wooden_sword

### Stone (9 goals)

charcoal, furnace, smoker, stone_axe, stone_hoe, stone_pickaxe, stone_shovel, stone_sword, torch

### Iron (16 goals)

blast_furnace, bucket, chain, hopper, iron_axe, iron_bars, iron_hoe, iron_nugget, iron_pickaxe, iron_shovel, iron_sword, rail, shears, smithing_table, stonecutter, tripwire_hook

### Golden (6 goals)

gold_ingot, golden_axe, golden_hoe, golden_pickaxe, golden_shovel, golden_sword

### Diamond (7 goals)

diamond, diamond_axe, diamond_hoe, diamond_pickaxe, diamond_shovel, diamond_sword, jukebox

### Redstone (6 goals)

activator_rail, compass, dropper, note_block, piston, redstone_torch

### Armor (13 goals)

diamond_boots, diamond_chestplate, diamond_helmet, diamond_leggings, golden_boots, golden_chestplate, golden_helmet, golden_leggings, iron_boots, iron_chestplate, iron_helmet, iron_leggings, shield

**总计: 67 goals ✅**

---

## 项目结构（更新版）

```
ns-fsm-baseline/
├── README.md
├── requirements.txt
├── config/
│   ├── goals_67.json               # 67 goals 定义（含 group、type）
│   └── hyperparams.yaml            # max_steps=100, max_attempts=3, runs=15
├── src/
│   ├── __init__.py
│   ├── env_wrapper.py              # MC-TextWorld 封装
│   ├── llm_interface.py            # Ollama/Qwen2.5 接口
│   ├── action_parser.py            # LLM 输出 → MC-TextWorld action 解析
│   ├── ground_truth.py             # 从 action_lib 提取 GT dependency graph
│   ├── react_agent.py              # ReAct baseline 实现
│   ├── reflexion_agent.py          # Reflexion baseline 实现
│   ├── prompts.py                  # 所有 prompt 模板
│   ├── metrics.py                  # SR, Action Accuracy, Error 分析
│   └── logger.py                   # Trajectory 记录
├── scripts/
│   ├── run_react.py                # 运行 ReAct 实验
│   ├── run_reflexion.py            # 运行 Reflexion 实验
│   ├── extract_ground_truth.py     # 提取 GT graph
│   └── error_analysis.py           # Error analysis 主脚本
└── results/
    ├── trajectories/               # 原始 trajectory JSON
    ├── figures/                    # 生成的图表
    └── tables/                     # 生成的表格数据
```

**删除的模块**（ReAct/Reflexion 不需要）：

- ~~dependency_graph.py~~（ReAct 不构建 dependency graph）
- ~~deckard_agent.py~~
- ~~sc_agent.py~~

**新增的模块**：

- `action_parser.py`（LLM 输出到环境 action 的转换，含 fuzzy matching）
- `react_agent.py`
- `reflexion_agent.py`

---

## 评测指标

### 核心指标


| 指标                     | 定义                                      | 用途            |
| ------------------------ | ----------------------------------------- | --------------- |
| **Success Rate (SR)**    | 成功完成 goal 的比例，按 group 分桶       | 主要结果指标    |
| **Average Steps**        | 成功 episode 的平均步数                   | 效率对比        |
| **Invalid Action Rate**  | LLM 生成的 action 中无法被环境执行的比例  | Error analysis  |
| **Repeated Action Rate** | 连续重复相同 action 的比例                | 死循环检测      |
| **Performance Decay**    | SR 随 plan length（group 难度）的下降斜率 | 核心 claim 支撑 |

### Error Analysis 指标


| 指标                     | 定义                                      | 支撑什么论点               |
| ------------------------ | ----------------------------------------- | -------------------------- |
| **Plan Knowledge Error** | LLM 不知道正确 recipe 的比例              | FSM 的合法转移集 T(s) 解决 |
| **Sequencing Error**     | LLM 知道 recipe 但执行顺序错误的比例      | FSM 的状态转移约束解决     |
| **Dead Loop Rate**       | 连续 ≥5 步重复相同 action pattern 的比例 | 可达性分析 + 回退机制解决  |
| **Cascade Failure Rate** | 一个错误导致后续全部失败的比例            | Datalog 前置条件检查解决   |
| **Reflexion Fix Rate**   | Reflexion 相比 ReAct 修复的失败比例       | 论证"反思只修复认知错误"   |

---

## LLM 调用量估算


| 阶段           | 计算                                            | 调用次数       |
| -------------- | ----------------------------------------------- | -------------- |
| ReAct 执行     | 67 goals × 15 runs × ~25 steps/episode        | **~25,000 次** |
| Reflexion 执行 | 67 goals × 15 runs × ~25 steps × ~2 attempts | **~50,000 次** |
| Reflexion 反思 | 67 goals × 15 runs × ~1.5 reflections         | **~1,500 次**  |
| **总计**       |                                                 | **~76,500 次** |

使用本地 Ollama + Qwen2.5-7B：

- 每次调用 ~1-2 秒（含生成）
- 总时间 ~76,500 × 1.5s ≈ **32 小时**
- 可通过减少 runs（15→5 做 pilot）或并行加速

---

## 实施步骤

### Phase 1: 基础设施（Day 1）

- [X]  MC-TextWorld 安装验证
- [X]  实现 `env_wrapper.py`
  - [X]  封装 reset/step/get_inventory
  - [X]  处理 action 成功/失败的返回
  - [X]  记录 trajectory
- [X]  实现 `llm_interface.py`
  - [X]  Ollama 连接（openai 库兼容接口）
  - [X]  超时/重试处理
  - [X]  Response 解析
- [X]  实现 `action_parser.py`
  - [X]  精确匹配 + fuzzy matching
  - [X]  从 action_lib.json 加载合法 action 列表
- [X]  实现 `ground_truth.py`
  - [X]  从 action_lib 反向构建 GT dependency graph
  - [X]  每个 goal 的最优 action sequence

### Phase 2: Agent 实现（Day 2）

- [ ]  实现 `prompts.py`
  - [ ]  ReAct system prompt
  - [ ]  ReAct step prompt
  - [ ]  Reflexion reflection prompt
  - [ ]
  - [ ]  Reflexion retry prompt (with reflection injection)
- [ ]  实现 `react_agent.py`
  - [ ]  Thought + Action 循环
  - [ ]  Trajectory 管理（保留最近 K 步）
  - [ ]  终止条件（success / max_steps / dead loop detection）
- [ ]  实现 `reflexion_agent.py`
  - [ ]  继承 ReAct 的单 episode 逻辑
  - [ ]  添加 episode-level reflection
  - [ ]  Reflection 注入机制
  - [ ]  多次尝试管理

### Phase 3: Pilot 测试（Day 2-3）

- [ ]  Wooden group（10 goals × 1 run）快速验证
  - [ ]  确认 ReAct 能完成简单任务（stick, crafting_table）
  - [ ]  确认 action_parser 正常工作
  - [ ]  确认 trajectory logging 正常
- [ ]  检查 LLM 输出格式，调整 prompt 如需要
- [ ]  估算完整实验时间

### Phase 4: 完整实验（Day 3-5）

- [ ]  ReAct: 67 goals × 15 runs
- [ ]  Reflexion: 67 goals × 15 runs
- [ ]  导出所有 trajectory 数据

### Phase 5: Error Analysis + 报告（Day 5-7）

- [ ]  计算所有指标（SR, Invalid Action Rate, Dead Loop Rate 等）
- [ ]  按 group 分桶的 SR 对比表
- [ ]  Performance decay 曲线（SR vs. group 难度）
- [ ]  挑选 3-5 个典型失败 case study
- [ ]  撰写 Baseline Reproduction → Error Analysis → Reflection

---

## 验证计划

### 自动化验证

1. **GT Graph**: 确认覆盖所有 67 个 goal 的完整依赖链
2. **Pilot test**: 先用 Wooden group (10 goals × 1 run) 验证全 pipeline
3. **Sanity check**: 用 GT graph 的 optimal action sequence 跑环境，确认 100% 成功

### 手动验证

1. 抽样检查 ReAct 的 Thought 输出质量
2. 检查 Reflexion 的 reflection 内容是否有意义
3. 确认 action_parser 的 fuzzy matching 不会引入系统性偏差

---

## 风险与应对


| 风险                             | 影响                                         | 应对                                                           |
| -------------------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| Qwen2.5-7B 的 Minecraft 知识太弱 | 所有 goal 都失败，error analysis 没有梯度    | 这恰好支撑 NS-FSM 的论点；确保 Wooden group 有足够成功率做对照 |
| LLM 输出格式不稳定               | action_parser 大量失败                       | 加强 prompt 中的格式约束；提供 few-shot examples               |
| 实验时间超预期                   | 无法在 deadline 前完成                       | 先跑 5 runs 做 pilot，确认趋势后再跑 15 runs                   |
| MC-TextWorld action 名称匹配问题 | LLM 输出`mine_log` 但环境需要 `mine_oak_log` | action_parser 的 fuzzy matching 层处理                         |
