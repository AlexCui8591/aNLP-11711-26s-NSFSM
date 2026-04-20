# NS-FSM 实验方案（NeurIPS 2026）

## 概览

本文档涵盖 NS-FSM（Neuro-Symbolic Finite State Machine）项目从系统搭建到最终实验对比的完整实验路线。核心思路分三步走：**搭建 NS-FSM 基础设施 → 数据采集与合成 → 多维度对比实验**。

---

## 第一阶段：NS-FSM 系统搭建

### 1.1 Ground Truth 规则编写

**目标**：为每个实验环境定义完备的合法动作空间与状态转移规则。

**MC-TextWorld 规则集**：

- 定义实体类型（`item`, `location`, `container`, `tool`）及其属性谓词（`at(X, Loc)`, `in_inventory(X)`, `is_open(X)` 等）
- 编写动作前置条件与后效果规则，覆盖 67 个目标的全部动作空间：
  - `go(Dir)` — 前置：`connected(CurLoc, Dir, NextLoc)`；后效果：更新 `at(player, NextLoc)`
  - `take(Item)` — 前置：`at(Item, CurLoc) ∧ at(player, CurLoc) ∧ ¬in_container(Item, _)`
  - `open(Container)` — 前置：`at(Container, CurLoc) ∧ ¬is_open(Container)`
  - `cook(Item, Tool)` — 前置：`in_inventory(Item) ∧ at(Tool, CurLoc) ∧ is_cookable(Item)`
  - 其他：`drop`, `put`, `examine`, `eat`, `close`, `insert` 等
- 编写目标完成判定规则：`goal_satisfied(G) :- ...`
- 异常状态规则：死锁检测、不可达状态标记

**Robotouille 规则集**：

- 定义烹饪领域实体：`ingredient`, `station`, `recipe`, `dish`
- 编写多步烹饪流程规则：
  - `pick_up(Ingredient)` → `move_to(Station)` → `place(Ingredient, Station)` → `cook/chop/fry(Ingredient)` → `plate(Dish)`
- 定义工作站约束（同一时间只能处理一个食材）、食材状态流转（raw → chopped → cooked）
- 并行任务协调规则（多道菜同时准备时的资源调度）

**交付物**：每个环境一份 `.dl`（Datalog）规则文件 + 对应的 JSON schema 定义文件。

### 1.2 Datalog 引擎集成

**技术选型**：pyDatalog 或 Soufflé（推荐 pyDatalog 用于快速迭代，Soufflé 用于大规模推理性能）

**集成架构**：

```
Agent Action → Datalog Query Engine → {valid, invalid, violation_type}
                    ↑
            Ground Truth Rules (.dl)
            Current State Facts
```

**实现步骤**：

1. **State Tracker 模块**：维护当前世界状态的 fact base，每步更新
2. **Action Validator 模块**：接收 LLM 生成的动作，查询 Datalog 判定合法性
3. **Violation Classifier**：对非法动作分类（前置条件违反 / 类型错误 / 目标冲突 / 不存在的动作）
4. **Feedback Generator**：根据违反类型生成结构化反馈，供三层 fallback 机制使用

**Generate-then-Verify 流程**：

```
LLM 自由生成动作 → Datalog 验证 → 合法：执行并更新状态
                                  → 非法：触发 fallback
```

> 注意：这里不是 constrained decoding——LLM 先自由生成，Datalog 做后验校验。这是 NS-FSM 和传统 constrained generation 的关键区别。

### 1.3 FSM 搜索机制

**FSM 状态定义**：

- 状态节点 = 世界状态的抽象表示（关键谓词的 truth assignment）
- 转移边 = 合法动作
- 目标状态 = `goal_satisfied(G)` 为真的状态集合

**搜索策略**：

- **在线阶段（推理时）**：BFS/DFS + Datalog 剪枝，在 fallback 触发时搜索最近的可达合法状态
- 三层 Fallback 机制：
  1. **Informed Retry**：基于 violation type 提供修正提示，让 LLM 重新生成
  2. **Forced Choice**：Datalog 枚举当前状态下所有合法动作，强制 LLM 从中选择
  3. **Full Reset**：回溯到最近的已知安全状态，重新规划

**Z3 SMT Solver（离线/数据合成阶段）**：

- 用于数据合成时验证规则一致性、生成 adversarial violation 样本
- 验证 FSM 规则的完备性：是否存在死锁状态、不可达目标
- 生成边界条件测试用例（属于第二阶段数据合成工作）

---

## 第二阶段：数据采集与 DPO 训练数据构造

### 2.1 Rollout 策略

**目标**：在两个环境中用 NS-FSM 引导 rollout，收集 (chosen, rejected) 对用于 DPO 训练。

**MC-TextWorld Rollout**：

| 配置项 | 规格 |
|--------|------|
| 环境规模 | 67 goals，多房间多物品交互 |
| 基座模型 | Qwen2.5-7B（Ollama 本地推理）/ Llama 3 70B（正式实验） |
| Rollout 模式 | NS-FSM-guided：每步 generate → verify → (execute \| fallback) |
| 每个 goal 轨迹数 | chosen 3-5 条 × rejected 3-5 条 |
| chosen 定义 | NS-FSM 验证通过的合法轨迹，最终达成 goal |
| rejected 定义 | 触发 fallback 前 LLM 的原始非法动作序列 |
| 预估数据量 | 67 goals × ~8 条/goal ≈ 500-600 条轨迹对 |

**Robotouille Rollout**：

| 配置项 | 规格 |
|--------|------|
| 环境规模 | 多食谱多食材并行烹饪任务 |
| 基座模型 | 同上 |
| 任务复杂度梯度 | 单菜（3-5 步）→ 双菜并行（8-12 步）→ 多菜协调（15+ 步） |
| 每个任务轨迹数 | chosen 3-5 条 × rejected 3-5 条 |
| rejected 来源 | 工作站冲突、食材状态跳步、顺序错误等 |
| 预估数据量 | ~400-500 条轨迹对 |

**DPO 数据格式**：

```json
{
  "prompt": "<task_description> + <current_state> + <history>",
  "chosen": "合法动作序列（NS-FSM verified）",
  "rejected": "LLM 原始生成的非法/次优动作序列",
  "violation_type": "precondition_fail | type_error | goal_conflict",
  "trajectory_length": 12,
  "environment": "mc_textworld | robotouille"
}
```

### 2.2 三类 DPO 数据构造

| DPO 层级 | 数据来源 | β 值 | 对比粒度 |
|-----------|----------|------|----------|
| **Step-DPO** | 单步合法 vs 单步非法动作 | 0.3 | 动作级：同一状态下的 chosen/rejected 动作对 |
| **Sem-DPO** | 语义合理 vs 语义偏移的规划 | 0.1 | 片段级：3-5 步子序列的语义一致性对比 |
| **Outcome-DPO** | 完成目标 vs 未完成目标的完整轨迹 | 0.2 | 轨迹级：整条轨迹的最终结果对比 |

**数据增强（Z3 辅助）**：

- 用 Z3 在离线阶段生成 adversarial perturbation：微调合法轨迹中的单步制造 subtle violation
- 生成 near-miss 样本：差一步就成功的轨迹（提供更强的 learning signal）

### 2.3 资源配置

**开发/调试阶段**：

| 资源 | 配置 |
|------|------|
| 本地推理 | Qwen2.5-7B via Ollama（单卡 A6000 / RTX 4090） |
| Rollout 机器 | 1× GPU 节点，预计 2-3 天完成两个环境的 rollout |
| 存储 | ~50GB（轨迹数据 + 状态快照 + 日志） |

**正式训练阶段**：

| 资源 | 配置 |
|------|------|
| 基座模型 | Llama 3 70B（128K context） |
| GPU 集群 | AWS 8× A100 80GB（或等效），DeepSpeed ZeRO-3 |
| DPO 训练时间 | 每层 DPO 约 12-18 小时（70B 模型，~1000 条数据） |
| 三层总训练 | 约 3-5 天（含 checkpoint 评估） |
| 两阶段 Curriculum | Phase 1：Step-DPO → Phase 2：Sem-DPO + Outcome-DPO 联合 |

---

## 第三阶段：实验设计

### 3.1 实验一：有 / 无 NS-FSM 架构对比（Ablation Study）

**目的**：验证 NS-FSM 的 generate-then-verify 架构本身（不含 DPO 训练）的增益。

| 对比组 | 描述 |
|--------|------|
| **Llama 3 70B (vanilla)** | 无任何结构约束，纯 LLM 自由生成 |
| **Llama 3 70B + NS-FSM** | 加入 Datalog 验证 + 三层 fallback，但不做 DPO 训练 |

**评估指标**：

| 指标 | 说明 |
|------|------|
| Goal Completion Rate (GCR) | 成功完成目标的比例 |
| Average Steps to Completion | 完成目标的平均步数（效率） |
| Invalid Action Rate (IAR) | 非法动作占总动作的比例 |
| Fallback Trigger Rate | 各层 fallback 被触发的频率分布 |
| Recovery Success Rate | fallback 后成功恢复的比例 |

**在两个环境上分别跑**，按任务复杂度（短 horizon / 中 horizon / 长 horizon）分层汇报。

### 3.2 实验二：与 Baseline / SOTA 方法对比

**对比方法**：

| 方法 | 类型 | 说明 |
|------|------|------|
| **ReAct** | Baseline | Reason + Act 交替，已在 MC-TextWorld 复现 |
| **Reflexion** | Baseline | 自反思 + 重试，已在 MC-TextWorld 复现 |
| **LATS** | SOTA | Language Agent Tree Search，搜索增强 |
| **StateFlow** | SOTA | 状态流图驱动的 agent 控制 |
| **Agent Q** | SOTA | Q-learning 风格的 agent 决策 |
| **NS-FSM (ours, w/o DPO)** | Ours | 仅架构，无训练 |

**实验设计**：

- 所有方法使用相同基座模型（Llama 3 70B）
- 统一 prompt 格式（task description + observation + action history）
- 在 MC-TextWorld（67 goals）和 Robotouille 上分别评估
- 对每个方法跑 3 次取均值 + 标准差
- 按 horizon 长度分桶（短 < 10 步 / 中 10-20 步 / 长 > 20 步）

**核心假设验证**：

> Agent 在长 horizon 任务上的失败主要源于 **instruction-following 能力瓶颈**，而非 "Lost in the Middle" 的上下文长度问题。

- 证据 1：vanilla LLM 的 IAR 在长 horizon 任务上的增长模式（线性 vs 突变）
- 证据 2：NS-FSM 的 fallback 触发分布（如果是上下文问题，应在序列中后段集中；如果是 instruction-following 问题，应在复杂动作处集中，与位置无关）
- 证据 3：对比 128K context 和 32K context 下的表现差异（如果差异不显著，支持 instruction-following 假设）

### 3.3 实验三：三层 DPO 强化后对比

**训练流程**：

```
Llama 3 70B (base)
    │
    ├── Step-DPO (β=0.3) ──→ Checkpoint-1
    │
    ├── + Sem-DPO (β=0.1) ──→ Checkpoint-2
    │
    └── + Outcome-DPO (β=0.2) ──→ Checkpoint-3 (final)
```

**两阶段 Curriculum Learning**：

- **Phase 1**：Step-DPO 单独训练（学会单步合法性）
- **Phase 2**：Sem-DPO + Outcome-DPO 联合训练（学会语义连贯性和目标导向规划）

**对比矩阵**：

| 模型 | 描述 |
|------|------|
| Llama 3 70B (vanilla) | 无结构、无训练 |
| NS-FSM (w/o DPO) | 有结构、无训练 |
| NS-FSM + Step-DPO only | 有结构 + 单步优化 |
| NS-FSM + Step + Sem DPO | 有结构 + 单步 + 语义优化 |
| NS-FSM + 三层 DPO (full) | 有结构 + 完整训练 |

**关键分析维度**：

1. **DPO 层级增量分析**：每加一层 DPO，GCR / IAR / 平均步数的变化曲线
2. **Curriculum vs 直接联合训练**：验证两阶段课程学习是否优于一次性联合训练所有三层
3. **NS-FSM 是否可"毕业"**：训练充分后，移除 NS-FSM 验证层（纯 LLM 推理），观察性能退化幅度
   - 如果退化很小 → NS-FSM 成功将结构化知识蒸馏进了 LLM
   - 如果退化很大 → 在线 verification 仍然必要，但 DPO 降低了 fallback 频率
4. **长 horizon 任务专项分析**：三层 DPO 是否显著缩小了长 horizon 与短 horizon 任务之间的性能差距

---

## 实验时间线

| 阶段 | 任务 | 预计周期 |
|------|------|----------|
| W1-W2 | 规则编写 + Datalog 集成 + FSM 搜索实现 | 2 周 |
| W3 | 两个环境的 rollout + 数据收集 | 1 周 |
| W4 | DPO 数据清洗 + 格式化 + Z3 数据增强 | 1 周 |
| W5-W6 | 实验一（有/无 NS-FSM ablation） | 1-2 周 |
| W6-W7 | 实验二（vs baseline / SOTA） | 1-2 周 |
| W8-W10 | 三层 DPO 训练 + 实验三 | 2-3 周 |
| W11-W12 | 结果分析 + 论文写作 | 2 周 |

> 总周期约 12 周，NeurIPS 2026 投稿截止前留出 2 周 buffer 做 rebuttal 准备。

---

## 关键风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| 70B 模型训练资源不足 | 无法完成三层 DPO | 先用 7B 模型验证 pipeline，确认有效后再 scale up |
| Robotouille 规则复杂度超预期 | 规则编写周期延长 | 先覆盖核心烹饪流程，逐步扩展边界情况 |
| DPO 数据质量不足 | 训练效果不佳 | 增加 rollout 轮次 + Z3 augmentation 补充 |
| Baseline 复现结果与原文不一致 | 对比不公平 | 联系原作者确认超参，使用官方代码库 |
| instruction-following 假设不成立 | 论文 motivation 需要调整 | 设计 context length 消融实验作为 backup 证据 |
