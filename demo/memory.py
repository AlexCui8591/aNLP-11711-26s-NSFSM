"""
结构化长上下文记忆管理器
全量记录 Raw Observation，不做压缩，
以测试 Lost in the Middle 现象。
"""


class StructuredMemory:

    def __init__(self, env):
        self.env = env
        self.step_history = []
        self.current_inventory = {}

    def reset(self):
        self.step_history = []
        self.current_inventory = {}

    def update(self, action_str, result_msg, current_inventory):
        self.step_history.append(f"{action_str} → {result_msg}")
        self.current_inventory = current_inventory

    # ------------------------------------------------------------------ #
    #  Prompt 生成
    # ------------------------------------------------------------------ #
    def get_prompt_context(self):
        cfg = self.env.difficulty_config
        survival = self.env.survival_enabled

        # ---- 目标 ----
        goal = f"【最终目标】合成 {cfg['target']}\n"
        if cfg.get("distractors"):
            goal += f"【支线任务】完成最终目标前，必须先拥有：{', '.join(cfg['distractors'])}\n"

        # ---- 生物群系地图 ----
        biome_text = "【世界地图】你可以在以下区域之间移动（使用 '移动 <地点>'）：\n"
        for name, info in self.env.BIOMES.items():
            acts = list(self.env.BIOME_ACTIONS.get(name, {}).keys())
            if name == "矿洞":
                acts.append("挖矿 <矿石>")
            biome_text += f"  · {name}：{info['description']}（可用动作：{', '.join(acts)}）\n"

        # ---- 合成配方 ----
        recipe_text = "【合成配方表】（标注 [需工作台] 的配方必须先拥有工作台）\n"
        for item, r in self.env.recipes.items():
            mats = " + ".join(f"{k}×{v}" for k, v in r["材料"].items())
            wb = " [需工作台]" if r.get("需要工作台") else ""
            recipe_text += f"  {mats} => {item}×{r['产出']}{wb}\n"

        # ---- 熔炼配方 ----
        smelt_text = "【熔炼配方】（需拥有熔炉）\n"
        for src, info in self.env.SMELTING.items():
            smelt_text += f"  {src} => {info['产出']}×{info['数量']}\n"

        # ---- 挖矿需求 ----
        mine_text = "【挖矿工具需求】（必须在矿洞中执行）\n"
        for ore, req in self.env.MINING_REQS.items():
            mine_text += f"  {ore}：需要 {' 或 '.join(req['tools'])}\n"

        # ---- 工具耐久 ----
        dur_info = self.env.get_tool_durability_info()
        if dur_info:
            dur_text = "【工具耐久】耐久归零时工具损坏，需重新合成：\n"
            for tool, d in dur_info.items():
                dur_text += f"  {tool}: {d}\n"
        else:
            dur_text = "【工具耐久】当前无工具。\n"

        # ---- 生存状态 ----
        surv_text = ""
        if survival:
            time_str = "夜晚 ⚠" if self.env._is_night() else "白天"
            surv_text = (
                f"【生存状态】HP: {self.env.hp}/{self.env.max_hp} | "
                f"饥饿: {self.env.hunger}/{self.env.max_hunger} | 时间: {time_str}\n"
                f"  · 每 3 步饥饿值 -1；饥饿 ≤5 时 HP 下降；饥饿 ≥15 时 HP 自然恢复\n"
                f"  · 夜晚无庇护会受到怪物攻击（矿洞危险系数最高）\n"
                f"  · 可用 '吃 <食物>' 恢复饥饿，可食用：苹果(+4) 面包(+5) 熟肉(+8) 生肉(+2)\n"
                f"  · 建造避难所或拥有床可在夜晚获得保护\n"
            )
            if self.env.has_shelter:
                surv_text += "  ★ 你已拥有避难所，夜晚安全。\n"

        # ---- 历史 ----
        hist = "无"
        if self.step_history:
            hist = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(self.step_history))

        # ---- 当前背包 ----
        inv = "，".join(f"{k}×{v}" for k, v in self.current_inventory.items() if v > 0)
        if not inv:
            inv = "空"

        # ---- 完整 Prompt ----
        prompt = f"""你是一个生存在 Minecraft 世界中的 Agent。请通过严格逻辑推理完成任务。

{goal}
{biome_text}
{recipe_text}
{smelt_text}
{mine_text}
【生存与合成流程指引】
⚠ 每次行动前必须执行以下三步：
  1. 读取【当前状态】中的背包、位置、工具耐久
  2. 检查【通用逃生规则】是否触发
  3. 对照【阶段判断表】执行第一个满足条件的步骤

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【通用逃生规则】（优先级最高，任何阶段均适用）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule-1  同一动作连续执行 3 次且背包无变化
        → 停止该动作，从阶段判断表顶部重新匹配

Rule-2  工具耐久 = 0 或工具损坏
        → 立即跳转到对应工具的合成步骤，重新合成后继续

Rule-3  HP ≤ 5
        → 立即执行 '吃 <食物>'，背包无食物则先 '打猎' 或 '采集 苹果'
        → 生存处理完毕前禁止执行其他动作

Rule-4  连续 5 步未推进主线（背包关键物品无变化）
        → 输出当前背包，重新从阶段判断表顶部匹配，不得重复上一个动作

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【阶段判断表】（从上到下逐条检查，执行第一个满足的）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[阶段0] 原木收集
  条件：背包无工作台 且 原木 < 4
  动作：砍树
  退出：原木 ≥ 4 → 立即进入阶段1，⛔ 禁止继续砍树

[阶段1] 基础工具合成
  条件：背包无工作台 且 原木 ≥ 4
  动作：依次执行 合成 木板 → 合成 木棍 → 合成 工作台
  退出：背包有工作台 → 进入阶段2
  ⛔ 此阶段禁止砍树

[阶段2] 木镐合成
  条件：有工作台 且 无木镐
  动作：合成 木镐
  退出：背包有木镐 → 进入阶段3
  ⛔ 禁止重复合成已有工具

[阶段3] 前往矿洞
  条件：有木镐 且 无石镐 且 位置 ≠ 矿洞
  动作：移动 矿洞

[阶段4] 圆石收集
  条件：有木镐 且 无石镐 且 位置 = 矿洞
  动作：挖矿 圆石
  退出：圆石 ≥ 8 → 进入阶段5
  失败处理：连续2次失败 → 检查木镐耐久，损坏则返回阶段2重新合成

[阶段5] 石器合成
  条件：有木镐 且 圆石 ≥ 8 且 无石镐
  动作：依次执行 合成 石镐 → 合成 熔炉
  退出：有石镐 且 有熔炉 → 进入阶段6

[阶段6] 铁矿收集
  条件：有石镐 且 铁矿石 < 6 且 位置 = 矿洞
  动作：挖矿 铁矿石
  退出：铁矿石 ≥ 6 → 进入阶段7
  失败处理：连续2次失败 → 检查石镐耐久，损坏则返回阶段5重新合成

[阶段6-前置] 若位置 ≠ 矿洞
  条件：有石镐 且 无铁锭 且 位置 ≠ 矿洞
  动作：移动 矿洞

[阶段7] 铁器合成
  条件：有熔炉 且 铁矿石 ≥ 6 且 无铁镐
  动作：依次执行 熔炼 铁矿石 → 合成 铁镐
  退出：有铁镐 → 进入阶段8

[阶段8] 钻石收集
  条件：有铁镐 且 钻石 < 3 且 位置 = 矿洞
  动作：挖矿 钻石
  退出：钻石 ≥ 3 → 进入阶段9
  失败处理：连续2次失败 → 检查铁镐耐久，损坏则返回阶段7重新合成

[阶段8-前置] 若位置 ≠ 矿洞
  条件：有铁镐 且 无足够钻石 且 位置 ≠ 矿洞
  动作：移动 矿洞

[阶段9] 钻石镐合成
  条件：钻石 ≥ 3 且 无钻石镐
  动作：合成 钻石镐
  退出：有钻石镐 → 进入阶段10

[阶段10] 高级材料收集
  条件：有钻石镐
  动作：按支线任务需求依次收集（黑曜石 / 皮革 / 甘蔗等）
  规则：每个支线目标独立检查背包，已有则跳过，不重复收集

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【全局禁止行为】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛔ 背包原木 ≥ 4 时禁止砍树
⛔ 已有某工具/工作台/熔炉时禁止重复合成
⛔ HP ≤ 5 时禁止执行任何非生存动作
⛔ 同一动作连续3次无收益时必须触发 Rule-1

【动作历史记录】(以下是你过去的所有动作和结果，这段记录会越来越长)
{hist}

【当前状态】
位置：{self.env.current_biome}
背包：{inv}
{dur_text.strip()}
{surv_text.strip()}

【可用动作指令】
1.  '砍树'           → 获得原木 (需在森林)
2.  '打猎'           → 获得生肉/皮革(森林) 或 生肉/羊毛(平原)
3.  '采集 <物品>'    → 苹果(森林) / 小麦(平原) / 甘蔗(平原) / 沙子(沙漠)
4.  '挖矿 <矿石>'   → 需在矿洞，且需对应等级的镐
5.  '熔炼 <原料>'    → 铁矿石/金矿石/沙子/生肉 (需熔炉)
6.  '合成 <物品>'    → 参考配方表 (部分需要工作台)
7.  '移动 <地点>'    → 森林 / 矿洞 / 平原 / 沙漠
8.  '吃 <食物>'      → 苹果/面包/熟肉/生肉 → 恢复饥饿值
9.  '建造 避难所'    → 消耗 16木板+16圆石+1门 → 夜晚安全
10. '睡觉'           → 需要床或避难所，夜晚使用
11. '打水'           → 需要铁桶

你是 Minecraft 生存助手。每次只输出如下格式，不要输出任何其他内容：

<thought>简短思考</thought>
<action>具体动作</action>

示例1：
<thought>需要木板，先砍树获取原木</thought>
<action>砍树</action>

示例2：
<thought>有了原木，合成木板</thought>
<action>合成 木板</action>

示例3：
<thought>需要去矿洞挖石头</thought>
<action>移动 矿洞</action>

现在请根据当前状态行动：
重要：你的回复必须以 <thought> 开始，以 </action> 结束。中间不要有其他内容。
"""
        return prompt


class OracleMemory:
    """
    Oracle 记忆管理器（无历史缓存版）
    不保留 action history，直接提供当前状态给模型。
    用于作为 Baseline 证明性能降低来自于上下文污染。
    """

    def __init__(self, env):
        self.env = env
        self.step_history = []  # 保留用于兼容 detector，但不塞入 prompt
        self.current_inventory = {}

    def reset(self):
        self.step_history = []
        self.current_inventory = {}

    def update(self, action_str, result_msg, current_inventory):
        # 记录给 detector 进行数据统计用
        self.step_history.append(f"{action_str} → {result_msg}")
        self.current_inventory = current_inventory

    def get_prompt_context(self):
        cfg = self.env.difficulty_config
        survival = self.env.survival_enabled

        # ---- 目标 ----
        goal = f"【最终目标】合成 {cfg['target']}\n"
        if cfg.get("distractors"):
            goal += f"【支线任务】完成最终目标前，必须先拥有：{', '.join(cfg['distractors'])}\n"

        # ---- 生物群系地图 ----
        biome_text = "【世界地图】你可以在以下区域之间移动（使用 '移动 <地点>'）：\n"
        for name, info in self.env.BIOMES.items():
            acts = list(self.env.BIOME_ACTIONS.get(name, {}).keys())
            if name == "矿洞":
                acts.append("挖矿 <矿石>")
            biome_text += f"  · {name}：{info['description']}（可用动作：{', '.join(acts)}）\n"

        # ---- 合成配方 ----
        recipe_text = "【合成配方表】（标注 [需工作台] 的配方必须先拥有工作台）\n"
        for item, r in self.env.recipes.items():
            mats = " + ".join(f"{k}×{v}" for k, v in r["材料"].items())
            wb = " [需工作台]" if r.get("需要工作台") else ""
            recipe_text += f"  {mats} => {item}×{r['产出']}{wb}\n"

        # ---- 熔炼配方 ----
        smelt_text = "【熔炼配方】（需拥有熔炉）\n"
        for src, info in self.env.SMELTING.items():
            smelt_text += f"  {src} => {info['产出']}×{info['数量']}\n"

        # ---- 挖矿需求 ----
        mine_text = "【挖矿工具需求】（必须在矿洞中执行）\n"
        for ore, req in self.env.MINING_REQS.items():
            mine_text += f"  {ore}：需要 {' 或 '.join(req['tools'])}\n"

        # ---- 工具耐久 ----
        dur_info = self.env.get_tool_durability_info()
        if dur_info:
            dur_text = "【工具耐久】耐久归零时工具损坏，需重新合成：\n"
            for tool, d in dur_info.items():
                dur_text += f"  {tool}: {d}\n"
        else:
            dur_text = "【工具耐久】当前无工具。\n"

        # ---- 生存状态 ----
        surv_text = ""
        if survival:
            time_str = "夜晚 ⚠" if self.env._is_night() else "白天"
            surv_text = (
                f"【生存状态】HP: {self.env.hp}/{self.env.max_hp} | "
                f"饥饿: {self.env.hunger}/{self.env.max_hunger} | 时间: {time_str}\n"
                f"  · 每 3 步饥饿值 -1；饥饿 ≤5 时 HP 下降；饥饿 ≥15 时 HP 自然恢复\n"
                f"  · 夜晚无庇护会受到怪物攻击（矿洞危险系数最高）\n"
                f"  · 可用 '吃 <食物>' 恢复饥饿，可食用：苹果(+4) 面包(+5) 熟肉(+8) 生肉(+2)\n"
                f"  · 建造避难所或拥有床可在夜晚获得保护\n"
            )
            if self.env.has_shelter:
                surv_text += "  ★ 你已拥有避难所，夜晚安全。\n"

        # ---- 当前背包 ----
        inv = "，".join(f"{k}×{v}" for k, v in self.current_inventory.items() if v > 0)
        if not inv:
            inv = "空"

        # ---- 完整 Prompt (移除动作历史记录) ----
        prompt = f"""你是一个生存在 Minecraft 世界中的 Agent。请通过严格逻辑推理完成任务。

{goal}
{biome_text}
{recipe_text}
{smelt_text}
{mine_text}
【生存与合成流程指引】(请务必遵循此发展阶梯)
1. 徒手阶段：在森林使用 '砍树' 获得原木，合成木板→木棍→工作台。
2. 木器阶段：用工作台合成 '木镐'。
3. 石器阶段：'移动 矿洞'，用木镐 '挖矿 圆石'，合成 '石镐' 和 '熔炉'。
4. 铁器阶段：用石镐 '挖矿 铁矿石'，'熔炼 铁矿石' 得铁锭，合成 '铁镐'。
5. 钻石阶段：用铁镐 '挖矿 钻石'，合成 '钻石镐'。
6. 高级阶段：钻石镐可挖黑曜石；纸+皮革→书；书+木板→书架；钻石+黑曜石+书→附魔台。
注意：不要跳过任何阶段！不要重复合成已有的工具/工作台/熔炉！
      工具有耐久度，损坏后需重新合成。注意管理工具状态。

【当前状态】
位置：{self.env.current_biome}
背包：{inv}
{dur_text.strip()}
{surv_text.strip()}

【可用动作指令】
1.  '砍树'           → 获得原木 (需在森林)
2.  '打猎'           → 获得生肉/皮革(森林) 或 生肉/羊毛(平原)
3.  '采集 <物品>'    → 苹果(森林) / 小麦(平原) / 甘蔗(平原) / 沙子(沙漠)
4.  '挖矿 <矿石>'   → 需在矿洞，且需对应等级的镐
5.  '熔炼 <原料>'    → 铁矿石/金矿石/沙子/生肉 (需熔炉)
6.  '合成 <物品>'    → 参考配方表 (部分需要工作台)
7.  '移动 <地点>'    → 森林 / 矿洞 / 平原 / 沙漠
8.  '吃 <食物>'      → 苹果/面包/熟肉/生肉 → 恢复饥饿值
9.  '建造 避难所'    → 消耗 16木板+16圆石+1门 → 夜晚安全
10. '睡觉'           → 需要床或避难所，夜晚使用
11. '打水'           → 需要铁桶

你是 Minecraft 生存助手。每次只输出如下格式，不要输出任何其他内容：

<thought>简短思考</thought>
<action>具体动作</action>

示例1：
<thought>需要木板，先砍树获取原木</thought>
<action>砍树</action>

示例2：
<thought>有了原木，合成木板</thought>
<action>合成 木板</action>

示例3：
<thought>需要去矿洞挖石头</thought>
<action>移动 矿洞</action>

现在请根据当前状态行动：
重要：你的回复必须以 <thought> 开始，以 </action> 结束。中间不要有其他内容。
"""
        return prompt
