"""
Lost in the Middle 检测器（合并版）
9 类信号：6 类核心 + 3 类增强环境信号
"""
import re
from collections import Counter


class LostInMiddleDetector:
    """检测 LLM Agent 在长上下文中的认知退化信号。"""

    def __init__(self):
        self.signals = []

    # ================================================================== #
    #  主入口
    # ================================================================== #
    def detect_all(self, step_num, action, thought, pre_inventory, history,
                   action_success, action_message, env_state=None):
        """对单步执行全部检测，返回本步检出的信号列表。"""
        sigs = []

        # 核心 6 类
        _a = self._detect_redundant_craft(step_num, action, pre_inventory)
        if _a: sigs.append(_a)
        _a = self._detect_redundant_gather(step_num, action, pre_inventory)
        if _a: sigs.append(_a)
        _a = self._detect_inventory_hallucination(step_num, thought, pre_inventory)
        if _a: sigs.append(_a)
        _a = self._detect_prerequisite_skip(step_num, action, action_success, action_message)
        if _a: sigs.append(_a)
        _a = self._detect_loop(step_num, action, history)
        if _a: sigs.append(_a)
        _a = self._detect_goal_drift(step_num, action, action_success, history)
        if _a: sigs.append(_a)

        # 增强环境 3 类（需要 env_state）
        if env_state:
            _a = self._detect_spatial_confusion(step_num, action, action_success, action_message)
            if _a: sigs.append(_a)
            _a = self._detect_survival_neglect(step_num, action, env_state)
            if _a: sigs.append(_a)
            _a = self._detect_tool_ignorance(step_num, action, thought, env_state)
            if _a: sigs.append(_a)

        self.signals.extend(sigs)
        return sigs

    def get_summary(self):
        by_type = Counter(s["type"] for s in self.signals)
        by_severity = Counter(s["severity"] for s in self.signals)
        return {
            "total": len(self.signals),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "first_signal_step": self.signals[0]["step"] if self.signals else None,
        }

    # ================================================================== #
    #  信号 1：冗余合成
    # ================================================================== #
    def _detect_redundant_craft(self, step_num, action, inventory):
        """已拥有的唯一物品（工作台、熔炉、各级镐）再次合成。"""
        unique_items = {"工作台", "熔炉", "木镐", "石镐", "铁镐", "钻石镐", "床", "附魔台"}
        if not action.startswith("合成"):
            return None
        item = action.replace("合成", "").strip()
        for sep in ("*", " x", "×"):
            if sep in item:
                item = item.split(sep)[0].strip()
        if item in unique_items and inventory.get(item, 0) >= 1:
            return {"type": "redundant_craft", "step": step_num,
                    "detail": f"背包已有{item}×{inventory[item]}，仍尝试合成",
                    "severity": "high"}
        return None

    # ================================================================== #
    #  信号 2：冗余采集
    # ================================================================== #
    def _detect_redundant_gather(self, step_num, action, inventory):
        """材料已远超需求仍在采集。"""
        if action == "砍树":
            total = inventory.get("木板", 0) + inventory.get("原木", 0) * 4
            if total >= 40:
                return {"type": "redundant_gather", "step": step_num,
                        "detail": f"木材充足(≈{total}木板当量)，仍在砍树",
                        "severity": "medium"}
        elif "圆石" in action and action.startswith("挖矿"):
            if inventory.get("圆石", 0) >= 25:
                return {"type": "redundant_gather", "step": step_num,
                        "detail": f"圆石已有{inventory['圆石']}个，仍在挖",
                        "severity": "medium"}
        elif "钻石" in action and action.startswith("挖矿"):
            if inventory.get("钻石", 0) >= 16:
                return {"type": "redundant_gather", "step": step_num,
                        "detail": f"钻石已有{inventory['钻石']}个，仍在挖",
                        "severity": "medium"}
        return None

    # ================================================================== #
    #  信号 3：背包幻觉
    # ================================================================== #
    def _detect_inventory_hallucination(self, step_num, thought, inventory):
        """思考中声称的物品数量与实际不符。"""
        if not thought:
            return None

        excl = r'，。！：\n需要求合制造配方产出'
        patterns = [
            (r'(?:我有|背包[里中]?有|已有|拥有|获得了?)[^' + excl +
             r']{0,8}?(\d+)[^' + excl + r']{0,3}?个[^' + excl +
             r']{0,4}?(铁锭|钻石|铁镐|石镐|木镐|熔炉|工作台|圆石|铁矿石'
             r'|木板|木棍|原木|黑曜石|金锭|煤炭|皮革|羊毛|小麦|甘蔗)', 'have'),
            (r'(?:完全没有|0个|从未拥有|背包[里中]?没有)[^' + excl +
             r']{0,8}?(铁锭|钻石|铁镐|石镐|木镐|熔炉|工作台|圆石|铁矿石'
             r'|木板|木棍|原木|黑曜石|金锭|煤炭|皮革|羊毛|小麦|甘蔗)', 'zero'),
        ]
        for pat, kind in patterns:
            for m in re.finditer(pat, thought):
                if kind == 'have':
                    claimed, item = int(m.group(1)), m.group(2)
                    actual = inventory.get(item, 0)
                    if abs(claimed - actual) > 1:
                        return {"type": "inventory_hallucination", "step": step_num,
                                "detail": f"声称有{item}×{claimed}，实际×{actual}",
                                "severity": "high"}
                elif kind == 'zero':
                    item = m.group(1)
                    if inventory.get(item, 0) >= 2:
                        return {"type": "inventory_hallucination", "step": step_num,
                                "detail": f"声称没有{item}，实际有×{inventory[item]}",
                                "severity": "high"}
        return None

    # ================================================================== #
    #  信号 4：前置跳过
    # ================================================================== #
    def _detect_prerequisite_skip(self, step_num, action, success, message):
        """跳过前置工具阶段（区别于单纯的材料不足）。"""
        if success:
            return None
        tool_keywords = ("工具", "需要以下工具", "镐", "没有可用的工具",
                         "需要工作台", "没有熔炉")
        mat_keywords = ("材料不足", "缺")
        is_tool = any(k in message for k in tool_keywords)
        is_mat = any(k in message for k in mat_keywords)
        if is_tool and not is_mat:
            return {"type": "prerequisite_skip", "step": step_num,
                    "detail": f"尝试{action}但缺少前置工具: {message}",
                    "severity": "high"}
        return None

    # ================================================================== #
    #  信号 5：动作循环 — 修复旧版 threshold bug
    # ================================================================== #
    def _detect_loop(self, step_num, action, history):
        """滑动窗口 8 步内同一动作出现 ≥5 次。"""
        WINDOW = 8
        THRESHOLD = 5
        if len(history) < WINDOW - 1:
            return None
        recent = []
        for h in history[-(WINDOW - 1):]:
            if "→" in h:
                a = h.split("→")[0].strip()
                if ":" in a:
                    a = a.split(":", 1)[1].strip()
                recent.append(a)
        recent.append(action)
        counts = Counter(recent)
        for act, cnt in counts.items():
            if act and cnt >= THRESHOLD:
                return {"type": "action_loop", "step": step_num,
                        "detail": f"最近{WINDOW}步中'{act}'出现{cnt}次",
                        "severity": "high"}
        return None

    # ================================================================== #
    #  信号 6：目标漂移 — 改为检查成功的主线动作
    # ================================================================== #
    def _detect_goal_drift(self, step_num, action, action_success, history):
        """最近 10 步无任何成功的主线推进。"""
        if len(history) < 15:
            return None
        mainline = {"木镐", "石镐", "铁镐", "钻石镐", "铁锭", "铁矿石",
                     "钻石", "熔炉", "熔炼", "黑曜石", "附魔台", "金锭",
                     "书", "纸", "书架"}
        for h in history[-10:]:
            if "成功" in h and any(kw in h for kw in mainline):
                return None
        return {"type": "goal_drift", "step": step_num,
                "detail": "最近10步无任何成功的主线推进动作",
                "severity": "medium"}

    # ================================================================== #
    #  信号 7（新）：空间困惑
    # ================================================================== #
    def _detect_spatial_confusion(self, step_num, action, success, message):
        """在错误的生物群系尝试地点限定动作。"""
        if not success and ("无法执行" in message or "需要前往" in message
                            or "需要在" in message):
            return {"type": "spatial_confusion", "step": step_num,
                    "detail": f"在错误地点执行{action}: {message}",
                    "severity": "high"}
        return None

    # ================================================================== #
    #  信号 8（新）：生存忽视
    # ================================================================== #
    def _detect_survival_neglect(self, step_num, action, env_state):
        """HP 或饥饿危急时仍执行非生存动作。"""
        hp = env_state.get("hp", 20)
        hunger = env_state.get("hunger", 20)
        survival_prefixes = ("吃", "建造", "睡觉", "移动")
        is_survival = any(action.startswith(p) for p in survival_prefixes)

        if hp <= 5 and not is_survival:
            return {"type": "survival_neglect", "step": step_num,
                    "detail": f"HP仅剩{hp}，仍在执行: {action}",
                    "severity": "high"}
        if hunger <= 3 and not action.startswith("吃"):
            inv = env_state.get("inventory", {})
            has_food = any(inv.get(f, 0) > 0
                          for f in ("苹果", "面包", "熟肉", "生肉"))
            if has_food:
                return {"type": "survival_neglect", "step": step_num,
                        "detail": f"饥饿值{hunger}且有食物，但未进食: {action}",
                        "severity": "medium"}
        return None

    # ================================================================== #
    #  信号 9（新）：工具耐久忽视
    # ================================================================== #
    def _detect_tool_ignorance(self, step_num, action, thought, env_state):
        """挖矿时工具即将损坏但思考中未提及。"""
        if not action.startswith("挖矿"):
            return None
        dur = env_state.get("tool_durability", {})
        for tool, remaining in dur.items():
            if "镐" in tool and remaining <= 2:
                if thought and "耐久" not in thought and "损坏" not in thought:
                    return {"type": "tool_ignorance", "step": step_num,
                            "detail": f"{tool}仅剩{remaining}耐久，思考中未提及",
                            "severity": "medium"}
        return None
