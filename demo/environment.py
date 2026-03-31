"""
增强版 Minecraft 生存 + 合成环境
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
真实 Minecraft 机制：
  · 30+ 合成 / 4 种熔炼配方
  · 4 大生物群系 + 移动
  · 工具耐久系统
  · HP / 饥饿值 / 昼夜循环
  · 夜晚怪物攻击（护甲减伤）
  · 随机采集产出
"""
import random


# ================================================================== #
#  难度配置（统一定义，供 main / experiment 共用）
# ================================================================== #
DIFFICULTY_CONFIGS = {
    "short": {
        "target": "石镐",
        "distractors": [],
        "survival": False,
        "max_steps": 30,
        "expected_steps": 15,
        "grace_cycles": 0,
        "description": "基础合成链：原木→工作台→木镐→移动矿洞→圆石→石镐",
    },
    "medium": {
        "target": "铁镐",
        "distractors": ["火把", "面包"],
        "survival": True,
        "max_steps": 60,
        "expected_steps": 35,
        "grace_cycles": 1,
        "description": "铁器时代：需要熔炼铁矿 + 生存管理 + 支线任务",
    },
    "long": {
        "target": "钻石镐",
        "distractors": ["火把", "床", "铁剑", "皮革胸甲"],
        "survival": True,
        "max_steps": 100,
        "expected_steps": 70,
        "grace_cycles": 1,
        "description": "钻石时代：完整科技树 + 多群系探索 + 生存挑战",
    },
    "extreme": {
        "target": "附魔台",
        "distractors": ["钻石剑", "钻石胸甲", "床", "书架", "盾牌", "面包"],
        "survival": True,
        "max_steps": 150,
        "expected_steps": 120,
        "grace_cycles": 2,
        "description": "终极挑战：附魔台(黑曜石+书+钻石) + 大量支线 + 全面生存压力",
    },
}


class MinecraftSurvivalEnv:
    """RL-style 文本环境：step() → (obs, reward, terminated, info)"""

    # ============================================================== #
    #  静态数据表
    # ============================================================== #

    # ---------- 合成配方 ----------
    RECIPES = {
        # 基础
        "木板":   {"材料": {"原木": 1}, "产出": 4, "需要工作台": False},
        "木棍":   {"材料": {"木板": 2}, "产出": 4, "需要工作台": False},
        "工作台": {"材料": {"木板": 4}, "产出": 1, "需要工作台": False},
        # 木质工具
        "木镐":   {"材料": {"木板": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        "木剑":   {"材料": {"木板": 2, "木棍": 1}, "产出": 1, "需要工作台": True},
        "木斧":   {"材料": {"木板": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        # 石质工具
        "石镐":   {"材料": {"圆石": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        "石剑":   {"材料": {"圆石": 2, "木棍": 1}, "产出": 1, "需要工作台": True},
        "石斧":   {"材料": {"圆石": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        # 铁质工具
        "铁镐":   {"材料": {"铁锭": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        "铁剑":   {"材料": {"铁锭": 2, "木棍": 1}, "产出": 1, "需要工作台": True},
        "铁斧":   {"材料": {"铁锭": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        # 钻石工具
        "钻石镐": {"材料": {"钻石": 3, "木棍": 2}, "产出": 1, "需要工作台": True},
        "钻石剑": {"材料": {"钻石": 2, "木棍": 1}, "产出": 1, "需要工作台": True},
        # 功能方块
        "熔炉":   {"材料": {"圆石": 8}, "产出": 1, "需要工作台": True},
        "火把":   {"材料": {"煤炭": 1, "木棍": 1}, "产出": 4, "需要工作台": False},
        "箱子":   {"材料": {"木板": 8}, "产出": 1, "需要工作台": True},
        "门":     {"材料": {"木板": 6}, "产出": 1, "需要工作台": True},
        "梯子":   {"材料": {"木棍": 7}, "产出": 3, "需要工作台": True},
        "床":     {"材料": {"羊毛": 3, "木板": 3}, "产出": 1, "需要工作台": True},
        "铁桶":   {"材料": {"铁锭": 3}, "产出": 1, "需要工作台": True},
        "盾牌":   {"材料": {"木板": 6, "铁锭": 1}, "产出": 1, "需要工作台": True},
        # 护甲
        "皮革胸甲": {"材料": {"皮革": 8}, "产出": 1, "需要工作台": True},
        "铁胸甲":   {"材料": {"铁锭": 8}, "产出": 1, "需要工作台": True},
        "钻石胸甲": {"材料": {"钻石": 8}, "产出": 1, "需要工作台": True},
        # 食物
        "面包":   {"材料": {"小麦": 3}, "产出": 1, "需要工作台": True},
        # 高级
        "纸":     {"材料": {"甘蔗": 3}, "产出": 3, "需要工作台": True},
        "书":     {"材料": {"纸": 3, "皮革": 1}, "产出": 1, "需要工作台": True},
        "书架":   {"材料": {"木板": 6, "书": 3}, "产出": 1, "需要工作台": True},
        "附魔台": {"材料": {"钻石": 2, "黑曜石": 4, "书": 1}, "产出": 1, "需要工作台": True},
    }

    # ---------- 熔炼配方 ----------
    SMELTING = {
        "铁矿石": {"产出": "铁锭", "数量": 1},
        "金矿石": {"产出": "金锭", "数量": 1},
        "沙子":   {"产出": "玻璃", "数量": 1},
        "生肉":   {"产出": "熟肉", "数量": 1},
    }

    # ---------- 挖矿工具等级 ----------
    MINING_REQS = {
        "圆石":   {"tools": ["木镐", "石镐", "铁镐", "钻石镐"], "yield": (1, 1)},
        "煤炭":   {"tools": ["木镐", "石镐", "铁镐", "钻石镐"], "yield": (1, 2)},
        "铁矿石": {"tools": ["石镐", "铁镐", "钻石镐"],         "yield": (1, 1)},
        "金矿石": {"tools": ["铁镐", "钻石镐"],                 "yield": (1, 1)},
        "红石":   {"tools": ["铁镐", "钻石镐"],                 "yield": (1, 3)},
        "钻石":   {"tools": ["铁镐", "钻石镐"],                 "yield": (1, 1)},
        "黑曜石": {"tools": ["钻石镐"],                          "yield": (1, 1)},
    }

    # ---------- 工具耐久 ----------
    TOOL_DURABILITY = {
        "木镐": 12, "木剑": 10, "木斧": 12,
        "石镐": 24, "石剑": 20, "石斧": 24,
        "铁镐": 50, "铁剑": 40, "铁斧": 50,
        "钻石镐": 120, "钻石剑": 100,
    }

    # ---------- 食物恢复值 ----------
    FOOD_RESTORE = {
        "苹果": 4, "面包": 5, "熟肉": 8, "生肉": 2,
    }

    # ---------- 生物群系 ----------
    BIOMES = {
        "森林": {"description": "茂密的森林，可以砍树、打猎和采集苹果", "danger": 1},
        "矿洞": {"description": "深邃的矿洞，可以挖掘各种矿石",       "danger": 3},
        "平原": {"description": "开阔的平原，可以放牧、收割小麦和甘蔗", "danger": 1},
        "沙漠": {"description": "炎热的沙漠，可以采集沙子",           "danger": 2},
    }

    # 各群系可用的采集动作 → {物品: (最小, 最大)}
    BIOME_ACTIONS = {
        "森林": {
            "砍树":     {"原木": (1, 2)},
            "打猎":     {"生肉": (1, 2), "皮革": (0, 2)},
            "采集 苹果": {"苹果": (1, 2)},
        },
        "矿洞": {},          # 挖矿由 MINING_REQS 单独处理
        "平原": {
            "打猎":     {"生肉": (1, 1), "羊毛": (1, 3)},
            "采集 小麦": {"小麦": (1, 3)},
            "采集 甘蔗": {"甘蔗": (1, 2)},
        },
        "沙漠": {
            "采集 沙子": {"沙子": (1, 3)},
        },
    }

    DAY_LENGTH = 20      # 一个昼夜循环的步数
    NIGHT_START = 14     # 第 14 步起为夜晚

    # ============================================================== #
    #  初始化 / 重置
    # ============================================================== #
    def __init__(self, difficulty_config, seed=None):
        self.difficulty_config = difficulty_config
        self.seed = seed
        self.rng = random.Random(seed)
        self.recipes = self.RECIPES                 # 供 memory.py 读取
        self.reset()

    def reset(self):
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        self.inventory = {}
        self.completed_items = set()
        self.hp = 20
        self.max_hp = 20
        self.hunger = 20
        self.max_hunger = 20
        self.current_biome = "森林"
        self.step_count = 0
        self.tool_durability_state = {}
        self.has_shelter = False
        self.survival_enabled = self.difficulty_config.get("survival", False)
        return dict(self.inventory)

    # ============================================================== #
    #  RL step
    # ============================================================== #
    def step(self, parsed_cmd):
        self.step_count += 1
        messages = []

        # 生存机制
        if self.survival_enabled:
            msg = self._tick_hunger()
            if msg:
                messages.append(msg)
            if self.hp <= 0:
                return (dict(self.inventory), 0.0, True,
                        {"success": False,
                         "message": "你因饥饿而死！游戏结束。",
                         "cause": "starvation"})

        # 执行动作
        if not parsed_cmd.is_valid:
            success, action_msg = False, f"动作失败：{parsed_cmd.error_msg}"
        else:
            success, action_msg = self._execute(parsed_cmd)
        messages.append(action_msg)

        # 记录所有获取过的物品 (包含合成后被立刻吃掉的物品)
        self.completed_items.update(self.inventory.keys())

        # 夜晚
        if self.survival_enabled:
            night_msg = self._tick_night()
            if night_msg:
                messages.append(night_msg)
            if self.hp <= 0:
                messages.append("你被怪物杀死了！游戏结束。")
                return (dict(self.inventory), 0.0, True,
                        {"success": success,
                         "message": "\n".join(messages),
                         "cause": "mob_death"})

        # 状态行
        if self.survival_enabled:
            time_str = "夜晚" if self._is_night() else "白天"
            messages.append(
                f"[HP:{self.hp}/{self.max_hp} 饥饿:{self.hunger}/{self.max_hunger} "
                f"位置:{self.current_biome} 时间:{time_str}]")

        goal = self._check_goal()
        return (dict(self.inventory), 1.0 if goal else 0.0, goal,
                {"success": success, "message": "\n".join(messages)})

    # ============================================================== #
    #  动作分发
    # ============================================================== #
    def _execute(self, cmd):
        t = cmd.action_type
        if t == "move":
            return self._do_move(cmd.target)
        if t in ("chop", "hunt", "gather"):
            return self._do_gather(cmd.raw_action)
        if t == "mine":
            return self._do_mine(cmd.target)
        if t == "smelt":
            return self._do_smelt(cmd.target)
        if t == "craft":
            return self._do_craft(cmd.target)
        if t == "eat":
            return self._do_eat(cmd.target)
        if t == "build_shelter":
            return self._do_build_shelter()
        if t == "sleep":
            return self._do_sleep()
        if t == "water":
            return self._do_water()
        if t == "end":
            return self._do_end()
        return False, f"动作失败：未知动作类型 '{t}'。"

    # ============================================================== #
    #  动作实现
    # ============================================================== #

    # ---------- 移动 ----------
    def _do_move(self, target):
        if target not in self.BIOMES:
            return False, f"动作失败：未知地点 '{target}'。可选：{', '.join(self.BIOMES)}"
        if target == self.current_biome:
            return False, f"动作失败：你已经在{target}了。"
        old = self.current_biome
        self.current_biome = target
        return True, f"动作成功：从{old}移动到{target}。{self.BIOMES[target]['description']}"

    # ---------- 采集（砍树 / 打猎 / 采集 X）----------
    def _do_gather(self, action):
        biome_acts = self.BIOME_ACTIONS.get(self.current_biome, {})
        if action not in biome_acts:
            valid = [b for b, a in self.BIOME_ACTIONS.items() if action in a]
            if valid:
                return False, (f"动作失败：'{action}' 在{self.current_biome}无法执行。"
                               f"需要前往：{', '.join(valid)}")
            return False, f"动作失败：未知的采集指令 '{action}'。"

        yields = biome_acts[action]
        gained = []
        for item, (lo, hi) in yields.items():
            qty = self.rng.randint(lo, hi)
            if qty > 0:
                self.inventory[item] = self.inventory.get(item, 0) + qty
                gained.append(f"{item}×{qty}")
        return True, f"动作成功：获得 {', '.join(gained)}" if gained else "动作成功：但这次什么也没得到。"

    # ---------- 挖矿 ----------
    def _do_mine(self, ore):
        if self.current_biome != "矿洞":
            return False, f"动作失败：挖矿需要在矿洞中进行。当前位置：{self.current_biome}"
        if ore not in self.MINING_REQS:
            return False, f"动作失败：未知矿石 '{ore}'。可挖掘：{', '.join(self.MINING_REQS)}"

        req = self.MINING_REQS[ore]
        used_tool = None
        for t in req["tools"]:
            if self.inventory.get(t, 0) > 0 and self.tool_durability_state.get(t, 0) > 0:
                used_tool = t
                break
        if not used_tool:
            return False, (f"动作失败：挖掘{ore}需要以下工具之一 {req['tools']}，"
                           f"你没有可用的工具（可能已损坏或未拥有）。")

        # 消耗耐久
        self.tool_durability_state[used_tool] -= 1
        dur_msg = ""
        if self.tool_durability_state[used_tool] <= 0:
            self._remove_item(used_tool, 1)
            del self.tool_durability_state[used_tool]
            dur_msg = f" ⚠ {used_tool}已损坏！"

        lo, hi = req["yield"]
        qty = self.rng.randint(lo, hi)
        self.inventory[ore] = self.inventory.get(ore, 0) + qty
        return True, f"动作成功：使用{used_tool}挖出{ore}×{qty}。{dur_msg}"

    # ---------- 熔炼 ----------
    def _do_smelt(self, material):
        if self.inventory.get("熔炉", 0) < 1:
            return False, "动作失败：你没有熔炉，无法熔炼。"
        if material not in self.SMELTING:
            return False, f"动作失败：'{material}' 无法熔炼。可熔炼：{', '.join(self.SMELTING)}"
        if self.inventory.get(material, 0) < 1:
            return False, f"动作失败：你没有{material}。"

        info = self.SMELTING[material]
        self._remove_item(material, 1)
        self.inventory[info["产出"]] = self.inventory.get(info["产出"], 0) + info["数量"]
        return True, f"动作成功：将{material}熔炼为{info['产出']}×{info['数量']}"

    # ---------- 合成 ----------
    def _do_craft(self, item):
        if item not in self.RECIPES:
            return False, f"动作失败：配方库中没有 '{item}' 的合成方法。"
        recipe = self.RECIPES[item]

        if recipe.get("需要工作台") and self.inventory.get("工作台", 0) < 1:
            return False, f"动作失败：合成{item}需要工作台，你还没有工作台。"

        missing = []
        for mat, need in recipe["材料"].items():
            have = self.inventory.get(mat, 0)
            if have < need:
                missing.append(f"缺{need - have}个{mat}")
        if missing:
            return False, f"动作失败：材料不足，无法合成{item}。({', '.join(missing)})"

        for mat, need in recipe["材料"].items():
            self._remove_item(mat, need)
        qty = recipe["产出"]
        self.inventory[item] = self.inventory.get(item, 0) + qty

        # 如果是工具，初始化耐久
        if item in self.TOOL_DURABILITY:
            self.tool_durability_state[item] = self.TOOL_DURABILITY[item]

        return True, f"动作成功：合成了{item}×{qty}"

    # ---------- 进食 ----------
    def _do_eat(self, food):
        if food not in self.FOOD_RESTORE:
            return False, f"动作失败：'{food}' 不可食用。可食用：{', '.join(self.FOOD_RESTORE)}"
        if self.inventory.get(food, 0) < 1:
            return False, f"动作失败：你没有{food}。"

        restore = self.FOOD_RESTORE[food]
        self._remove_item(food, 1)
        old = self.hunger
        self.hunger = min(self.max_hunger, self.hunger + restore)
        return True, (f"动作成功：吃了{food}，恢复{self.hunger - old}点饥饿值。"
                      f"(饥饿:{self.hunger}/{self.max_hunger})")

    # ---------- 建造避难所 ----------
    def _do_build_shelter(self):
        needs = {"门": 1, "木板": 16, "圆石": 16}
        missing = []
        for item, need in needs.items():
            have = self.inventory.get(item, 0)
            if have < need:
                missing.append(f"缺{need - have}个{item}")
        if missing:
            hint = "（提示：门需先合成，配方为6木板）" if self.inventory.get("门", 0) < 1 else ""
            return False, f"动作失败：建造避难所材料不足。({', '.join(missing)}){hint}"

        for item, need in needs.items():
            self._remove_item(item, need)
        self.has_shelter = True
        return True, "动作成功：消耗16木板+16圆石+1门，建造了避难所！夜晚将更加安全。"

    # ---------- 睡觉 ----------
    def _do_sleep(self):
        if self.inventory.get("床", 0) < 1 and not self.has_shelter:
            return False, "动作失败：需要床或避难所才能安全睡觉。"
        if not self._is_night():
            return False, "动作失败：只能在夜晚睡觉。"
        return True, "动作成功：你安全地度过了夜晚。"

    # ---------- 打水 ----------
    def _do_water(self):
        if self.inventory.get("铁桶", 0) < 1:
            return False, "动作失败：你没有铁桶。"
        self.inventory["桶装水"] = self.inventory.get("桶装水", 0) + 1
        return True, "动作成功：用铁桶装满了水，获得桶装水×1"

    # ---------- 结束 ----------
    def _do_end(self):
        if self._check_goal():
            return True, "动作成功：恭喜！所有目标已完成！"
        else:
            missing = []
            target = self.difficulty_config["target"]
            if self.inventory.get(target, 0) < 1:
                missing.append(f"【最终目标】{target}")
            for d in self.difficulty_config.get("distractors", []):
                if d not in self.completed_items:
                    missing.append(f"【支线任务】{d}")
            return False, f"动作失败：任务尚未完成，还缺少：{', '.join(missing)}"

    # ============================================================== #
    #  生存系统
    # ============================================================== #
    def _is_night(self):
        return (self.step_count % self.DAY_LENGTH) >= self.NIGHT_START

    def _tick_hunger(self):
        """每 4 步饥饿 -1；低饥饿时扣 HP，高饥饿时回 HP。"""
        if self.step_count % 4 == 0:
            self.hunger = max(0, self.hunger - 1)

        if self.hunger <= 0:
            self.hp = max(0, self.hp - 2)
            return f"⚠ 你极度饥饿，HP-2！(HP:{self.hp}/{self.max_hp})"
        if self.hunger <= 5:
            self.hp = max(0, self.hp - 1)
            return f"⚠ 你很饿，HP-1！(HP:{self.hp}/{self.max_hp})"
        if self.hunger >= 15 and self.hp < self.max_hp:
            self.hp = min(self.max_hp, self.hp + 1)
        return None

    def _tick_night(self):
        """夜晚无庇护时受到怪物攻击，护甲可减伤。"""
        if not self._is_night():
            return ""
            
        grace_cycles = self.difficulty_config.get("grace_cycles", 0)
        current_cycle = self.step_count // self.DAY_LENGTH
        if current_cycle < grace_cycles:
            return "夜晚降临，但在安全期内，没有怪物出现。"

        if self.has_shelter or self.inventory.get("床", 0) > 0:
            return "夜晚降临，你在安全的庇护中度过。"

        # 护甲减伤
        armor = 0
        if self.inventory.get("钻石胸甲", 0) > 0:
            armor = 8
        elif self.inventory.get("铁胸甲", 0) > 0:
            armor = 5
        elif self.inventory.get("皮革胸甲", 0) > 0:
            armor = 3

        danger = self.BIOMES[self.current_biome]["danger"]
        raw = self.rng.randint(1, 3) * danger
        
        has_torch = self.inventory.get("火把", 0) > 0
        if has_torch:
            raw = max(1, int(raw * 0.5))

        actual = max(1, raw - armor)
        self.hp = max(0, self.hp - actual)

        torch_txt = "借着火把的微光抵御了部分袭击，" if has_torch else "黑暗中难以防卫，"
        armor_txt = f"(护甲减免{raw - actual}点)" if armor > 0 else ""
        return (f"⚠ 夜晚降临！{torch_txt}受到{actual}点伤害{armor_txt}！"
                f"(HP:{self.hp}/{self.max_hp})")

    # ============================================================== #
    #  目标判定
    # ============================================================== #
    def _check_goal(self):
        target = self.difficulty_config["target"]
        if self.inventory.get(target, 0) < 1:
            return False
        for d in self.difficulty_config.get("distractors", []):
            if d not in self.completed_items:
                return False
        return True

    # ============================================================== #
    #  查询接口（供 memory / detector 使用）
    # ============================================================== #
    def get_env_state(self):
        """返回当前环境状态快照，供 detector 使用。"""
        return {
            "hp": self.hp,
            "max_hp": self.max_hp,
            "hunger": self.hunger,
            "max_hunger": self.max_hunger,
            "biome": self.current_biome,
            "is_night": self._is_night(),
            "has_shelter": self.has_shelter,
            "tool_durability": dict(self.tool_durability_state),
            "inventory": dict(self.inventory),
            "step": self.step_count,
            "survival_enabled": self.survival_enabled,
        }

    def get_tool_durability_info(self):
        info = {}
        for tool, rem in self.tool_durability_state.items():
            mx = self.TOOL_DURABILITY.get(tool, "?")
            info[tool] = f"{rem}/{mx}"
        return info

    # ============================================================== #
    #  内部工具
    # ============================================================== #
    def _remove_item(self, item, count):
        self.inventory[item] = self.inventory.get(item, 0) - count
        if self.inventory[item] <= 0:
            del self.inventory[item]
