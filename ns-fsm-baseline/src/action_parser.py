import json
import os
import re
import sys

# 加载 ActionLibrary 用于 get_candidate_actions
mc_path = os.path.join(os.path.dirname(__file__), '..', '..', 'MC-TextWorld')
sys.path.insert(0, mc_path)
try:
    from mctextworld.action import ActionLibrary
except ImportError as exc:  # pragma: no cover - depends on optional external package
    ActionLibrary = None
    _MC_TEXTWORLD_ACTION_IMPORT_ERROR = exc
else:
    _MC_TEXTWORLD_ACTION_IMPORT_ERROR = None


# MC-TextWorld 中所有木材变体
_WOOD_VARIANTS = ["oak", "birch", "spruce", "jungle", "acacia", "dark_oak"]

# 常见 LLM 简写 → 标准物品名映射
_ITEM_ALIASES = {
    "log":           "oak_log",
    "planks":        "oak_planks",
    "boat":          "oak_boat",
    "wood":          "oak_log",
    "iron":          "iron_ingot",
    "gold":          "gold_ingot",
    "golden":        "gold_ingot",
    "diamond":       "diamond",
    "coal":          "coal",
    "stone":         "cobblestone",
    "cobble":        "cobblestone",
}


class ActionParser:
    def __init__(self, action_lib_path: str = None):
        """
        初始化 ActionParser。
        若未提供 action_lib_path，自动从 MC-TextWorld 目录加载。
        """
        if action_lib_path is None:
            action_lib_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'MC-TextWorld',
                'mctextworld', 'action_lib.json'
            )
            if not os.path.exists(action_lib_path):
                action_lib_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "config",
                    "action_lib_summary.json",
                )

        with open(action_lib_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # action_lib.json 可能是 {name: [...]} 或 [...] 两种格式
        self._action_lib_dict = self._normalize_action_library(raw)

        self.valid_actions = set(self._action_lib_dict.keys())
        self.action_types = {"mine", "craft", "smelt"}
        self._action_library = ActionLibrary() if ActionLibrary is not None else None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def parse(self, text: str) -> tuple:
        """
        从 LLM 输出文本中提取并验证 MC-TextWorld action。

        流程：
          1. 提取阶段：从文本中找 raw_action 字符串
               - 优先匹配 "Action: xxx" 格式
               - 退而匹配文本末尾最近的 type_item 格式词
          2. 精确匹配：直接查 valid_actions
          3. 模糊匹配：常见别名 → 木材变体 → oak 通用兜底
          4. 全部失败 → 返回 (None, reason)

        返回：
          (action_str, reason)
          - action_str: 合法的 MC-TextWorld action 字符串，或 None
          - reason: 说明字符串（成功时为 "ok"，失败时描述原因）
        """
        raw_action = self._extract_raw_action(text)
        if not raw_action:
            return None, "LLM 输出中未找到 Action: 字段或 type_item 格式的词"

        # 精确匹配
        if raw_action in self.valid_actions:
            return raw_action, "ok"

        # 模糊匹配
        matched = self._fuzzy_match(raw_action)
        if matched:
            return matched, f"模糊匹配：{raw_action} → {matched}"

        return None, f"无法解析动作：'{raw_action}'（不在 action_lib 中，模糊匹配也失败）"

    def get_candidate_actions(self, inventory: dict) -> list:
        """
        返回当前 inventory 下所有可执行的 action 列表。
        用于在 prompt 中向 LLM 展示合法选项，减少无效输出。
        """
        if self._action_library is not None:
            candidates = self._action_library.get_candidate_actions(inventory)
            return [action for action in candidates if action != "no_op"]

        raise ImportError(
            "MC-TextWorld ActionLibrary is required to compute executable actions. "
            f"Could not import mctextworld.action.ActionLibrary from {mc_path}."
        ) from _MC_TEXTWORLD_ACTION_IMPORT_ERROR

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_raw_action(self, text: str) -> str:
        """从文本中提取原始 action 字符串（尚未验证合法性）。"""
        # 优先：显式 "Action: xxx" 格式
        match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()

        # 退而求其次：从文本末尾向前找 type_item 格式的词
        words = text.split()
        for w in reversed(words):
            w_clean = re.sub(r"[^a-zA-Z_]", "", w.lower())
            if any(w_clean.startswith(t + "_") for t in self.action_types):
                return w_clean

        return ""

    def _fuzzy_match(self, raw_action: str) -> str:
        """
        对无法精确匹配的 raw_action 进行模糊匹配。

        策略（按优先级）：
          1. 别名替换：通过 _ITEM_ALIASES 将 LLM 简写映射到标准物品名
          2. 木材变体枚举：尝试所有 6 种木材变体（oak/birch/spruce/...）
          3. oak 通用兜底：type_item → type_oak_item
        """
        parts = raw_action.split("_", 1)
        if len(parts) != 2:
            return ""
        act_type, item = parts

        candidates = []

        # 策略 1：别名替换
        if item in _ITEM_ALIASES:
            candidates.append(f"{act_type}_{_ITEM_ALIASES[item]}")

        # 策略 2：木材变体枚举（仅当 item 不含变体前缀时）
        if not any(item.startswith(v + "_") for v in _WOOD_VARIANTS):
            for variant in _WOOD_VARIANTS:
                candidates.append(f"{act_type}_{variant}_{item}")

        # 策略 3：oak 通用兜底
        candidates.append(f"{act_type}_oak_{item}")

        for c in candidates:
            if c in self.valid_actions:
                return c

        return ""

    @staticmethod
    def _normalize_action_library(raw):
        if isinstance(raw, list):
            action_lib = {}
            for entry in raw:
                action_lib.setdefault(entry["action"], []).append(entry)
            return action_lib
        return {str(action): list(variants) for action, variants in dict(raw).items()}
