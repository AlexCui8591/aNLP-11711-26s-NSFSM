class ParsedCommand:
    """结构化的动作指令"""
    def __init__(self, raw_action, action_type, target=None, is_valid=True, error_msg=""):
        self.raw_action = raw_action
        self.action_type = action_type
        self.target = target
        self.is_valid = is_valid
        self.error_msg = error_msg


def parse_command(action_str):
    """
    解析 LLM 输出的动作指令，返回 ParsedCommand。
    支持的动作类型：
      砍树, 打猎, 打水, 睡觉, 建造 避难所,
      移动 <地点>, 采集 <物品>, 挖矿 <矿石>, 熔炼 <原料>,
      合成 <物品>, 吃 <食物>
    """
    if not action_str:
        return ParsedCommand(action_str, None, is_valid=False, error_msg="动作指令为空。")

    action = action_str.strip()

    # ---- 单词动作 ----
    if action == "砍树":
        return ParsedCommand(action, "chop")
    elif action == "打猎":
        return ParsedCommand(action, "hunt")
    elif action == "打水":
        return ParsedCommand(action, "water")
    elif action == "睡觉":
        return ParsedCommand(action, "sleep")
    elif action in ("建造 避难所", "建造避难所"):
        return ParsedCommand("建造 避难所", "build_shelter")
    elif action == "结束":
        return ParsedCommand(action, "end")

    # ---- 带目标的动作 ----
    elif action.startswith("移动"):
        target = action.replace("移动", "").strip()
        if not target:
            return ParsedCommand(action, "move", is_valid=False,
                                 error_msg="缺少移动目标。可选：森林/矿洞/平原/沙漠")
        return ParsedCommand(action, "move", target)

    elif action.startswith("采集"):
        target = action.replace("采集", "").strip()
        if not target:
            return ParsedCommand(action, "gather", is_valid=False,
                                 error_msg="缺少采集目标。如：采集 苹果/小麦/甘蔗/沙子")
        return ParsedCommand(f"采集 {target}", "gather", target)

    elif action.startswith("挖矿"):
        target = action.replace("挖矿", "").strip()
        if not target:
            return ParsedCommand(action, "mine", is_valid=False, error_msg="缺少挖矿目标。")
        return ParsedCommand(action, "mine", target)

    elif action.startswith("熔炼"):
        target = action.replace("熔炼", "").strip()
        if not target:
            return ParsedCommand(action, "smelt", is_valid=False, error_msg="缺少熔炼目标。")
        return ParsedCommand(action, "smelt", target)

    elif action.startswith("合成"):
        target = action.replace("合成", "").strip()
        if not target:
            return ParsedCommand(action, "craft", is_valid=False, error_msg="缺少合成目标。")
        # 容错：去除 LLM 可能加上的数量后缀
        for sep in ("*", " x", "×"):
            if sep in target:
                target = target.split(sep)[0].strip()
        return ParsedCommand(f"合成 {target}", "craft", target)

    elif action.startswith("吃"):
        target = action.replace("吃", "").strip()
        if not target:
            return ParsedCommand(action, "eat", is_valid=False,
                                 error_msg="缺少食物名称。如：吃 面包/熟肉/苹果")
        return ParsedCommand(action, "eat", target)

    return ParsedCommand(action, "unknown", is_valid=False,
                         error_msg=f"无法识别的指令 '{action}'。请参考可用动作格式。")
