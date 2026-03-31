"""
组件验证测试脚本
验证 ActionParser、GroundTruth、LLMInterface 三个核心模块的接口行为。

运行方式：
    python -m pytest ns-fsm-baseline/scripts/test_components.py -v
    python ns-fsm-baseline/scripts/test_components.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# 路径设置
_script_dir = os.path.dirname(os.path.abspath(__file__))
_baseline_dir = os.path.dirname(_script_dir)
_project_dir = os.path.dirname(_baseline_dir)
sys.path.insert(0, os.path.join(_baseline_dir, "src"))
sys.path.insert(0, os.path.join(_project_dir, "MC-TextWorld"))


# ===========================================================================
# TestActionParser
# ===========================================================================

class TestActionParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from action_parser import ActionParser
        action_lib_path = os.path.join(_baseline_dir, "config", "action_lib_summary.json")
        cls.parser = ActionParser(action_lib_path=action_lib_path)

    # --- 提取 + 精确匹配 ---

    def test_exact_match(self):
        """Action: 前缀 + 合法 action → 精确匹配，reason 为 'ok'"""
        action, reason = self.parser.parse("Action: mine_oak_log")
        self.assertEqual(action, "mine_oak_log")
        self.assertEqual(reason, "ok")

    def test_exact_match_craft(self):
        """craft 类精确匹配"""
        action, reason = self.parser.parse("Action: craft_crafting_table")
        self.assertEqual(action, "craft_crafting_table")
        self.assertEqual(reason, "ok")

    # --- 模糊匹配：别名替换 ---

    def test_fuzzy_log(self):
        """mine_log → mine_oak_log（别名替换）"""
        action, reason = self.parser.parse("Action: mine_log")
        self.assertEqual(action, "mine_oak_log")
        self.assertNotEqual(reason, "ok")     # reason 应说明是模糊匹配

    def test_fuzzy_planks(self):
        """craft_planks → craft_oak_planks（别名替换）"""
        action, reason = self.parser.parse("Action: craft_planks")
        self.assertEqual(action, "craft_oak_planks")

    def test_fuzzy_wood_variant_birch(self):
        """mine_birch_log：若 action_lib 中存在，应精确匹配"""
        action, reason = self.parser.parse("Action: mine_birch_log")
        # birch_log 在 MC-TextWorld action_lib 中存在
        self.assertIsNotNone(action)
        self.assertIn("birch", action)

    # --- 回退匹配：无 Action: 前缀 ---

    def test_fallback_no_prefix(self):
        """文本末尾含 type_item 格式词，无 Action: 前缀 → 回退提取"""
        action, reason = self.parser.parse("I think I should craft_stick now")
        self.assertIsNotNone(action)

    def test_fallback_picks_last_word(self):
        """多个 type_item 词时优先取最后一个"""
        action, reason = self.parser.parse("mine_oak_log then craft_stick")
        self.assertIsNotNone(action)
        # 从末尾向前扫，应匹配 craft_stick
        self.assertIn("stick", action)

    # --- 失败场景 ---

    def test_invalid_action(self):
        """完全无法匹配的动作 → 返回 None"""
        action, reason = self.parser.parse("Action: fly_dragon")
        self.assertIsNone(action)
        self.assertIn("fly_dragon", reason)

    def test_empty_text(self):
        """空字符串 → 返回 None"""
        action, reason = self.parser.parse("")
        self.assertIsNone(action)

    def test_no_action_in_text(self):
        """纯自然语言，无 action 格式 → 返回 None"""
        action, reason = self.parser.parse("I need to think about this carefully.")
        self.assertIsNone(action)

    # --- get_candidate_actions ---

    def test_candidate_actions_empty_inventory(self):
        """空背包：只有 mine 类 action 可执行（无需材料）"""
        candidates = self.parser.get_candidate_actions({})
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        for act in candidates:
            self.assertTrue(act.startswith("mine_"),
                            f"空背包下不应出现非 mine action: {act}")

    def test_candidate_actions_with_log(self):
        """有 oak_log 时应出现 craft_oak_planks"""
        candidates = self.parser.get_candidate_actions({"oak_log": 1})
        self.assertIn("craft_oak_planks", candidates)

    def test_candidate_actions_with_planks(self):
        """有足够 planks 时应出现 craft_crafting_table"""
        candidates = self.parser.get_candidate_actions({"oak_planks": 4})
        self.assertIn("craft_crafting_table", candidates)


# ===========================================================================
# TestGroundTruth
# ===========================================================================

class TestGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ground_truth import GroundTruth
        cls.gt = GroundTruth()

    # --- is_achievable ---

    def test_is_achievable_valid(self):
        self.assertTrue(self.gt.is_achievable("wooden_pickaxe"))

    def test_is_achievable_invalid(self):
        self.assertFalse(self.gt.is_achievable("fake_item_xyz"))

    def test_is_achievable_iron_pickaxe(self):
        self.assertTrue(self.gt.is_achievable("iron_pickaxe"))

    # --- get_optimal_sequence ---

    def test_optimal_sequence_is_list(self):
        seq = self.gt.get_optimal_sequence("wooden_pickaxe")
        self.assertIsInstance(seq, list)
        self.assertGreater(len(seq), 0)

    def test_optimal_sequence_ends_with_goal_action(self):
        """序列最后一个 action 应是 craft_wooden_pickaxe"""
        seq = self.gt.get_optimal_sequence("wooden_pickaxe")
        self.assertEqual(seq[-1], "craft_wooden_pickaxe")

    def test_optimal_sequence_mine_before_craft(self):
        """mine_oak_log 必须在 craft_oak_planks 之前"""
        seq = self.gt.get_optimal_sequence("wooden_pickaxe")
        self.assertIn("mine_oak_log", seq)
        self.assertIn("craft_oak_planks", seq)
        self.assertLess(seq.index("mine_oak_log"), seq.index("craft_oak_planks"))

    def test_optimal_sequence_planks_before_pickaxe(self):
        """craft_oak_planks 必须在 craft_wooden_pickaxe 之前"""
        seq = self.gt.get_optimal_sequence("wooden_pickaxe")
        self.assertLess(
            seq.index("craft_oak_planks"),
            seq.index("craft_wooden_pickaxe")
        )

    def test_optimal_sequence_no_duplicates(self):
        """序列中每个 action 只出现一次（已去重）"""
        seq = self.gt.get_optimal_sequence("wooden_pickaxe")
        self.assertEqual(len(seq), len(set(seq)))

    def test_optimal_sequence_invalid_goal(self):
        """无效 goal → 返回空列表"""
        seq = self.gt.get_optimal_sequence("fake_item_xyz")
        self.assertEqual(seq, [])

    # --- get_all_required_items ---

    def test_all_required_items_wooden_pickaxe(self):
        items = self.gt.get_all_required_items("wooden_pickaxe")
        self.assertIsInstance(items, set)
        for expected in ["oak_log", "oak_planks", "stick", "crafting_table"]:
            self.assertIn(expected, items,
                          f"wooden_pickaxe 的依赖中应包含 {expected}")

    def test_all_required_items_invalid(self):
        items = self.gt.get_all_required_items("fake_item_xyz")
        self.assertEqual(items, set())

    # --- get_dependency_depth ---

    def test_dependency_depth_leaf(self):
        """oak_log 是叶节点（mine action，无依赖），深度 == 0"""
        depth = self.gt.get_dependency_depth("oak_log")
        self.assertEqual(depth, 0)

    def test_dependency_depth_stick(self):
        """stick → oak_planks → oak_log，深度 == 2"""
        depth = self.gt.get_dependency_depth("stick")
        self.assertEqual(depth, 2)

    def test_dependency_depth_deeper_for_iron(self):
        """iron_pickaxe 的依赖链比 wooden_pickaxe 更深"""
        iron_depth = self.gt.get_dependency_depth("iron_pickaxe")
        wood_depth = self.gt.get_dependency_depth("wooden_pickaxe")
        self.assertGreater(iron_depth, wood_depth)

    def test_dependency_depth_invalid(self):
        """无效 goal → 返回 -1"""
        depth = self.gt.get_dependency_depth("fake_item_xyz")
        self.assertEqual(depth, -1)

    # --- get_direct_deps ---

    def test_get_direct_deps_wooden_pickaxe(self):
        """wooden_pickaxe 的直接依赖应含 stick、oak_planks、crafting_table"""
        deps = self.gt.get_direct_deps("wooden_pickaxe")
        self.assertIsInstance(deps, dict)
        for key in ["stick", "oak_planks", "crafting_table"]:
            self.assertIn(key, deps)

    def test_get_direct_deps_unknown_item(self):
        """未知 item → 返回空字典"""
        deps = self.gt.get_direct_deps("fake_item_xyz")
        self.assertEqual(deps, {})

    # --- get_required_actions ---

    def test_get_required_actions_type(self):
        actions = self.gt.get_required_actions("wooden_pickaxe")
        self.assertIsInstance(actions, set)

    def test_get_required_actions_contains_mine(self):
        actions = self.gt.get_required_actions("wooden_pickaxe")
        self.assertIn("mine_oak_log", actions)


# ===========================================================================
# TestLLMInterface
# ===========================================================================

class TestLLMInterface(unittest.TestCase):
    """
    LLMInterface 测试。
    parse_* 方法无需网络，直接测试。
    generate() 用 unittest.mock 模拟，无需 Ollama 在线。
    """

    @classmethod
    def setUpClass(cls):
        from llm_interface import LLMInterface
        # patch OpenAI 避免连接 Ollama
        with patch("llm_interface.OpenAI"):
            cls.llm = LLMInterface()

    # --- parse_react_response ---

    def test_parse_react_standard(self):
        """标准 ReAct 格式：Thought + Action 均正确提取"""
        text = "Thought: I need wood first.\nAction: mine_oak_log"
        thought, action = self.llm.parse_react_response(text)
        self.assertIn("wood", thought)
        self.assertEqual(action, "mine_oak_log")

    def test_parse_react_only_action(self):
        """只有 Action: 没有 Thought: → thought 为空串，action 正确"""
        text = "Action: craft_stick"
        thought, action = self.llm.parse_react_response(text)
        self.assertEqual(thought, "")
        self.assertEqual(action, "craft_stick")

    def test_parse_react_no_action(self):
        """无 Action: 字段 → action 为空串"""
        text = "Thought: I'm not sure what to do."
        thought, action = self.llm.parse_react_response(text)
        self.assertNotEqual(thought, "")
        self.assertEqual(action, "")

    def test_parse_react_case_insensitive(self):
        """action: 小写前缀也能正确解析"""
        text = "thought: need planks\naction: craft_oak_planks"
        thought, action = self.llm.parse_react_response(text)
        self.assertEqual(action, "craft_oak_planks")

    def test_parse_react_multiline_thought(self):
        """多行 Thought 内容完整提取"""
        text = "Thought: First I need logs.\nThen I need planks.\nAction: mine_oak_log"
        thought, action = self.llm.parse_react_response(text)
        self.assertIn("logs", thought)
        self.assertEqual(action, "mine_oak_log")

    def test_parse_react_empty_text(self):
        """空字符串 → 均为空串"""
        thought, action = self.llm.parse_react_response("")
        self.assertEqual(thought, "")
        self.assertEqual(action, "")

    # --- parse_reflection ---

    def test_parse_reflection_with_prefix(self):
        """有 Reflection: 前缀 → 提取后续内容"""
        text = "Reflection: I failed because I didn't mine enough logs."
        result = self.llm.parse_reflection(text)
        self.assertIn("failed", result)
        self.assertNotIn("Reflection:", result)

    def test_parse_reflection_no_prefix(self):
        """无 Reflection: 前缀 → 返回完整文本"""
        text = "I should have mined more resources first."
        result = self.llm.parse_reflection(text)
        self.assertEqual(result, text)

    def test_parse_reflection_case_insensitive(self):
        """REFLECTION: 大写也能识别"""
        text = "REFLECTION: Next time I'll craft planks first."
        result = self.llm.parse_reflection(text)
        self.assertIn("planks", result)

    def test_parse_reflection_empty_text(self):
        """空字符串 → 返回空字符串"""
        result = self.llm.parse_reflection("")
        self.assertEqual(result, "")

    # --- generate()：mock 测试 ---

    def test_generate_mock_success(self):
        """mock 正常返回 → generate() 返回该字符串"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Thought: ok\nAction: mine_oak_log  "
        self.llm.client.chat.completions.create = MagicMock(return_value=mock_response)

        result = self.llm.generate("system", "user")
        self.assertEqual(result, "Thought: ok\nAction: mine_oak_log")

    def test_generate_retry_on_failure(self):
        """前两次抛异常，第三次成功 → generate() 最终返回成功结果"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Action: craft_stick"

        self.llm.max_retries = 2
        self.llm.client.chat.completions.create = MagicMock(
            side_effect=[
                Exception("timeout"),
                Exception("timeout"),
                mock_response,
            ]
        )

        with patch("time.sleep"):   # 不真正等待
            result = self.llm.generate("system", "user")

        self.assertEqual(result, "Action: craft_stick")

    def test_generate_max_retry_exceeded(self):
        """始终抛异常 → 超过 max_retries 后 raise"""
        self.llm.max_retries = 2
        self.llm.client.chat.completions.create = MagicMock(
            side_effect=Exception("connection refused")
        )

        with patch("time.sleep"):
            with self.assertRaises(Exception):
                self.llm.generate("system", "user")


# ===========================================================================
# 入口
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
