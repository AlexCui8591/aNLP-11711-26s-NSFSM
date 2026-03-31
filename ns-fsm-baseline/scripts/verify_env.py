"""
MC-TextWorld 环境验证脚本
确认：
1. action_lib 加载正常，列出所有 action 及其结构
2. env.reset() 返回格式正确
3. env.step() 基本操作正常（craft_crafting_table 完整流程）
4. 统计 action_lib 中的 item 种类和 action 数量
"""
import json
import sys
import os

# 添加 MC-TextWorld 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'MC-TextWorld'))

from mctextworld.simulator import Env
from mctextworld.action import ActionLibrary


def test_action_lib():
    """测试 1: 检查 action_lib 加载"""
    print("=" * 60)
    print("TEST 1: ActionLibrary 加载")
    print("=" * 60)
    
    action_lib = ActionLibrary()
    all_actions = list(action_lib.action_lib.keys())
    
    print(f"  总 action 数量: {len(all_actions)}")
    
    # 按 type 分类统计
    type_counts = {"mine": 0, "craft": 0, "smelt": 0, "other": 0}
    for act_name in all_actions:
        prefix = act_name.split("_")[0]
        if prefix in type_counts:
            type_counts[prefix] += 1
        else:
            type_counts["other"] += 1
    
    print(f"  按类型统计: {type_counts}")
    
    # 列出前 20 个 action 作为示例
    print(f"\n  前 20 个 action 示例:")
    for act_name in sorted(all_actions)[:20]:
        variants = action_lib.action_lib[act_name]
        print(f"    {act_name}: {len(variants)} variant(s)")
        for v in variants[:1]:  # 只显示第一个 variant
            print(f"      output: {v.get('output', {})}")
            print(f"      precondition: {v.get('precondition', {})}")
            print(f"      tool: {v.get('tool', {})}")
    
    # 收集所有可产出的 item
    all_output_items = set()
    for act_name, variants in action_lib.action_lib.items():
        for v in variants:
            for item in v.get('output', {}).keys():
                all_output_items.add(item)
    
    print(f"\n  所有可产出的 item 数量: {len(all_output_items)}")
    print(f"  items: {sorted(all_output_items)[:30]}...")
    
    return action_lib, all_output_items


def test_env_reset():
    """测试 2: env.reset() 返回格式"""
    print("\n" + "=" * 60)
    print("TEST 2: Env.reset() 格式验证")
    print("=" * 60)
    
    env = Env(
        task_name="Obtain a crafting_table",
        init_inv={},
        task_obj={"crafting_table": 1}
    )
    env.maximum_step = 100
    
    obs, reward, done, info = env.reset()
    
    print(f"  obs type: {type(obs)}")
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  obs['inventory']: {obs['inventory']}")
    print(f"  obs['position']: {obs['position']}")
    print(f"  obs['biome']: {obs['biome']}")
    print(f"  reward: {reward}")
    print(f"  done: {done}")
    print(f"  info: {info}")
    
    assert isinstance(obs, dict), "obs should be dict"
    assert 'inventory' in obs, "obs should have 'inventory'"
    assert 'position' in obs, "obs should have 'position'"
    assert 'biome' in obs, "obs should have 'biome'"
    print("  ✅ reset() 格式正确")
    
    return env


def test_basic_actions(env):
    """测试 3: 基本 action 执行流程"""
    print("\n" + "=" * 60)
    print("TEST 3: 基本 action 执行 (crafting_table 完整流程)")
    print("=" * 60)
    
    env.reset()
    
    # Step 1: mine_log (获取原木)
    actions_sequence = [
        ("mine_oak_log", "获取 oak_log"),
        ("mine_oak_log", "再获取 oak_log"),
        ("craft_oak_planks", "oak_log → oak_planks"),
        ("craft_oak_planks", "再做 oak_planks"),
        ("craft_crafting_table", "oak_planks → crafting_table"),
    ]
    
    for action, desc in actions_sequence:
        obs, reward, done, info = env.step(action)
        success = info.get('action success', False)
        status = "✅" if success else "❌"
        print(f"  {status} {action} ({desc})")
        print(f"     inventory: {obs['inventory']}")
        if not success:
            print(f"     message: {info.get('message', 'N/A')}")
        if done:
            print(f"     🎉 DONE! Task completed!")
            break
    
    return done


def test_candidate_actions():
    """测试 4: 查看给定 inventory 下的合法 action"""
    print("\n" + "=" * 60)
    print("TEST 4: get_candidate_actions() 测试")
    print("=" * 60)
    
    action_lib = ActionLibrary()
    
    # 空 inventory
    empty_inv = {}
    candidates_empty = action_lib.get_candidate_actions(empty_inv)
    print(f"  空 inventory 下可执行 action: {len(candidates_empty)}")
    print(f"    actions: {candidates_empty[:10]}...")
    
    # 有 log 的 inventory
    log_inv = {"log": 2}
    candidates_log = action_lib.get_candidate_actions(log_inv)
    print(f"\n  有 2 个 log 时可执行 action: {len(candidates_log)}")
    print(f"    actions: {candidates_log[:15]}...")
    
    # 有 planks 的 inventory
    planks_inv = {"planks": 8}
    candidates_planks = action_lib.get_candidate_actions(planks_inv)
    print(f"\n  有 8 个 planks 时可执行 action: {len(candidates_planks)}")
    print(f"    actions: {candidates_planks[:15]}...")


def test_goal_items_coverage(action_lib):
    """测试 5: 检查典型 goal item 是否可达"""
    print("\n" + "=" * 60)
    print("TEST 5: Goal Items 可达性检查")
    print("=" * 60)
    
    # 典型的 goal items (按 group)
    sample_goals = {
        "Wood": ["crafting_table", "stick", "oak_planks"],
        "Stone": ["furnace", "stone_pickaxe", "torch"],
        "Iron": ["iron_pickaxe", "iron_sword", "blast_furnace", "shield"],
        "Gold": ["golden_sword", "golden_pickaxe"],
        "Diamond": ["diamond_pickaxe", "diamond_sword", "jukebox"],
        "Redstone": ["piston", "compass", "redstone_torch"],
        "Armor": ["diamond_chestplate", "iron_chestplate"],
    }
    
    all_actions = action_lib.action_lib
    
    total = 0
    found = 0
    not_found = []
    
    for group, items in sample_goals.items():
        print(f"\n  [{group}]")
        for item in items:
            total += 1
            # 查找能产出这个 item 的 action
            producing_actions = []
            for act_name, variants in all_actions.items():
                for v in variants:
                    if item in v.get('output', {}):
                        producing_actions.append(act_name)
            
            if producing_actions:
                found += 1
                print(f"    ✅ {item} → 可通过 {producing_actions} 获得")
            else:
                not_found.append(item)
                print(f"    ❌ {item} → 未找到产出 action！")
    
    print(f"\n  总结: {found}/{total} items 可达")
    if not_found:
        print(f"  未找到的 items: {not_found}")


def dump_action_lib_summary(action_lib):
    """额外：导出 action_lib 摘要到 JSON 用于后续分析"""
    print("\n" + "=" * 60)
    print("BONUS: 导出 action_lib 摘要")
    print("=" * 60)
    
    summary = {}
    for act_name, variants in action_lib.action_lib.items():
        summary[act_name] = []
        for v in variants:
            summary[act_name].append({
                "type": v.get("type", "unknown"),
                "output": v.get("output", {}),
                "precondition": v.get("precondition", {}),
                "tool": v.get("tool", {}),
            })
    
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'action_lib_summary.json'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 已导出到 {os.path.abspath(output_path)}")
    print(f"     共 {len(summary)} 个 action")


if __name__ == "__main__":
    print("🔍 MC-TextWorld 环境验证\n")
    
    # Test 1: ActionLibrary
    action_lib, all_items = test_action_lib()
    
    # Test 2: reset()
    env = test_env_reset()
    
    # Test 3: 基本 action flow
    success = test_basic_actions(env)
    
    # Test 4: candidate actions
    test_candidate_actions()
    
    # Test 5: goal items coverage
    test_goal_items_coverage(action_lib)
    
    # Bonus: 导出 action_lib 摘要
    dump_action_lib_summary(action_lib)
    
    print("\n" + "=" * 60)
    print("✅ 所有验证完成！")
    print("=" * 60)
