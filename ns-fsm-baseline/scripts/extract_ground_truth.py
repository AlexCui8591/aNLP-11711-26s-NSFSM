import json
import os
import argparse

def evaluate_variant_cost(variant):
    """
    Evaluates the cost of a specific recipe variant.
    Lower is better.
    We prefer variants with fewer total required item types,
    and we tie-break to prefer "oak" related items.
    """
    precondition = variant.get("precondition", {})
    tool = variant.get("tool", {})
    
    # Cost = total number of distinct precondition items + tools
    cost = len(precondition) + len(tool)
    
    # Tie-breaker penalty: subtract a small amount if picking oak
    penalty = 0.0
    for item in list(precondition.keys()) + list(tool.keys()):
        if "oak" not in item and any("oak" in k for k in (precondition.keys() | tool.keys())):
           # Not specifically penalty, just we want oak to be mathematically slightly lower cost
           pass
    
    # Calculate penalty
    # Count how many non-oak items there are that have wood variants.
    # To keep it simple, if an item string has "spruce", "birch", "jungle", "acacia", "dark_oak", "crimson", "warped", add penalty
    wood_types = ["spruce", "birch", "jungle", "acacia", "dark_oak", "crimson", "warped", "bamboo"]
    for item in precondition.keys():
        if any(wt in item for wt in wood_types):
            penalty += 0.1
            
    return cost + penalty

def build_dependency_graph(item_name: str, action_lib: dict, memo: dict = None, visited: set = None):
    """
    Recursively builds the ground truth dependency graph for a target item.
    """
    if memo is None: memo = {}
    if visited is None: visited = set()
    
    if item_name in memo:
        return memo[item_name]
        
    # Cycle detection
    if item_name in visited:
        # Detected a cycle! (e.g., iron_block -> iron_ingot -> iron_block)
        return {"action": None, "type": "cycle", "dependencies": {}}
        
    visited.add(item_name)
    
    # Find all actions that output this item
    producing_actions = []
    for action_name, variants in action_lib.items():
        for variant in variants:
            if item_name in variant.get("output", {}):
                producing_actions.append((action_name, variant))
                
    if not producing_actions:
        visited.remove(item_name)
        # It's an un-craftable base material / invalid
        return {"action": None, "type": "unknown", "dependencies": {}}
        
    # Pick the simplest variant
    producing_actions.sort(key=lambda x: evaluate_variant_cost(x[1]))
    best_action_name, best_variant = producing_actions[0]
    
    act_type = best_variant.get("type", "unknown")
    precondition = best_variant.get("precondition", {})
    tool = best_variant.get("tool", {})
    
    node = {
        "action": best_action_name,
        "type": act_type,
        "precondition": precondition,
        "tool": tool,
        "dependencies": {}
    }
    
    # Base case: mine actions have empty preconditions (and are leaves, minus potential tools)
    if act_type == "mine" and not tool:
        memo[item_name] = node
        visited.remove(item_name)
        return node
        
    # Recursive case: process preconditions and tools
    all_deps = list(precondition.keys()) + list(tool.keys())
    for dep_item in all_deps:
        node["dependencies"][dep_item] = build_dependency_graph(dep_item, action_lib, memo, visited)
        
    memo[item_name] = node
    visited.remove(item_name)
    return node

def flatten_graph_for_ega(graph_node):
    """
    Flattens the nested dependency graph into a {item: {dep: qty}} dict for EGA calculation.
    """
    flat = {}
    
    def traverse(item, node):
        if item in flat: return
        
        # Add entry for this item
        deps = {}
        for k, v in node.get("precondition", {}).items(): deps[k] = v
        for k, v in node.get("tool", {}).items(): deps[k] = v
        flat[item] = deps
        
        for child_item, child_node in node.get("dependencies", {}).items():
            traverse(child_item, child_node)
            
    # Assuming graph_node is the top-level dict matching the root item. 
    # But wait, we don't pass root name. We need to parse.
    # Actually wait, the caller passes the named node.
    pass

def flatten_full_library(memo: dict) -> dict:
    flat = {}
    for item, node in memo.items():
        deps = {}
        for k, v in node.get("precondition", {}).items(): deps[k] = v
        for k, v in node.get("tool", {}).items(): deps[k] = v
        flat[item] = deps
    return flat

def main():
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    action_lib_path = os.path.join(config_dir, "action_lib_summary.json")
    goals_path = os.path.join(config_dir, "goals_67.json")
    output_graph_path = os.path.join(config_dir, "ground_truth_graph.json")
    output_flat_path = os.path.join(config_dir, "ground_truth_flat.json")
    
    with open(action_lib_path, "r", encoding="utf-8") as f:
        action_lib = json.load(f)
        
    with open(goals_path, "r", encoding="utf-8") as f:
        goals_data = json.load(f)
        
    # Extract all required goal items
    all_goal_items = []
    for group, group_data in goals_data.items():
        for req in group_data["goals"]:
            all_goal_items.append(req["goal"])
            
    print(f"Goal items to process: {len(all_goal_items)}")
            
    memo = {}
    full_graph = {}
    
    for goal_item in all_goal_items:
        full_graph[goal_item] = build_dependency_graph(goal_item, action_lib, memo, set())
        
    print(f"Graph generated. Total unique items in dependency chains: {len(memo)}")
    
    with open(output_graph_path, "w", encoding="utf-8") as f:
        json.dump(full_graph, f, indent=2)
        
    flat_graph = flatten_full_library(memo)
    with open(output_flat_path, "w", encoding="utf-8") as f:
        json.dump(flat_graph, f, indent=2)
        
    print(f"Saved nested graph to {output_graph_path}")
    print(f"Saved flat graph to {output_flat_path}")

if __name__ == "__main__":
    main()
