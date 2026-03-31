"""
Ground-truth dependency helpers for task analysis and evaluation.
"""

import json
import os


class GroundTruth:
    def __init__(self, graph_path: str = None, flat_path: str = None):
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

        if graph_path is None:
            graph_path = os.path.join(config_dir, "ground_truth_graph.json")
        if flat_path is None:
            flat_path = os.path.join(config_dir, "ground_truth_flat.json")

        with open(graph_path, "r", encoding="utf-8") as f:
            self._graph = json.load(f)

        with open(flat_path, "r", encoding="utf-8") as f:
            self._flat = json.load(f)

        self._node_index = {}
        for item, node in self._graph.items():
            self._index_node(item, node)

    def get_optimal_sequence(self, goal: str) -> list:
        node = self._get_node(goal)
        if not node:
            return []

        sequence = []
        seen = set()

        def dfs(cur: dict):
            if not cur or cur.get("action") is None:
                return
            for dep_node in cur.get("dependencies", {}).values():
                dfs(dep_node)
            action = cur["action"]
            if action not in seen:
                seen.add(action)
                sequence.append(action)

        dfs(node)
        return sequence

    def get_all_required_items(self, goal: str) -> set:
        node = self._get_node(goal)
        if not node:
            return set()

        items = set()

        def dfs(cur: dict):
            if not cur:
                return
            for item_name, dep_node in cur.get("dependencies", {}).items():
                items.add(item_name)
                dfs(dep_node)

        dfs(node)
        return items

    def get_dependency_depth(self, goal: str) -> int:
        node = self._get_node(goal)
        if not node:
            return -1

        def dfs(cur: dict) -> int:
            dependencies = cur.get("dependencies", {}) if cur else {}
            if not dependencies:
                return 0
            return 1 + max(dfs(child) for child in dependencies.values())

        return dfs(node)

    def is_achievable(self, goal: str) -> bool:
        node = self._get_node(goal)
        return node is not None and node.get("action") is not None

    def get_node(self, item: str) -> dict:
        return self._get_node(item) or {}

    def get_direct_deps(self, item: str) -> dict:
        return self._flat.get(item, {})

    def get_required_actions(self, goal: str) -> set:
        return set(self.get_optimal_sequence(goal))

    def _index_node(self, item: str, node: dict):
        if not item or not node:
            return

        self._node_index.setdefault(item, node)
        for dep_item, dep_node in node.get("dependencies", {}).items():
            self._index_node(dep_item, dep_node)

    def _get_node(self, item: str):
        return self._graph.get(item) or self._node_index.get(item)
