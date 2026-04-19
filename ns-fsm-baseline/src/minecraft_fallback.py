"""Shared lightweight Minecraft fallback simulator.

The real Minecraft experiments should use MC-TextWorld when it is installed.
This module is only a dependency-light substitute for smoke tests, local
debugging, and environments where the optional MC-TextWorld package is missing.
It executes high-level actions by applying action_lib preconditions, tools, and
outputs to an inventory dictionary.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Mapping


def default_action_lib_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "action_lib_summary.json",
    )


def load_action_library(path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    with open(path or default_action_lib_path(), "r", encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_action_library(raw)


def normalize_action_library(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, list):
        action_lib: dict[str, list[dict[str, Any]]] = {}
        for entry in raw:
            action_lib.setdefault(entry["action"], []).append(entry)
        return action_lib
    return {str(action): list(variants) for action, variants in dict(raw).items()}


class MinecraftFallbackSimulator:
    """Inventory-only simulator derived from the MC-TextWorld action library."""

    def __init__(
        self,
        action_library: Mapping[str, list[Mapping[str, Any]]] | None = None,
        action_lib_path: str | None = None,
    ):
        raw = action_library if action_library is not None else load_action_library(action_lib_path)
        self.action_library = normalize_action_library(raw)

    def all_actions(self, include_no_op: bool = False) -> list[str]:
        actions = sorted(self.action_library)
        if include_no_op:
            return actions
        return [action for action in actions if action != "no_op"]

    def candidate_actions(self, inventory: Mapping[str, int]) -> list[str]:
        candidates = []
        for action_name, variants in self.action_library.items():
            if action_name == "no_op":
                continue
            if any(self.has_requirements(variant, inventory) for variant in variants):
                candidates.append(action_name)
        return sorted(candidates)

    def step(
        self,
        action_name: str,
        inventory: Mapping[str, int],
    ) -> tuple[bool, str, dict[str, int]]:
        variants = self.action_library.get(action_name)
        if not variants:
            return False, f"Unknown action '{action_name}' - not in the action library", dict(inventory)

        for variant in variants:
            if not self.has_requirements(variant, inventory):
                continue
            updated = deepcopy(dict(inventory))
            for item, qty in variant.get("precondition", {}).items():
                updated[item] = updated.get(item, 0) - int(qty)
                if updated[item] <= 0:
                    updated.pop(item, None)
            for item, qty in variant.get("output", {}).items():
                updated[item] = updated.get(item, 0) + int(qty)
            return True, "Action executed in fallback simulator.", updated

        return False, self.diagnose_failure(action_name, inventory), dict(inventory)

    def diagnose_failure(self, action_name: str, inventory: Mapping[str, int]) -> str:
        variants = self.action_library.get(action_name)
        if not variants:
            return f"Unknown action '{action_name}' - not in the action library"

        best_msg = None
        best_missing_count = float("inf")
        for variant in variants:
            missing_materials = {}
            missing_tools = {}
            for item, need in variant.get("precondition", {}).items():
                have = int(inventory.get(item, 0))
                need = int(need)
                if have < need:
                    missing_materials[item] = f"need {need}, have {have}"
            for item, need in variant.get("tool", {}).items():
                have = int(inventory.get(item, 0))
                need = int(need)
                if have < need:
                    missing_tools[item] = f"need {need}, have {have}"

            missing_count = len(missing_materials) + len(missing_tools)
            if missing_count == 0 or missing_count >= best_missing_count:
                continue
            best_missing_count = missing_count
            parts = []
            if missing_tools:
                parts.append(
                    "missing tool: "
                    + ", ".join(f"{item} ({reason})" for item, reason in missing_tools.items())
                )
            if missing_materials:
                parts.append(
                    "missing materials: "
                    + ", ".join(f"{item} ({reason})" for item, reason in missing_materials.items())
                )
            best_msg = f"Failed {action_name}: " + "; ".join(parts)

        return best_msg or f"Action '{action_name}' failed (preconditions not met)"

    @staticmethod
    def has_requirements(
        variant: Mapping[str, Any],
        inventory: Mapping[str, int],
    ) -> bool:
        for item, qty in variant.get("precondition", {}).items():
            if int(inventory.get(item, 0)) < int(qty):
                return False
        for item, qty in variant.get("tool", {}).items():
            if int(inventory.get(item, 0)) < int(qty):
                return False
        return True
