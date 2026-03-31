"""
兼容性别名 — 旧代码中 `from minecraft_env import MinecraftCraftingEnv`
会自动映射到新的 MinecraftSurvivalEnv。
"""
from environment import MinecraftSurvivalEnv as MinecraftCraftingEnv
from environment import DIFFICULTY_CONFIGS

__all__ = ["MinecraftCraftingEnv", "DIFFICULTY_CONFIGS"]
