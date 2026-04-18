"""FSM template registry."""

from .base import FSMTemplate, get_template, list_templates, register_template
from .code_repair import CodeRepairFSM
from .generic_tool_use import GenericToolUseFSM
from .symbolic_planning import SymbolicPlanningFSM

register_template("generic_tool_use", GenericToolUseFSM)
register_template("structured_tool_use", GenericToolUseFSM)
register_template("symbolic_planning", SymbolicPlanningFSM)
register_template("code_repair", CodeRepairFSM)

__all__ = [
    "CodeRepairFSM",
    "FSMTemplate",
    "GenericToolUseFSM",
    "SymbolicPlanningFSM",
    "get_template",
    "list_templates",
    "register_template",
]
