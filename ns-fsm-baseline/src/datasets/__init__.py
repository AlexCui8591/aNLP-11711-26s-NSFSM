"""Dataset adapters for the NS-FSM pipeline."""

from .base import DatasetAdapter, StepResult, TaskSpec, task_spec_to_dict
from .generic import GenericScenarioAdapter
from .registry import get_adapter, get_adapter_cls, list_datasets, register_dataset
from .structured import StructuredScenarioAdapter
from .swe_bench import SWEBenchAdapter

__all__ = [
    "DatasetAdapter",
    "GenericScenarioAdapter",
    "StepResult",
    "StructuredScenarioAdapter",
    "SWEBenchAdapter",
    "TaskSpec",
    "get_adapter",
    "get_adapter_cls",
    "list_datasets",
    "register_dataset",
    "task_spec_to_dict",
]
