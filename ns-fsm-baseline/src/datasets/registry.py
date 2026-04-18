"""Dataset adapter registry.

The registry keeps benchmark adapters optional. Unknown datasets fall back to
generic mode so arbitrary user scenarios can still enter the pipeline.
"""

from __future__ import annotations

from typing import Any, Type

try:
    from .base import DatasetAdapter
    from .generic import GenericScenarioAdapter
    from .structured import StructuredScenarioAdapter
except ImportError:  # pragma: no cover - supports direct src execution
    from base import DatasetAdapter
    from generic import GenericScenarioAdapter
    from structured import StructuredScenarioAdapter


DATASET_REGISTRY: dict[str, Type[DatasetAdapter]] = {
    "generic": GenericScenarioAdapter,
    "structured": StructuredScenarioAdapter,
}


def register_dataset(name: str, adapter_cls: Type[DatasetAdapter]) -> None:
    """Register or replace a dataset adapter class."""

    if not name:
        raise ValueError("Dataset name cannot be empty.")
    DATASET_REGISTRY[name] = adapter_cls


def get_adapter_cls(name: str | None) -> Type[DatasetAdapter]:
    """Return an adapter class, falling back to generic when unknown."""

    dataset_name = (name or "generic").strip()
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name]

    if dataset_name == "minecraft":
        try:
            try:
                from .minecraft import MinecraftAdapter
            except ImportError:  # pragma: no cover
                from minecraft import MinecraftAdapter
            register_dataset("minecraft", MinecraftAdapter)
            return MinecraftAdapter
        except Exception:
            return GenericScenarioAdapter

    if dataset_name in {"swe_bench", "swe-bench", "swebench"}:
        try:
            try:
                from .swe_bench import SWEBenchAdapter
            except ImportError:  # pragma: no cover
                from swe_bench import SWEBenchAdapter
            register_dataset("swe_bench", SWEBenchAdapter)
            return SWEBenchAdapter
        except Exception:
            return GenericScenarioAdapter

    return GenericScenarioAdapter


def get_adapter(name: str | None = None, **kwargs: Any) -> DatasetAdapter:
    """Instantiate an adapter by name."""

    adapter_cls = get_adapter_cls(name)
    return adapter_cls(**kwargs)


def list_datasets() -> list[str]:
    """Return known dataset names without importing optional adapters."""

    names = set(DATASET_REGISTRY)
    names.add("minecraft")
    names.add("swe_bench")
    return sorted(names)


def registry_warning(requested_name: str | None, adapter: DatasetAdapter) -> str:
    """Return a human-readable fallback warning when applicable."""

    requested = requested_name or "generic"
    if requested != adapter.dataset_name:
        return f"Unknown or unavailable dataset '{requested}', using '{adapter.dataset_name}'."
    return ""
