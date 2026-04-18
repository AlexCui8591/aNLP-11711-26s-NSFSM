"""Build RuntimeFSM objects from validated designs or reusable templates."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from .datasets.base import TaskSpec, task_spec_to_dict
    from .fsm import RuntimeFSM
    from .fsm_templates import FSMTemplate, get_template
    from .fsm_validator import FSMDesignValidator
except ImportError:  # pragma: no cover - supports direct src execution
    from datasets.base import TaskSpec, task_spec_to_dict
    from fsm import RuntimeFSM
    from fsm_templates import FSMTemplate, get_template
    from fsm_validator import FSMDesignValidator


class FSMBuilder:
    """Factory for runtime FSMs."""

    def __init__(self, validator: FSMDesignValidator | None = None):
        self.validator = validator or FSMDesignValidator()

    def build(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
        template: FSMTemplate | None = None,
        fsm_design: Mapping[str, Any] | None = None,
        allow_fallback: bool = True,
    ) -> RuntimeFSM:
        if fsm_design is not None:
            return self.from_design(
                fsm_design=fsm_design,
                task_spec=task_spec,
                adapter=adapter,
                allow_fallback=allow_fallback,
            )
        return self.from_template(task_spec=task_spec, adapter=adapter, template=template)

    def from_design(
        self,
        fsm_design: Mapping[str, Any],
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
        allow_fallback: bool = True,
    ) -> RuntimeFSM:
        validation = self.validator.validate(
            fsm_design,
            task_spec=task_spec,
            adapter=adapter,
            allow_fallback=allow_fallback,
        )
        if not validation["valid"]:
            raise ValueError(f"FSM design failed validation: {validation['errors']}")
        runtime = RuntimeFSM(validation["fsm_design"], task_spec=task_spec, adapter=adapter)
        runtime.validation = validation
        return runtime

    def from_template(
        self,
        task_spec: TaskSpec | Mapping[str, Any],
        adapter: Any | None = None,
        template: FSMTemplate | None = None,
    ) -> RuntimeFSM:
        spec = task_spec_to_dict(task_spec)
        selected_template = template or get_template(spec.get("task_type"))
        design = selected_template.build_design(spec, adapter)
        runtime = self.from_design(
            fsm_design=design,
            task_spec=spec,
            adapter=adapter,
            allow_fallback=True,
        )
        runtime.template_name = selected_template.name
        return runtime


def build_fsm(
    task_spec: TaskSpec | Mapping[str, Any],
    adapter: Any | None = None,
    fsm_design: Mapping[str, Any] | None = None,
    template: FSMTemplate | None = None,
) -> RuntimeFSM:
    return FSMBuilder().build(
        task_spec=task_spec,
        adapter=adapter,
        template=template,
        fsm_design=fsm_design,
    )
