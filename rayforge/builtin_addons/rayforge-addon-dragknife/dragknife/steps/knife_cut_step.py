from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, ClassVar, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.part import Part

from rayforge.core.step import Step
from rayforge.core.varset import LengthVar, VarSet

from ..knife_head_var import KnifeHeadVar

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


class KnifeCutStep(Step):
    """Base class for knife assembler-driven steps.

    Subclasses set ``REQUIRED_MACHINE_CAPS``, ``ASSEMBLER_NAME``,
    ``HEAD_CAPABILITY`` and ``KNIFE_TRANSFORMER_NAME`` and override
    ``_default_knife_transformer`` and ``_knife_transformer_instance``
    to configure the internal knife compensation transformer that the
    pipeline applies after contour assembly.
    """

    ASSEMBLER_NAME: ClassVar[str] = ""
    KNIFE_TRANSFORMER_NAME: ClassVar[str] = ""
    HEAD_CAPABILITY: ClassVar[Any] = None

    def __init__(self, name=None, typelabel=None):
        self.offset_mm: float = 0.5
        self.safe_z: float = 2.0
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)

    @classmethod
    def create(
        cls,
        context: RayforgeContext,
        name=None,
        **kwargs,
    ) -> KnifeCutStep:
        machine = context.machine
        step = cls(name=name)
        step.per_step_transformers_dicts = []
        if machine is None:
            step.selected_head_uid = None
            return step
        head = next(
            (
                h
                for h in machine.heads
                if h.machine_capability is cls.HEAD_CAPABILITY
            ),
            machine.get_default_head(),
        )
        if head is None:
            step.selected_head_uid = None
            return step
        step.selected_head_uid = head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        step.apply_head_defaults(head)
        step.sync_knife_transformer()
        return step

    def apply_head_defaults(self, head) -> None:
        """Seed step settings from the selected head's defaults."""
        offset = getattr(head, "offset_mm", None)
        if offset is not None:
            self.offset_mm = float(offset)

    @classmethod
    def _default_knife_transformer(cls):
        """A knife compensation transformer with default settings."""
        raise NotImplementedError

    def _knife_transformer_instance(self):
        """The knife compensation transformer for this step's state."""
        raise NotImplementedError

    def sync_knife_transformer(self) -> None:
        """Update the internal knife transformer dict from the step
        settings. The transformer is an implementation detail; users
        configure the knife through this step's settings."""
        name = self.KNIFE_TRANSFORMER_NAME
        for t_dict in self.per_workpiece_transformers_dicts:
            if t_dict.get("name") == name:
                t_dict.update(self._knife_transformer_instance().to_dict())
                return

    def set_offset_mm(self, value: float):
        value = max(0.0, float(value))
        if self.offset_mm == value:
            return
        self.offset_mm = value
        self.sync_knife_transformer()
        self.updated.send(self)

    def set_safe_z(self, value: float):
        value = max(0.0, float(value))
        if self.safe_z == value:
            return
        self.safe_z = value
        self.sync_knife_transformer()
        self.updated.send(self)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({"offset_mm": self.offset_mm, "safe_z": self.safe_z})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnifeCutStep:
        step = cast("KnifeCutStep", super().from_dict(data))
        step.offset_mm = max(0.0, float(data.get("offset_mm", step.offset_mm)))
        step.safe_z = max(0.0, float(data.get("safe_z", step.safe_z)))
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset({"offset_mm", "safe_z"})

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        return [cls._default_knife_transformer().to_dict()], []

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                KnifeHeadVar(
                    cls.HEAD_CAPABILITY,
                    description=_(
                        "Knife head used for this step; the machine's "
                        "first knife is used when unset"
                    ),
                ),
                LengthVar(
                    key="offset_mm",
                    label=_("Blade Offset"),
                    description=_(
                        "Distance between the blade tip and the "
                        "machine's pivot point"
                    ),
                    default=0.5,
                    min_val=0.0,
                    max_val=20.0,
                ),
                LengthVar(
                    key="safe_z",
                    label=_("Safe Z Height"),
                    description=_("Height to retract between moves"),
                    default=2.0,
                    min_val=0.0,
                    max_val=50.0,
                ),
                *Step.recipe_varset().vars,
            ]
        )

    @classmethod
    def recipe_varset_groups(cls) -> list[tuple[str, VarSet]]:
        full = cls.recipe_varset()
        knife_keys = {v.key for v in KnifeCutStep.recipe_varset()}
        knife_vars = [v for v in full if v.key in knife_keys]
        step_vars = [v for v in full if v.key not in knife_keys]
        knife_description = _(
            "Knife head, blade geometry, and retraction height for "
            "this operation."
        )
        knife_vs = VarSet(vars=knife_vars, description=knife_description)
        groups: list[tuple[str, VarSet]] = [(_("Knife"), knife_vs)]
        if step_vars:
            groups.append((_("Step Settings"), VarSet(vars=step_vars)))
        return groups

    def populate_payload(self, payload, machine: Machine):
        super().populate_payload(payload, machine)
        # The renderer colours ops by power and treats zero as a "no
        # cut" state. A knife always cuts at full force while down.
        payload.power = 1.0

    def build_spec(
        self, machine: Machine, workpiece: WorkPiece
    ) -> ContourSpec:
        """The raygeo contour assembler spec: cut on the line."""
        return ContourSpec(
            offset_mm=0.0,
            cut_side="centerline",
            overcut=0.0,
            cut_order="inside_outside",
            remove_inner=False,
            arc_tolerance=machine.arc_tolerance,
            allow_arcs=machine.supports_arcs,
            supports_curves=machine.supports_curves,
        )

    def build_compute_payload(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> tuple[Part, ComputePayload]:
        part = workpiece.to_part()
        if part is None:
            part = Part(size_mm=workpiece.size)
        return part, ComputePayload(
            assembler=Assembler(self.build_spec(machine, workpiece)),
            cut_speed=self.cut_speed,
        )

    def assembler_token_params(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict[str, Any]:
        return {
            "offset_mm": self.offset_mm,
            "safe_z": self.safe_z,
        }
