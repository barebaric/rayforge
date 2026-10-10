from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.geo import Geometry
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.part import Part

from rayforge.core.capability import MachineCapability
from rayforge.core.cut_side import CutOrder
from rayforge.core.varset import LabeledChoiceVar, LengthVar, VarSet
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector_with_raster_fallback,
)
from rayforge.pipeline.transformer.registry import transformer_registry

from ..offset_fill import offset_fill_rings
from .laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


def _fill_direction_choices() -> list[tuple[str, str]]:
    return [
        (_("Outside → Inside"), CutOrder.OUTSIDE_INSIDE.name),
        (_("Inside → Outside"), CutOrder.INSIDE_OUTSIDE.name),
    ]


class OffsetFillStep(LaserStep):
    """Fills closed shapes with rings parallel to their outline."""

    TYPELABEL = _("Offset Fill")
    ICON = "step-offset-fill-symbolic"
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})
    ASSEMBLER_NAME = "offset_fill"

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *LaserStep.recipe_varset().vars,
                LengthVar(
                    key="line_interval_mm",
                    label=_("Line Interval"),
                    description=_(
                        "Distance between neighbouring fill rings. "
                        "Defaults to the laser spot size"
                    ),
                    default=None,
                    min_val=0.01,
                    max_val=50.0,
                ),
                LengthVar(
                    key="offset_mm",
                    label=_("Offset"),
                    description=_(
                        "Distance of the first ring inside the outline"
                    ),
                    default=0.0,
                    min_val=0.0,
                    max_val=20.0,
                ),
                LabeledChoiceVar(
                    key="fill_direction",
                    label=_("Fill Direction"),
                    description=_("Order in which the rings are cut"),
                    choices=_fill_direction_choices(),
                    default=CutOrder.OUTSIDE_INSIDE.name,
                    allow_none=False,
                ),
            ]
        )

    def __init__(self, name: str | None = None, typelabel: str | None = None):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.power = 0.8
        self.line_interval_mm: float | None = None
        self.offset_mm = 0.0
        self.fill_direction = CutOrder.OUTSIDE_INSIDE.name

    def get_assembler_kwargs(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict:
        spot_x, _spot_y = LaserHead.get_spot_size(
            self.get_selected_laser(machine)
        )
        interval = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else spot_x
        )
        return {
            "line_interval_mm": interval,
            "offset_mm": self.offset_mm,
            "cut_order": str(self.fill_direction).lower(),
            "arc_tolerance": machine.arc_tolerance,
            "allow_arcs": machine.supports_arcs,
            "supports_curves": machine.supports_curves,
        }

    def build_compute_payload(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> tuple[Part, ComputePayload]:
        """Build a :class:`Part` with one face per fill ring, in the
        chosen fill direction, and a :class:`ComputePayload` that cuts
        the faces in that order along their centre line with the contour
        assembler.
        """
        source = build_part_vector_with_raster_fallback(
            workpiece, self.pixels_per_mm
        )
        kwargs = self.get_assembler_kwargs(machine, workpiece)
        inside_out = kwargs["cut_order"] == "inside_outside"
        part = Part(size_mm=workpiece.size)
        rings: list[Geometry] = []
        if source is not None and source.has_geometry():
            for face_id in source.face_ids:
                face_geo = source.face(face_id).geometry
                if face_geo is None or face_geo.is_empty():
                    continue
                face_rings = offset_fill_rings(
                    face_geo,
                    step=kwargs["line_interval_mm"],
                    offset=kwargs["offset_mm"],
                )
                if inside_out:
                    face_rings.reverse()
                rings.extend(face_rings)
        for index, ring in enumerate(rings):
            part.add_face(f"ring-{index}", ring)
        spec = ContourSpec(
            offset_mm=0.0,
            cut_side="centerline",
            overcut=0.0,
            cut_order=kwargs["cut_order"],
            remove_inner=False,
            arc_tolerance=kwargs["arc_tolerance"],
            allow_arcs=kwargs["allow_arcs"],
            supports_curves=kwargs["supports_curves"],
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict | None:
        return self.get_assembler_kwargs(machine, workpiece)

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["line_interval_mm"] = self.line_interval_mm
        result["offset_mm"] = self.offset_mm
        result["fill_direction"] = self.fill_direction
        return result

    @classmethod
    def from_dict(cls, data: dict) -> OffsetFillStep:
        step = cast("OffsetFillStep", super().from_dict(data))
        step.line_interval_mm = data.get("line_interval_mm")
        step.offset_mm = data.get("offset_mm", 0.0)
        step.fill_direction = data.get(
            "fill_direction", CutOrder.OUTSIDE_INSIDE.name
        )
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        Smooth = transformer_registry.get("Smooth")
        CropTransformer = transformer_registry.get("CropTransformer")
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert Smooth is not None
        assert CropTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            Smooth(enabled=False, amount=20).to_dict(),
            CropTransformer(enabled=False).to_dict(),
            optimize_dict,
        ], [
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: RayforgeContext,
        name: str | None = None,
        **kwargs,
    ) -> OffsetFillStep:
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_laser_head()
        if default_head is None:
            raise ValueError("Machine has no laser heads configured.")

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()
        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_head_uid = default_head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        step.cut_speed = min(machine.max_cut_speed, 500)
        params = machine.get_pwm_params(default_head)
        if params is not None:
            step.frequency = params.frequency
            step.pulse_width = params.pulse_width
        return step
