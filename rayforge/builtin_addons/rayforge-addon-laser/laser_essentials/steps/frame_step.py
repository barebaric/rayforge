from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Protocol, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.frame import FrameSpec
from raygeo.ops.part import Part

from rayforge.core.capability import MachineCapability
from rayforge.core.cut_side import CutSide
from rayforge.core.step import legacy_producer_params
from rayforge.core.varset import LabeledChoiceVar, LengthVar, VarSet
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector_with_raster_fallback,
)
from rayforge.pipeline.transformer.registry import transformer_registry

from .laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine

    class LeadInOutTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


class FrameStep(LaserStep):
    TYPELABEL = _("Frame")
    ICON = "step-frame-symbolic"
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})
    ASSEMBLER_NAME = "frame"

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *LaserStep.recipe_varset().vars,
                LabeledChoiceVar(
                    key="cut_side",
                    label=_("Cut Side"),
                    choices=[(cs.label(), cs.name) for cs in CutSide],
                    default="CENTERLINE",
                    allow_none=False,
                ),
                LengthVar(
                    key="offset_mm",
                    label=_("Offset"),
                    description=_(
                        "Shifts the frame inward/outward per Cut Side "
                        "(none on Centerline). Defaults to kerf "
                        "compensation for the head"
                    ),
                    default=0.0,
                    sensitive_when=lambda v: v.get("cut_side") != "CENTERLINE",
                ),
            ]
        )

    def __init__(self, name: str | None = None, typelabel: str | None = None):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.power = 0.8
        self.offset_mm = 0.0
        self.cut_side = "CENTERLINE"

    def get_operation_mode_short(self):
        try:
            return CutSide[self.cut_side].label()
        except (KeyError, TypeError):
            return None

    def get_assembler_kwargs(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict:
        kwargs: dict = {}
        kwargs["cut_side"] = str(self.cut_side).lower()
        kwargs["offset_mm"] = self.offset_mm
        return kwargs

    def build_compute_payload(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> tuple[Part, ComputePayload]:
        """Build a :class:`Part` (from the workpiece's vector
        geometry) and a :class:`ComputePayload` carrying a
        :class:`FrameSpec`.

        When the workpiece has no vector boundaries, the source is
        rendered to pixels and traced into geometry before assembling.
        """
        part = build_part_vector_with_raster_fallback(
            workpiece, self.pixels_per_mm
        )
        kwargs = self.get_assembler_kwargs(machine, workpiece)
        spec = FrameSpec(
            offset_mm=kwargs["offset_mm"],
            cut_side=kwargs["cut_side"],
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict | None:
        return self.get_assembler_kwargs(machine, workpiece)

    def apply_import_settings(self, settings: dict) -> None:
        """Apply importer-provided settings this step owns."""
        super().apply_import_settings(settings)
        offset_mm = settings.get("offset_mm")
        if offset_mm is not None:
            self.offset_mm = offset_mm

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["cut_side"] = self.cut_side
        data["offset_mm"] = self.offset_mm
        return data

    @classmethod
    def from_dict(cls, data: dict) -> FrameStep:
        step = cast("FrameStep", super().from_dict(data))
        legacy = legacy_producer_params(data)
        step.cut_side = data.get(
            "cut_side",
            legacy.get("cut_side", legacy.get("kerf_mode", "CENTERLINE")),
        )
        if "offset_mm" in data:
            step.offset_mm = data["offset_mm"]
        else:
            path_offset = data.get(
                "path_offset_mm",
                legacy.get("path_offset_mm", legacy.get("offset_mm", 0.0)),
            )
            step.offset_mm = path_offset + (data.get("kerf_mm", 0.0) / 2.0)
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        TabOpsTransformer = transformer_registry.get("TabOpsTransformer")
        CropTransformer = transformer_registry.get("CropTransformer")
        MergeLinesTransformer = transformer_registry.get(
            "MergeLinesTransformer"
        )
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert LeadInOutTransformer is not None
        assert TabOpsTransformer is not None
        assert CropTransformer is not None
        assert MergeLinesTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            LeadInOutTransformer(
                enabled=False, lead_in_mm=0, lead_out_mm=0, auto=True
            ).to_dict(),
            TabOpsTransformer().to_dict(),
            CropTransformer(enabled=False).to_dict(),
            optimize_dict,
        ], [
            MergeLinesTransformer().to_dict(),
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: RayforgeContext,
        name: str | None = None,
        **kwargs,
    ) -> FrameStep:
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
        step.offset_mm = default_head.kerf_mm
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        # Operating feed defaults are machine-derived: the machine only
        # exposes its ceiling, so the default is that ceiling, bounded by
        # the operation's typical feed rate.
        step.cut_speed = min(machine.max_cut_speed, 500)
        params = machine.get_pwm_params(default_head)
        if params is not None:
            step.frequency = params.frequency
            step.pulse_width = params.pulse_width

        LeadInOutTransformer = cast(
            "LeadInOutTransformerType",
            transformer_registry.get("LeadInOutTransformer"),
        )
        if LeadInOutTransformer:
            calc = LeadInOutTransformer.calculate_auto_distance
            auto_distance = calc(step.cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        return step
