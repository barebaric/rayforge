from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.frame import FrameSpec
from raygeo.ops.part import Part

from rayforge.core.capability import MachineCapability, StepCapability
from rayforge.core.cut_side import CutSide
from rayforge.core.varset import FloatVar, LabeledChoiceVar, VarSet
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector_with_raster_fallback,
)
from rayforge.pipeline.transformer.registry import transformer_registry

from ..capabilities import CUT, SCORE, WITH_KERF
from .laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


class FrameStep(LaserStep):
    TYPELABEL = _("Frame")
    ICON = "step-frame-symbolic"
    CAPABILITIES: Tuple[StepCapability, ...] = (CUT, SCORE, WITH_KERF)
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})
    ASSEMBLER_NAME = "frame"

    RECIPE_KEYS: Tuple[str, ...] = LaserStep.RECIPE_KEYS + (
        "cut_side",
        "path_offset_mm",
    )

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
                ),
                FloatVar(
                    key="path_offset_mm",
                    label=_("Path Offset"),
                    default=0.0,
                ),
            ]
        )

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.power = 0.8
        self.kerf_mm = 0.1
        self.path_offset_mm = 0.0
        self.cut_side = "CENTERLINE"

    def get_operation_mode_short(self):
        try:
            return CutSide[self.cut_side].label()
        except (KeyError, TypeError):
            return None

    def get_assembler_kwargs(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["cut_side"] = str(self.cut_side).lower()
        kwargs["path_offset_mm"] = self.path_offset_mm
        kwargs["kerf_mm"] = self.kerf_mm
        return kwargs

    def build_compute_payload(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> "Tuple[Part, ComputePayload]":
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
            kerf_mm=kwargs["kerf_mm"],
            path_offset_mm=kwargs["path_offset_mm"],
            cut_side=kwargs["cut_side"],
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> Optional[dict]:
        return self.get_assembler_kwargs(machine, workpiece)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["cut_side"] = self.cut_side
        data["path_offset_mm"] = self.path_offset_mm
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FrameStep":
        step = cast("FrameStep", super().from_dict(data))
        step.cut_side = data.get("cut_side", "CENTERLINE")
        step.path_offset_mm = data.get("path_offset_mm", 0.0)
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
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
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "FrameStep":
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
        step.kerf_mm = default_head.spot_size_mm[0]
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

        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        if LeadInOutTransformer:
            calc = getattr(LeadInOutTransformer, "calculate_auto_distance")
            auto_distance = calc(step.cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        return step
