from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

from rayforge.core.capability import CUT, SCORE, WITH_KERF, Capability
from rayforge.core.step import Step
from rayforge.pipeline.assembler.registry import assembler_registry
from rayforge.core.cut_side import CutSide
from rayforge.pipeline.stage.assembler_helpers import (
    MachineDefaults,
    build_part_vector,
    make_artifact,
    wrap_assembler_result,
)
from rayforge.pipeline.transformer.registry import transformer_registry
from raygeo.ops import Ops

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser
    from rayforge.pipeline.artifact import WorkPieceArtifact


class FrameStep(Step):
    TYPELABEL = _("Frame")
    ICON = "step-frame-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (CUT, SCORE, WITH_KERF)
    ASSEMBLER_NAME = "frame"
    SET_POWER = True

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.path_offset_mm = 0.0
        self.cut_side = "CENTERLINE"

    def get_operation_mode_short(self):
        try:
            return CutSide[self.cut_side].label()
        except (KeyError, TypeError):
            return None

    def get_assembler_kwargs(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["cut_side"] = str(self.cut_side).lower()
        kwargs["path_offset_mm"] = self.path_offset_mm
        kwargs["kerf_mm"] = machine_defaults.kerf_mm
        return kwargs

    def assemble_on_surface(
        self,
        workpiece: "WorkPiece",
        laser: "Laser",
        generation_id: int,
        surface: Any = None,
        pixels_per_mm: Optional[Tuple[float, float]] = None,
        *,
        machine_defaults: "MachineDefaults",
        y_offset_mm: float = 0.0,
        computed_auto_levels: Optional[Tuple[int, int]] = None,
    ) -> "WorkPieceArtifact":
        use_surface = surface is not None and (
            not workpiece.boundaries or workpiece.boundaries.is_empty()
        )
        part = build_part_vector(
            workpiece,
            surface=surface if use_surface else None,
            normalize_windings=self.NORMALIZE_WINDINGS,
        )
        if part is None or not part.has_geometry():
            return make_artifact(
                Ops(), workpiece, generation_id, is_vector=self.IS_VECTOR
            )
        kwargs = self.get_assembler_kwargs(machine_defaults, workpiece)
        result = assembler_registry.assemble(
            self.ASSEMBLER_NAME, part, **kwargs
        )
        set_power = machine_defaults.step_power if self.SET_POWER else None
        return wrap_assembler_result(
            result,
            workpiece,
            laser,
            generation_id,
            split_contours=self.SPLIT_CONTOURS,
            set_power=set_power,
            is_vector=self.IS_VECTOR,
        )

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
        default_head = machine.get_default_head()

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()

        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        if LeadInOutTransformer:
            calc = getattr(LeadInOutTransformer, "calculate_auto_distance")
            auto_distance = calc(machine.max_cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = default_head.uid
        step.kerf_mm = default_head.spot_size_mm[0]
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)
        return step
