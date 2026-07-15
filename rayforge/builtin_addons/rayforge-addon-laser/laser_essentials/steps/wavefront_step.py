from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

from rayforge.core.capability import CUT, Capability
from rayforge.core.step import Step
from rayforge.pipeline.assembler.registry import assembler_registry
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


class WavefrontStep(Step):
    TYPELABEL = _("Wavefront")
    ICON = "step-contour-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (CUT,)
    ASSEMBLER_NAME = "wavefront"
    NORMALIZE_WINDINGS = True

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.step_over_mm: Optional[float] = None
        self.offset_mm = 0.0
        self.area_tolerance = 0.01

    def get_assembler_kwargs(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["offset_mm"] = self.offset_mm
        kwargs["area_tolerance"] = self.area_tolerance
        kwargs["step_over"] = (
            self.step_over_mm
            if self.step_over_mm is not None
            else machine_defaults.step_over
        )
        kwargs["precision"] = machine_defaults.arc_tolerance
        kwargs["cut_feed_rate"] = machine_defaults.cut_speed
        kwargs["cut_power"] = machine_defaults.step_power
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
        set_power = (
            machine_defaults.step_power if self.SET_POWER else None
        )
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
        result = super().to_dict()
        result["step_over_mm"] = self.step_over_mm
        result["offset_mm"] = self.offset_mm
        result["area_tolerance"] = self.area_tolerance
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "WavefrontStep":
        step = cast("WavefrontStep", super().from_dict(data))
        step.step_over_mm = data.get("step_over_mm", None)
        step.offset_mm = data.get("offset_mm", 0.0)
        step.area_tolerance = data.get("area_tolerance", 0.01)
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        CropTransformer = transformer_registry.get("CropTransformer")
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert CropTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            CropTransformer(enabled=False).to_dict(),
            optimize_dict,
        ], [
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "WavefrontStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()
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
