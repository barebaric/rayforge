from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import numpy as np
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops import Ops
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.shrinkwrap import ShrinkwrapSpec
from raygeo.ops.part import Part
from raygeo.ops.part.image_source import WholeImageSource

from rayforge.core.capability import CUT, SCORE, WITH_KERF, Capability
from rayforge.core.cut_side import CutSide
from rayforge.core.step import Step
from rayforge.image.tracing import prepare_surface
from rayforge.pipeline.assembler.registry import assembler_registry
from rayforge.pipeline.stage.assembler_helpers import (
    MachineDefaults,
    build_part_vector,
    make_artifact,
    wrap_assembler_result,
)
from rayforge.pipeline.transformer.registry import transformer_registry

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser
    from rayforge.pipeline.artifact import WorkPieceArtifact


class ShrinkWrapStep(Step):
    TYPELABEL = _("Shrink Wrap")
    ICON = "step-shrinkwrap-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (CUT, SCORE, WITH_KERF)
    ASSEMBLER_NAME = "shrinkwrap"
    SET_POWER = True

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.gravity = 0.0
        self.path_offset_mm = 0.0
        self.cut_side = "CENTERLINE"

    def get_operation_mode_short(self):
        if not self.cut_side:
            return None
        try:
            return CutSide[self.cut_side].label()
        except KeyError:
            return None

    def get_assembler_kwargs(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["cut_side"] = self.cut_side.lower()
        kwargs["gravity"] = self.gravity
        kwargs["path_offset_mm"] = self.path_offset_mm
        kwargs["kerf_mm"] = machine_defaults.kerf_mm
        kwargs["arc_tolerance"] = machine_defaults.arc_tolerance
        kwargs["allow_arcs"] = machine_defaults.allow_arcs
        kwargs["supports_curves"] = machine_defaults.supports_curves
        return kwargs

    def build_compute_payload(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> "Tuple[Part, ComputePayload]":
        """Build a :class:`Part` with vector geometry and a boolean
        image, and a :class:`ComputePayload` carrying a
        :class:`ShrinkwrapSpec`."""
        part = _build_shrinkwrap_part(workpiece)
        kwargs = self.get_assembler_kwargs(machine_defaults, workpiece)
        spec = ShrinkwrapSpec(
            gravity=kwargs["gravity"],
            kerf_mm=kwargs["kerf_mm"],
            path_offset_mm=kwargs["path_offset_mm"],
            cut_side=kwargs["cut_side"],
            arc_tolerance=kwargs["arc_tolerance"],
            allow_arcs=kwargs["allow_arcs"],
            supports_curves=kwargs["supports_curves"],
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> Optional[dict]:
        return self.get_assembler_kwargs(machine_defaults, workpiece)

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
        part = build_part_vector(
            workpiece,
            surface=surface,
            normalize_windings=self.NORMALIZE_WINDINGS,
        )

        if surface is not None:
            assert part is not None
            boolean_image = prepare_surface(surface)
            if not np.any(boolean_image):
                return make_artifact(
                    Ops(),
                    workpiece,
                    generation_id,
                    is_vector=self.IS_VECTOR,
                )
            part.image = boolean_image

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

    def requires_full_render(self) -> bool:
        return True

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["gravity"] = self.gravity
        result["path_offset_mm"] = self.path_offset_mm
        result["cut_side"] = self.cut_side
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ShrinkWrapStep":
        step = cast("ShrinkWrapStep", super().from_dict(data))
        step.gravity = data.get("gravity", 0.0)
        step.path_offset_mm = data.get("path_offset_mm", 0.0)
        step.cut_side = data.get("cut_side", "CENTERLINE")
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        Smooth = transformer_registry.get("Smooth")
        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        TabOpsTransformer = transformer_registry.get("TabOpsTransformer")
        CropTransformer = transformer_registry.get("CropTransformer")
        MergeLinesTransformer = transformer_registry.get(
            "MergeLinesTransformer"
        )
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert Smooth is not None
        assert LeadInOutTransformer is not None
        assert TabOpsTransformer is not None
        assert CropTransformer is not None
        assert MergeLinesTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            Smooth(enabled=False, amount=20).to_dict(),
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
    ) -> "ShrinkWrapStep":
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

        # step.cut_speed is only final after the loop above.
        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        if LeadInOutTransformer:
            calc = getattr(LeadInOutTransformer, "calculate_auto_distance")
            auto_distance = calc(step.cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        return step


def _build_shrinkwrap_part(workpiece: "WorkPiece") -> Part:
    """Build a :class:`Part` for the shrinkwrap assembler.

    Uses the workpiece's vector boundaries when available, falling
    back to a traced boolean image from a rendered surface.  The
    boolean image is attached as a :class:`WholeImageSource` so the
    Rust assembler can read it on a rayon worker.
    """
    part = build_part_vector(workpiece)
    if part is None or not part.has_geometry():
        size = workpiece.size
        if size[0] <= 0 or size[1] <= 0:
            return Part(size_mm=size)
        px_per_mm = (50.0, 50.0)
        target_w = max(1, int(size[0] * px_per_mm[0]))
        target_h = max(1, int(size[1] * px_per_mm[1]))
        surface = workpiece.render_to_pixels(target_w, target_h)
        if surface is None:
            return Part(size_mm=size)
        boolean = prepare_surface(surface)
        if not np.any(boolean):
            return Part(size_mm=size)
        part = Part(size_mm=size)
        part.image_source = WholeImageSource(boolean)
        return part
    return part
