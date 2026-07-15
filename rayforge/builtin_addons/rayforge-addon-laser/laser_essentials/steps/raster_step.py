from __future__ import annotations

from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    cast,
)

import numpy as np
from raygeo.ops import Ops
from raygeo.ops.types import SectionType

from rayforge.core.capability import ENGRAVE, Capability
from rayforge.core.step import Step
from rayforge.image.dither import DitherAlgorithm
from rayforge.pipeline.assembler.registry import assembler_registry
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    MachineDefaults,
    build_part_raster,
    compute_raster_auto_levels,
    make_artifact,
    preprocess_raster_image,
)
from rayforge.pipeline.transformer.registry import transformer_registry


if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser
    from rayforge.pipeline.artifact import WorkPieceArtifact

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


class EngraveStep(Step):
    TYPELABEL = _("Engrave")
    ICON = "step-raster-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (ENGRAVE,)
    ASSEMBLER_NAME = "raster"
    IS_VECTOR = False
    ALWAYS_WRAP = True
    SECTION_TYPE = SectionType.RASTER_FILL

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.scan_angle = 0.0
        self.depth_mode = "POWER_MODULATION"
        self.invert = False
        self.auto_levels = True
        self.black_point = 0
        self.white_point = 255
        self.threshold = 128
        self.line_interval_mm = None
        self.sample_interval_mm = None
        self.min_power = 0.0
        self.max_power = 1.0
        self.num_power_levels = 25
        self.offset_x_mm = 0.0
        self.offset_y_mm = 0.0
        self.scan_mode = "SEGMENTED"
        self.cross_hatch = False
        self.num_depth_levels = 5
        self.z_step_down = 0.0
        self.angle_increment = 0.0
        self.dither_algorithm = None

    def get_operation_mode_short(self):
        if not self.depth_mode:
            return None
        try:
            return DepthMode[self.depth_mode].short_name
        except KeyError:
            return None

    def get_assembler_kwargs(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        line_interval = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else machine_defaults.line_interval_mm
        )
        step_power = machine_defaults.step_power
        return {
            "mode": DepthMode[self.depth_mode].raygeo_name,
            "line_interval_mm": line_interval,
            "sample_interval_mm": self.sample_interval_mm,
            "min_power": self.min_power,
            "max_power": self.max_power,
            "step_power": step_power,
            "num_power_levels": self.num_power_levels,
            "angle": self.scan_angle,
            "offset_x_mm": self.offset_x_mm,
            "offset_y_mm": self.offset_y_mm,
            "scan_mode": self.scan_mode.lower(),
            "cross_hatch": self.cross_hatch,
            "num_depth_levels": self.num_depth_levels,
            "z_step_down": self.z_step_down,
            "angle_increment": self.angle_increment,
        }

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["scan_angle"] = self.scan_angle
        result["depth_mode"] = self.depth_mode
        result["invert"] = self.invert
        result["auto_levels"] = self.auto_levels
        result["black_point"] = self.black_point
        result["white_point"] = self.white_point
        result["threshold"] = self.threshold
        result["line_interval_mm"] = self.line_interval_mm
        result["sample_interval_mm"] = self.sample_interval_mm
        result["min_power"] = self.min_power
        result["max_power"] = self.max_power
        result["num_power_levels"] = self.num_power_levels
        result["offset_x_mm"] = self.offset_x_mm
        result["offset_y_mm"] = self.offset_y_mm
        result["scan_mode"] = self.scan_mode
        result["cross_hatch"] = self.cross_hatch
        result["num_depth_levels"] = self.num_depth_levels
        result["z_step_down"] = self.z_step_down
        result["angle_increment"] = self.angle_increment
        result["dither_algorithm"] = (
            self.dither_algorithm.value
            if self.dither_algorithm
            else None
        )
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "EngraveStep":
        step = cast("EngraveStep", super().from_dict(data))
        step.scan_angle = data.get("scan_angle", 0.0)
        step.depth_mode = data.get("depth_mode", "POWER_MODULATION")
        step.invert = data.get("invert", False)
        step.auto_levels = data.get("auto_levels", True)
        step.black_point = data.get("black_point", 0)
        step.white_point = data.get("white_point", 255)
        step.threshold = data.get("threshold", 128)
        step.line_interval_mm = data.get("line_interval_mm", None)
        step.sample_interval_mm = data.get("sample_interval_mm", None)
        step.min_power = data.get("min_power", 0.0)
        step.max_power = data.get("max_power", 1.0)
        step.num_power_levels = data.get("num_power_levels", 25)
        step.offset_x_mm = data.get("offset_x_mm", 0.0)
        step.offset_y_mm = data.get("offset_y_mm", 0.0)
        step.scan_mode = data.get("scan_mode", "SEGMENTED")
        step.cross_hatch = data.get("cross_hatch", False)
        step.num_depth_levels = data.get("num_depth_levels", 5)
        step.z_step_down = data.get("z_step_down", 0.0)
        step.angle_increment = data.get("angle_increment", 0.0)
        dither_val = data.get("dither_algorithm")
        if dither_val is not None:
            step.dither_algorithm = DitherAlgorithm(dither_val)
        return step

    def prepare(
        self,
        workpiece: "WorkPiece",
        settings: Dict[str, Any],
        resolved_params: Dict[str, Any],
    ) -> None:
        self._computed_auto_levels = None
        if not self.auto_levels:
            return
        self._computed_auto_levels = compute_raster_auto_levels(
            workpiece,
            settings["pixels_per_mm"],
            invert=self.invert,
        )

    def should_skip_workpiece(self, workpiece: "WorkPiece") -> bool:
        fills = workpiece.fills
        return fills is not None and len(fills) == 0

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
        assert pixels_per_mm is not None
        assert surface is not None

        part = build_part_raster(workpiece, pixels_per_mm)
        rp = self.get_assembler_kwargs(machine_defaults, workpiece)

        width_px = surface.get_width()
        height_px = surface.get_height()

        depth_mode = DepthMode[self.depth_mode]

        if width_px == 0 or height_px == 0:
            final_ops = Ops()
            final_ops.ops_section_start(
                self.SECTION_TYPE, workpiece.uid,
                raster_mode=depth_mode.raster_mode,
            )
            final_ops.ops_section_end(
                self.SECTION_TYPE, raster_mode=depth_mode.raster_mode,
            )
            return make_artifact(
                final_ops,
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(0, 0),
            )

        image, alpha = preprocess_raster_image(
            surface,
            mode=depth_mode,
            invert=self.invert,
            auto_levels=self.auto_levels,
            computed_auto_levels=computed_auto_levels,
            black_point=self.black_point,
            white_point=self.white_point,
            threshold=self.threshold,
            dither_algorithm=self.dither_algorithm,
            laser_spot_x_mm=laser.spot_size_mm[0],
            pixels_per_mm_x=pixels_per_mm[0],
        )
        if image is None:
            return make_artifact(
                Ops(),
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(width_px, height_px),
            )
        part.image = image

        spot_y = laser.spot_size_mm[1]
        line_interval_mm = rp.get("line_interval_mm") or spot_y
        x_offset_mm = workpiece.bbox[0]
        y_off_mm = workpiece.bbox[1] + y_offset_mm
        sample_interval_mm = rp.get(
            "sample_interval_mm"
        ) or laser.spot_size_mm[0]
        step_power = machine_defaults.step_power
        alpha_arr = (
            (alpha * 255).astype(np.uint8) if alpha is not None else None
        )

        result = assembler_registry.assemble(
            self.ASSEMBLER_NAME,
            part,
            alpha=alpha_arr,
            mode=depth_mode.raygeo_name,
            line_interval_mm=line_interval_mm,
            sample_interval_mm=sample_interval_mm,
            min_power=rp.get("min_power", 0),
            max_power=rp.get("max_power", 100),
            step_power=step_power,
            num_power_levels=rp.get("num_power_levels", 256),
            angle=self.scan_angle,
            offset_x_mm=x_offset_mm,
            offset_y_mm=y_off_mm,
            scan_mode=rp.get("scan_mode", "segmented").lower(),
            cross_hatch=rp.get("cross_hatch", False),
            num_depth_levels=rp.get("num_depth_levels", 1),
            z_step_down=rp.get("z_step_down", 0.0),
            angle_increment=rp.get("angle_increment", 0),
        )

        final_ops = result.ops
        if final_ops.len() > 2:
            head_ops = Ops()
            head_ops.set_head(laser.uid)
            head_ops.extend(final_ops)
            final_ops = head_ops

        return make_artifact(
            final_ops,
            workpiece,
            generation_id,
            is_vector=self.IS_VECTOR,
            source_dimensions=(width_px, height_px),
        )

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        OverscanTransformer = transformer_registry.get("OverscanTransformer")
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert OverscanTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            OverscanTransformer(
                enabled=True, distance_mm=0, auto=True
            ).to_dict(),
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
    ) -> "EngraveStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()

        OverscanTransformer = cast(
            "OverscanTransformerType",
            transformer_registry.get("OverscanTransformer"),
        )
        assert OverscanTransformer is not None
        auto_distance = OverscanTransformer.calculate_auto_distance(
            machine.max_cut_speed, machine.acceleration
        )
        for t in per_wp:
            if t.get("name") == "OverscanTransformer":
                t["distance_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = default_head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)
        return step
