import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import cairo
import numpy as np
from raygeo.image.scan import ScanMode
from raygeo.ops import Ops
from raygeo.ops.assembly import AssemblyResult
from raygeo.ops.types import SectionType

from rayforge.image.dither import DitherAlgorithm
from rayforge.pipeline.assembler.registry import assembler_registry
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    build_part_raster,
    compute_raster_auto_levels,
    make_artifact,
    preprocess_raster_image,
    wrap_assembler_result,
)
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser

logger = logging.getLogger(__name__)


class Rasterizer(OpsProducer):
    """
    Generates raster engraving paths from a grayscale surface.
    Supports multiple modes: power modulation, constant power, dithering,
    and multi-pass engraving with Z stepping.
    """

    def __init__(
        self,
        scan_angle: float = 0.0,
        depth_mode: DepthMode = DepthMode.POWER_MODULATION,
        scan_mode: ScanMode = ScanMode.SEGMENTED,
        threshold: int = 128,
        dither_algorithm: Optional[
            DitherAlgorithm
        ] = DitherAlgorithm.FLOYD_STEINBERG,
        cross_hatch: bool = False,
        speed: float = 3000.0,
        min_power: float = 0.0,
        max_power: float = 1.0,
        num_depth_levels: int = 5,
        num_power_levels: int = 25,
        z_step_down: float = 0.0,
        invert: bool = False,
        line_interval_mm: Optional[float] = None,
        sample_interval_mm: Optional[float] = None,
        black_point: int = 0,
        white_point: int = 255,
        auto_levels: bool = True,
        angle_increment: float = 0.0,
    ):
        self.scan_angle = scan_angle
        self.depth_mode = depth_mode
        self.scan_mode = scan_mode
        self.threshold = threshold
        self.dither_algorithm = dither_algorithm
        self.cross_hatch = cross_hatch
        self.speed = speed
        self.min_power = min_power
        self.max_power = max_power
        self.num_depth_levels = num_depth_levels
        self.num_power_levels = num_power_levels
        self.z_step_down = z_step_down
        self.invert = invert
        self.line_interval_mm = line_interval_mm
        self.sample_interval_mm = sample_interval_mm
        self.black_point = black_point
        self.white_point = white_point
        self.auto_levels = auto_levels
        self.angle_increment = angle_increment
        self._computed_auto_levels: Optional[Tuple[int, int]] = None

    def prepare(
        self,
        workpiece: "WorkPiece",
        settings: Dict[str, Any],
    ) -> None:
        """
        Compute global auto levels from a low-resolution preview.

        This ensures consistent black/white points across all chunks
        when processing large images in chunks.
        """
        self._computed_auto_levels = None

        if not self.auto_levels:
            return

        self._computed_auto_levels = compute_raster_auto_levels(
            workpiece,
            settings["pixels_per_mm"],
            invert=self.invert,
        )

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        generation_id: int,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("Rasterizer requires a workpiece context.")
        if surface.get_format() != cairo.FORMAT_ARGB32:
            raise ValueError("Unsupported Cairo surface format")

        width_px = surface.get_width()
        height_px = surface.get_height()

        if width_px == 0 or height_px == 0:
            final_ops = Ops()
            final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)
            final_ops.ops_section_end(SectionType.RASTER_FILL)
            return make_artifact(
                final_ops,
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(0, 0),
            )

        line_interval_mm = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else laser.spot_size_mm[1]
        )
        x_offset_mm = workpiece.bbox[0]
        y_offset_mm = workpiece.bbox[1] + y_offset_mm

        image, alpha = preprocess_raster_image(
            surface,
            mode=self.depth_mode,
            invert=self.invert,
            auto_levels=self.auto_levels,
            computed_auto_levels=self._computed_auto_levels,
            black_point=self.black_point,
            white_point=self.white_point,
            threshold=self.threshold,
            dither_algorithm=self.dither_algorithm,
            laser_spot_x_mm=laser.spot_size_mm[0],
            pixels_per_mm_x=pixels_per_mm[0],
        )

        if image is None:
            return wrap_assembler_result(
                AssemblyResult(),
                workpiece,
                laser,
                generation_id,
                section_type=SectionType.RASTER_FILL,
                is_vector=False,
                source_dimensions=(width_px, height_px),
            )

        step_power = settings.get("power", 1.0) if settings else 1.0
        sample_interval_mm = (
            self.sample_interval_mm
            if self.sample_interval_mm is not None
            else laser.spot_size_mm[0]
        )

        part = build_part_raster(workpiece, pixels_per_mm)
        part.image = image

        result = assembler_registry.assemble(
            "raster",
            part,
            alpha=(
                (alpha * 255).astype(np.uint8) if alpha is not None else None
            ),
            mode=self.depth_mode.raygeo_name,
            line_interval_mm=line_interval_mm,
            sample_interval_mm=sample_interval_mm,
            min_power=self.min_power,
            max_power=self.max_power,
            step_power=step_power,
            num_power_levels=self.num_power_levels,
            angle=self.scan_angle,
            offset_x_mm=x_offset_mm,
            offset_y_mm=y_offset_mm,
            scan_mode=self.scan_mode.name.lower(),
            cross_hatch=self.cross_hatch,
            num_depth_levels=self.num_depth_levels,
            z_step_down=self.z_step_down,
            angle_increment=self.angle_increment,
        )

        return wrap_assembler_result(
            result,
            workpiece,
            laser,
            generation_id,
            section_type=SectionType.RASTER_FILL,
            is_vector=False,
            source_dimensions=(width_px, height_px),
            always_wrap=True,
        )

    def is_vector_producer(self) -> bool:
        return False

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {
                "scan_angle": self.scan_angle,
                "depth_mode": self.depth_mode.name,
                "scan_mode": self.scan_mode.name,
                "threshold": self.threshold,
                "dither_algorithm": (
                    self.dither_algorithm.value
                    if self.dither_algorithm
                    else None
                ),
                "cross_hatch": self.cross_hatch,
                "speed": self.speed,
                "min_power": self.min_power,
                "max_power": self.max_power,
                "num_depth_levels": self.num_depth_levels,
                "num_power_levels": self.num_power_levels,
                "z_step_down": self.z_step_down,
                "invert": self.invert,
                "line_interval_mm": self.line_interval_mm,
                "sample_interval_mm": self.sample_interval_mm,
                "black_point": self.black_point,
                "white_point": self.white_point,
                "auto_levels": self.auto_levels,
                "angle_increment": self.angle_increment,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rasterizer":
        """
        Deserializes a dictionary into a Rasterizer instance.
        Handles backward compatibility for legacy type names and params.
        """
        old_type = data.get("type")
        params_in = dict(data.get("params", {}))

        if old_type == "Rasterizer" and "depth_mode" not in params_in:
            params_in["depth_mode"] = "CONSTANT_POWER"
            if "direction_degrees" in params_in:
                params_in["scan_angle"] = params_in.pop("direction_degrees")
        elif old_type == "DitherRasterizer":
            params_in["depth_mode"] = "DITHER"

        valid_params = {
            "scan_angle",
            "depth_mode",
            "scan_mode",
            "threshold",
            "dither_algorithm",
            "cross_hatch",
            "speed",
            "min_power",
            "max_power",
            "num_depth_levels",
            "num_power_levels",
            "z_step_down",
            "invert",
            "line_interval_mm",
            "sample_interval_mm",
            "black_point",
            "white_point",
            "auto_levels",
            "angle_increment",
        }

        init_args = {k: v for k, v in params_in.items() if k in valid_params}

        if "num_depth_levels" in init_args:
            init_args["num_depth_levels"] = int(init_args["num_depth_levels"])

        if "num_power_levels" in init_args:
            init_args["num_power_levels"] = int(init_args["num_power_levels"])

        depth_mode_str = init_args.get(
            "depth_mode", DepthMode.POWER_MODULATION.name
        )
        try:
            init_args["depth_mode"] = DepthMode[depth_mode_str]
        except KeyError:
            init_args["depth_mode"] = DepthMode.POWER_MODULATION

        scan_mode_str = init_args.pop("scan_mode", ScanMode.SEGMENTED.name)
        scan_mode_map = {
            "SEGMENTED": ScanMode.SEGMENTED,
            "FULL_SWEEP": ScanMode.FULL_SWEEP,
            "Segmented": ScanMode.SEGMENTED,
            "FullSweep": ScanMode.FULL_SWEEP,
        }
        init_args["scan_mode"] = scan_mode_map.get(
            scan_mode_str, ScanMode.SEGMENTED
        )

        dither_algorithm_str = init_args.get("dither_algorithm")
        if dither_algorithm_str is not None:
            try:
                init_args["dither_algorithm"] = DitherAlgorithm(
                    dither_algorithm_str
                )
            except ValueError:
                init_args["dither_algorithm"] = DitherAlgorithm.FLOYD_STEINBERG
        else:
            init_args["dither_algorithm"] = None

        return cls(**init_args)


DepthEngraver = Rasterizer
DitherRasterizer = Rasterizer
