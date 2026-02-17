import cairo
import numpy as np
import logging
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...core.ops import Ops, SectionType
from ...image.image_util import surface_to_grayscale
from ...shared.tasker.progress import ProgressContext
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer
from .raster_util import (
    find_segments,
    convert_y_to_output,
    calculate_ymax_mm,
    find_bounding_box,
    find_mask_bounding_box,
    generate_horizontal_scan_positions,
    resample_rows,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


class DepthMode(Enum):
    POWER_MODULATION = auto()
    MULTI_PASS = auto()


class DepthEngraver(OpsProducer):
    """
    Generates depth-engraving paths from a grayscale surface.
    """

    def __init__(
        self,
        scan_angle: float = 0.0,
        bidirectional: bool = True,
        depth_mode: DepthMode = DepthMode.POWER_MODULATION,
        speed: float = 3000.0,
        min_power: float = 0.0,
        max_power: float = 1.0,
        num_depth_levels: int = 5,
        z_step_down: float = 0.0,
        invert: bool = False,
        line_interval_mm: Optional[float] = None,
    ):
        self.scan_angle = scan_angle
        self.bidirectional = bidirectional
        self.depth_mode = depth_mode
        self.speed = speed
        self.min_power = min_power
        self.max_power = max_power
        self.num_depth_levels = num_depth_levels
        self.z_step_down = z_step_down
        self.invert = invert
        self.line_interval_mm = line_interval_mm

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("DepthEngraver requires a workpiece context.")
        if surface.get_format() != cairo.FORMAT_ARGB32:
            raise ValueError("Unsupported Cairo surface format")

        final_ops = Ops()
        final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)

        width_px = surface.get_width()
        height_px = surface.get_height()

        line_interval_mm = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else laser.spot_size_mm[1]
        )
        x_offset_mm = workpiece.bbox[0]
        y_offset_mm = workpiece.bbox[1] + y_offset_mm

        if width_px > 0 and height_px > 0:
            gray_image, alpha = surface_to_grayscale(surface)

            if self.invert:
                alpha_mask = alpha > 0
                gray_image[alpha_mask] = 255 - gray_image[alpha_mask]

            if self.depth_mode == DepthMode.POWER_MODULATION:
                step_power = settings.get("power", 1.0) if settings else 1.0
                mode_ops = self._run_power_modulation(
                    gray_image,
                    pixels_per_mm,
                    x_offset_mm,
                    y_offset_mm,
                    line_interval_mm,
                    step_power,
                )
            else:
                mode_ops = self._run_multi_pass(
                    gray_image,
                    pixels_per_mm,
                    x_offset_mm,
                    y_offset_mm,
                    line_interval_mm,
                )

            if not mode_ops.is_empty():
                final_ops.set_laser(laser.uid)
                final_ops.extend(mode_ops)

        final_ops.ops_section_end(SectionType.RASTER_FILL)

        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(width_px, height_px),
            generation_size=workpiece.size,
        )

    def _run_power_modulation(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
        step_power: float = 1.0,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = gray_image.shape
        ymax_mm = calculate_ymax_mm((width_px, height_px), pixels_per_mm)
        px_per_mm_x, px_per_mm_y = pixels_per_mm

        bbox = find_bounding_box(gray_image)
        if bbox is None:
            return ops

        y_min_px, y_max_px = bbox[0], bbox[1]
        y_coords_mm, y_coords_px = generate_horizontal_scan_positions(
            y_min_px,
            y_max_px,
            height_px,
            pixels_per_mm,
            line_interval_mm,
            offset_y_mm,
        )

        if len(y_coords_mm) == 0:
            return ops

        resampled_gray = resample_rows(gray_image, y_coords_px)

        power_range = self.max_power - self.min_power
        power_fractions = (
            self.min_power + (1.0 - resampled_gray / 255.0) * power_range
        )
        power_fractions = power_fractions * step_power
        power_image = (power_fractions * 255).astype(np.uint8)

        is_reversed = False
        y_pixel_center_offset_mm = 0.5 / px_per_mm_y

        for i, y_mm in enumerate(y_coords_mm):
            row_power_values = power_image[i, :]

            if np.any(row_power_values > 0):
                segments = find_segments(row_power_values)

                if self.bidirectional and is_reversed:
                    segments = segments[::-1]

                line_y_mm = y_mm + y_pixel_center_offset_mm
                final_y_mm = float(convert_y_to_output(line_y_mm, ymax_mm))

                for start_idx, end_idx in segments:
                    power_slice = row_power_values[start_idx:end_idx]

                    start_x = start_idx / px_per_mm_x
                    end_x = end_idx / px_per_mm_x

                    if self.bidirectional and is_reversed:
                        ops.move_to(end_x, final_y_mm, 0.0)
                        ops.scan_to(
                            start_x,
                            final_y_mm,
                            0.0,
                            bytearray(power_slice[::-1]),
                        )
                    else:
                        ops.move_to(start_x, final_y_mm, 0.0)
                        ops.scan_to(
                            end_x,
                            final_y_mm,
                            0.0,
                            bytearray(power_slice),
                        )

                if self.bidirectional:
                    is_reversed = not is_reversed

        if not np.isclose(self.scan_angle, 0.0):
            height_mm = height_px / px_per_mm_y
            center_x = (width_px / px_per_mm_x) / 2
            center_y = height_mm / 2
            ops.rotate(self.scan_angle, center_x, center_y)

        return ops

    def _run_multi_pass(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
    ) -> Ops:
        ops = Ops()

        pass_map = np.ceil(
            ((255 - gray_image) / 255.0) * self.num_depth_levels
        ).astype(int)

        for pass_level in range(1, self.num_depth_levels + 1):
            mask = (pass_map >= pass_level).astype(np.uint8)
            if not np.any(mask):
                continue

            z_offset = -((pass_level - 1) * self.z_step_down)
            pass_ops = self._rasterize_mask(
                mask,
                pixels_per_mm,
                offset_x_mm,
                offset_y_mm,
                line_interval_mm,
                z_offset,
            )
            ops.extend(pass_ops)

        return ops

    def _rasterize_mask(
        self,
        mask: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
        z: float,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = mask.shape
        ymax_mm = calculate_ymax_mm((width_px, height_px), pixels_per_mm)
        px_per_mm_x, px_per_mm_y = pixels_per_mm

        bbox = find_mask_bounding_box(mask)
        if bbox is None:
            return ops

        y_min_px, y_max_px, x_min, x_max = bbox
        y_coords_mm, y_coords_px = generate_horizontal_scan_positions(
            y_min_px,
            y_max_px,
            height_px,
            pixels_per_mm,
            line_interval_mm,
            offset_y_mm,
        )

        if len(y_coords_mm) == 0:
            return ops

        resampled_mask = resample_rows(mask, y_coords_px)
        resampled_mask = (resampled_mask > 0.5).astype(np.uint8)

        is_reversed = False
        y_pixel_center_offset_mm = 0.5 / px_per_mm_y

        for i, y_mm in enumerate(y_coords_mm):
            row = resampled_mask[i, x_min : x_max + 1]

            if not np.any(row):
                continue

            segments = find_segments(row)

            if self.bidirectional and is_reversed:
                segments = segments[::-1]

            line_y_mm = y_mm + y_pixel_center_offset_mm
            final_y_mm = float(convert_y_to_output(line_y_mm, ymax_mm))

            for start_px, end_px in segments:
                content_start_mm_x = (x_min + start_px) / px_per_mm_x
                content_end_mm_x = (x_min + end_px - 1 + 0.5) / px_per_mm_x

                if self.bidirectional and is_reversed:
                    ops.move_to(content_end_mm_x, final_y_mm, z)
                    ops.line_to(content_start_mm_x, final_y_mm, z)
                else:
                    ops.move_to(content_start_mm_x, final_y_mm, z)
                    ops.line_to(content_end_mm_x, final_y_mm, z)

            if self.bidirectional:
                is_reversed = not is_reversed

        return ops

    def is_vector_producer(self) -> bool:
        return False

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {
                "scan_angle": self.scan_angle,
                "bidirectional": self.bidirectional,
                "depth_mode": self.depth_mode.name,
                "speed": self.speed,
                "min_power": self.min_power,
                "max_power": self.max_power,
                "num_depth_levels": self.num_depth_levels,
                "z_step_down": self.z_step_down,
                "invert": self.invert,
                "line_interval_mm": self.line_interval_mm,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepthEngraver":
        """
        Deserializes a dictionary into a DepthEngraver instance.
        """
        params_in = data.get("params", {})

        init_args = {
            "scan_angle": 0.0,
            "bidirectional": True,
            "speed": 3000.0,
            "min_power": 0.0,
            "max_power": 100.0,
            "num_depth_levels": 5,
            "z_step_down": 0.0,
            "invert": False,
            "line_interval_mm": None,
        }
        init_args.update(params_in)

        depth_mode_str = init_args.get(
            "depth_mode", DepthMode.POWER_MODULATION.name
        )
        try:
            init_args["depth_mode"] = DepthMode[depth_mode_str]
        except KeyError:
            init_args["depth_mode"] = DepthMode.POWER_MODULATION

        return cls(**init_args)
