import cairo
import numpy as np
import math
import logging
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any
from ..coord import CoordinateSystem
from ..artifact import WorkPieceArtifact, VertexData, TextureData
from .base import OpsProducer
from ...core.ops import (
    Ops,
    SectionType,
)

if TYPE_CHECKING:
    from ...machine.models.laser import Laser
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.proxy import BaseExecutionContext

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
    ):
        self.scan_angle = scan_angle
        self.bidirectional = bidirectional
        self.depth_mode = depth_mode
        self.speed = speed
        self.min_power = min_power
        self.max_power = max_power
        self.num_depth_levels = num_depth_levels
        self.z_step_down = z_step_down

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        proxy: Optional["BaseExecutionContext"] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("DepthEngraver requires a workpiece context.")
        if surface.get_format() != cairo.FORMAT_ARGB32:
            raise ValueError("Unsupported Cairo surface format")

        final_ops = Ops()
        # Always start with the section marker
        final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)

        width_px = surface.get_width()
        height_px = surface.get_height()

        # Initialize power texture data with zeros (transparent)
        power_texture_data = np.zeros((height_px, width_px), dtype=np.uint8)

        # The line interval is now derived from the laser's spot size
        line_interval_mm = laser.spot_size_mm[1]

        if width_px > 0 and height_px > 0:
            stride = surface.get_stride()
            buf = surface.get_data()
            data_with_padding = np.ndarray(
                shape=(height_px, stride // 4, 4), dtype=np.uint8, buffer=buf
            )
            data = data_with_padding[:, :width_px, :]

            alpha = data[:, :, 3].astype(np.float32) / 255.0

            r = data[:, :, 2].astype(np.float32)
            g = data[:, :, 1].astype(np.float32)
            b = data[:, :, 0].astype(np.float32)

            alpha_safe = np.maximum(alpha, 1e-6)

            r_unpremult = r / alpha_safe
            g_unpremult = g / alpha_safe
            b_unpremult = b / alpha_safe

            r_unpremult = np.clip(r_unpremult, 0, 255)
            g_unpremult = np.clip(g_unpremult, 0, 255)
            b_unpremult = np.clip(b_unpremult, 0, 255)

            r_blended = 255.0 - (255.0 - r_unpremult) * alpha
            g_blended = 255.0 - (255.0 - g_unpremult) * alpha
            b_blended = 255.0 - (255.0 - b_unpremult) * alpha

            gray_image = (
                0.2989 * r_blended + 0.5870 * g_blended + 0.1140 * b_blended
            ).astype(np.uint8)

            if self.depth_mode == DepthMode.POWER_MODULATION:
                (
                    mode_ops,
                    resampled_power,
                    scan_y_px,
                ) = self._run_power_modulation(
                    gray_image.astype(np.uint8),
                    pixels_per_mm,
                    y_offset_mm,
                    line_interval_mm,
                )
                # Paint the resampled power data into the texture at the
                # correct scan lines to ensure perfect alignment.
                for i, y_px_float in enumerate(scan_y_px):
                    y_px_int = int(round(y_px_float))
                    if 0 <= y_px_int < height_px:
                        power_texture_data[y_px_int, :] = resampled_power[i, :]

            else:  # Multi-Pass mode
                if not np.isclose(self.scan_angle, 0):
                    logger.warning(
                        "Angled scanning is not supported for Multi-Pass "
                        "depth engraving. Defaulting to horizontal "
                        "(0 degrees)."
                    )
                mode_ops = self._run_multi_pass(
                    gray_image.astype(np.uint8),
                    pixels_per_mm,
                    y_offset_mm,
                    line_interval_mm,
                )
                # For multi-pass, the texture represents the final depth map.
                pass_map = np.ceil(
                    ((255 - gray_image) / 255.0) * self.num_depth_levels
                ).astype(int)

                # Power is proportional to the number of passes at a pixel.
                num_levels = self.num_depth_levels
                if num_levels > 0:
                    power_fractions = pass_map / float(num_levels)
                    power_texture_data = (power_fractions * 255).astype(
                        np.uint8
                    )
                else:
                    # If no levels, texture is all zeros.
                    power_texture_data = np.zeros_like(
                        pass_map, dtype=np.uint8
                    )

            if not mode_ops.is_empty():
                final_ops.set_laser(laser.uid)
                final_ops.extend(mode_ops)

        # Always close with the section end marker
        final_ops.ops_section_end(SectionType.RASTER_FILL)

        # Calculate dimensions and position in millimeters
        px_per_mm_x, px_per_mm_y = pixels_per_mm
        dimensions_mm = (width_px / px_per_mm_x, height_px / px_per_mm_y)
        position_mm = (0.0, y_offset_mm)

        vertex_data = VertexData()
        texture_data = TextureData(
            power_texture_data=power_texture_data,
            dimensions_mm=dimensions_mm,
            position_mm=position_mm,
        )

        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(width_px, height_px),
            generation_size=workpiece.size,
            vertex_data=vertex_data,
            texture_data=texture_data,
        )

    def _run_power_modulation(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: Tuple[float, float],
        y_offset_mm: float,
        line_interval_mm: float,
    ) -> Tuple[Ops, np.ndarray, np.ndarray]:
        ops = Ops()
        height_px, width_px = gray_image.shape
        px_per_mm_x, px_per_mm_y = pixels_per_mm
        height_mm = height_px / px_per_mm_y

        occupied_rows = np.any(gray_image < 255, axis=1)
        if not np.any(occupied_rows):
            return ops, np.array([]), np.array([])

        y_min_px, y_max_px = np.where(occupied_rows)[0][[0, -1]]

        y_min_mm = y_min_px / px_per_mm_y
        y_max_mm = (y_max_px + 1) / px_per_mm_y

        global_y_min_mm = y_offset_mm + y_min_mm
        num_intervals = math.ceil(global_y_min_mm / line_interval_mm)
        first_scan_y_mm_global = num_intervals * line_interval_mm
        first_scan_y_mm_local = first_scan_y_mm_global - y_offset_mm

        scan_y_coords_mm = np.arange(
            first_scan_y_mm_local, y_max_mm, line_interval_mm
        )
        if len(scan_y_coords_mm) == 0:
            return ops, np.array([]), np.array([])

        scan_y_coords_px = scan_y_coords_mm * px_per_mm_y
        scan_y_coords_px = np.clip(scan_y_coords_px, 0, height_px - 1)

        y0 = np.floor(scan_y_coords_px).astype(int)
        y1 = np.ceil(scan_y_coords_px).astype(int)
        y_frac = scan_y_coords_px - y0

        row0_values = gray_image[y0, :]
        row1_values = gray_image[y1, :]

        resampled_gray = (
            row0_values * (1 - y_frac[:, np.newaxis])
            + row1_values * y_frac[:, np.newaxis]
        )

        # Interpolate grayscale value into the modulation range. The resulting
        # power fractions are relative to the step's master power setting,
        # which is applied by a preceding SetPowerCommand.
        power_range = self.max_power - self.min_power
        power_fractions = (
            self.min_power + (1.0 - resampled_gray / 255.0) * power_range
        )

        # Convert power fractions (0.0-1.0) to bytes (0-255) for the command
        power_image = (power_fractions * 255).astype(np.uint8)

        is_reversed = False
        y_pixel_center_offset_mm = 0.5 / px_per_mm_y

        for i, y_mm in enumerate(scan_y_coords_mm):
            row_power_values = power_image[i, :]

            if np.any(row_power_values > 0):
                # Find contiguous segments of "on" pixels and create a separate
                # ScanLine for each, avoiding long scanlines over empty space.
                is_on = row_power_values > 0
                padded = np.concatenate(([False], is_on, [False]))
                diffs = np.diff(padded.astype(int))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]

                line_segments = list(zip(starts, ends))
                if self.bidirectional and is_reversed:
                    line_segments.reverse()

                line_y_mm = y_mm + y_pixel_center_offset_mm
                final_y_mm = float(height_mm - line_y_mm)

                for start_idx, end_idx in line_segments:
                    power_slice = row_power_values[start_idx:end_idx]

                    start_x = start_idx / px_per_mm_x
                    end_x = end_idx / px_per_mm_x

                    start_pt = (start_x, final_y_mm, 0.0)
                    end_pt = (end_x, final_y_mm, 0.0)

                    if self.bidirectional and is_reversed:
                        ops.move_to(*end_pt)
                        ops.scan_to(
                            start_pt[0],
                            start_pt[1],
                            start_pt[2],
                            bytearray(power_slice[::-1]),
                        )
                    else:
                        ops.move_to(*start_pt)
                        ops.scan_to(
                            end_pt[0],
                            end_pt[1],
                            end_pt[2],
                            bytearray(power_slice),
                        )
                if self.bidirectional:
                    is_reversed = not is_reversed

        if not np.isclose(self.scan_angle, 0.0):
            center_x = (width_px / px_per_mm_x) / 2
            center_y = height_mm / 2
            ops.rotate(self.scan_angle, center_x, center_y)

        return ops, power_image, scan_y_coords_px

    def _run_multi_pass(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: Tuple[float, float],
        y_offset_mm: float,
        line_interval_mm: float,
    ) -> Ops:
        ops = Ops()
        height_px = gray_image.shape[0]
        height_mm = height_px / pixels_per_mm[1]

        pass_map = np.ceil(
            ((255 - gray_image) / 255.0) * self.num_depth_levels
        ).astype(int)

        for pass_level in range(1, self.num_depth_levels + 1):
            mask = (pass_map >= pass_level).astype(np.uint8)
            if not np.any(mask):
                continue

            z_offset = -((pass_level - 1) * self.z_step_down)
            pass_ops = self._rasterize_mask_horizontally(
                mask,
                pixels_per_mm,
                height_mm,
                y_offset_mm,
                z_offset,
                line_interval_mm,
            )
            ops.extend(pass_ops)
        return ops

    def _rasterize_mask_horizontally(
        self,
        mask: np.ndarray,
        pixels_per_mm: Tuple[float, float],
        height_mm: float,
        y_offset_mm: float,
        z: float,
        line_interval_mm: float,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = mask.shape
        px_per_mm_x, px_per_mm_y = pixels_per_mm
        is_reversed = False

        # Find the bounding box of the occupied area
        occupied_rows = np.any(mask, axis=1)
        occupied_cols = np.any(mask, axis=0)

        if not np.any(occupied_rows) or not np.any(occupied_cols):
            return ops  # No occupied area, return an empty path

        y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
        x_min, x_max = np.where(occupied_cols)[0][[0, -1]]

        # Calculate dimensions in millimeters
        y_min_mm = y_min / px_per_mm_y

        # Align to global grid
        global_y_min_mm = y_offset_mm + y_min_mm
        num_intervals = math.ceil(global_y_min_mm / line_interval_mm)
        first_global_y_mm = num_intervals * line_interval_mm
        y_start_mm = first_global_y_mm - y_offset_mm

        # Correction for vertical alignment: center the raster line in
        # the pixel.
        y_pixel_center_offset_mm = 0.5 / px_per_mm_y

        # The content ends at the bottom edge of the last occupied pixel row
        # (y_max).
        # The loop should include any raster line that starts before this edge.
        y_extent_mm = (y_max + 1) / px_per_mm_y

        # Iterate over rows in millimeters (floating-point)
        y_step_mm = line_interval_mm
        for y_mm in np.arange(y_start_mm, y_extent_mm, y_step_mm):
            # Convert y_mm to pixel coordinates (floating-point)
            y_px = int(round(y_mm * px_per_mm_y))
            if y_px >= height_px:  # Ensure we don't go out of bounds
                continue

            row = mask[y_px, x_min : x_max + 1]

            if np.any(row):
                diff = np.diff(np.hstack(([0], row, [0])))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                if self.bidirectional and is_reversed:
                    starts, ends = starts[::-1], ends[::-1]

                line_y_mm = y_mm + y_pixel_center_offset_mm
                final_y_mm = float(height_mm - line_y_mm)

                for start_px, end_px in zip(starts, ends):
                    content_start_mm_x = (x_min + start_px + 0.5) / px_per_mm_x
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
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepthEngraver":
        """
        Deserializes a dictionary into a DepthEngraver instance.
        """
        params_in = data.get("params", {})

        # Create a new dictionary with defaults, then update with loaded data
        init_args = {
            "scan_angle": 0.0,
            "bidirectional": True,
            "speed": 3000.0,
            "min_power": 0.0,
            "max_power": 100.0,
            "num_depth_levels": 5,
            "z_step_down": 0.0,
        }
        init_args.update(params_in)

        # Handle the enum conversion
        depth_mode_str = init_args.get(
            "depth_mode", DepthMode.POWER_MODULATION.name
        )
        try:
            init_args["depth_mode"] = DepthMode[depth_mode_str]
        except KeyError:
            init_args["depth_mode"] = DepthMode.POWER_MODULATION

        return cls(**init_args)
