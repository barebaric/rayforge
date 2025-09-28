import cairo
import numpy as np
import math
import logging
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    ScanLinePowerCommand,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece

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
        line_interval: float = 0.1,
        bidirectional: bool = True,
        overscan: float = 2.0,
        depth_mode: DepthMode = DepthMode.POWER_MODULATION,
        speed: float = 3000.0,
        min_power: float = 0.0,
        max_power: float = 100.0,
        power: float = 100.0,
        num_depth_levels: int = 5,
        z_step_down: float = 0.0,
    ):
        self.scan_angle = scan_angle
        self.line_interval = line_interval
        self.bidirectional = bidirectional
        self.overscan = overscan
        self.depth_mode = depth_mode
        self.speed = speed
        self.min_power = min_power
        self.max_power = max_power
        self.power = power
        self.num_depth_levels = num_depth_levels
        self.z_step_down = z_step_down

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError("DepthEngraver requires a workpiece context.")
        if surface.get_format() != cairo.FORMAT_ARGB32:
            raise ValueError("Unsupported Cairo surface format")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.RASTER_FILL, workpiece.uid)
        )

        width_px = surface.get_width()
        height_px = surface.get_height()
        if width_px == 0 or height_px == 0:
            final_ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
            return PipelineArtifact(
                ops=final_ops,
                is_scalable=False,
                source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            )

        stride = surface.get_stride()
        buf = surface.get_data()
        data_with_padding = np.ndarray(
            shape=(height_px, stride // 4, 4), dtype=np.uint8, buffer=buf
        )
        data = data_with_padding[:, :width_px, :]

        alpha = data[:, :, 3]
        gray_image = (
            0.2989 * data[:, :, 2]
            + 0.5870 * data[:, :, 1]
            + 0.1140 * data[:, :, 0]
        )
        gray_image[alpha == 0] = 255

        if self.depth_mode == DepthMode.POWER_MODULATION:
            mode_ops = self._run_power_modulation(
                gray_image.astype(np.uint8), pixels_per_mm, y_offset_mm
            )
        else:
            if not np.isclose(self.scan_angle, 0):
                logger.warning(
                    "Angled scanning is not supported for Multi-Pass "
                    "depth engraving. Defaulting to horizontal (0 degrees)."
                )
            mode_ops = self._run_multi_pass(
                gray_image.astype(np.uint8), pixels_per_mm, y_offset_mm
            )

        final_ops.extend(mode_ops)
        final_ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
        return PipelineArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )

    def _run_power_modulation(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: Tuple[float, float],
        y_offset_mm: float,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = gray_image.shape
        px_per_mm_x, px_per_mm_y = pixels_per_mm
        height_mm = height_px / px_per_mm_y

        # Convert grayscale to power values (0-255)
        power_image = ((1.0 - gray_image / 255.0) * self.max_power).astype(
            np.uint8
        )
        power_image = np.clip(power_image, self.min_power, self.max_power)

        is_reversed = False

        occupied_rows = np.any(power_image > 0, axis=1)
        if not np.any(occupied_rows):
            return ops

        y_min, y_max = np.where(occupied_rows)[0][[0, -1]]

        y_min_mm = y_min / px_per_mm_y
        global_y_min_mm = y_offset_mm + y_min_mm
        num_intervals = math.ceil(global_y_min_mm / self.line_interval)
        first_global_y_mm = num_intervals * self.line_interval
        y_start_mm = first_global_y_mm - y_offset_mm
        y_pixel_center_offset_mm = 0.5 / px_per_mm_y
        y_extent_mm = (y_max + 1) / px_per_mm_y

        y_step_mm = self.line_interval
        for y_mm in np.arange(y_start_mm, y_extent_mm, y_step_mm):
            y_px = int(round(y_mm * px_per_mm_y))
            if y_px >= height_px:
                continue

            row_power_values = power_image[y_px, :]
            if not np.any(row_power_values > 0):
                is_reversed = not is_reversed if self.bidirectional else False
                continue

            line_y_mm = y_mm + y_pixel_center_offset_mm
            final_y_mm = float(height_mm - line_y_mm)

            # Define the geometry for the FULL scan line
            start_full_mm_x = (0 + 0.5) / px_per_mm_x
            end_full_mm_x = (width_px - 1 + 0.5) / px_per_mm_x

            start_pt = (start_full_mm_x, final_y_mm, 0.0)
            end_pt = (end_full_mm_x, final_y_mm, 0.0)

            if self.bidirectional and is_reversed:
                start_pt, end_pt = end_pt, start_pt
                final_power_values = row_power_values[::-1]
            else:
                final_power_values = row_power_values

            # A scanline is a discrete operation; always move to the start.
            ops.move_to(*start_pt)
            ops.add(
                ScanLinePowerCommand(
                    end_pt,
                    bytearray(final_power_values.astype(np.uint8)),
                )
            )

            is_reversed = not is_reversed

        if not np.isclose(self.scan_angle, 0.0):
            center_x = (width_px / px_per_mm_x) / 2
            center_y = height_mm / 2
            ops.rotate(self.scan_angle, center_x, center_y)

        return ops

    def _run_multi_pass(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: Tuple[float, float],
        y_offset_mm: float,
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
                mask, pixels_per_mm, height_mm, y_offset_mm, z_offset
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
        num_intervals = math.ceil(global_y_min_mm / self.line_interval)
        first_global_y_mm = num_intervals * self.line_interval
        y_start_mm = first_global_y_mm - y_offset_mm

        # Correction for vertical alignment: center the raster line in
        # the pixel.
        y_pixel_center_offset_mm = 0.5 / px_per_mm_y

        # The content ends at the bottom edge of the last occupied pixel row
        # (y_max).
        # The loop should include any raster line that starts before this edge.
        y_extent_mm = (y_max + 1) / px_per_mm_y

        # Iterate over rows in millimeters (floating-point)
        y_step_mm = self.line_interval
        for y_mm in np.arange(y_start_mm, y_extent_mm, y_step_mm):
            # Convert y_mm to pixel coordinates (floating-point)
            y_px = int(round(y_mm * px_per_mm_y))
            if y_px >= height_px:  # Ensure we don't go out of bounds
                continue

            row = mask[y_px, x_min : x_max + 1]

            if not np.any(row):
                is_reversed = not is_reversed if self.bidirectional else False
                continue

            diff = np.diff(np.hstack(([0], row, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            if self.bidirectional and is_reversed:
                starts, ends = starts[::-1], ends[::-1]

            # The Y coordinate for the line, adjusted to be in the
            # center of the pixel row it represents.
            line_y_mm = y_mm + y_pixel_center_offset_mm
            final_y_mm = float(height_mm - line_y_mm)

            for start_px, end_px in zip(starts, ends):
                content_start_mm_x = (x_min + start_px + 0.5) / px_per_mm_x
                content_end_mm_x = (x_min + end_px - 1 + 0.5) / px_per_mm_x

                if self.bidirectional and is_reversed:
                    # Move to the right side (with overscan) and draw left
                    ops.move_to(
                        content_end_mm_x + self.overscan, final_y_mm, z
                    )
                    ops.line_to(
                        content_start_mm_x - self.overscan, final_y_mm, z
                    )
                else:
                    # Move to the left side (with overscan) and draw right
                    ops.move_to(
                        content_start_mm_x - self.overscan, final_y_mm, z
                    )
                    ops.line_to(
                        content_end_mm_x + self.overscan, final_y_mm, z
                    )

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
                "line_interval": self.line_interval,
                "bidirectional": self.bidirectional,
                "overscan": self.overscan,
                "depth_mode": self.depth_mode.name,
                "speed": self.speed,
                "min_power": self.min_power,
                "max_power": self.max_power,
                "power": self.power,
                "num_depth_levels": self.num_depth_levels,
                "z_step_down": self.z_step_down,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepthEngraver":
        """Deserializes a dictionary into a DepthEngraver instance."""
        params = data.get("params", {}).copy()
        # Handle enum conversion during deserialization
        depth_mode_str = params.get("depth_mode", "POWER_MODULATION")
        try:
            depth_mode = DepthMode[depth_mode_str]
        except KeyError:
            depth_mode = DepthMode.POWER_MODULATION
        params["depth_mode"] = depth_mode
        return cls(**params)
