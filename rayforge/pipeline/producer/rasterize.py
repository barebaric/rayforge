import cairo
import numpy as np
import math
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...core.ops import Ops, SectionType
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...shared.tasker.proxy import BaseExecutionContext

logger = logging.getLogger(__name__)


def rasterize_horizontally(
    surface,
    ymax,
    pixels_per_mm=(10, 10),
    raster_size_mm=0.1,
    y_offset_mm=0.0,
    threshold=128,
):
    """
    Generate an engraving path for a Cairo surface, focusing on horizontal
    movement.

    Args:
        surface: A Cairo surface containing a black and white image.
        pixels_per_mm: Resolution of the image in pixels per millimeter.
        raster_size_mm: Distance between horizontal engraving lines in
                        millimeters.
        y_offset_mm: The absolute vertical offset of this surface chunk
                     from the top of the entire workpiece (in mm).
        threshold: The brightness value (0-255) to consider black.

    Returns:
        A Ops object containing the optimized engraving path.
    """
    surface_format = surface.get_format()
    if surface_format != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    # Convert surface to a NumPy array
    width = surface.get_width()
    height = surface.get_height()
    if width == 0 or height == 0:
        return Ops()
    data = np.frombuffer(surface.get_data(), dtype=np.uint8)
    data = data.reshape((height, width, 4))

    # Extract BGRA channels
    blue = data[:, :, 0]  # Blue channel
    green = data[:, :, 1]  # Green channel
    red = data[:, :, 2]  # Red channel
    alpha = data[:, :, 3]  # Alpha channel

    # Convert to grayscale (weighted average of RGB channels)
    bw_image = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    # Threshold to black and white
    bw_image = (bw_image < threshold).astype(np.uint8)

    # Optionally handle transparency (e.g., treat fully transparent
    # pixels as white)
    bw_image[alpha == 0] = 0  # Set fully transparent pixels to white (0)

    # Find the bounding box of the occupied area
    occupied_rows = np.any(bw_image, axis=1)
    occupied_cols = np.any(bw_image, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return Ops()  # No occupied area, return an empty path

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]

    # Calculate dimensions in millimeters
    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm

    # Convert bounding box to millimeters
    y_min_mm = y_min / pixels_per_mm_y

    ops = Ops()

    # If this is a chunk of a larger image, its raster lines must align with
    # a global grid. The y_offset_mm tells us where this chunk starts.
    # We calculate the first raster line's Y-position based on this global
    # grid, ensuring seamless transitions between chunks.
    global_y_min_mm = y_offset_mm + y_min_mm
    # Find the first multiple of raster_size_mm that is >= global_y_min_mm
    first_global_y_mm = (
        math.ceil(global_y_min_mm / raster_size_mm) * raster_size_mm
    )
    # Convert it back to a local coordinate for our loop
    y_start_mm = first_global_y_mm - y_offset_mm

    # Correction for vertical alignment: center the raster line in the pixel.
    y_pixel_center_offset_mm = 0.5 / pixels_per_mm_y

    # The content ends at the bottom edge of the last occupied pixel row
    # (y_max).
    # The loop should include any raster line that starts before this edge.
    y_extent_mm = (y_max + 1) / pixels_per_mm_y

    # Iterate over rows in millimeters (floating-point) within the bounding box
    for y_mm in np.arange(y_start_mm, y_extent_mm, raster_size_mm):
        # Convert y_mm to pixel coordinates (floating-point)
        y_px = y_mm * pixels_per_mm_y

        # Use nearest neighbor instead of interpolation
        y1 = int(round(y_px))
        if y1 >= height:  # Ensure we don't go out of bounds
            continue

        row = bw_image[y1, x_min : x_max + 1]

        # Find the start and end of black segments in the current row
        black_segments = np.where(np.diff(np.hstack(([0], row, [0]))))[
            0
        ].reshape(-1, 2)
        for start, end in black_segments:
            if row[start] == 1:  # Only process black segments
                # Use center-to-center toolpath convention for X-axis.
                # A segment from pixel `i_start` to `i_end` runs from the
                # center of `i_start` to the center of `i_end`.
                # Absolute start pixel index: x_min + start
                # Absolute end pixel index: x_min + end - 1
                start_mm = (x_min + start + 0.5) / pixels_per_mm_x
                end_mm = (x_min + end - 1 + 0.5) / pixels_per_mm_x

                # The Y coordinate for the line, adjusted to be in the
                # center of the pixel row it represents.
                line_y_mm = y_mm + y_pixel_center_offset_mm

                # Move to the start of the black segment
                ops.move_to(start_mm, ymax - line_y_mm)
                # Draw a line to the end of the black segment
                ops.line_to(end_mm, ymax - line_y_mm)

    return ops


def rasterize_vertically(
    surface,
    ymax,
    pixels_per_mm=(10, 10),
    raster_size_mm=0.1,
    x_offset_mm=0.0,
    threshold=128,
):
    """
    Generate an engraving path for a Cairo surface, focusing on vertical
    movement.

    Args:
        surface: A Cairo surface containing a black and white image.
        pixels_per_mm: Resolution of the image in pixels per millimeter.
        raster_size_mm: Distance between vertical engraving lines in
                        millimeters.
        x_offset_mm: The absolute horizontal offset of this surface chunk
                     from the left of the entire workpiece (in mm).
        threshold: The brightness value (0-255) to consider black.

    Returns:
        A Ops object containing the optimized engraving path.
    """
    surface_format = surface.get_format()
    if surface_format != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    # Convert surface to a NumPy array
    width = surface.get_width()
    height = surface.get_height()
    if width == 0 or height == 0:
        return Ops()
    data = np.frombuffer(surface.get_data(), dtype=np.uint8)
    data = data.reshape((height, width, 4))

    # Extract BGRA channels
    blue = data[:, :, 0]
    green = data[:, :, 1]
    red = data[:, :, 2]
    alpha = data[:, :, 3]

    # Convert to grayscale and threshold
    bw_image = (
        0.2989 * red + 0.5870 * green + 0.1140 * blue < threshold
    ).astype(np.uint8)
    bw_image[alpha == 0] = 0

    # Find bounding box
    occupied_rows = np.any(bw_image, axis=1)
    occupied_cols = np.any(bw_image, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return Ops()

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]

    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm
    x_min_mm = x_min / pixels_per_mm_x

    ops = Ops()

    global_x_min_mm = x_offset_mm + x_min_mm
    first_global_x_mm = (
        math.ceil(global_x_min_mm / raster_size_mm) * raster_size_mm
    )
    x_start_mm = first_global_x_mm - x_offset_mm

    x_pixel_center_offset_mm = 0.5 / pixels_per_mm_x
    x_extent_mm = (x_max + 1) / pixels_per_mm_x

    for x_mm in np.arange(x_start_mm, x_extent_mm, raster_size_mm):
        x_px = x_mm * pixels_per_mm_x
        x1 = int(round(x_px))
        if x1 >= width:
            continue

        col = bw_image[y_min : y_max + 1, x1]

        black_segments = np.where(np.diff(np.hstack(([0], col, [0]))))[
            0
        ].reshape(-1, 2)
        for start, end in black_segments:
            if col[start] == 1:
                start_mm = (y_min + start + 0.5) / pixels_per_mm_y
                end_mm = (y_min + end - 1 + 0.5) / pixels_per_mm_y

                line_x_mm = x_mm + x_pixel_center_offset_mm

                ops.move_to(line_x_mm, ymax - start_mm)
                ops.line_to(line_x_mm, ymax - end_mm)

    return ops


class Rasterizer(OpsProducer):
    """
    Generates rastered movements (using only straight lines)
    across filled pixels in the surface.
    """

    def __init__(self, cross_hatch: bool = False, threshold: int = 128):
        super().__init__()
        self.cross_hatch = cross_hatch
        self.threshold = threshold

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "params": {
                "cross_hatch": self.cross_hatch,
                "threshold": self.threshold,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rasterizer":
        params = data.get("params", {})
        return cls(
            cross_hatch=params.get("cross_hatch", False),
            threshold=params.get("threshold", 128),
        )

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
            raise ValueError("Rasterizer requires a workpiece context.")

        final_ops = Ops()
        # Always add section markers
        final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)

        width = surface.get_width()
        height = surface.get_height()
        logger.debug(f"Rasterizer received surface: {width}x{height} pixels")
        logger.debug(f"Rasterizer received pixels_per_mm: {pixels_per_mm}")

        raster_ops = Ops()
        if width > 0 and height > 0:
            ymax = height / pixels_per_mm[1]
            raster_ops = rasterize_horizontally(
                surface,
                ymax,  # y max for axis inversion
                pixels_per_mm,
                laser.spot_size_mm[1],
                y_offset_mm=y_offset_mm,
                threshold=self.threshold,
            )

            if self.cross_hatch:
                logger.info("Cross-hatch enabled, performing vertical pass.")
                x_offset_mm = workpiece.bbox[0]
                vertical_ops = rasterize_vertically(
                    surface,
                    ymax,
                    pixels_per_mm,
                    laser.spot_size_mm[0],
                    x_offset_mm=x_offset_mm,
                    threshold=self.threshold,
                )
                raster_ops.extend(vertical_ops)

        if not raster_ops.is_empty():
            final_ops.set_laser(laser.uid)
            final_ops.set_power((settings or {}).get("power", 0))
            final_ops.extend(raster_ops)

        final_ops.ops_section_end(SectionType.RASTER_FILL)

        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=workpiece.size,
        )

    def is_vector_producer(self) -> bool:
        return False
