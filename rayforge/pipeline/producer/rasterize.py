import cairo
import numpy as np
import math
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any, Tuple
from ...core.ops import Ops, SectionType
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...shared.tasker.proxy import BaseExecutionContext

logger = logging.getLogger(__name__)


def _validate_surface_format(surface) -> None:
    """Validate that the surface is in ARGB32 format."""
    surface_format = surface.get_format()
    if surface_format != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")


def _surface_to_binarized_array(
    surface, threshold: int, invert: bool
) -> np.ndarray:
    """Convert Cairo surface to binarized grayscale array."""
    width = surface.get_width()
    height = surface.get_height()
    data = np.frombuffer(surface.get_data(), dtype=np.uint8)
    data = data.reshape((height, width, 4))

    blue = data[:, :, 0]
    green = data[:, :, 1]
    red = data[:, :, 2]
    alpha = data[:, :, 3]

    grayscale = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    if invert:
        bw_image = (grayscale > threshold).astype(np.uint8)
    else:
        bw_image = (grayscale < threshold).astype(np.uint8)

    bw_image[alpha == 0] = 0
    return bw_image


def _find_bounding_box(
    bw_image: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box of occupied pixels."""
    occupied_rows = np.any(bw_image, axis=1)
    occupied_cols = np.any(bw_image, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return None

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max


def _find_black_segments(line: np.ndarray) -> np.ndarray:
    """Find start and end indices of black segments in a line."""
    return np.where(np.diff(np.hstack(([0], line, [0]))))[0].reshape(-1, 2)


def rasterize(
    surface,
    ymax: float,
    pixels_per_mm: Tuple[float, float],
    raster_size_mm: float,
    offset_mm: float,
    threshold: int = 128,
    invert: bool = False,
    horizontal: bool = True,
) -> Ops:
    """
    Generate an engraving path for a Cairo surface.

    Args:
        surface: A Cairo surface containing a black and white image.
        ymax: Maximum Y coordinate for axis inversion.
        pixels_per_mm: Resolution of the image in pixels per millimeter.
        raster_size_mm: Distance between engraving lines in millimeters.
        offset_mm: The absolute offset of this surface chunk from the
                   edge of the entire workpiece (in mm).
        threshold: The brightness value (0-255) to consider black.
        invert: If True, invert the area to raster (engrave white areas).
        horizontal: If True, raster horizontally; otherwise vertically.

    Returns:
        A Ops object containing the optimized engraving path.
    """
    width = surface.get_width()
    height = surface.get_height()
    if width == 0 or height == 0:
        return Ops()

    _validate_surface_format(surface)
    bw_image = _surface_to_binarized_array(surface, threshold, invert)

    bbox = _find_bounding_box(bw_image)
    if bbox is None:
        return Ops()

    y_min, y_max, x_min, x_max = bbox
    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm

    ops = Ops()

    if horizontal:
        y_min_mm = y_min / pixels_per_mm_y
        global_y_min_mm = offset_mm + y_min_mm
        first_global_y_mm = (
            math.ceil(global_y_min_mm / raster_size_mm) * raster_size_mm
        )
        y_start_mm = first_global_y_mm - offset_mm
        y_pixel_center_offset_mm = 0.5 / pixels_per_mm_y
        y_extent_mm = (y_max + 1) / pixels_per_mm_y

        for y_mm in np.arange(y_start_mm, y_extent_mm, raster_size_mm):
            y_px = y_mm * pixels_per_mm_y
            y1 = int(round(y_px))
            if y1 >= height:
                continue

            row = bw_image[y1, x_min : x_max + 1]
            black_segments = _find_black_segments(row)

            for start, end in black_segments:
                if row[start] == 1:
                    start_mm = (x_min + start + 0.5) / pixels_per_mm_x
                    end_mm = (x_min + end - 1 + 0.5) / pixels_per_mm_x
                    line_y_mm = y_mm + y_pixel_center_offset_mm

                    ops.move_to(float(start_mm), float(ymax - line_y_mm))
                    ops.line_to(float(end_mm), float(ymax - line_y_mm))
    else:
        x_min_mm = x_min / pixels_per_mm_x
        global_x_min_mm = offset_mm + x_min_mm
        first_global_x_mm = (
            math.ceil(global_x_min_mm / raster_size_mm) * raster_size_mm
        )
        x_start_mm = first_global_x_mm - offset_mm
        x_pixel_center_offset_mm = 0.5 / pixels_per_mm_x
        x_extent_mm = (x_max + 1) / pixels_per_mm_x

        for x_mm in np.arange(x_start_mm, x_extent_mm, raster_size_mm):
            x_px = x_mm * pixels_per_mm_x
            x1 = int(round(x_px))
            if x1 >= width:
                continue

            col = bw_image[y_min : y_max + 1, x1]
            black_segments = _find_black_segments(col)

            for start, end in black_segments:
                if col[start] == 1:
                    start_mm = (y_min + start + 0.5) / pixels_per_mm_y
                    end_mm = (y_min + end - 1 + 0.5) / pixels_per_mm_y
                    line_x_mm = x_mm + x_pixel_center_offset_mm

                    ops.move_to(float(line_x_mm), float(ymax - start_mm))
                    ops.line_to(float(line_x_mm), float(ymax - end_mm))

    return ops


def rasterize_horizontally(
    surface,
    ymax,
    pixels_per_mm=(10, 10),
    raster_size_mm=0.1,
    y_offset_mm=0.0,
    threshold=128,
    invert=False,
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
        invert: If True, invert the area to raster (engrave white areas).

    Returns:
        A Ops object containing the optimized engraving path.
    """
    return rasterize(
        surface,
        ymax,
        pixels_per_mm,
        raster_size_mm,
        y_offset_mm,
        threshold,
        invert,
        horizontal=True,
    )


def rasterize_vertically(
    surface,
    ymax,
    pixels_per_mm=(10, 10),
    raster_size_mm=0.1,
    x_offset_mm=0.0,
    threshold=128,
    invert=False,
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
        invert: If True, invert the area to raster (engrave white areas).

    Returns:
        A Ops object containing the optimized engraving path.
    """
    return rasterize(
        surface,
        ymax,
        pixels_per_mm,
        raster_size_mm,
        x_offset_mm,
        threshold,
        invert,
        horizontal=False,
    )


class Rasterizer(OpsProducer):
    """
    Generates rastered movements (using only straight lines)
    across filled pixels in the surface.
    """

    def __init__(
        self,
        cross_hatch: bool = False,
        threshold: int = 128,
        invert: bool = False,
    ):
        super().__init__()
        self.cross_hatch = cross_hatch
        self.threshold = threshold
        self.invert = invert

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "params": {
                "cross_hatch": self.cross_hatch,
                "threshold": self.threshold,
                "invert": self.invert,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rasterizer":
        params = data.get("params", {})
        return cls(
            cross_hatch=params.get("cross_hatch", False),
            threshold=params.get("threshold", 128),
            invert=params.get("invert", False),
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
                ymax,
                pixels_per_mm,
                laser.spot_size_mm[1],
                y_offset_mm=y_offset_mm,
                threshold=self.threshold,
                invert=self.invert,
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
                    invert=self.invert,
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
