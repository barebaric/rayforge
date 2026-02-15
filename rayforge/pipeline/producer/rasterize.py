import cairo
import numpy as np
import math
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any, Tuple
from ...core.ops import Ops, SectionType
from ...shared.tasker.progress import ProgressContext
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


def _validate_surface_format(surface) -> None:
    """Validate that the surface is in ARGB32 format.

    Args:
        surface: Cairo surface to validate.

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    surface_format = surface.get_format()
    if surface_format != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")


def _surface_to_binarized_array(
    surface, threshold: int, invert: bool
) -> np.ndarray:
    """Convert Cairo surface to binarized grayscale array.

    Args:
        surface: Cairo surface to convert.
        threshold: Brightness value (0-255) for binarization.
        invert: If True, invert the binarization logic.

    Returns:
        2D numpy array with values 0 (white) or 1 (black).
    """
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
    """Find the bounding box of occupied (black) pixels.

    Args:
        bw_image: 2D numpy array with binary values.

    Returns:
        Tuple of (y_min, y_max, x_min, x_max) or None if image is empty.
    """
    occupied_rows = np.any(bw_image, axis=1)
    occupied_cols = np.any(bw_image, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return None

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max


def _find_black_segments(line: np.ndarray) -> np.ndarray:
    """Find start and end indices of black segments in a line.

    Args:
        line: 1D numpy array with binary values.

    Returns:
        2D numpy array of shape (n, 2) where each row contains
        [start_index, end_index] for a black segment.
    """
    return np.where(np.diff(np.hstack(([0], line, [0]))))[0].reshape(-1, 2)


def _line_pixels(
    start: Tuple[float, float],
    end: Tuple[float, float],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Get pixel coordinates along a line using Bresenham's algorithm.

    Args:
        start: (x, y) start coordinates in pixels
        end: (x, y) end coordinates in pixels
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Array of (x, y) pixel coordinates along the line
    """
    x0, y0 = int(round(start[0])), int(round(start[1]))
    x1, y1 = int(round(end[0])), int(round(end[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    pixels = []
    x, y = x0, y0

    while True:
        if 0 <= x < width and 0 <= y < height:
            pixels.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(pixels, dtype=np.int32)


def _calculate_raster_params(
    bbox: Tuple[int, int, int, int],
    pixels_per_mm: Tuple[float, float],
    raster_size_mm: float,
    direction_degrees: float,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
) -> Dict[str, Any]:
    """Calculate raster parameters from bounding box and angle.

    Args:
        bbox: Bounding box as (y_min, y_max, x_min, x_max).
        pixels_per_mm: Resolution as (x, y) pixels per millimeter.
        raster_size_mm: Distance between raster lines in millimeters.
        direction_degrees: Raster direction in degrees.
        offset_x_mm: Global X offset for line alignment across chunks.
        offset_y_mm: Global Y offset for line alignment across chunks.

    Returns:
        Dictionary containing calculated parameters:
            - angle_rad: Direction angle in radians.
            - cos_a: Cosine of the direction angle.
            - sin_a: Sine of the direction angle.
            - center_x_mm: Center X coordinate in millimeters.
            - center_y_mm: Center Y coordinate in millimeters.
            - diag_mm: Diagonal length of bounding box in millimeters.
            - perp_cos: Cosine of perpendicular angle.
            - perp_sin: Sine of perpendicular angle.
            - first_line_index: First line index aligned to global grid.
            - last_line_index: Last line index aligned to global grid.
    """
    y_min, y_max, x_min, x_max = bbox
    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm

    angle_rad = math.radians(direction_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    bbox_width_mm = (x_max - x_min + 1) / pixels_per_mm_x
    bbox_height_mm = (y_max - y_min + 1) / pixels_per_mm_y

    center_x_mm = (x_min + x_max + 1) / (2 * pixels_per_mm_x)
    center_y_mm = (y_min + y_max + 1) / (2 * pixels_per_mm_y)

    diag_mm = math.sqrt(bbox_width_mm**2 + bbox_height_mm**2)

    perp_angle_rad = angle_rad + math.pi / 2
    perp_cos = math.cos(perp_angle_rad)
    perp_sin = math.sin(perp_angle_rad)

    global_center_perp = (center_x_mm + offset_x_mm) * perp_cos + (
        center_y_mm + offset_y_mm
    ) * perp_sin

    perp_extent_start = global_center_perp - diag_mm / 2
    perp_extent_end = global_center_perp + diag_mm / 2

    first_line_global_perp = math.ceil(perp_extent_start / raster_size_mm)
    last_line_global_perp = math.floor(perp_extent_end / raster_size_mm)

    return {
        "angle_rad": angle_rad,
        "cos_a": cos_a,
        "sin_a": sin_a,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "diag_mm": diag_mm,
        "perp_cos": perp_cos,
        "perp_sin": perp_sin,
        "first_line_index": first_line_global_perp,
        "last_line_index": last_line_global_perp,
        "global_center_perp": global_center_perp,
    }


def _get_raster_line_coords(
    params: Dict[str, Any],
    line_index: int,
    raster_size_mm: float,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Calculate start and end coordinates for a raster line.

    Args:
        params: Raster parameters from _calculate_raster_params.
        line_index: Global line index (multiple of raster_size_mm).
        raster_size_mm: Distance between raster lines in millimeters.
        offset_x_mm: Global X offset for line alignment.
        offset_y_mm: Global Y offset for line alignment.

    Returns:
        Tuple of (start_x_mm, start_y_mm, end_x_mm, end_y_mm).
    """
    center_x_mm = params["center_x_mm"]
    center_y_mm = params["center_y_mm"]
    diag_mm = params["diag_mm"]
    cos_a = params["cos_a"]
    sin_a = params["sin_a"]
    perp_cos = params["perp_cos"]
    perp_sin = params["perp_sin"]
    global_center_perp = params["global_center_perp"]

    line_global_perp = line_index * raster_size_mm
    line_local_perp = line_global_perp - global_center_perp

    line_center_x_mm = center_x_mm + line_local_perp * perp_cos
    line_center_y_mm = center_y_mm + line_local_perp * perp_sin

    half_diag = diag_mm / 2
    start_x_mm = line_center_x_mm - half_diag * cos_a
    start_y_mm = line_center_y_mm - half_diag * sin_a
    end_x_mm = line_center_x_mm + half_diag * cos_a
    end_y_mm = line_center_y_mm + half_diag * sin_a

    return start_x_mm, start_y_mm, end_x_mm, end_y_mm


def _process_raster_segments(
    pixels: np.ndarray,
    values: np.ndarray,
    segments: np.ndarray,
    reverse: bool,
    line_start_mm: Tuple[float, float],
    line_end_mm: Tuple[float, float],
    pixels_per_mm: Tuple[float, float],
) -> list:
    """Process raster segments and convert to millimeter coordinates.

    Uses the line's mm coordinates for the globally-aligned axis and
    pixel-based coordinates for the local axis to preserve segment accuracy.

    Args:
        pixels: Array of (x, y) pixel coordinates along the raster line.
        values: Binary values at each pixel position.
        segments: Array of [start_idx, end_idx] pairs for black segments.
        reverse: If True, swap start/end coordinates for each segment.
        line_start_mm: (x, y) start of the raster line in mm.
        line_end_mm: (x, y) end of the raster line in mm.
        pixels_per_mm: Resolution as (x, y) pixels per millimeter.

    Returns:
        List of tuples (start_mm_x, start_mm_y, end_mm_x, end_mm_y).
    """
    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm
    segment_coords = []

    sx, sy = line_start_mm
    ex, ey = line_end_mm

    is_horizontal = abs(ey - sy) < abs(ex - sx)

    for start_idx, end_idx in segments:
        if values[start_idx] == 1:
            if reverse:
                seg_start_px = pixels[end_idx - 1]
                seg_end_px = pixels[start_idx]
            else:
                seg_start_px = pixels[start_idx]
                seg_end_px = pixels[end_idx - 1]

            if is_horizontal:
                line_y_mm = sy
                start_mm_x = seg_start_px[0] / pixels_per_mm_x
                end_mm_x = seg_end_px[0] / pixels_per_mm_x
                start_mm_y = line_y_mm
                end_mm_y = line_y_mm
            else:
                line_x_mm = sx
                start_mm_y = seg_start_px[1] / pixels_per_mm_y
                end_mm_y = seg_end_px[1] / pixels_per_mm_y
                start_mm_x = line_x_mm
                end_mm_x = line_x_mm

            segment_coords.append((start_mm_x, start_mm_y, end_mm_x, end_mm_y))

    return segment_coords


def _add_segments_to_ops(
    ops: Ops,
    segment_coords: list,
    ymax: float,
    last_seg_end: Optional[Tuple[float, float]],
) -> Tuple[Ops, Optional[Tuple[float, float]]]:
    """Add processed segments to ops with proper coordinate conversion.

    Args:
        ops: Ops object to add commands to.
        segment_coords: List of (start_x, start_y, end_x, end_y) tuples.
        ymax: Maximum Y coordinate for axis inversion.
        last_seg_end: Previous segment end position for distance checking.

    Returns:
        Tuple of (updated_ops, new_last_seg_end).
    """
    for start_mm_x, start_mm_y, end_mm_x, end_mm_y in segment_coords:
        if last_seg_end is not None:
            dist_sq = (start_mm_x - last_seg_end[0]) ** 2 + (
                start_mm_y - last_seg_end[1]
            ) ** 2
            if dist_sq > 0.01:
                ops.move_to(float(start_mm_x), float(ymax - start_mm_y))
        else:
            ops.move_to(float(start_mm_x), float(ymax - start_mm_y))

        ops.line_to(float(end_mm_x), float(ymax - end_mm_y))
        last_seg_end = (end_mm_x, end_mm_y)

    return ops, last_seg_end


def rasterize_at_angle(
    surface,
    ymax: float,
    pixels_per_mm: Tuple[float, float],
    raster_size_mm: float,
    direction_degrees: float,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
    threshold: int = 128,
    invert: bool = False,
) -> Ops:
    """
    Generate an engraving path for a Cairo surface at a specified angle.

    Args:
        surface: A Cairo surface containing a black and white image.
        ymax: Maximum Y coordinate for axis inversion.
        pixels_per_mm: Resolution of the image in pixels per millimeter.
        raster_size_mm: Distance between engraving lines in millimeters.
        direction_degrees: Raster direction in degrees (0 = horizontal,
                          90 = vertical).
        offset_x_mm: The absolute horizontal offset of this surface chunk
                     from the left of the entire workpiece (in mm).
        offset_y_mm: The absolute vertical offset of this surface chunk
                     from the top of the entire workpiece (in mm).
        threshold: The brightness value (0-255) to consider black.
        invert: If True, invert the area to raster (engrave white areas).

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

    params = _calculate_raster_params(
        bbox,
        pixels_per_mm,
        raster_size_mm,
        direction_degrees,
        offset_x_mm,
        offset_y_mm,
    )
    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm

    ops = Ops()

    for i in range(params["first_line_index"], params["last_line_index"] + 1):
        start_x_mm, start_y_mm, end_x_mm, end_y_mm = _get_raster_line_coords(
            params, i, raster_size_mm, offset_x_mm, offset_y_mm
        )

        start_x_px = start_x_mm * pixels_per_mm_x
        start_y_px = start_y_mm * pixels_per_mm_y
        end_x_px = end_x_mm * pixels_per_mm_x
        end_y_px = end_y_mm * pixels_per_mm_y

        pixels = _line_pixels(
            (start_x_px, start_y_px), (end_x_px, end_y_px), width, height
        )

        if len(pixels) == 0:
            continue

        values = bw_image[pixels[:, 1], pixels[:, 0]]
        segments = _find_black_segments(values)

        if len(segments) == 0:
            continue

        reverse = (i % 2) != 0
        if reverse:
            segments = segments[::-1]

        segment_coords = _process_raster_segments(
            pixels,
            values,
            segments,
            reverse,
            (start_x_mm, start_y_mm),
            (end_x_mm, end_y_mm),
            pixels_per_mm,
        )

        if segment_coords:
            ops = _add_segments_to_ops(ops, segment_coords, ymax, None)[0]

    return ops


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
        direction_degrees: float = 0.0,
    ):
        super().__init__()
        self.cross_hatch = cross_hatch
        self.threshold = threshold
        self.invert = invert
        self.direction_degrees = direction_degrees

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "params": {
                "cross_hatch": self.cross_hatch,
                "threshold": self.threshold,
                "invert": self.invert,
                "direction_degrees": self.direction_degrees,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rasterizer":
        params = data.get("params", {})
        return cls(
            cross_hatch=params.get("cross_hatch", False),
            threshold=params.get("threshold", 128),
            invert=params.get("invert", False),
            direction_degrees=params.get("direction_degrees", 0.0),
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
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("Rasterizer requires a workpiece context.")

        final_ops = Ops()
        final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)

        width = surface.get_width()
        height = surface.get_height()
        logger.debug(f"Rasterizer received surface: {width}x{height} pixels")
        logger.debug(f"Rasterizer received pixels_per_mm: {pixels_per_mm}")

        surface_width_mm = width / pixels_per_mm[0]
        surface_height_mm = height / pixels_per_mm[1]

        raster_ops = Ops()
        if width > 0 and height > 0:
            ymax = height / pixels_per_mm[1]
            x_offset_mm = workpiece.bbox[0]

            raster_ops = rasterize_at_angle(
                surface,
                ymax,
                pixels_per_mm,
                laser.spot_size_mm[1],
                self.direction_degrees,
                offset_x_mm=x_offset_mm,
                offset_y_mm=y_offset_mm,
                threshold=self.threshold,
                invert=self.invert,
            )

            if self.cross_hatch:
                logger.info("Cross-hatch enabled, performing second pass.")
                perp_direction = (self.direction_degrees + 90) % 360
                cross_hatch_ops = rasterize_at_angle(
                    surface,
                    ymax,
                    pixels_per_mm,
                    laser.spot_size_mm[0],
                    perp_direction,
                    offset_x_mm=x_offset_mm,
                    offset_y_mm=y_offset_mm,
                    threshold=self.threshold,
                    invert=self.invert,
                )
                raster_ops.extend(cross_hatch_ops)

        if not raster_ops.is_empty():
            final_ops.set_laser(laser.uid)
            final_ops.set_power((settings or {}).get("power", 0))
            final_ops.extend(raster_ops)

        final_ops.ops_section_end(SectionType.RASTER_FILL)

        logger.debug(
            f"Rasterizer creating artifact: "
            f"surface_px=({width}x{height}), "
            f"surface_mm=({surface_width_mm:.2f}x{surface_height_mm:.2f}), "
            f"workpiece.size=({workpiece.size[0]:.2f}x"
            f"{workpiece.size[1]:.2f}), "
            f"source_dimensions=({surface_width_mm:.2f}x"
            f"{surface_height_mm:.2f})"
        )
        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(surface_width_mm, surface_height_mm),
            generation_size=workpiece.size,
        )

    def is_vector_producer(self) -> bool:
        return False
