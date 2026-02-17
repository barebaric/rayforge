import numpy as np
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...core.ops import Ops, SectionType
from ...image.image_util import surface_to_binary
from ...shared.tasker.progress import ProgressContext
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer
from .raster_util import (
    ScanLine,
    generate_scan_lines,
    find_segments,
    convert_y_to_output,
    find_mask_bounding_box,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


def _process_segments_for_scan_line(
    scan_line: ScanLine,
    bw_image: np.ndarray,
    pixels_per_mm: tuple,
    reverse: bool,
) -> list:
    """Process a scan line and extract segment coordinates in mm.

    Args:
        scan_line: The ScanLine to process.
        bw_image: Binary image array.
        pixels_per_mm: Resolution as (x, y) pixels per millimeter.
        reverse: If True, reverse segment order.

    Returns:
        List of (start_x_mm, start_y_mm, end_x_mm, end_y_mm) tuples.
    """
    if len(scan_line.pixels) == 0:
        return []

    values = bw_image[scan_line.pixels[:, 1], scan_line.pixels[:, 0]]
    segments = find_segments(values)

    if len(segments) == 0:
        return []

    if reverse:
        segments = segments[::-1]

    segment_coords = []
    for start_idx, end_idx in segments:
        if values[start_idx] == 1:
            if reverse:
                seg_start_px = scan_line.pixels[end_idx - 1]
                seg_end_px = scan_line.pixels[start_idx]
            else:
                seg_start_px = scan_line.pixels[start_idx]
                seg_end_px = scan_line.pixels[end_idx - 1]

            start_mm = scan_line.pixel_to_mm(
                seg_start_px[0], seg_start_px[1], pixels_per_mm
            )
            end_mm = scan_line.pixel_to_mm(
                seg_end_px[0], seg_end_px[1], pixels_per_mm
            )

            segment_coords.append(
                (start_mm[0], start_mm[1], end_mm[0], end_mm[1])
            )

    return segment_coords


def rasterize_at_angle(
    surface,
    ymax: float,
    pixels_per_mm: tuple,
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

    bw_image = surface_to_binary(surface, threshold, invert)

    bbox = find_mask_bounding_box(bw_image)
    if bbox is None:
        return Ops()

    ops = Ops()

    for scan_line in generate_scan_lines(
        bbox=bbox,
        image_size=(width, height),
        pixels_per_mm=pixels_per_mm,
        line_interval_mm=raster_size_mm,
        direction_degrees=direction_degrees,
        offset_x_mm=offset_x_mm,
        offset_y_mm=offset_y_mm,
    ):
        reverse = (scan_line.index % 2) != 0

        segment_coords = _process_segments_for_scan_line(
            scan_line, bw_image, pixels_per_mm, reverse
        )

        for start_x_mm, start_y_mm, end_x_mm, end_y_mm in segment_coords:
            ops.move_to(
                float(start_x_mm), float(convert_y_to_output(start_y_mm, ymax))
            )
            ops.line_to(
                float(end_x_mm), float(convert_y_to_output(end_y_mm, ymax))
            )

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
        line_interval_mm: Optional[float] = None,
    ):
        super().__init__()
        self.cross_hatch = cross_hatch
        self.threshold = threshold
        self.invert = invert
        self.direction_degrees = direction_degrees
        self.line_interval_mm = line_interval_mm

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "params": {
                "cross_hatch": self.cross_hatch,
                "threshold": self.threshold,
                "invert": self.invert,
                "direction_degrees": self.direction_degrees,
                "line_interval_mm": self.line_interval_mm,
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
            line_interval_mm=params.get("line_interval_mm"),
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
            line_interval = (
                self.line_interval_mm
                if self.line_interval_mm is not None
                else laser.spot_size_mm[1]
            )
            cross_hatch_interval = (
                self.line_interval_mm
                if self.line_interval_mm is not None
                else laser.spot_size_mm[0]
            )

            raster_ops = rasterize_at_angle(
                surface,
                ymax,
                pixels_per_mm,
                line_interval,
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
                    cross_hatch_interval,
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
            f"source_dimensions=({width}x{height})"
        )
        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(width, height),
            generation_size=workpiece.size,
        )

    def is_vector_producer(self) -> bool:
        return False
