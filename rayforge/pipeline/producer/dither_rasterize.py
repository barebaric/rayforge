import cairo
import numpy as np
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...core.ops import Ops, SectionType
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer
from .raster_util import (
    find_segments,
    convert_y_to_output,
    calculate_ymax_mm,
    find_mask_bounding_box,
    generate_horizontal_scan_positions,
    resample_rows,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...shared.tasker.progress import ProgressContext

logger = logging.getLogger(__name__)


class DitherRasterizer(OpsProducer):
    """
    Generates rastered movements using dithering instead of threshold.
    This producer applies dithering algorithms to convert grayscale images
    to binary patterns for engraving.
    """

    def __init__(
        self,
        dither_algorithm: str = "floyd_steinberg",
        invert: bool = False,
        bidirectional: bool = True,
        line_interval_mm: Optional[float] = None,
    ):
        super().__init__()
        self.dither_algorithm = dither_algorithm
        self.invert = invert
        self.bidirectional = bidirectional
        self.line_interval_mm = line_interval_mm

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "params": {
                "dither_algorithm": self.dither_algorithm,
                "invert": self.invert,
                "bidirectional": self.bidirectional,
                "line_interval_mm": self.line_interval_mm,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DitherRasterizer":
        params = data.get("params", {})
        return cls(
            dither_algorithm=params.get("dither_algorithm", "floyd_steinberg"),
            invert=params.get("invert", False),
            bidirectional=params.get("bidirectional", True),
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
        context: Optional["ProgressContext"] = None,
    ) -> WorkPieceArtifact:
        from ...image.dither import surface_to_dithered_array

        if workpiece is None:
            raise ValueError("DitherRasterizer requires a workpiece context.")
        if surface.get_format() != cairo.FORMAT_ARGB32:
            raise ValueError("Unsupported Cairo surface format")

        final_ops = Ops()
        final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)

        width_px = surface.get_width()
        height_px = surface.get_height()
        surface_width_mm = width_px / pixels_per_mm[0]
        surface_height_mm = height_px / pixels_per_mm[1]

        logger.debug(
            f"DitherRasterizer: surface={width_px}x{height_px} px, "
            f"pixels_per_mm={pixels_per_mm}, "
            f"surface_mm=({surface_width_mm:.2f}x{surface_height_mm:.2f})"
        )

        if width_px > 0 and height_px > 0:
            from ...image.dither import DitherAlgorithm

            dithered_mask = surface_to_dithered_array(
                surface,
                DitherAlgorithm(self.dither_algorithm),
                self.invert,
            )

            occupied_pixels = np.sum(dithered_mask)
            logger.debug(
                f"DitherRasterizer: dithered_mask has {occupied_pixels} "
                f"occupied pixels (invert={self.invert})"
            )

            line_interval = (
                self.line_interval_mm
                if self.line_interval_mm is not None
                else laser.spot_size_mm[1]
            )
            x_offset_mm = workpiece.bbox[0]
            adjusted_y_offset_mm = workpiece.bbox[1] + y_offset_mm

            raster_ops = self._rasterize_mask(
                dithered_mask,
                pixels_per_mm,
                x_offset_mm,
                adjusted_y_offset_mm,
                line_interval,
            )
            final_ops.extend(raster_ops)

        if not final_ops.is_empty():
            final_ops.set_laser(laser.uid)
            final_ops.set_power((settings or {}).get("power", 1.0))

        final_ops.ops_section_end(SectionType.RASTER_FILL)

        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(width_px, height_px),
            generation_size=workpiece.size,
        )

    def _rasterize_mask(
        self,
        mask: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
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

        resampled_mask = resample_rows(mask.astype(np.float32), y_coords_px)
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

                power_data = bytearray([255] * (end_px - start_px))

                if self.bidirectional and is_reversed:
                    ops.move_to(content_end_mm_x, final_y_mm, 0.0)
                    ops.scan_to(
                        content_start_mm_x,
                        final_y_mm,
                        0.0,
                        power_data[::-1],
                    )
                else:
                    ops.move_to(content_start_mm_x, final_y_mm, 0.0)
                    ops.scan_to(
                        content_end_mm_x,
                        final_y_mm,
                        0.0,
                        power_data,
                    )

            if self.bidirectional:
                is_reversed = not is_reversed

        return ops

    def is_vector_producer(self) -> bool:
        return False
