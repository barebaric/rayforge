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
            raster_ops = self._rasterize_mask_horizontally(
                dithered_mask,
                pixels_per_mm,
                y_offset_mm,
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

    def _rasterize_mask_horizontally(
        self,
        mask: np.ndarray,
        pixels_per_mm: Tuple[float, float],
        y_offset_mm: float,
        line_interval_mm: float,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = mask.shape
        px_per_mm_x, px_per_mm_y = pixels_per_mm
        height_mm = height_px / px_per_mm_y

        occupied_rows = np.any(mask, axis=1)
        occupied_cols = np.any(mask, axis=0)

        if not np.any(occupied_rows) or not np.any(occupied_cols):
            return ops

        y_min_px, y_max_px = np.where(occupied_rows)[0][[0, -1]]
        x_min, x_max = np.where(occupied_cols)[0][[0, -1]]

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
            return ops

        scan_y_coords_px = scan_y_coords_mm * px_per_mm_y
        scan_y_coords_px = np.clip(scan_y_coords_px, 0, height_px - 1)

        y0 = np.floor(scan_y_coords_px).astype(int)
        y1 = np.clip(np.ceil(scan_y_coords_px).astype(int), 0, height_px - 1)

        y_pixel_center_offset_mm = 0.5 / px_per_mm_y

        for i, y_mm in enumerate(scan_y_coords_mm):
            row_y0 = mask[y0[i], x_min : x_max + 1]
            row_y1 = mask[y1[i], x_min : x_max + 1]

            frac = scan_y_coords_px[i] - y0[i]
            if frac < 0.5:
                row = row_y0
            else:
                row = row_y1

            if not np.any(row):
                continue

            if self.bidirectional:
                line_index = round((y_mm + y_offset_mm) / line_interval_mm)
                is_reversed = (line_index % 2) != 0
            else:
                is_reversed = False

            diff = np.diff(np.hstack(([0], row, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            if is_reversed:
                starts, ends = starts[::-1], ends[::-1]

            line_y_mm = y_mm + y_pixel_center_offset_mm
            final_y_mm = float(height_mm - line_y_mm)

            for start_px, end_px in zip(starts, ends):
                content_start_mm_x = (x_min + start_px) / px_per_mm_x
                content_end_mm_x = (x_min + end_px) / px_per_mm_x

                power_data = bytearray([255] * (end_px - start_px))

                if is_reversed:
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

        return ops

    def is_vector_producer(self) -> bool:
        return False
