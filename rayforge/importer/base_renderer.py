from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Generator, Tuple, cast
import cairo
import math
import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ..core.workpiece import WorkPiece

logger = logging.getLogger(__name__)

CAIRO_MAX_DIMENSION = 16384


class Renderer(ABC):
    """
    An abstract base class for any object that can render a WorkPiece
    to a pixel surface. Renderers are typically stateless singletons.
    """

    @abstractmethod
    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Returns the natural (untransformed) size of the content in mm,
        or None if the size cannot be determined.
        """
        pass

    @abstractmethod
    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the workpiece to a Cairo surface of specific pixel dimensions.
        """
        pass

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Hook for rendering source data to a vips image.
        Subclasses that support pyvips should override this.
        """
        surface = self.render_to_pixels(workpiece, width, height)
        if not surface:
            return None
        h, w = surface.get_height(), surface.get_width()
        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )
        # Cairo surface data is BGRA, Vips expects RGBA for color operations.
        b, g, r, a = vips_image[0], vips_image[1], vips_image[2], vips_image[3]
        return r.bandjoin([g, b, a])

    def get_or_create_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        A cache-aware helper to get a vips image for the workpiece.
        It checks the workpiece's internal cache before performing an
        expensive render operation.
        """
        key = (width, height)
        if key in workpiece._render_cache:
            return workpiece._render_cache[key]

        image = self._render_to_vips_image(workpiece, width, height)
        if image:
            workpiece._render_cache[key] = image
        return image

    def _calculate_chunk_layout(
        self,
        real_width: int,
        real_height: int,
        max_chunk_width: Optional[int],
        max_chunk_height: Optional[int],
        max_memory_size: Optional[int],
    ) -> Tuple[int, int, int, int]:
        bytes_per_pixel = 4
        effective_max_width = min(
            max_chunk_width
            if max_chunk_width is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )
        chunk_width = min(real_width, effective_max_width)
        possible_heights = []
        effective_max_height = min(
            max_chunk_height
            if max_chunk_height is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )
        possible_heights.append(effective_max_height)
        if max_memory_size is not None and chunk_width > 0:
            height_from_mem = math.floor(
                max_memory_size / (chunk_width * bytes_per_pixel)
            )
            possible_heights.append(height_from_mem)
        chunk_height = min(real_height, *possible_heights)
        chunk_width = max(1, chunk_width)
        chunk_height = max(1, chunk_height)
        cols = math.ceil(real_width / chunk_width)
        rows = math.ceil(real_height / chunk_height)
        return chunk_width, cols, chunk_height, rows

    def render_chunk(
        self,
        workpiece: "WorkPiece",
        width_px: float,
        height_px: float,
        max_chunk_width: Optional[int] = None,
        max_chunk_height: Optional[int] = None,
        max_memory_size: Optional[int] = None,
        overlap_x: int = 1,
        overlap_y: int = 0,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        if all(
            arg is None
            for arg in [max_chunk_width, max_chunk_height, max_memory_size]
        ):
            raise ValueError(
                "At least one of max_chunk_width, max_chunk_height, "
                "or max_memory_size must be provided."
            )

        vips_image = self.get_or_create_vips_image(
            workpiece, round(width_px), round(height_px)
        )
        if not vips_image or not isinstance(vips_image, pyvips.Image):
            logger.warning("Failed to load image for chunking.")
            return

        real_width = cast(int, vips_image.width)
        real_height = cast(int, vips_image.height)
        if not real_width or not real_height:
            return

        chunk_width, cols, chunk_height, rows = self._calculate_chunk_layout(
            real_width,
            real_height,
            max_chunk_width,
            max_chunk_height,
            max_memory_size,
        )

        for row in range(rows):
            for col in range(cols):
                left = col * chunk_width
                top = row * chunk_height
                width = min(chunk_width + overlap_x, real_width - left)
                height = min(chunk_height + overlap_y, real_height - top)

                if width <= 0 or height <= 0:
                    continue

                chunk: pyvips.Image = vips_image.crop(left, top, width, height)
                if chunk.bands < 4:
                    chunk = chunk.bandjoin(255)

                # Vips RGBA -> Cairo BGRA
                b, g, r, a = chunk[2], chunk[1], chunk[0], chunk[3]
                bgra_chunk = b.bandjoin([g, r, a])
                mem_buffer: memoryview = bgra_chunk.write_to_memory()

                surface = cairo.ImageSurface.create_for_data(
                    mem_buffer,
                    cairo.FORMAT_ARGB32,
                    chunk.width,
                    chunk.height,
                    chunk.width * 4,
                )
                yield surface, (left, top)
