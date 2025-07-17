from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple
import cairo
import math
import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

logger = logging.getLogger(__name__)


class Renderer(ABC):
    """
    An abstract base class that defines the interface for all renderers.

    Each concrete renderer instance is created for a specific piece of
    image data and is responsible for managing its own internal state and
    implementation details.
    """

    label: Optional[str] = None
    mime_types: Optional[Tuple[str, ...]] = None
    extensions: Optional[Tuple[str, ...]] = None

    @abstractmethod
    def __init__(self, data: bytes):
        """
        The constructor that all subclasses must implement. It is
        responsible for receiving the raw byte data and preparing it for
        all subsequent rendering operations.
        """
        pass

    def _calculate_chunk_layout(
        self,
        real_width: int,
        real_height: int,
        max_chunk_width: Optional[int],
        max_chunk_height: Optional[int],
        max_memory_size: Optional[int],
    ) -> Tuple[int, int, int, int]:
        """
        Calculates the optimal chunk dimensions and grid layout.

        It determines the chunk width and height based on the provided
        constraints and returns the dimensions along with the number of
        columns and rows required to cover the entire image.
        """
        # Determine chunk width and number of columns.
        if max_chunk_width is None or max_chunk_width >= real_width:
            chunk_width = real_width
            cols = 1
        else:
            chunk_width = max_chunk_width
            cols = math.ceil(real_width / chunk_width)

        # Determine chunk height and number of rows.
        possible_heights = []
        if max_chunk_height is not None:
            possible_heights.append(max_chunk_height)

        if max_memory_size is not None and chunk_width > 0:
            bytes_per_pixel = 4  # cairo.FORMAT_ARGB32
            height_from_mem = math.floor(
                max_memory_size / (chunk_width * bytes_per_pixel)
            )
            possible_heights.append(height_from_mem)

        if not possible_heights:
            # This case is prevented by the caller's validation.
            # As a fallback, use a default small height.
            chunk_height = 20
        else:
            chunk_height = min(possible_heights)

        chunk_height = max(1, chunk_height)
        rows = math.ceil(real_height / chunk_height)

        return chunk_width, cols, chunk_height, rows

    @abstractmethod
    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns the natural (untransformed) size of the image in mm.

        If the source document uses pixel units, the px_factor is used
        to convert those dimensions to millimeters.
        """
        pass

    @abstractmethod
    def get_aspect_ratio(self) -> float:
        """
        Returns the natural (untransformed) aspect ratio of the image.
        """
        pass

    @abstractmethod
    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the image to a Cairo surface of specific pixel dimensions.
        """
        pass

    @abstractmethod
    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders the source data to a vips image of specific dimensions.
        This is a hook for the Template Method pattern used by render_chunk.
        """
        pass

    def render_chunk(
        self,
        width_px: int,
        height_px: int,
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

        vips_image = self._render_to_vips_image(width_px, height_px)
        if not isinstance(vips_image, pyvips.Image):
            logger.warning("Failed to load image for chunking.")
            return

        real_width = vips_image.width
        real_height = vips_image.height
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
                if chunk.bands == 3:
                    chunk = chunk.bandjoin(255)

                b, g, r, a = chunk[2], chunk[1], chunk[0], chunk[3]
                bgra_chunk = b.bandjoin([g, r, a])
                buf: bytes = bgra_chunk.write_to_memory()
                surface = cairo.ImageSurface.create_for_data(
                    buf,
                    cairo.FORMAT_ARGB32,
                    chunk.width,
                    chunk.height,
                    chunk.width * 4,
                )
                yield surface, (left, top)
