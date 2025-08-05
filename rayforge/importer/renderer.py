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

# Cairo has a hard limit on surface dimensions, often 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 16384


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
        constraints, ensuring no dimension exceeds Cairo's internal limit.
        The contract is that returned chunks will not EXCEED the maximums,
        not that they will meet them.
        """
        bytes_per_pixel = 4  # cairo.FORMAT_ARGB32

        # 1. Determine the absolute maximum width allowed.
        # This is the lesser of the user's request and Cairo's hard limit.
        # If the user provides no limit, we still must respect Cairo's limit.
        effective_max_width = min(
            max_chunk_width
            if max_chunk_width is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )

        # 2. Determine the chunk_width for our tiling plan.
        # It cannot be larger than the image itself or the effective max
        # width.
        chunk_width = min(real_width, effective_max_width)

        # 3. Determine the absolute maximum height allowed from all
        # constraints.
        possible_heights = []

        # Constraint from max_chunk_height parameter and Cairo limit
        effective_max_height = min(
            max_chunk_height
            if max_chunk_height is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )
        possible_heights.append(effective_max_height)

        # Constraint from max_memory_size parameter
        if max_memory_size is not None and chunk_width > 0:
            height_from_mem = math.floor(
                max_memory_size / (chunk_width * bytes_per_pixel)
            )
            possible_heights.append(height_from_mem)

        # 4. The final chunk_height is the most restrictive of all
        # possibilities.
        # It also cannot be larger than the image's real height.
        chunk_height = min(real_height, *possible_heights)

        # 5. Ensure dimensions are at least 1 pixel.
        chunk_width = max(1, chunk_width)
        chunk_height = max(1, chunk_height)

        # 6. Calculate the number of rows and columns for the tiling loop.
        cols = math.ceil(real_width / chunk_width)
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
