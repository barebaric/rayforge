from typing import Generator, Optional, Tuple, cast
import cairo
import io
import math
import numpy as np
import logging
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips  # type: ignore
from .renderer import Renderer


logger = logging.Logger(__name__)


# Base class for rendering with pyvips
class VipsRenderer(Renderer):
    @classmethod
    def get_vips_loader(cls):
        """
        Return the pyvips loader function for the specific format.
        """
        raise NotImplementedError

    @classmethod
    def get_vips_loader_args(cls):
        """
        Return kwargs for the pyvips loader function.
        """
        return {}

    @classmethod
    def _render_to_vips_image(
        cls, data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Internal helper. Default raster implementation that loads an image
        and resizes it to the exact pixel dimensions, stretching if necessary.
        """
        try:
            image = cls.get_vips_loader()(data, **cls.get_vips_loader_args())
            if (
                not isinstance(image, pyvips.Image)
                or image.width == 0
                or image.height == 0
            ):
                return None

            # Calculate separate horizontal and vertical scale factors and
            # use .resize() to force a non-uniform scale.
            h_scale = width / image.width
            v_scale = height / image.height
            return image.resize(h_scale, vscale=v_scale)

        except pyvips.Error as e:
            logger.error(f"Failed to render to vips image: {e}")
            return None

    @classmethod
    def prepare(cls, data: bytes) -> bytes:
        """
        Prepare the input data for rendering.
        This default implementation crops the content.
        """
        return cls._crop_to_content(data)

    @classmethod
    def _crop_to_content(cls, data: bytes) -> bytes:
        """
        Crop the content of the given data.
        This default implementation returns the data unchanged.
        """
        return data

    @classmethod
    def get_aspect_ratio(cls, data: bytes) -> float:
        width_mm, height_mm = cls.get_natural_size(data)
        if width_mm is None or height_mm is None:
            return 1.0  # Default to square aspect ratio
        return width_mm / height_mm

    @classmethod
    def render_to_pixels(
        cls, data: bytes, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Public method that uses the internal helper to get a vips image
        and converts it to a Cairo surface.
        """
        final_image = cls._render_to_vips_image(data, width, height)
        if not isinstance(final_image, pyvips.Image):
            raise RuntimeError('failed to render image')

        buf: bytes = final_image.write_to_buffer('.png')  # type: ignore
        return cairo.ImageSurface.create_from_png(io.BytesIO(buf))

    @classmethod
    def render_chunk(
        cls,
        data: bytes,
        width_px: int,
        height_px: int,
        chunk_width: int = 10000,
        chunk_height: int = 20,
        overlap_x: int = 1,
        overlap_y: int = 0,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        # Use the robust internal helper instead of the old, buggy
        # get_vips_image.
        vips_image = cls._render_to_vips_image(data, width_px, height_px)

        if not isinstance(vips_image, pyvips.Image):
            logger.warning("Failed to load image for chunking.")
            return

        # The rest of the chunking logic can now proceed, confident that
        # vips_image has the exact, correct dimensions.
        real_width = cast(int, vips_image.width)
        real_height = cast(int, vips_image.height)
        cols = math.ceil(real_width / chunk_width)
        rows = math.ceil(real_height / chunk_height)

        for row in range(rows):
            for col in range(cols):
                left = col * chunk_width
                top = row * chunk_height
                width = min(chunk_width + overlap_x, real_width - left)
                height = min(chunk_height + overlap_y, real_height - top)
                chunk: pyvips.Image = vips_image.crop(left, top, width, height)

                if chunk.bands == 3:
                    chunk = chunk.bandjoin(255)

                buf: bytes = chunk.write_to_memory()
                surface = cairo.ImageSurface.create_for_data(
                    buf,
                    cairo.FORMAT_ARGB32,
                    chunk.width,
                    chunk.height,
                    chunk.width * 4,
                )
                yield surface, (left, top)

    @classmethod
    def _get_margins(
        cls, data: bytes
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the content margins of an image as percentages.
        Assumes the image has an alpha channel.
        """
        # Load the image using the class's VIPS loader
        kwargs = cls.get_vips_loader_args()
        vips_image = cls.get_vips_loader()(data, **kwargs)
        if not isinstance(vips_image, pyvips.Image):
            logger.error("Failed to load image for cropping")
            return 0, 0, 0, 0

        # Ensure the image has an alpha channel (band 3)
        if cast(int, vips_image.bands) < 4:
            vips_image = vips_image.bandjoin(255)  # Add alpha if missing
        if not isinstance(vips_image, pyvips.Image):
            logger.error("Failed to add alpha channel to image for cropping")
            return 0, 0, 0, 0

        # Extract the alpha channel and get dimensions
        alpha = vips_image[3]
        assert alpha, "Unexpected alpha channel type"
        width = alpha.width
        height = alpha.height

        # Convert alpha channel to NumPy array
        alpha_np = alpha.numpy()

        # Compute sum along columns (axis 0) and rows (axis 1)
        columns_sum = alpha_np.sum(axis=0)  # Sum of each column
        rows_sum = alpha_np.sum(axis=1)    # Sum of each row

        # Find left and right margins
        if np.any(columns_sum):
            left = np.nonzero(columns_sum)[0][0]    # First column with content
            right = np.nonzero(columns_sum)[0][-1]  # Last column with content
        else:
            left = width   # No content, set to full width
            right = -1     # Indicates no content

        # Find top and bottom margins
        if np.any(rows_sum):
            top = np.nonzero(rows_sum)[0][0]          # First row with content
            bottom = np.nonzero(rows_sum)[0][-1]      # Last row with content
        else:
            top = height   # No content, set to full height
            bottom = -1    # Indicates no content

        # Handle case where there is no content
        if left >= width or right < 0 or top >= height or bottom < 0:
            return 0, 0, 0, 0

        # Calculate margins as percentages
        left_pct = left / width
        right_pct = (width - right - 1) / width
        top_pct = top / height
        bottom_pct = (height - bottom - 1) / height

        return left_pct, top_pct, right_pct, bottom_pct
