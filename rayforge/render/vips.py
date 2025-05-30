from typing import Generator, Optional, Tuple
import cairo
import io
import math
import numpy as np
from .renderer import Renderer


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
    def get_vips_image(cls,
                       data,
                       width_px=None,
                       height_px=None,
                       pixels_per_mm: Optional[Tuple[float, float]] = None):
        """
        Return the pyvips image by using the loader function for the
        specific format.
        """
        if pixels_per_mm is None or None in pixels_per_mm:
            assert width_px and height_px, \
                   "Need either width/height or pixels_per_mm"
            natsize = cls.get_natural_size(data, px_factor=.1)
            pixels_per_mm = width_px/natsize[0], height_px/natsize[1]

        dpi = max(*pixels_per_mm) * 25.4
        kwargs = cls.get_vips_loader_args()
        image = cls.get_vips_loader()(data, dpi=dpi, **kwargs)

        if width_px or height_px:
            hscale = 1.0 if width_px is None else width_px / image.width
            vscale = 1.0 if height_px is None else height_px / image.height
            image = image.resize(hscale, vscale=vscale)

        return image

    @classmethod
    def prepare(cls, data):
        return cls._crop_to_content(data)

    @classmethod
    def _crop_to_content(cls, data):
        return data

    @classmethod
    def get_aspect_ratio(cls, data: bytes) -> float:
        width_mm, height_mm = cls.get_natural_size(data)
        if width_mm is None or height_mm is None:
            return 1.0  # Default to square aspect ratio
        return width_mm / height_mm

    @classmethod
    def render_workpiece(cls,
                         data,
                         width=None,
                         height=None,
                         pixels_per_mm=(25, 25)):
        """Render the full image at the specified size."""
        image = cls.get_vips_image(data, width, height, pixels_per_mm)
        buf = image.write_to_buffer('.png')
        return cairo.ImageSurface.create_from_png(io.BytesIO(buf))

    @classmethod
    def render_chunk(
        cls,
        data,
        width_px,
        height_px,
        chunk_width=10000,
        chunk_height=20,
        overlap_x=1,
        overlap_y=0,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        vips_image = cls.get_vips_image(data, width_px, height_px)

        # Resize to exact target dimensions
        hscale = width_px / vips_image.width
        vscale = height_px / vips_image.height
        vips_image = vips_image.resize(hscale, vscale=vscale)

        # Calculate number of chunks needed
        cols = math.ceil(vips_image.width / chunk_width)
        rows = math.ceil(vips_image.height / chunk_height)

        for row in range(rows):
            for col in range(cols):
                # Calculate chunk boundaries
                left = col * chunk_width
                top = row * chunk_height
                width = min(chunk_width+overlap_x, vips_image.width - left)
                height = min(chunk_height+overlap_y, vips_image.height - top)

                # Extract chunk
                chunk = vips_image.crop(left, top, width, height)

                # Ensure chunk has 4 bands (RGBA)
                if chunk.bands == 3:
                    chunk_rgba = chunk.bandjoin(255)  # Add opaque alpha
                elif chunk.bands == 4:
                    chunk_rgba = chunk  # Use existing alpha
                else:
                    raise ValueError(
                        f"Unexpected number of bands: {chunk.bands}"
                    )

                # Create Cairo surface for the chunk
                buf = chunk_rgba.write_to_memory()
                surface = cairo.ImageSurface.create_for_data(
                    buf,
                    cairo.FORMAT_ARGB32,
                    chunk.width,
                    chunk.height,
                    chunk_rgba.width*4  # Stride for RGBA
                )

                # Yield surface + position metadata
                yield surface, (left, top)

    @classmethod
    def _get_margins(cls, data):
        # Load the image using the class's VIPS loader
        kwargs = cls.get_vips_loader_args()
        vips_image = cls.get_vips_loader()(data, **kwargs)

        # Ensure the image has an alpha channel (band 3)
        if vips_image.bands < 4:
            vips_image = vips_image.bandjoin(255)  # Add alpha if missing

        # Extract the alpha channel and get dimensions
        alpha = vips_image[3]
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
            return (0, 0, 0, 0)

        # Calculate margins as percentages
        left_pct = left / width
        right_pct = (width - right - 1) / width
        top_pct = top / height
        bottom_pct = (height - bottom - 1) / height

        return left_pct, top_pct, right_pct, bottom_pct
