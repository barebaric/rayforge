import io
from typing import Generator, Optional, Tuple, Dict
import warnings
import logging
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
from pypdf import PdfReader
from ..util.unit import to_mm
from .renderer import Renderer
import cairo
import math

logger = logging.getLogger(__name__)


class PDFRenderer(Renderer):
    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    def __init__(self, data: bytes):
        """
        Initializes the renderer.
        """
        self.raw_data = data
        self._natural_size_cache: Dict[
            float, Tuple[Optional[float], Optional[float]]
        ] = {}
        self._vips_image_cache: Dict[Tuple[int, int], pyvips.Image] = {}

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates the natural size from the PDF's media box.
        The result is cached in an instance dictionary to correctly handle
        different px_factor arguments and to avoid re-parsing the file.
        """
        if px_factor in self._natural_size_cache:
            return self._natural_size_cache[px_factor]

        try:
            reader = PdfReader(io.BytesIO(self.raw_data))
            page = reader.pages[0]
            media_box = page.mediabox
            width_pt = float(media_box.width)
            height_pt = float(media_box.height)
            size = (
                to_mm(width_pt, "pt", px_factor),
                to_mm(height_pt, "pt", px_factor),
            )
            self._natural_size_cache[px_factor] = size
            return size
        except Exception as e:
            logger.error(f"Failed to get natural size from PDF: {e}")
            return None, None

    def get_aspect_ratio(self) -> float:
        """
        Calculates aspect ratio on-the-fly using the cached get_natural_size.
        """
        w, h = self.get_natural_size()
        if w and h and h > 0:
            return w / h
        return 1.0

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders the PDF to a vips image, caching the result in an instance
        dictionary. This is the most expensive operation for this renderer.
        """
        key = width, height
        if key in self._vips_image_cache:
            return self._vips_image_cache[key]

        try:
            nat_w_mm, _ = self.get_natural_size()
            if nat_w_mm and nat_w_mm > 0:
                nat_w_in = nat_w_mm / 25.4
                target_dpi = width / nat_w_in
            else:
                target_dpi = 300
            image = pyvips.Image.pdfload_buffer(self.raw_data, dpi=target_dpi)
            if not isinstance(image, pyvips.Image) or image.width == 0:
                return None
            h_scale = width / image.width
            v_scale = height / image.height
            final_image = image.resize(h_scale, vscale=v_scale)

            self._vips_image_cache[key] = final_image
            return final_image
        except pyvips.Error as e:
            logger.error(f"Error rendering PDF to vips image: {e}")
            return None

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self._render_to_vips_image(width, height)
        if not isinstance(final_image, pyvips.Image):
            return None
        if final_image.bands < 4:
            final_image = final_image.bandjoin(255)
        b, g, r, a = (
            final_image[2],
            final_image[1],
            final_image[0],
            final_image[3],
        )
        bgra_image = b.bandjoin([g, r, a])
        mem_buffer = bgra_image.write_to_memory()
        return cairo.ImageSurface.create_for_data(
            mem_buffer,
            cairo.FORMAT_ARGB32,
            final_image.width,
            final_image.height,
            final_image.width * 4,
        )

    def render_chunk(
        self,
        width_px: int,
        height_px: int,
        chunk_width: int = 10000,
        chunk_height: int = 20,
        overlap_x: int = 1,
        overlap_y: int = 0,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        vips_image = self._render_to_vips_image(width_px, height_px)
        if not isinstance(vips_image, pyvips.Image):
            logger.warning("Failed to load image for chunking.")
            return
        real_width = vips_image.width
        real_height = vips_image.height
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
