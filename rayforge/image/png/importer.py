import warnings
from typing import Optional
import logging

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import TraceSpec, VectorizationSpec
from .. import image_util
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from .renderer import PNG_RENDERER

logger = logging.getLogger(__name__)


class PngImporter(Importer):
    label = "PNG files"
    mime_types = ("image/png",)
    extensions = (".png",)
    is_bitmap = True

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        if vectorization_spec is None:
            vectorization_spec = TraceSpec()

        if not isinstance(vectorization_spec, TraceSpec):
            logger.error("PngImporter requires a TraceSpec to trace.")
            return None

        try:
            image = pyvips.Image.pngload_buffer(
                self.raw_data, access=pyvips.Access.RANDOM
            )
            logger.info(
                f"Successfully loaded PNG with pyvips: "
                f"{image.width}x{image.height}, "
                f"{image.bands} bands, format {image.format}"
            )
        except pyvips.Error as e:
            logger.error(
                f"pyvips failed to load PNG buffer: {e}", exc_info=True
            )
            return None

        metadata = image_util.extract_vips_metadata(image)
        metadata["image_format"] = "PNG"

        # Extract physical dimensions from the VIPS image
        width_mm, height_mm = image_util.get_physical_size_mm(image)

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PNG_RENDERER,
            metadata=metadata,
            width_px=image.width,
            height_px=image.height,
            width_mm=width_mm,
            height_mm=height_mm,
        )

        normalized_image = image_util.normalize_to_rgba(image)
        if not normalized_image:
            logger.error("Failed to normalize image to RGBA format.")
            return None
        logger.info("Normalized image to RGBA.")

        surface = image_util.vips_rgba_to_cairo_surface(normalized_image)
        logger.debug(
            f"Converted to cairo surface: "
            f"{surface.get_width()}x{surface.get_height()}"
        )

        # Trace the surface to get vector geometries
        geometries = trace_surface(surface, vectorization_spec)

        # Use the helper to create a single, masked workpiece
        items = image_util.create_single_workpiece_from_trace(
            geometries,
            source,
            image,
            vectorization_spec,
            self.source_file.stem,
        )

        return ImportPayload(source=source, items=items)
