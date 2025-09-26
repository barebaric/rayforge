import warnings
from typing import Optional
import logging

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_importer import Importer, ImportPayload
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...core.import_source import ImportSource
from ..tracing import trace_surface
from .. import image_util
from .renderer import JPG_RENDERER

logger = logging.getLogger(__name__)


class JpgImporter(Importer):
    label = "JPEG files"
    mime_types = ("image/jpeg",)
    extensions = (".jpg", ".jpeg")

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        if not vector_config:
            logger.error("JpgImporter requires a vector_config to trace.")
            return None

        try:
            image = pyvips.Image.jpegload_buffer(
                self.raw_data, access=pyvips.Access.RANDOM
            )
            logger.info(
                f"Successfully loaded JPEG with pyvips: "
                f"{image.width}x{image.height}, "
                f"{image.bands} bands, format {image.format}"
            )
        except pyvips.Error as e:
            logger.error(
                f"pyvips failed to load JPEG buffer: {e}", exc_info=True
            )
            return None

        metadata = image_util.extract_vips_metadata(image)
        metadata["image_format"] = "JPEG"

        source = ImportSource(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=JPG_RENDERER,
            vector_config=vector_config,
            metadata=metadata,
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

        width_mm, height_mm = image_util.get_physical_size_mm(image)
        pixels_per_mm = (image.width / width_mm, image.height / height_mm)

        geometries = trace_surface(surface, pixels_per_mm)
        combined_geo = Geometry()

        if geometries:
            logger.info(f"Successfully traced {len(geometries)} vector paths.")
            for geo in geometries:
                combined_geo.commands.extend(geo.commands)
        else:
            logger.warning(
                "Tracing did not produce any vector geometries. "
                "Creating a workpiece with a frame around the image instead."
            )
            # Create a rectangle representing the full image boundary.
            combined_geo.move_to(0, 0)
            combined_geo.line_to(width_mm, 0)
            combined_geo.line_to(width_mm, height_mm)
            combined_geo.line_to(0, height_mm)
            combined_geo.close_path()

        final_wp = WorkPiece(name=self.source_file.stem, vectors=combined_geo)
        final_wp.import_source_uid = source.uid
        final_wp.set_size(width_mm, height_mm)
        final_wp.pos = (0, 0)

        return ImportPayload(source=source, items=[final_wp])
