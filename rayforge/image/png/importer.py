import warnings
from typing import Optional
import logging

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...core.import_source import ImportSource
from ...core.matrix import Matrix
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from .. import image_util
from .renderer import PNG_RENDERER

logger = logging.getLogger(__name__)


class PngImporter(Importer):
    label = "PNG files"
    mime_types = ("image/png",)
    extensions = (".png",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        if not vector_config:
            logger.error("PngImporter requires a vector_config to trace.")
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
                f"pyvips failed to load PNG '{self.source_file.name}': {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error loading PNG '{self.source_file.name}': {e}",
                exc_info=True
            )
            return None

        metadata = image_util.extract_vips_metadata(image)
        metadata["image_format"] = "PNG"

        source = ImportSource(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PNG_RENDERER,
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

        geometries = trace_surface(surface)
        combined_geo = Geometry()

        if geometries:
            logger.info(f"Successfully traced {len(geometries)} vector paths.")
            for geo in geometries:
                geo.close_gaps()
                combined_geo.commands.extend(geo.commands)
        else:
            logger.warning(
                "Tracing did not produce any vector geometries. "
                "Creating a workpiece with a frame around the image instead."
            )
            combined_geo.move_to(0, 0)
            combined_geo.line_to(image.width, 0)
            combined_geo.line_to(image.width, image.height)
            combined_geo.line_to(0, image.height)
            combined_geo.close_path()

        # 1. Calculate independent scale factors for X and Y to map the
        #    pixel geometry into a 1x1 unit square.
        if image.width > 0 and image.height > 0:
            norm_scale_x = 1.0 / image.width
            norm_scale_y = 1.0 / image.height

            # 2. Create a non-uniform scaling matrix.
            normalization_matrix = Matrix.scale(norm_scale_x, norm_scale_y)

            # 3. Apply this transform to the geometry data.
            combined_geo.transform(normalization_matrix.to_4x4_numpy())

        final_wp = WorkPiece(name=self.source_file.stem, vectors=combined_geo)
        final_wp.import_source_uid = source.uid

        width_mm, height_mm = image_util.get_physical_size_mm(image)

        # This call is now architecturally sound. It applies the physical size
        # via the matrix to a true 1x1 normalized shape.
        final_wp.set_size(width_mm, height_mm)
        final_wp.pos = (0, 0)

        return ImportPayload(source=source, items=[final_wp])
