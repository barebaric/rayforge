import warnings
from typing import Optional
import logging
import cairo

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pyvips
    except ImportError:
        raise ImportError("The BMP importer requires the pyvips library.")

from ...core.workpiece import WorkPiece
from ..base_importer import Importer, ImportPayload
from ...core.vectorization_spec import VectorizationSpec, TraceSpec
from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.matrix import Matrix
from ...core.generation_config import GenerationConfig
from ..tracing import trace_surface
from .parser import parse_bmp
from .renderer import BMP_RENDERER

logger = logging.getLogger(__name__)


class BmpImporter(Importer):
    label = "BMP files"
    mime_types = ("image/bmp",)
    extensions = (".bmp",)
    is_bitmap = True

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        if not isinstance(vectorization_spec, TraceSpec):
            logger.error("BmpImporter requires a TraceSpec to trace.")
            return None

        # Step 1: Create the SourceAsset to hold the original data and config
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=BMP_RENDERER,
        )

        # Step 2: Use the parser to get clean pixel data and metadata.
        parsed_data = parse_bmp(self.raw_data)
        if not parsed_data:
            logger.error(
                "BMP file could not be parsed. It may be compressed or in an "
                "unsupported format."
            )
            return None

        rgba_bytes, width, height, dpi_x, dpi_y = parsed_data

        try:
            # Step 3: Create a clean pyvips image from the RGBA buffer.
            image = pyvips.Image.new_from_memory(
                rgba_bytes, width, height, 4, "uchar"
            )
            # Explicitly set the color interpretation.
            # This tells pyvips that the 4 bands are sRGB + Alpha, which is
            # required for band extraction to work reliably.
            image = image.copy(interpretation=pyvips.Interpretation.SRGB)

        except pyvips.Error as e:
            logger.error(
                "Failed to create pyvips image from parsed BMP data: %s", e
            )
            return None

        # Step 4: Proceed with the known-good pyvips image for tracing.
        # Avoid division by zero if DPI is missing/invalid.
        dpi_x = dpi_x or 96.0
        dpi_y = dpi_y or 96.0
        width_mm = width * (25.4 / dpi_x)
        height_mm = height * (25.4 / dpi_y)

        # Convert to BGRA for Cairo (which expects ARGB32 in machine
        # byte order, effectively BGRA on little-endian systems).
        b, g, r, a = image[2], image[1], image[0], image[3]
        bgra_image = b.bandjoin([g, r, a])

        # Create a self-managed Cairo surface and copy data into it row-by-row.
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        cairo_buffer = surface.get_data()
        vips_pixel_data = bgra_image.write_to_memory()
        cairo_stride = surface.get_stride()
        vips_row_bytes = width * 4

        for y in range(height):
            cairo_row_start = y * cairo_stride
            vips_row_start = y * vips_row_bytes

            cairo_buffer[
                cairo_row_start : cairo_row_start + vips_row_bytes
            ] = vips_pixel_data[
                vips_row_start : vips_row_start + vips_row_bytes
            ]

        surface.mark_dirty()

        # Step 5: Trace the surface and create the WorkPiece.
        geometries = trace_surface(surface)

        # Always combine geometries into a single WorkPiece, even if the
        # list is empty. An empty list results in a WorkPiece with empty
        # vectors.
        combined_geo = Geometry()
        if geometries:
            for geo in geometries:
                geo.close_gaps()
                combined_geo.commands.extend(geo.commands)

        # Get the pixel-space bounding box for the segmentation mask.
        min_x, min_y, max_x, max_y = combined_geo.rect()
        mask_geo = Geometry()
        mask_geo.move_to(min_x, min_y)
        mask_geo.line_to(max_x, min_y)
        mask_geo.line_to(max_x, max_y)
        mask_geo.line_to(min_x, max_y)
        mask_geo.close_path()

        # Normalize the pixel-based geometry to a 1x1 unit square.
        if width > 0 and height > 0:
            norm_scale_x = 1.0 / width
            norm_scale_y = 1.0 / height
            normalization_matrix = Matrix.scale(norm_scale_x, norm_scale_y)
            combined_geo.transform(normalization_matrix.to_4x4_numpy())

        # Create the GenerationConfig.
        gen_config = GenerationConfig(
            source_asset_uid=source.uid,
            segment_mask_geometry=mask_geo,
            vectorization_spec=vectorization_spec,
        )

        # Create the WorkPiece with the normalized vectors and new config.
        final_wp = WorkPiece(
            name=self.source_file.stem,
            vectors=combined_geo,
            generation_config=gen_config,
        )

        # Apply the final physical size via the matrix. This is now correct.
        final_wp.set_size(width_mm, height_mm)
        final_wp.pos = (0, 0)

        return ImportPayload(source=source, items=[final_wp])
