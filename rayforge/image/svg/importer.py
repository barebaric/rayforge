import io
from typing import Optional, List
import logging
from xml.etree import ElementTree as ET
from svgelements import (
    SVG,
    Path,
    Move,
    Line,
    Close,
    Arc,
    CubicBezier,
    QuadraticBezier,
)

from ...core.geo import Geometry
from ...core.import_source import ImportSource
from ...core.item import DocItem
from ...core.matrix import Matrix
from ...core.vectorization_config import TraceConfig
from ...core.workpiece import WorkPiece
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from .renderer import SVG_RENDERER
from .svgutil import get_natural_size, trim_svg, PPI

logger = logging.getLogger(__name__)


class SvgImporter(Importer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        """
        Generates DocItems from SVG data.

        If vector_config is provided, it renders the SVG to a bitmap and
        traces it. This is robust but may lose fidelity.

        If vector_config is None, it attempts to parse the SVG path and
        shape data directly for a high-fidelity vector import.
        """
        # Process the SVG: trim it
        trimmed_data = trim_svg(self.raw_data)

        # Create import source.
        source = ImportSource(
            source_file=self.source_file,
            original_data=self.raw_data,
            working_data=trimmed_data,
            renderer=SVG_RENDERER,
        )

        # Read metadata.
        metadata = {}
        try:
            # Get size of original, untrimmed SVG
            untrimmed_size = get_natural_size(self.raw_data)
            if untrimmed_size:
                metadata["untrimmed_width_mm"] = untrimmed_size[0]
                metadata["untrimmed_height_mm"] = untrimmed_size[1]

            # Get size of the new, trimmed SVG
            trimmed_size = get_natural_size(trimmed_data)
            if trimmed_size:
                metadata["trimmed_width_mm"] = trimmed_size[0]
                metadata["trimmed_height_mm"] = trimmed_size[1]

            # Get viewBox from trimmed SVG for direct import
            root = ET.fromstring(trimmed_data)
            vb_str = root.get("viewBox")
            if vb_str:
                metadata["viewbox"] = tuple(map(float, vb_str.split()))

            source.metadata.update(metadata)
        except Exception as e:
            logger.warning(f"Could not calculate SVG metadata: {e}")

        if vector_config is not None:
            # Path 1: Render to bitmap and trace
            items = self._get_doc_items_from_trace(source, vector_config)
        else:
            # Path 2: Direct vector parsing
            items = self._get_doc_items_direct(source)

        if not items:
            return None

        return ImportPayload(source=source, items=items)

    def _get_doc_items_from_trace(
        self, source: ImportSource, vector_config: TraceConfig
    ) -> Optional[List[DocItem]]:
        """
        Renders trimmed SVG data to a bitmap, traces it, and creates a
        WorkPiece.
        """
        size_mm = None
        if source.metadata:
            w = source.metadata.get("trimmed_width_mm")
            h = source.metadata.get("trimmed_height_mm")
            if w is not None and h is not None:
                size_mm = (w, h)

        # If we can't determine a size, we can't trace. Return None.
        if not size_mm or not size_mm[0] or not size_mm[1]:
            logger.warning("failed to find a size")
            return None

        if not source.working_data:
            logger.error("source has no data to trace")
            return None

        w_mm, h_mm = size_mm
        w_px, h_px = 2048, 2048

        surface = SVG_RENDERER.render_to_pixels_from_data(
            source.working_data, w_px, h_px
        )

        wp = WorkPiece(name=self.source_file.stem)
        wp.import_source_uid = source.uid

        if surface:
            geometries = trace_surface(surface)
            if geometries:
                combined_geo = Geometry()
                for geo in geometries:
                    combined_geo.commands.extend(geo.commands)

                # Normalize the pixel-based geometry to a 1x1 unit square
                if surface.get_width() > 0 and surface.get_height() > 0:
                    norm_x = 1.0 / surface.get_width()
                    norm_y = 1.0 / surface.get_height()
                    norm_matrix = Matrix.scale(norm_x, norm_y)
                    combined_geo.transform(norm_matrix.to_4x4_numpy())

                wp.vectors = combined_geo

        # Always set the size. If tracing failed, the workpiece will be empty
        # but correctly sized.
        wp.set_size(size_mm[0], size_mm[1])

        return [wp]

    def _get_doc_items_direct(
        self, source: ImportSource
    ) -> Optional[List[DocItem]]:
        """
        Parses trimmed SVG vector data directly, handling viewBox and unit
        conversions to ensure the vector geometry matches the final size.
        """
        if not source.working_data:
            logger.error("source has no data to trace")
            return None

        try:
            # Correctly wrap the trimmed byte data in an in-memory stream.
            svg_stream = io.BytesIO(source.working_data)
            svg = SVG.parse(svg_stream, ppi=PPI)
        except Exception as e:
            logger.error(f"Failed to parse SVG for direct import: {e}")
            return None

        # --- Establish Authoritative Dimensions and Transformation ---
        # The key to matching the rendered output is to honor the `viewBox`.
        # The `width` and `height` attributes define the final rendered size.
        if (
            not source.metadata.get("viewbox")
            or not source.metadata.get("trimmed_width_mm")
            or not source.metadata.get("trimmed_height_mm")
        ):
            logger.warning(
                "SVG is missing viewBox, width, or height attributes; "
                "falling back to trace method for direct import."
            )
            # Fallback to tracing if essential attributes for direct import
            # are missing.
            return self._get_doc_items_from_trace(source, TraceConfig())

        logger.info(f"SVG raw width/height: {svg.width}, {svg.height}")
        logger.info(f"SVG viewBox: {svg.viewbox}")

        # Get final dimensions in millimeters from the trimmed SVG metadata.
        final_width_mm = source.metadata.get("trimmed_width_mm", 0.0)
        final_height_mm = source.metadata.get("trimmed_height_mm", 0.0)

        logger.info(
            f"Final dimensions: {final_width_mm:.3f}mm x "
            f"{final_height_mm:.3f}mm"
        )

        # Get the bounding box of the SVG content from svgelements.
        try:
            bbox = svg.bbox(with_stroke=True)
            if bbox is None:
                # This can happen if the SVG contains no visible geometry.
                logger.warning(
                    "svgelements could not determine SVG bounds"
                    " (no visible content); "
                    "falling back to trace method."
                )
                return self._get_doc_items_from_trace(source, TraceConfig())
            min_x, min_y, max_x, max_y = bbox
        except Exception as e:
            logger.warning(
                f"Error calculating SVG bounds with svgelements ({e}); "
                "falling back to trace method."
            )
            return self._get_doc_items_from_trace(source, TraceConfig())

        geo = Geometry()
        # Average scale for tolerance adjustment
        # Use max to avoid division by zero if one dimension is zero
        avg_dim = max(final_width_mm, final_height_mm, 1.0)
        avg_scale = avg_dim / 960  # Assuming typical viewBox size
        tolerance = 0.1 / avg_scale if avg_scale > 1e-9 else 0.1
        logger.debug(
            f"Average scale estimate: {avg_scale:.6f}, "
            f"tolerance: {tolerance:.6f}"
        )

        for shape in svg.elements():
            # svgelements provides a unified way to handle all shapes by
            # converting them to a Path object.
            try:
                path = Path(shape)
                path.reify()  # Apply transforms
            except (AttributeError, TypeError):
                continue  # Skip non-shape elements like <defs>

            for seg in path:
                # Add checks to ensure segment points are not None
                if seg.end is None or seg.end.x is None or seg.end.y is None:
                    continue

                end_x, end_y = float(seg.end.x), float(seg.end.y)

                if isinstance(seg, Move):
                    geo.move_to(end_x, end_y)
                elif isinstance(seg, Line):
                    geo.line_to(end_x, end_y)
                elif isinstance(seg, Close):
                    geo.close_path()
                elif isinstance(seg, Arc):
                    # Ensure all points for arc calculation are valid.
                    if (
                        seg.start is None
                        or seg.start.x is None
                        or seg.start.y is None
                        or seg.center is None
                        or seg.center.x is None
                        or seg.center.y is None
                    ):
                        continue

                    start_x, start_y = float(seg.start.x), float(seg.start.y)
                    center_x, center_y = (
                        float(seg.center.x),
                        float(seg.center.y),
                    )

                    center_offset_x = center_x - start_x
                    center_offset_y = center_y - start_y
                    # SVG sweep_flag=1 is CCW, 0 is CW.
                    is_clockwise = seg.sweep == 0
                    geo.arc_to(
                        end_x,
                        end_y,
                        center_offset_x,
                        center_offset_y,
                        clockwise=is_clockwise,
                    )
                elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                    # Linearize the curve into a series of line segments.
                    length = seg.length()
                    if length is None or length <= 1e-9:
                        # Just draw a line to the end if curve is invalid
                        geo.line_to(end_x, end_y)
                        continue

                    # Use a tolerance adjusted for estimated scale
                    num_steps = max(2, int(length / tolerance))

                    # Iterate from t=0 to t=1 to get points along the curve.
                    for i in range(1, num_steps + 1):
                        t = i / num_steps
                        p = seg.point(t)
                        if (
                            p is not None
                            and p.x is not None
                            and p.y is not None
                        ):
                            px, py = float(p.x), float(p.y)
                            geo.line_to(px, py)

        logger.info(
            f"Pre-transform bounds: x=[{min_x:.3f}, {max_x:.3f}], "
            f"y=[{min_y:.3f}, {max_y:.3f}]"
        )
        content_width = max_x - min_x
        content_height = max_y - min_y
        logger.info(
            f"Content size: {content_width:.3f} x {content_height:.3f}"
        )

        if content_width <= 0 or content_height <= 0:
            logger.warning(
                "Invalid content bounds; falling back to trace method."
            )
            return self._get_doc_items_from_trace(source, TraceConfig())

        # First, translate the Y-down geometry to its own origin.
        translation_matrix = Matrix.translation(-min_x, -min_y)
        geo.transform(translation_matrix.to_4x4_numpy())

        # Second, scale the translated geometry to a 1x1 unit box.
        norm_matrix = Matrix.scale(1.0 / content_width, 1.0 / content_height)
        geo.transform(norm_matrix.to_4x4_numpy())

        # Third, flip the now-normalized Y-down geometry to be Y-up.
        # This is a scale of -1 on Y, followed by a translation of +1 on Y
        # to move it back into the positive quadrant.
        flip_matrix = Matrix.scale(1.0, -1.0)
        flip_matrix = flip_matrix.post_translate(0, -1.0)
        geo.transform(flip_matrix.to_4x4_numpy())

        # Create the final workpiece with the now-normalized, Y-up vectors.
        wp = WorkPiece(name=self.source_file.stem, vectors=geo)
        wp.import_source_uid = source.uid

        # Set the size to the final millimeter dimensions.
        wp.set_size(final_width_mm, final_height_mm)
        wp.pos = (0, 0)

        logger.info(
            f"Workpiece set size: {final_width_mm:.3f}mm x "
            f"{final_height_mm:.3f}mm"
        )

        return [wp]
