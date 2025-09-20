import io
from typing import Optional, List
import logging
import numpy as np
from svgelements import (
    SVG,
    Path,
    Move,
    Line,
    Close,
    Arc,
    CubicBezier,
    QuadraticBezier,
    Length,
)

from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...shared.util.tracing import trace_surface
from ..base_importer import Importer
from .renderer import SVG_RENDERER

logger = logging.getLogger(__name__)

# A standard fallback conversion factor for pixel units when no other
# context is provided. Corresponds to 96 DPI.
# (1 inch / 96 px) * 25.4 mm/inch
PPI = 96.0
MM_PER_PX_FALLBACK = 25.4 / PPI


class SvgImporter(Importer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List[DocItem]]:
        """
        Generates DocItems from SVG data.

        If vector_config is provided, it renders the SVG to a bitmap and
        traces it. This is robust but may lose fidelity.

        If vector_config is None, it attempts to parse the SVG path and
        shape data directly for a high-fidelity vector import.
        """
        if vector_config is not None:
            # Path 1: Render to bitmap and trace
            return self._get_doc_items_from_trace(vector_config)
        else:
            # Path 2: Direct vector parsing
            return self._get_doc_items_direct()

    def _get_doc_items_from_trace(
        self, vector_config: TraceConfig
    ) -> Optional[List[DocItem]]:
        # This is the original tracing implementation.
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=SVG_RENDERER,
            data=self.raw_data,
        )

        try:
            # Get the SVG's natural size in millimeters.
            size_mm = SVG_RENDERER.get_natural_size(
                wp, px_factor=MM_PER_PX_FALLBACK
            )
            if size_mm and size_mm[0] and size_mm[1]:
                wp.set_size(size_mm[0], size_mm[1])
            else:
                size_mm = None
        except Exception:
            return [wp]

        if not size_mm:
            return [wp]

        w_mm, h_mm = size_mm
        # Render at a reasonable resolution for tracing
        w_px, h_px = 2048, 2048
        surface = SVG_RENDERER.render_to_pixels(wp, w_px, h_px)
        if not surface:
            return [wp]

        pixels_per_mm = (w_px / w_mm, h_px / h_mm)

        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return [wp]

        # Combine all traced paths into a single Geometry object.
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)

        # Update the workpiece with the generated vectors.
        wp.vectors = combined_geo
        return [wp]

    def _get_doc_items_direct(self) -> Optional[List[DocItem]]:
        """
        Parses SVG vector data directly, handling viewBox and unit conversions
        to ensure the vector geometry matches the rendered size.
        """
        try:
            # Correctly wrap the raw byte data in an in-memory stream object.
            svg_stream = io.BytesIO(self.raw_data)
            svg = SVG.parse(svg_stream, ppi=PPI)
        except Exception as e:
            logger.error(f"Failed to parse SVG for direct import: {e}")
            return None

        # --- Establish Authoritative Dimensions and Transformation ---
        # The key to matching the rendered output is to honor the `viewBox`.
        # The `width` and `height` attributes define the final rendered size.
        if svg.viewbox is None or svg.width is None or svg.height is None:
            logger.warning(
                "SVG is missing viewBox, width, or height attributes; "
                "falling back to trace method for direct import."
            )
            # Fallback to tracing if essential attributes for direct import
            # are missing.
            return self._get_doc_items_from_trace(TraceConfig())

        logger.info(f"SVG raw width/height: {svg.width}, {svg.height}")
        logger.info(f"SVG viewBox: {svg.viewbox}")

        # Get full rendered dimensions in millimeters.
        full_width_mm = Length(svg.width).value(ppi=PPI) * MM_PER_PX_FALLBACK
        full_height_mm = Length(svg.height).value(ppi=PPI) * MM_PER_PX_FALLBACK

        logger.info(
            f"Full dimensions: {full_width_mm:.3f}mm x {full_height_mm:.3f}mm"
        )

        # Get margins to trim
        wp_temp = WorkPiece(
            source_file=self.source_file,
            renderer=SVG_RENDERER,
            data=self.raw_data,
        )
        left, top, right, bottom = SVG_RENDERER._get_margins(wp_temp)
        logger.info(
            f"Margins: left={left:.4f}, top={top:.4f}, "
            f"right={right:.4f}, bottom={bottom:.4f}"
        )

        # Calculate trimmed dimensions
        final_width_mm = full_width_mm * (1 - left - right)
        final_height_mm = full_height_mm * (1 - top - bottom)
        logger.info(
            f"Final dimensions: {final_width_mm:.3f}mm x "
            f"{final_height_mm:.3f}mm"
        )

        geo = Geometry()
        # Average scale for tolerance adjustment
        # (using full dimensions as fallback)
        avg_scale = (full_width_mm + full_height_mm) / 2 / 960
        # Assuming typical viewBox size
        tolerance = 0.1 / avg_scale  # 0.1mm in final units
        logger.debug(
            f"Average scale estimate: {avg_scale:.6f}, "
            f"tolerance: {tolerance:.6f}"
        )

        # Track bounds before transformation
        min_x, min_y, max_x, max_y = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
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

                # Update pre-transform bounds
                min_x = min(min_x, end_x)
                min_y = min(min_y, end_y)
                max_x = max(max_x, end_x)
                max_y = max(max_y, end_y)

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
                            # Update bounds for bezier points too
                            min_x = min(min_x, px)
                            min_y = min(min_y, py)
                            max_x = max(max_x, px)
                            max_y = max(max_y, py)

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
            return self._get_doc_items_from_trace(TraceConfig())
            # Fallback

        # Calculate scales based on content bounds to fill the final dimensions
        scale_x = final_width_mm / content_width
        scale_y = final_height_mm / content_height

        logger.debug(
            f"Content-based scale factors: x={scale_x:.6f}, y={scale_y:.6f}"
        )

        # Build 4x4 transformation matrix
        # Order: translate content to origin -> scale -> flip Y ->
        # translate Y to bottom
        t_origin = np.array(
            [[1, 0, 0, -min_x], [0, 1, 0, -min_y], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        s = np.array(
            [
                [scale_x, 0, 0, 0],
                [0, scale_y, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        f_y = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        t_height = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, final_height_mm],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transform_matrix = t_height @ f_y @ s @ t_origin
        logger.debug(f"4x4 Transform matrix:\n{transform_matrix}")

        # Apply the transformation to the entire generated geometry.
        geo.transform(transform_matrix)

        # Get post-transform bounds
        post_min_x, post_min_y, post_max_x, post_max_y = geo.rect()
        logger.info(
            f"Post-transform bounds: x=[{post_min_x:.3f}, "
            f"{post_max_x:.3f}], y=[{post_min_y:.3f}, "
            f"{post_max_y:.3f}]"
        )
        logger.info(
            f"Post-transform size: {post_max_x - post_min_x:.3f} x "
            f"{post_max_y - post_min_y:.3f}"
        )

        # Create the final workpiece.
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=SVG_RENDERER,
            data=self.raw_data,
            vectors=geo,
        )

        # Set the size to the final millimeter dimensions.
        wp.set_size(final_width_mm, final_height_mm)
        wp.pos = (0, 0)  # FileCmd will center it later.

        logger.info(
            f"Workpiece set size: {final_width_mm:.3f}mm x "
            f"{final_height_mm:.3f}mm"
        )
        logger.info(
            f"Vector bounds vs workpiece size: "
            f"{post_max_x - post_min_x:.3f} vs {final_width_mm:.3f} | "
            f"{post_max_y - post_min_y:.3f} vs {final_height_mm:.3f}"
        )

        return [wp]
