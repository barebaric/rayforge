import logging
import math
from typing import Optional, TYPE_CHECKING, List, Tuple
import warnings
from ...core.geo import Geometry
from ...core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
)
from ..base_renderer import Renderer, RenderSpecification

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.workpiece import RenderContext
    from ...image.structures import ImportResult


logger = logging.getLogger(__name__)


def _geometry_to_svg_path(
    geometry: Geometry,
    width: int,
    height: int,
    stroke_width: float = 1.0,
) -> str:
    """
    Converts a normalized (0-1) Geometry object into an SVG path string,
    scaled to the target pixel dimensions.
    """
    path_data = []
    # Cairo/SVG use Y-down, but our normalized geometry is Y-up. We flip Y.
    # Flip matrix: Scale Y by -1, then translate by height.
    # y' = -y * height + height = height * (1 - y)
    for cmd_type, x, y, z, p1, p2, p3, p4 in geometry.iter_commands():
        if cmd_type == CMD_TYPE_MOVE:
            path_data.append(f"M {x * width:.3f} {height * (1 - y):.3f}")
        elif cmd_type == CMD_TYPE_LINE:
            path_data.append(f"L {x * width:.3f} {height * (1 - y):.3f}")
        elif cmd_type == CMD_TYPE_ARC:
            i, j, cw = p1, p2, p3  # Unpack arc params from generic slots
            # This requires converting center-offset format to SVG's
            # endpoint + radius format.

            ex_px = x * width
            ey_px = height * (1 - y)

            radius = math.hypot(i, j)
            radius_x_px = radius * width
            radius_y_px = radius * height

            # Large arc flag is 1 if sweep is > 180 degrees.
            # For sketches, we assume arcs are <= 180.
            large_arc = 0

            # Sweep flag in SVG (Y-down):
            # 1: Positive angle direction (Clockwise)
            # 0: Negative angle direction (Counter-Clockwise)
            #
            # Since visual direction is preserved (Top stays Top),
            # Source CW (True) maps to SVG CW (1).
            sweep = 1 if bool(cw) else 0

            path_data.append(
                f"A {radius_x_px:.3f} {radius_y_px:.3f} 0 {large_arc} {sweep} "
                f"{ex_px:.3f} {ey_px:.3f}"
            )
        elif cmd_type == CMD_TYPE_BEZIER:
            c1x, c1y, c2x, c2y = p1, p2, p3, p4
            c1x_px = c1x * width
            c1y_px = height * (1 - c1y)  # Y-flip
            c2x_px = c2x * width
            c2y_px = height * (1 - c2y)  # Y-flip
            ex_px = x * width
            ey_px = height * (1 - y)  # Y-flip
            path_data.append(
                f"C {c1x_px:.3f} {c1y_px:.3f} {c2x_px:.3f} {c2y_px:.3f} "
                f"{ex_px:.3f} {ey_px:.3f}"
            )

    return " ".join(path_data)


class SketchRenderer(Renderer):
    """
    Renders a sketch's "design view" by generating an in-memory SVG
    and rasterizing it with Vips. It handles both fills and strokes.
    """

    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: Tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        """
        Specifies that 'boundaries' and 'fills' geometries are required for
        rendering sketches.
        """
        kwargs = {
            "boundaries": source_context.boundaries,
            "fills": source_context.fills,
        }
        return RenderSpecification(
            width=target_size[0],
            height=target_size[1],
            data=source_context.data,  # data is b"" for sketches
            kwargs=kwargs,
            apply_mask=False,
        )

    def render_preview_image(
        self,
        import_result: "ImportResult",
        target_width: int,
        target_height: int,
    ) -> Optional[pyvips.Image]:
        """
        Generates a preview by rendering the sketch's vectorized geometry.
        """
        from ...core.matrix import Matrix

        vec_result = import_result.vectorization_result
        if not vec_result:
            return None

        # The geometry from the importer is in the sketch's native (mm)
        # coordinate system. We need to normalize it to a 1x1 box for the
        # SVG path conversion helper.
        merged_boundaries = Geometry()
        for geo in vec_result.geometries_by_layer.values():
            if geo:
                merged_boundaries.extend(geo)

        if merged_boundaries.is_empty():
            return None

        # Normalize boundaries
        min_x, min_y, max_x, max_y = merged_boundaries.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)
        norm_matrix = Matrix.scale(
            1.0 / width, 1.0 / height
        ) @ Matrix.translation(-min_x, -min_y)
        normalized_boundaries = merged_boundaries.copy()
        normalized_boundaries.transform(norm_matrix.to_4x4_numpy())

        # Normalize fills using the same transformation matrix
        normalized_fills = []
        for fill_list in vec_result.fills_by_layer.values():
            for fill_geo in fill_list:
                norm_fill = fill_geo.copy()
                norm_fill.transform(norm_matrix.to_4x4_numpy())
                normalized_fills.append(norm_fill)

        return self.render_base_image(
            data=b"",  # Data is not used for sketch rendering
            width=target_width,
            height=target_height,
            boundaries=normalized_boundaries,
            fills=normalized_fills,
        )

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        """
        Renders the sketch's vector data to a pyvips Image.
        It expects 'boundaries' (strokes) and optionally 'fills'
        (solid regions) in kwargs, as Geometry objects.
        """
        logger.debug(
            f"SketchRenderer.render_base_image called. "
            f"width={width}, height={height}"
        )

        boundaries: Optional[Geometry] = kwargs.get("boundaries")
        fills: Optional[List[Geometry]] = kwargs.get("fills")

        if not boundaries and not fills:
            return pyvips.Image.black(width, height)

        svg_parts = [
            f'<svg width="{width}" height="{height}" '
            'xmlns="http://www.w3.org/2000/svg">'
        ]

        # 1. Render Fills first (as dark grey shapes, no stroke)
        if fills:
            for fill_geo in fills:
                path_d = _geometry_to_svg_path(fill_geo, width, height)
                if path_d:
                    svg_parts.append(
                        f'<path d="{path_d}" fill="#1A1A1A" stroke="none" />'
                    )

        # 2. Render Boundaries on top (as black strokes, no fill)
        if boundaries:
            # Scale stroke width to be roughly 1px regardless of size
            stroke_width = 1.0
            path_d = _geometry_to_svg_path(
                boundaries, width, height, stroke_width=stroke_width
            )
            if path_d:
                svg_parts.append(
                    f'<path d="{path_d}" fill="none" stroke="black" '
                    f'stroke-width="{stroke_width}" stroke-linecap="round" '
                    'stroke-linejoin="round" />'
                )

        svg_parts.append("</svg>")
        svg_string = "".join(svg_parts)

        try:
            try:
                import cairosvg

                png_bytes = cairosvg.svg2png(
                    bytestring=svg_string.encode("utf-8"),
                    output_width=width,
                    output_height=height,
                )
                return pyvips.Image.pngload_buffer(
                    png_bytes, access=pyvips.Access.RANDOM
                )
            except ImportError:
                logger.error("CairoSVG is not available for sketch rendering.")
            except (pyvips.Error, ValueError, TypeError, Exception) as e:
                logger.error(
                    "CairoSVG fallback failed to render sketch SVG: %s",
                    e,
                    exc_info=True,
                )

            try:
                svg_loader = getattr(pyvips.Image, "svgload_buffer")
            except AttributeError:
                svg_loader = None

            if svg_loader:
                return svg_loader(svg_string.encode("utf-8"))

            logger.error(
                "No SVG renderer succeeded for sketch "
                "(CairoSVG/libvips unavailable)."
            )
        except pyvips.Error as e:
            logger.error(f"Failed to render sketch SVG with Vips: {e}")
            logger.debug(f"Failed SVG content:\n{svg_string}")
            return None


# Create a singleton instance for use by the importer
SKETCH_RENDERER = SketchRenderer()
