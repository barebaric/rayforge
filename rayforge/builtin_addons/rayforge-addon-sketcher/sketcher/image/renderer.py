import logging
import math
from typing import Optional, TYPE_CHECKING, Tuple
import warnings
from rayforge.core.geo import Geometry
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
)
from rayforge.image.base_renderer import Renderer, RenderSpecification
from rayforge.image.structures import FillRenderData, FillStyle
from rayforge.image.svg.svg_fallback import (
    SVG_LOAD_AVAILABLE,
    render_svg_to_cairo,
    cairo_surface_to_vips,
)
from rayforge.shared.util.colors import ColorRGBA

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from rayforge.core.source_asset_segment import SourceAssetSegment
    from rayforge.core.workpiece import RenderContext
    from rayforge.image.structures import ImportResult


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
    for cmd_type, x, y, z, p1, p2, p3, p4 in geometry.iter_commands():
        if cmd_type == CMD_TYPE_MOVE:
            path_data.append(f"M {x * width:.3f} {height * (1 - y):.3f}")
        elif cmd_type == CMD_TYPE_LINE:
            path_data.append(f"L {x * width:.3f} {height * (1 - y):.3f}")
        elif cmd_type == CMD_TYPE_ARC:
            i, j, cw = p1, p2, p3

            ex_px = x * width
            ey_px = height * (1 - y)

            radius = math.hypot(i, j)
            radius_x_px = radius * width
            radius_y_px = radius * height

            large_arc = 0

            sweep = 1 if bool(cw) else 0

            path_data.append(
                f"A {radius_x_px:.3f} {radius_y_px:.3f} 0 {large_arc} {sweep} "
                f"{ex_px:.3f} {ey_px:.3f}"
            )
        elif cmd_type == CMD_TYPE_BEZIER:
            c1x, c1y, c2x, c2y = p1, p2, p3, p4
            c1x_px = c1x * width
            c1y_px = height * (1 - c1y)
            c2x_px = c2x * width
            c2y_px = height * (1 - c2y)
            ex_px = x * width
            ey_px = height * (1 - y)
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
            data=source_context.data,
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
        from rayforge.core.matrix import Matrix

        vec_result = import_result.vectorization_result
        if not vec_result:
            return None

        merged_boundaries = Geometry()
        for geo in vec_result.geometries_by_layer.values():
            if geo:
                merged_boundaries.extend(geo)

        if merged_boundaries.is_empty():
            return None

        min_x, min_y, max_x, max_y = merged_boundaries.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)
        norm_matrix = Matrix.scale(
            1.0 / width, 1.0 / height
        ) @ Matrix.translation(-min_x, -min_y)
        normalized_boundaries = merged_boundaries.copy()
        normalized_boundaries.transform(norm_matrix.to_4x4_numpy())

        normalized_fills = []
        for fill_list in vec_result.fills_by_layer.values():
            for fill_data in fill_list:
                norm_fill = fill_data.geometry.copy()
                norm_fill.transform(norm_matrix.to_4x4_numpy())
                normalized_fills.append(
                    FillRenderData(
                        geometry=norm_fill,
                        style=fill_data.style,
                        color=fill_data.color,
                        gradient_stops=fill_data.gradient_stops,
                        gradient_angle=fill_data.gradient_angle,
                    )
                )

        return self.render_base_image(
            data=b"",
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
        (FillRenderData or Geometry objects) in kwargs.
        """
        logger.debug(
            f"SketchRenderer.render_base_image called. "
            f"width={width}, height={height}"
        )

        boundaries: Optional[Geometry] = kwargs.get("boundaries")
        fills = kwargs.get("fills")

        if not boundaries and not fills:
            return None

        svg_parts = [
            f'<svg width="{width}" height="{height}" '
            'xmlns="http://www.w3.org/2000/svg">'
        ]

        if fills:
            for fill_data in fills:
                path_d = _geometry_to_svg_path(
                    fill_data.geometry, width, height
                )
                if path_d:
                    fill_svg = self._fill_to_svg(fill_data, path_d)
                    svg_parts.append(fill_svg)

        if boundaries:
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
        svg_bytes = svg_string.encode("utf-8")

        try:
            if SVG_LOAD_AVAILABLE:
                image = pyvips.Image.svgload_buffer(svg_bytes)
            else:
                surface = render_svg_to_cairo(svg_bytes, width, height)
                if not surface:
                    logger.error("Failed to render sketch SVG with Cairo.")
                    logger.debug(f"Failed SVG content:\n{svg_string}")
                    return None
                image = cairo_surface_to_vips(surface)
                if not image:
                    logger.error("Failed to convert Cairo surface to pyvips.")
                    return None
            return image
        except pyvips.Error as e:
            logger.error(f"Failed to render sketch SVG with Vips: {e}")
            logger.debug(f"Failed SVG content:\n{svg_string}")
            return None

    def _fill_to_svg(self, fill_data: FillRenderData, path_d: str) -> str:
        """Convert FillRenderData to an SVG path element."""
        if fill_data.style == FillStyle.SOLID:
            color = self._rgba_to_svg_color(fill_data.color)
            return f'<path d="{path_d}" fill="{color}" stroke="none" />'

        if fill_data.style == FillStyle.LINEAR_GRADIENT:
            return self._linear_gradient_to_svg(fill_data, path_d)

        if fill_data.style == FillStyle.RADIAL_GRADIENT:
            return self._radial_gradient_to_svg(fill_data, path_d)

        color = self._rgba_to_svg_color(fill_data.color)
        return f'<path d="{path_d}" fill="{color}" stroke="none" />'

    def _rgba_to_svg_color(self, color: ColorRGBA) -> str:
        """Convert RGBA tuple to SVG color string."""
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        return f"rgb({r},{g},{b})"

    def _linear_gradient_to_svg(
        self, fill_data: FillRenderData, path_d: str
    ) -> str:
        """Create SVG with linear gradient."""
        import uuid

        grad_id = f"grad_{uuid.uuid4().hex[:8]}"

        angle_rad = math.radians(fill_data.gradient_angle)
        x1 = 50 - 50 * math.cos(angle_rad)
        y1 = 50 - 50 * math.sin(angle_rad)
        x2 = 50 + 50 * math.cos(angle_rad)
        y2 = 50 + 50 * math.sin(angle_rad)

        stops = self._get_gradient_stops(fill_data)
        gradient_svg = (
            f'<linearGradient id="{grad_id}" '
            f'x1="{x1}%" y1="{y1}%" x2="{x2}%" y2="{y2}%" '
            f'gradientUnits="userSpaceOnUse">'
            f"{stops}"
            f"</linearGradient>"
        )

        return (
            f"<defs>{gradient_svg}</defs>"
            f'<path d="{path_d}" fill="url(#{grad_id})" stroke="none" />'
        )

    def _radial_gradient_to_svg(
        self, fill_data: FillRenderData, path_d: str
    ) -> str:
        """Create SVG with radial gradient."""
        import uuid

        grad_id = f"grad_{uuid.uuid4().hex[:8]}"

        stops = self._get_gradient_stops(fill_data)
        gradient_svg = (
            f'<radialGradient id="{grad_id}" '
            f'cx="50%" cy="50%" r="50%" fx="50%" fy="50%">'
            f"{stops}"
            f"</radialGradient>"
        )

        return (
            f"<defs>{gradient_svg}</defs>"
            f'<path d="{path_d}" fill="url(#{grad_id})" stroke="none" />'
        )

    def _get_gradient_stops(self, fill_data: FillRenderData) -> str:
        """Generate SVG gradient stop elements."""
        if not fill_data.gradient_stops:
            color = self._rgba_to_svg_color(fill_data.color)
            opacity = fill_data.color[3]
            return (
                f'<stop offset="0%" stop-color="{color}" '
                f'stop-opacity="{opacity}"/>'
                f'<stop offset="100%" stop-color="{color}" '
                f'stop-opacity="{opacity}"/>'
            )

        stops = []
        for pos, color in fill_data.gradient_stops:
            svg_color = self._rgba_to_svg_color(color)
            opacity = color[3]
            stops.append(
                f'<stop offset="{pos * 100}%" stop-color="{svg_color}" '
                f'stop-opacity="{opacity}"/>'
            )
        return "".join(stops)


SKETCH_RENDERER = SketchRenderer()
