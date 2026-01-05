import logging
import math
from typing import Optional, TYPE_CHECKING, List
import warnings
from ...core.geo import Geometry
from ...core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
)
from ..base_renderer import Renderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass

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
    for cmd_type, x, y, z, i, j, cw in geometry.iter_commands():
        if cmd_type == CMD_TYPE_MOVE:
            path_data.append(f"M {x * width:.3f} {height * (1 - y):.3f}")
        elif cmd_type == CMD_TYPE_LINE:
            path_data.append(f"L {x * width:.3f} {height * (1 - y):.3f}")
        elif cmd_type == CMD_TYPE_ARC:
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
    return " ".join(path_data)


class SketchRenderer(Renderer):
    """
    Renders a sketch's "design view" by generating an in-memory SVG
    and rasterizing it with Vips. It handles both fills and strokes.
    """

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
            # Use svgload_buffer which is highly optimized
            image = pyvips.Image.svgload_buffer(svg_string.encode("utf-8"))
            return image
        except pyvips.Error as e:
            logger.error(f"Failed to render sketch SVG with Vips: {e}")
            logger.debug(f"Failed SVG content:\n{svg_string}")
            return None


# Create a singleton instance for use by the importer
SKETCH_RENDERER = SketchRenderer()
