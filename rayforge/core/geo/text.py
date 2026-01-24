# flake8: noqa: E402
import cairo
import logging
import gi

gi.require_version("PangoCairo", "1.0")

from gi.repository import PangoCairo
from .geometry import Geometry
from .font_config import FontConfig
from typing import Optional, List

logger = logging.getLogger(__name__)


def get_available_font_families() -> List[str]:
    """
    Get a list of available font families from the system.

    This function uses Pango to discover available fonts in a
    platform-independent way. It returns a sorted list of font family
    names including generic font families (sans-serif, serif, monospace)
    and all available system fonts.

    Returns:
        A sorted list of font family names available on the system.
    """
    font_map = PangoCairo.font_map_get_default()
    families = []

    try:
        font_families = font_map.list_families()
        for family in font_families:
            name = family.get_name()
            if name and name not in families:
                families.append(name)
    except Exception as e:
        logger.warning(f"Error getting font families: {e}")
        return _get_fallback_fonts()

    families.sort(key=str.lower)

    generic_fonts = ["sans-serif", "serif", "monospace"]
    for generic in generic_fonts:
        if generic not in families:
            families.insert(0, generic)

    return families


def _get_fallback_fonts() -> List[str]:
    """
    Get a fallback list of fonts when system font discovery fails.

    Returns:
        A list of common font family names as fallback.
    """
    return [
        "sans-serif",
        "serif",
        "monospace",
        "Arial",
        "Helvetica",
        "Times New Roman",
        "Courier New",
        "Verdana",
        "Georgia",
        "Palatino",
        "Garamond",
        "Bookman",
        "Comic Sans MS",
        "Trebuchet MS",
        "Arial Black",
        "Impact",
    ]


def text_to_geometry(
    text: str,
    font_config: Optional[FontConfig] = None,
) -> Geometry:
    """
    Generates a Geometry object representing the vector path of the given text.

    The geometry is generated starting at the origin (0, 0), which corresponds
    to the beginning of the text baseline. The dimensions and coordinates are
    determined by the font metrics and are not normalized.

    Args:
        text: The string content to render.
        font_config: The font configuration to use.

    Returns:
        A Geometry object containing the vector contours of the text.
    """
    if font_config is None:
        font_config = FontConfig()

    if not text:
        return Geometry()

    # Use a RecordingSurface since we only care about the path, not raster
    # pixels. It also handles unbounded drawing operations gracefully.
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(surface)

    # Configure font selection
    slant = (
        cairo.FONT_SLANT_ITALIC
        if font_config.italic
        else cairo.FONT_SLANT_NORMAL
    )
    weight = (
        cairo.FONT_WEIGHT_BOLD
        if font_config.bold
        else cairo.FONT_WEIGHT_NORMAL
    )
    ctx.select_font_face(font_config.font_family, slant, weight)
    ctx.set_font_size(font_config.font_size)

    # Position at origin. Text is drawn relative to the baseline.
    ctx.move_to(0, 0)
    ctx.text_path(text)

    # Use copy_path() to retrieve the path with curves preserved.
    raw_path = list(ctx.copy_path())

    # Filter the path to remove redundant or non-geometric commands.
    # 1. Consecutive MOVE_TO commands: keep only the last one
    #    (e.g., 0,0 -> GlyphStart).
    # 2. Trailing MOVE_TO commands: remove them (cursor advance after text).

    clean_path = []
    for i, cmd in enumerate(raw_path):
        type_, _ = cmd
        if type_ == cairo.PATH_MOVE_TO:
            # If next command is also MOVE_TO, skip this one
            if (
                i + 1 < len(raw_path)
                and raw_path[i + 1][0] == cairo.PATH_MOVE_TO
            ):
                continue
        clean_path.append(cmd)

    # Remove trailing Move if present (often added by Cairo for text cursor
    # position)
    if clean_path and clean_path[-1][0] == cairo.PATH_MOVE_TO:
        clean_path.pop()

    geo = Geometry()

    for type_, points in clean_path:
        if type_ == cairo.PATH_MOVE_TO:
            geo.move_to(points[0], points[1])
        elif type_ == cairo.PATH_LINE_TO:
            geo.line_to(points[0], points[1])
        elif type_ == cairo.PATH_CURVE_TO:
            # Cairo curves are cubic Béziers.
            # Points layout: (x1, y1, x2, y2, x3, y3)
            # Control Point 1: (points[0], points[1])
            # Control Point 2: (points[2], points[3])
            # End Point:       (points[4], points[5])
            geo.bezier_to(
                x=points[4],
                y=points[5],
                c1x=points[0],
                c1y=points[1],
                c2x=points[2],
                c2y=points[3],
            )
        elif type_ == cairo.PATH_CLOSE_PATH:
            geo.close_path()

    return geo
