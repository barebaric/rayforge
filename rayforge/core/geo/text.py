# flake8: noqa: E402
import cairo
import logging
from gi.repository import Pango, PangoCairo
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

    Uses Pango for text layout so that kerning and other OpenType features
    are applied correctly.  The geometry is generated starting at the
    origin (0, 0), which corresponds to the beginning of the text
    baseline.

    To avoid Pango's integer-rounding hinting at small font sizes (which
    causes glyph overlap), the layout is rendered at a larger internal
    size and then scaled down.

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

    # Render at a larger size to avoid hinting artifacts, then scale down.
    render_scale = max(1.0, 100.0 / font_config.font_size)
    render_size = font_config.font_size * render_scale

    render_config = FontConfig(
        font_family=font_config.font_family,
        font_size=render_size,
        bold=font_config.bold,
        italic=font_config.italic,
    )

    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(surface)

    layout, _ = render_config._create_pango_layout(ctx)
    layout.set_text(text, -1)

    baseline_y = layout.get_baseline() / Pango.SCALE

    PangoCairo.layout_path(ctx, layout)

    raw_path = list(ctx.copy_path())

    # Filter the path to remove redundant or non-geometric commands.
    # 1. Consecutive MOVE_TO commands: keep only the last one
    #    (e.g., 0,0 -> GlyphStart).
    # 2. Trailing MOVE_TO commands: remove them (cursor advance after text).

    clean_path = []
    for i, cmd in enumerate(raw_path):
        type_, points = cmd
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

    inv = 1.0 / render_scale
    baseline_y_scaled = baseline_y * inv

    geo = Geometry()

    for type_, points in clean_path:
        if type_ == cairo.PATH_MOVE_TO:
            geo.move_to(points[0] * inv, points[1] * inv - baseline_y_scaled)
        elif type_ == cairo.PATH_LINE_TO:
            geo.line_to(points[0] * inv, points[1] * inv - baseline_y_scaled)
        elif type_ == cairo.PATH_CURVE_TO:
            geo.bezier_to(
                x=points[4] * inv,
                y=points[5] * inv - baseline_y_scaled,
                c1x=points[0] * inv,
                c1y=points[1] * inv - baseline_y_scaled,
                c2x=points[2] * inv,
                c2y=points[3] * inv - baseline_y_scaled,
            )
        elif type_ == cairo.PATH_CLOSE_PATH:
            geo.close_path()

    return geo
