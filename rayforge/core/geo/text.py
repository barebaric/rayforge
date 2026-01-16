import cairo
from .geometry import Geometry


def text_to_geometry(
    text: str,
    font_family: str = "sans-serif",
    font_size: float = 10.0,
    is_bold: bool = False,
    is_italic: bool = False,
) -> Geometry:
    """
    Generates a Geometry object representing the vector path of the given text.

    The geometry is generated starting at the origin (0, 0), which corresponds
    to the beginning of the text baseline. The dimensions and coordinates are
    determined by the font metrics and are not normalized.

    Args:
        text: The string content to render.
        font_family: The font family name (e.g., "Arial", "sans-serif").
        font_size: The font size in geometry units (typically mm).
        is_bold: Whether to use a bold weight.
        is_italic: Whether to use an italic slant.

    Returns:
        A Geometry object containing the vector contours of the text.
    """
    if not text:
        return Geometry()

    # Use a RecordingSurface since we only care about the path, not raster
    # pixels. It also handles unbounded drawing operations gracefully.
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(surface)

    # Configure font selection
    slant = cairo.FONT_SLANT_ITALIC if is_italic else cairo.FONT_SLANT_NORMAL
    weight = cairo.FONT_WEIGHT_BOLD if is_bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face(font_family, slant, weight)
    ctx.set_font_size(font_size)

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
