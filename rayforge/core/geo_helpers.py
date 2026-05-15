import cairo
from raygeo import Geometry
from rayforge.core.font_config import FontConfig
from rayforge.core.text import text_to_geometry


def geometry_from_cairo_path(path_data: cairo.Path) -> Geometry:
    geo = Geometry()
    for path_type, points in path_data:
        if path_type == cairo.PATH_MOVE_TO:
            geo.move_to(points[0], points[1])
        elif path_type == cairo.PATH_LINE_TO:
            geo.line_to(points[0], points[1])
        elif path_type == cairo.PATH_CLOSE_PATH:
            geo.close_path()
    return geo


def geometry_from_text(text: str, font_config=None) -> Geometry:
    if font_config is None:
        font_config = FontConfig()
    base_geo = text_to_geometry(text, font_config=font_config)
    geo = Geometry()
    geo.extend(base_geo)
    return geo
