import cairo
from raygeo.geo import Geometry


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
