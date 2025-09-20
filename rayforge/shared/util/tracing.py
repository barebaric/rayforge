import cairo
import numpy as np
import cv2
import potrace
from typing import Tuple, List
from ...core.geo import Geometry

BORDER_SIZE = 2


def _prepare_surface_for_potrace(surface: cairo.ImageSurface) -> np.ndarray:
    """
    Prepares a Cairo surface for Potrace by converting it to a NumPy
    array of dtype=bool. Dark areas of the source image will be `True`.
    """
    surface_format = surface.get_format()
    channels = 4 if surface_format == cairo.FORMAT_ARGB32 else 3

    width, height = surface.get_width(), surface.get_height()
    buf = surface.get_data()
    img = (
        np.frombuffer(buf, dtype=np.uint8)
        .reshape(height, width, channels)
        .copy()
    )

    border_color = [255] * channels
    img = cv2.copyMakeBorder(
        img,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )

    if channels == 4:
        alpha = img[:, :, 3]
        img[alpha == 0] = 255

    gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY if channels == 3 else cv2.COLOR_BGRA2GRAY
    )
    return gray < 128


def _curves_to_geometry(
    curves: List[potrace.Curve], scale_x: float, scale_y: float, height_px: int
) -> List[Geometry]:
    """
    Converts Potrace curves into a list of separate Geometry objects, scaled
    to millimeter units.
    """
    geometries = []

    def _transform_point(p: Tuple[float, float]) -> Tuple[float, float]:
        px, py = p
        ops_px = px - BORDER_SIZE
        ops_py = height_px - (py - BORDER_SIZE)
        return ops_px / scale_x, ops_py / scale_y

    for curve in curves:
        geo = Geometry()
        start_pt = _transform_point(curve.start_point)
        geo.move_to(start_pt[0], start_pt[1])

        for segment in curve:
            if segment.is_corner:
                c = _transform_point(segment.c)
                end = _transform_point(segment.end_point)
                geo.line_to(c[0], c[1])
                geo.line_to(end[0], end[1])
            else:
                # Approximate bezier curve with line segments
                last_cmd = geo.commands[-1]
                if last_cmd.end is None:
                    # This should not happen in a valid path, but it makes
                    # the code robust and satisfies the type checker.
                    continue
                start_ops = last_cmd.end[:2]

                start_px = np.array(
                    [
                        (start_ops[0] * scale_x) + BORDER_SIZE,
                        (height_px - (start_ops[1] * scale_y)) + BORDER_SIZE,
                    ]
                )
                c1_px = np.array(segment.c1)
                c2_px = np.array(segment.c2)
                end_px = np.array(segment.end_point)

                for t in np.linspace(0, 1, 20)[1:]:
                    p_px = (
                        (1 - t) ** 3 * start_px
                        + 3 * (1 - t) ** 2 * t * c1_px
                        + 3 * (1 - t) * t**2 * c2_px
                        + t**3 * end_px
                    )
                    pt = _transform_point(tuple(p_px))
                    geo.line_to(pt[0], pt[1])

        geo.close_path()
        geometries.append(geo)

    return geometries


def trace_surface(
    surface: cairo.ImageSurface, pixels_per_mm: Tuple[float, float]
) -> List[Geometry]:
    """
    Traces a Cairo surface and returns a list of Geometry objects, one for
    each distinct path found. Coordinates are in millimeters.
    """
    boolean_image = _prepare_surface_for_potrace(surface)

    # Use aggressive parameters for high fidelity
    potrace_path = potrace.Bitmap(boolean_image).trace(
        turdsize=1,
        opttolerance=0.055,
        alphamax=0,
        turnpolicy=potrace.TURNPOLICY_MINORITY,
    )

    if not potrace_path:
        return []

    return _curves_to_geometry(
        list(potrace_path),
        pixels_per_mm[0],
        pixels_per_mm[1],
        surface.get_height(),
    )
