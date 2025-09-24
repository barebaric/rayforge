import cairo
import numpy as np
import cv2
import potrace
from typing import Tuple, List
import logging
from ..core.geo import Geometry
from .hull import get_enclosing_hull, polygon_to_geometry
from .denoise import denoise_boolean_image

logger = logging.getLogger(__name__)

BORDER_SIZE = 2
# A safety limit to prevent processing pathologically complex images.
# If potrace generates more paths than this, we fall back to convex hulls.
MAX_VECTORS_LIMIT = 25000


def _get_hulls_from_image(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> List[Geometry]:
    """
    Fallback function: Finds contours in the image, calculates the convex hull
    for each, and returns them as a list of Geometry objects.

    Args:
        boolean_image: The clean boolean image containing only major shapes.
        scale_x: Pixels per millimeter (X).
        scale_y: Pixels per millimeter (Y).
        height_px: Original height of the source surface in pixels.

    Returns:
        A list of Geometry objects, each representing a convex hull.
    """
    geometries = []
    img_uint8 = boolean_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # A hull requires at least 3 points, which is checked inside
        # polygon_to_geometry.
        hull_points = cv2.convexHull(contour)
        geo = polygon_to_geometry(
            hull_points, scale_x, scale_y, height_px, BORDER_SIZE
        )
        if geo:
            geometries.append(geo)

    return geometries


def prepare_surface(surface: cairo.ImageSurface) -> np.ndarray:
    """
    Prepares a Cairo surface for Potrace, including an adaptive denoising
    pipeline to remove small, irrelevant features before tracing.
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

    # Determine if we should use the alpha channel for tracing.
    # This is true only if the image has an alpha channel AND it's not
    # completely opaque (which would make the alpha channel useless).
    use_alpha = channels == 4 and not np.all(img[:, :, 3] == 255)

    if use_alpha:
        # For transparent images, the "background" is transparency.
        # The border must also be transparent to avoid being traced.
        border_color = [0, 0, 0, 0]
        img_with_border = cv2.copyMakeBorder(
            img,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        alpha = img_with_border[:, :, 3]
        boolean_image = alpha > 10
    else:
        # For opaque images, we add a guaranteed white border.
        border_color = [255] * channels
        img_with_border = cv2.copyMakeBorder(
            img,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        gray = cv2.cvtColor(
            img_with_border,
            cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY,
        )

        # Use a background-aware heuristic for all opaque images.
        # This handles both light-on-dark and dark-on-light content.
        otsu_threshold, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # We know the background color because we added a white border.
        # If the background is brighter than the threshold, we trace dark
        # objects. If the background is darker, we trace light objects.
        if 255 > otsu_threshold:
            threshold_type = cv2.THRESH_BINARY_INV
        else:
            threshold_type = cv2.THRESH_BINARY

        _, thresh_img = cv2.threshold(
            gray, otsu_threshold, 255, threshold_type
        )
        boolean_image = thresh_img > 0

    # Delegate all denoising logic to the dedicated function.
    return denoise_boolean_image(boolean_image)


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
                last_cmd = geo.commands[-1]
                if last_cmd.end is None:
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
    Traces a Cairo surface and returns a list of Geometry objects. It now
    includes an adaptive pre-processing step to handle noisy images and a
    fallback mechanism for overly complex vector results.
    """
    cleaned_boolean_image = prepare_surface(surface)

    if not np.any(cleaned_boolean_image):
        return []

    # Use aggressive parameters for high fidelity
    potrace_result = potrace.Bitmap(cleaned_boolean_image).trace(
        turdsize=1,
        opttolerance=0.055,
        alphamax=0,
        turnpolicy=potrace.TURNPOLICY_MINORITY,
    )

    if not potrace_result:
        # If Potrace fails or produces no path for a non-empty image,
        # fall back to returning a single convex hull of the entire shape.
        geo = get_enclosing_hull(
            cleaned_boolean_image,
            pixels_per_mm[0],
            pixels_per_mm[1],
            surface.get_height(),
            BORDER_SIZE,
        )
        return [geo] if geo else []

    # Convert iterable Path to a list to get its length.
    potrace_path_list = list(potrace_result)

    # Fallback Mechanism for excessive complexity
    if len(potrace_path_list) >= MAX_VECTORS_LIMIT:
        # Image is too complex, fall back to convex hulls of major components
        return _get_hulls_from_image(
            cleaned_boolean_image,
            pixels_per_mm[0],
            pixels_per_mm[1],
            surface.get_height(),
        )

    # Normal High-Fidelity Path
    return _curves_to_geometry(
        potrace_path_list,
        pixels_per_mm[0],
        pixels_per_mm[1],
        surface.get_height(),
    )
