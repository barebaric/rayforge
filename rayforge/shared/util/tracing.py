import cairo
import numpy as np
import cv2
import potrace
from typing import Tuple, List
import logging
from ...core.geo import Geometry

logger = logging.getLogger(__name__)

BORDER_SIZE = 2
# A safety limit to prevent processing pathologically complex images.
# If potrace generates more paths than this, we fall back to convex hulls.
MAX_VECTORS_LIMIT = 25000


def _get_component_areas(boolean_image: np.ndarray) -> np.ndarray:
    """
    Analyzes a boolean image and returns the areas (in pixels) of all
    distinct components.

    Args:
        boolean_image: A NumPy array of dtype=bool.

    Returns:
        A NumPy array of integer areas for each detected component.
    """
    if not np.any(boolean_image):
        return np.array([], dtype=int)

    # Convert boolean image to uint8 for OpenCV
    img_uint8 = boolean_image.astype(np.uint8)

    # Find and get stats for each component.
    # The first label (0) is the background.
    output = cv2.connectedComponentsWithStats(img_uint8, connectivity=8)
    _num_labels, _labels, stats, _centroids = output

    # We only care about the areas of the actual components,
    # not the background.
    # stats[0] is the background, so we slice from index 1.
    areas = stats[1:, cv2.CC_STAT_AREA]
    return areas


def _find_adaptive_area_threshold(areas: np.ndarray) -> int:
    """
    Analyzes component areas to find a dynamic threshold for separating
    content from noise by identifying the "knee" in the size histogram.

    Args:
        areas: A NumPy array of component areas.

    Returns:
        The minimum area (in pixels) a component should have to be considered
        "content".
    """
    if areas.size == 0:
        return 0

    bin_counts = np.bincount(areas)
    if len(bin_counts) <= 1:
        return 0

    # Find the most frequent area (ignoring background at index 0).
    most_common_area_idx = int(np.argmax(bin_counts[1:])) + 1
    peak_count = bin_counts[most_common_area_idx]

    # FIX: A new, more robust heuristic to detect clean geometric images.
    # If the most common feature is large, it's not noise.
    if most_common_area_idx > 10:
        return 2

    # If the peak count is very low, it's also likely a clean image.
    if peak_count < 10:
        return 2

    drop_off_threshold = peak_count * 0.01

    # Find the last area index that is considered part of the noise cluster.
    # Noise is defined as any area with a count above the drop-off threshold.
    significant_indices = np.where(bin_counts > drop_off_threshold)[0]
    if significant_indices.size > 0:
        # The threshold is one pixel larger than the last noisy area size.
        last_noisy_area = significant_indices[-1]
        threshold = last_noisy_area + 1
    else:
        # Should not happen if peak_count >= 10, but as a fallback:
        threshold = 2

    return max(2, threshold)


def _filter_image_by_component_area(
    boolean_image: np.ndarray, min_area: int
) -> np.ndarray:
    """
    Removes all components from a boolean image that have an area smaller
    than the provided minimum area.

    Args:
        boolean_image: The source boolean image.
        min_area: The minimum number of pixels for a component to be kept.

    Returns:
        A new boolean image with small components removed.
    """
    if min_area <= 1:
        return boolean_image.copy()

    img_uint8 = boolean_image.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_uint8, connectivity=8
    )

    # Create a new blank image
    filtered_image = np.zeros_like(boolean_image, dtype=bool)

    # Iterate through each component (skipping background label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # If the component is large enough, add it to our new image
            filtered_image[labels == i] = True

    return filtered_image


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
    # findContours requires uint8 format
    img_uint8 = boolean_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    def _transform_point(p: Tuple[float, float]) -> Tuple[float, float]:
        px, py = p
        ops_px = px - BORDER_SIZE
        ops_py = height_px - (py - BORDER_SIZE)
        return ops_px / scale_x, ops_py / scale_y

    for contour in contours:
        if len(contour) < 3:
            continue  # A hull requires at least 3 points

        hull_points = cv2.convexHull(contour).squeeze(axis=1)

        geo = Geometry()
        start_pt = _transform_point(tuple(hull_points[0]))
        geo.move_to(start_pt[0], start_pt[1])

        for point in hull_points[1:]:
            pt = _transform_point(tuple(point))
            geo.line_to(pt[0], pt[1])

        geo.close_path()
        geometries.append(geo)

    return geometries


def _get_single_hull_from_image(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> List[Geometry]:
    """
    Fallback for when Potrace fails. Calculates a single convex hull that
    encompasses all content in the image.

    Args:
        boolean_image: The boolean image containing all shapes.
        scale_x: Pixels per millimeter (X).
        scale_y: Pixels per millimeter (Y).
        height_px: Original height of the source surface in pixels.

    Returns:
        A list containing a single Geometry object for the enclosing hull,
        or an empty list if no content was found.
    """
    img_uint8 = boolean_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Combine all points from all contours into a single array
    all_points = np.vstack(contours)
    if len(all_points) < 3:
        return []

    hull_points = cv2.convexHull(all_points).squeeze(axis=1)

    def _transform_point(p: Tuple[float, float]) -> Tuple[float, float]:
        px, py = p
        ops_px = px - BORDER_SIZE
        ops_py = height_px - (py - BORDER_SIZE)
        return ops_px / scale_x, ops_py / scale_y

    geo = Geometry()
    start_pt = _transform_point(tuple(hull_points[0]))
    geo.move_to(start_pt[0], start_pt[1])

    for point in hull_points[1:]:
        pt = _transform_point(tuple(point))
        geo.line_to(pt[0], pt[1])

    geo.close_path()
    return [geo]


def _prepare_surface_for_potrace(surface: cairo.ImageSurface) -> np.ndarray:
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

    # Denoising for images that have actual
    # distinct components (like text on a solid background).
    component_areas = _get_component_areas(boolean_image)
    if component_areas.size > 1:
        min_area_threshold = _find_adaptive_area_threshold(component_areas)
        cleaned_image = _filter_image_by_component_area(
            boolean_image, min_area_threshold
        )
        return cleaned_image

    return boolean_image


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
    cleaned_boolean_image = _prepare_surface_for_potrace(surface)

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
        return _get_single_hull_from_image(
            cleaned_boolean_image,
            pixels_per_mm[0],
            pixels_per_mm[1],
            surface.get_height(),
        )

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
