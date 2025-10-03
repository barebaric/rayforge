import cairo
import numpy as np
import cv2
import vtracer
import xml.etree.ElementTree as ET
import re
from typing import Tuple, List
import logging
from ..core.geo import Geometry
from .hull import get_enclosing_hull, get_hulls_from_image
from .denoise import denoise_boolean_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BORDER_SIZE = 2
# A safety limit to prevent processing pathologically complex images.
# If the generates more paths than this, we fall back to convex hulls.
MAX_VECTORS_LIMIT = 25000


def _get_image_from_surface(
    surface: cairo.ImageSurface,
) -> Tuple[np.ndarray, int]:
    """Extracts image data from a Cairo surface."""
    logger.debug("Entering _get_image_from_surface")
    surface_format = surface.get_format()
    channels = 4 if surface_format == cairo.FORMAT_ARGB32 else 3
    width, height = surface.get_width(), surface.get_height()
    buf = surface.get_data()
    img = (
        np.frombuffer(buf, dtype=np.uint8)
        .reshape(height, width, channels)
        .copy()
    )
    return img, channels


def _get_boolean_image_from_alpha(img: np.ndarray) -> np.ndarray:
    """
    Creates a boolean image from the alpha channel, adding a transparent
    border.
    """
    logger.debug("Entering _get_boolean_image_from_alpha")
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
    return alpha > 10


def _get_boolean_image_from_color(
    img: np.ndarray, channels: int
) -> np.ndarray:
    """
    Creates a boolean image from color channels, adding a white border and
    using Otsu's thresholding.
    """
    logger.debug("Entering _get_boolean_image_from_color")
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
    otsu_threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_type = (
        cv2.THRESH_BINARY_INV if 255 > otsu_threshold else cv2.THRESH_BINARY
    )
    _, thresh_img = cv2.threshold(gray, otsu_threshold, 255, threshold_type)
    return thresh_img > 0


def prepare_surface(surface: cairo.ImageSurface) -> np.ndarray:
    """
    Prepares a Cairo surface for tracing, including an adaptive denoising
    pipeline to remove small, irrelevant features before tracing.
    """
    logger.debug("Entering prepare_surface")
    img, channels = _get_image_from_surface(surface)

    use_alpha = channels == 4 and not np.all(img[:, :, 3] == 255)

    if use_alpha:
        boolean_image = _get_boolean_image_from_alpha(img)
    else:
        boolean_image = _get_boolean_image_from_color(img, channels)

    return denoise_boolean_image(boolean_image)


def _parse_svg_transform(transform_str: str) -> np.ndarray:
    """Parses an SVG transform attribute string (translate only)."""
    logger.debug("Entering _parse_svg_transform")
    matrix = np.identity(3)
    if not transform_str:
        return matrix
    match = re.search(
        r"translate\(\s*([-\d.eE]+)\s*,?\s*([-\d.eE]+)?\s*\)",
        transform_str,
    )
    if match:
        tx = float(match.group(1))
        ty = float(match.group(2)) if match.group(2) is not None else 0.0
        matrix[0, 2] = tx
        matrix[1, 2] = ty
    return matrix


def _apply_svg_transform(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Applies a 3x3 transformation matrix to a 2D point."""
    logger.debug("Entering _apply_svg_transform")
    vec = np.array([point[0], point[1], 1])
    transformed_vec = matrix @ vec
    return transformed_vec[:2]


def _flatten_bezier(
    start: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    end: np.ndarray,
    num_steps=20,
) -> List[np.ndarray]:
    """Flattens a cubic Bezier curve into a list of points."""
    logger.debug("Entering _flatten_bezier")
    points = []
    t_values = np.linspace(0, 1, num_steps)[1:]  # Exclude start point
    for t in t_values:
        one_minus_t = 1 - t
        p = (
            (one_minus_t**3 * start)
            + (3 * one_minus_t**2 * t * c1)
            + (3 * one_minus_t * t**2 * c2)
            + (t**3 * end)
        )
        points.append(p)
    return points


def _transform_point_for_geometry(
    p: Tuple[float, float],
    height_px: int,
    scale_x: float,
    scale_y: float,
) -> Tuple[float, float]:
    """Transforms a point from SVG coordinates to geometry coordinates."""
    logger.debug("Entering _transform_point_for_geometry")
    px, py = p
    ops_px = px - BORDER_SIZE
    ops_py = height_px - (py - BORDER_SIZE)
    return ops_px / scale_x, ops_py / scale_y


def _parse_path_coords(coords_str: str) -> List[float]:
    """Parses coordinate strings from SVG path data."""
    logger.debug("Entering _parse_path_coords")
    return [
        float(c)
        for c in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", coords_str)
    ]


def _process_moveto_command(
    cmd: str,
    coords: List[float],
    current_pos: np.ndarray,
    transform: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> Tuple[Geometry, np.ndarray, np.ndarray]:
    """Processes an SVG 'M' or 'm' command."""
    logger.debug(f"Entering _process_moveto_command with cmd: {cmd}")
    current_geo = Geometry()
    point_coords = coords[0:2]

    if cmd == "m":
        current_pos += np.array(point_coords)
    else:
        current_pos = np.array(point_coords)
    start_of_subpath = current_pos
    tp = _apply_svg_transform(current_pos, transform)
    geo_pt = _transform_point_for_geometry(
        tuple(tp), height_px, scale_x, scale_y
    )
    current_geo.move_to(geo_pt[0], geo_pt[1])

    # Handle implicit lineto commands that can follow a moveto
    for i in range(2, len(coords), 2):
        if cmd == "m":  # Implicit linetos are relative
            current_pos += np.array(coords[i : i + 2])
        else:  # Implicit linetos are absolute
            current_pos = np.array(coords[i : i + 2])
        tp = _apply_svg_transform(current_pos, transform)
        geo_pt = _transform_point_for_geometry(
            tuple(tp), height_px, scale_x, scale_y
        )
        current_geo.line_to(geo_pt[0], geo_pt[1])
    return current_geo, current_pos, start_of_subpath


def _process_lineto_command(
    cmd: str,
    coords: List[float],
    current_pos: np.ndarray,
    current_geo: Geometry,
    transform: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> np.ndarray:
    """Processes SVG 'L', 'l', 'H', 'h', 'V', 'v' commands."""
    logger.debug(f"Entering _process_lineto_command with cmd: {cmd}")
    if cmd == "L":
        current_pos = np.array(coords[0:2])
    elif cmd == "l":
        current_pos += np.array(coords[0:2])
    elif cmd == "H":
        current_pos[0] = coords[0]
    elif cmd == "h":
        current_pos[0] += coords[0]
    elif cmd == "V":
        current_pos[1] = coords[0]
    elif cmd == "v":
        current_pos[1] += coords[0]

    tp = _apply_svg_transform(current_pos, transform)
    geo_pt = _transform_point_for_geometry(
        tuple(tp), height_px, scale_x, scale_y
    )
    current_geo.line_to(geo_pt[0], geo_pt[1])
    return current_pos


def _process_curveto_command(
    cmd: str,
    coords: List[float],
    current_pos: np.ndarray,
    current_geo: Geometry,
    transform: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> np.ndarray:
    """Processes SVG 'C' or 'c' commands."""
    logger.debug(f"Entering _process_curveto_command with cmd: {cmd}")
    if cmd == "C":
        c1, c2, end = (
            np.array(coords[0:2]),
            np.array(coords[2:4]),
            np.array(coords[4:6]),
        )
    else:  # cmd == "c"
        c1 = current_pos + np.array(coords[0:2])
        c2 = current_pos + np.array(coords[2:4])
        end = current_pos + np.array(coords[4:6])

    for p in _flatten_bezier(current_pos, c1, c2, end):
        tp = _apply_svg_transform(p, transform)
        geo_pt = _transform_point_for_geometry(
            tuple(tp), height_px, scale_x, scale_y
        )
        current_geo.line_to(geo_pt[0], geo_pt[1])
    return end


def _process_closepath_command(
    current_geo: Geometry,
    current_pos: np.ndarray,
    start_of_subpath: np.ndarray,
) -> np.ndarray:
    """Processes an SVG 'Z' or 'z' command."""
    logger.debug("Entering _process_closepath_command")
    current_geo.close_path()
    return start_of_subpath


def _parse_svg_path_tokens(path_data: str):
    """Parses SVG path data into command and coordinate tokens."""
    logger.debug("Entering _parse_svg_path_tokens")
    return re.findall(r"([MmLlHhVvCcZz])([^MmLlHhVvCcZz]*)", path_data)


def _svg_path_data_to_geometries(
    path_data: str,
    transform: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> List[Geometry]:
    """
    Parses an SVG path 'd' attribute and converts it into Geometry objects.
    Each subpath (starting with 'M' or 'm') becomes a new Geometry object.
    """
    logger.debug("Entering _svg_path_data_to_geometries")
    geometries = []
    current_geo = None
    current_pos = np.array([0.0, 0.0])
    start_of_subpath = np.array([0.0, 0.0])

    tokens = _parse_svg_path_tokens(path_data)

    for cmd, coords_str in tokens:
        coords = _parse_path_coords(coords_str)

        if cmd.lower() == "m":
            current_geo, current_pos, start_of_subpath = (
                _process_moveto_command(
                    cmd,
                    coords,
                    current_pos,
                    transform,
                    scale_x,
                    scale_y,
                    height_px,
                )
            )
            geometries.append(current_geo)
        elif current_geo is None:
            continue
        elif cmd.lower() in ["l", "h", "v"]:
            current_pos = _process_lineto_command(
                cmd,
                coords,
                current_pos,
                current_geo,
                transform,
                scale_x,
                scale_y,
                height_px,
            )
        elif cmd.lower() == "c":
            current_pos = _process_curveto_command(
                cmd,
                coords,
                current_pos,
                current_geo,
                transform,
                scale_x,
                scale_y,
                height_px,
            )
        elif cmd.lower() == "z":
            current_pos = _process_closepath_command(
                current_geo, current_pos, start_of_subpath
            )

    return [g for g in geometries if g.commands]


def _traverse_svg_node(
    node: ET.Element,
    parent_transform: np.ndarray,
    all_geometries: List[Geometry],
    scale_x: float,
    scale_y: float,
    height_px: int,
):
    """Recursively traverses SVG nodes to extract path data."""
    logger.debug(f"Entering _traverse_svg_node for node: {node.tag}")
    local_transform = _parse_svg_transform(node.get("transform", ""))
    transform = parent_transform @ local_transform
    if node.tag.endswith("path"):
        path_data = node.get("d")
        if path_data:
            geos = _svg_path_data_to_geometries(
                path_data, transform, scale_x, scale_y, height_px
            )
            all_geometries.extend(geos)
    for child in node:
        _traverse_svg_node(
            child, transform, all_geometries, scale_x, scale_y, height_px
        )


def _svg_string_to_geometries(
    svg_str: str,
    scale_x: float,
    scale_y: float,
    height_px: int,
) -> List[Geometry]:
    """
    Parses an SVG string from vtracer and converts all path elements into
    a list of Geometry objects.
    """
    logger.debug("Entering _svg_string_to_geometries")
    all_geometries = []
    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        logger.error("Failed to parse SVG string from vtracer.")
        return []

    _traverse_svg_node(
        root, np.identity(3), all_geometries, scale_x, scale_y, height_px
    )
    return all_geometries


def _fallback_to_enclosing_hull(
    cleaned_boolean_image: np.ndarray,
    pixels_per_mm_x: float,
    pixels_per_mm_y: float,
    surface_height: int,
) -> List[Geometry]:
    """Generates an enclosing hull as a fallback."""
    logger.debug("Entering _fallback_to_enclosing_hull")
    geo = get_enclosing_hull(
        cleaned_boolean_image,
        pixels_per_mm_x,
        pixels_per_mm_y,
        surface_height,
        BORDER_SIZE,
    )
    return [geo] if geo else []


def _encode_image_to_png(
    cleaned_boolean_image: np.ndarray,
) -> Tuple[bool, bytes]:
    """Encodes a boolean image to PNG bytes."""
    logger.debug("Entering _encode_image_to_png")
    img_uint8 = (~cleaned_boolean_image * 255).astype(np.uint8)
    success, png_bytes_buffer = cv2.imencode(".png", img_uint8)
    if not success:
        logger.error("Failed to encode boolean image to PNG for vtracer.")
        return False, b""
    return True, png_bytes_buffer.tobytes()


def _convert_png_to_svg_with_vtracer(png_bytes: bytes) -> str:
    """Converts PNG bytes to SVG string using vtracer."""
    logger.debug("Entering _convert_png_to_svg_with_vtracer")
    return vtracer.convert_raw_image_to_svg(
        img_bytes=png_bytes,
        img_format="png",
        colormode="binary",
        mode="polygon",
        filter_speckle=26,
        length_threshold=3.5,
    )


def _extract_svg_from_raw_output(raw_output: str) -> str:
    """Extracts valid SVG content from vtracer's raw output."""
    logger.debug("Entering _extract_svg_from_raw_output")
    try:
        start = raw_output.index("<svg")
        end = raw_output.rindex("</svg>") + len("</svg>")
        return raw_output[start:end]
    except ValueError:
        logger.warning("Could not find valid <svg> tags in vtracer output.")
        raise


def _count_svg_subpaths(svg_str: str) -> int:
    """Counts the total number of sub-paths in an SVG string."""
    logger.debug("Entering _count_svg_subpaths")
    root = ET.fromstring(svg_str)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall(".//svg:path", ns)
    total_sub_paths = 0
    for path in paths:
        path_data = path.get("d", "")
        count = path_data.count("m") + path_data.count("M")
        total_sub_paths += max(1, count)
    return total_sub_paths


def _fallback_to_hulls_from_image(
    cleaned_boolean_image: np.ndarray,
    surface_height: int,
) -> List[Geometry]:
    """Generates convex hulls from an image as a fallback."""
    logger.debug("Entering _fallback_to_hulls_from_image")
    return get_hulls_from_image(
        cleaned_boolean_image,
        1.0,
        1.0,
        surface_height,
        BORDER_SIZE,
    )


def trace_surface(
    surface: cairo.ImageSurface,
) -> List[Geometry]:
    """
    Traces a Cairo surface and returns a list of Geometry objects. It uses
    vtracer for high-quality vectorization, includes an adaptive pre-processing
    step to handle noisy images, and a fallback mechanism for overly complex
    vector results.
    """
    logger.debug("Entering trace_surface")
    cleaned_boolean_image = prepare_surface(surface)

    if not np.any(cleaned_boolean_image):
        logger.debug("No shapes found in the cleaned image, returning empty.")
        return []

    success, png_bytes = _encode_image_to_png(cleaned_boolean_image)
    if not success:
        return _fallback_to_enclosing_hull(
            cleaned_boolean_image,
            1.0,  # scale_x = 1 (pixel units)
            1.0,  # scale_y = 1 (pixel units)
            surface.get_height(),
        )

    try:
        raw_output = _convert_png_to_svg_with_vtracer(png_bytes)
        svg_str = _extract_svg_from_raw_output(raw_output)
    except Exception as e:
        logger.error(f"vtracer failed: {e}")
        return _fallback_to_enclosing_hull(
            cleaned_boolean_image,
            1.0,  # scale_x = 1 (pixel units)
            1.0,  # scale_y = 1 (pixel units)
            surface.get_height(),
        )

    try:
        total_sub_paths = _count_svg_subpaths(svg_str)
        if total_sub_paths == 0:
            logger.warning(
                "vtracer produced 0 sub-paths, falling back to hulls."
            )
            return _fallback_to_hulls_from_image(
                cleaned_boolean_image,
                surface.get_height(),
            )
        if total_sub_paths >= MAX_VECTORS_LIMIT:
            logger.warning(
                f"vtracer produced {total_sub_paths} sub-paths, exceeding "
                f"limit of {MAX_VECTORS_LIMIT}. Falling back to convex hulls."
            )
            return _fallback_to_hulls_from_image(
                cleaned_boolean_image,
                surface.get_height(),
            )
    except ET.ParseError:
        logger.error("Failed to parse SVG from vtracer, falling back.")
        return _fallback_to_enclosing_hull(
            cleaned_boolean_image,
            1.0,  # scale_x = 1 (pixel units)
            1.0,  # scale_y = 1 (pixel units)
            surface.get_height(),
        )

    return _svg_string_to_geometries(svg_str, 1.0, 1.0, surface.get_height())
