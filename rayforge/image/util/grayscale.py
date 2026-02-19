"""
Grayscale and binary image conversion utilities for Cairo surfaces.
"""

from typing import Tuple
import cairo
import numpy


def compute_auto_levels(
    gray_image: numpy.ndarray,
    clip_percent: float = 1.0,
) -> Tuple[int, int]:
    """
    Compute black and white points automatically using percentile clipping.

    Uses a heuristic that clips a percentage of pixels at both ends of the
    histogram to automatically determine the black and white points.

    Args:
        gray_image: uint8 numpy array with grayscale values (0-255).
        clip_percent: Percentage of pixels to clip at each end (0-100).
            Default is 1.0, meaning the bottom 1% and top 1% are clipped.

    Returns:
        Tuple of (black_point, white_point) as integers in range 0-255.
        black_point will be at least 1 less than white_point.
    """
    if gray_image.size == 0:
        return 0, 255

    flat = gray_image.flatten()
    if flat.size == 0:
        return 0, 255

    lower_percentile = clip_percent
    upper_percentile = 100.0 - clip_percent

    black_point = int(numpy.percentile(flat, lower_percentile))
    white_point = int(numpy.percentile(flat, upper_percentile))

    black_point = max(0, min(253, black_point))
    white_point = max(black_point + 2, min(255, white_point))

    return black_point, white_point


def normalize_grayscale(
    gray_image: numpy.ndarray,
    black_point: int = 0,
    white_point: int = 255,
) -> numpy.ndarray:
    """
    Normalize grayscale values based on black and white points.

    Stretches the histogram so that:
    - Values at or below black_point become 0 (black)
    - Values at or above white_point become 255 (white)
    - Values in between are linearly interpolated

    Args:
        gray_image: uint8 numpy array with grayscale values (0-255).
        black_point: Input value that maps to black (0). Default 0.
        white_point: Input value that maps to white (255). Default 255.

    Returns:
        Normalized uint8 numpy array with values 0-255.

    Raises:
        ValueError: If black_point >= white_point.
    """
    if black_point >= white_point:
        raise ValueError(
            f"black_point ({black_point}) must be less than "
            f"white_point ({white_point})"
        )

    clipped = numpy.clip(gray_image, black_point, white_point).astype(
        numpy.float32
    )
    normalized = (clipped - black_point) / (white_point - black_point) * 255.0

    return normalized.astype(numpy.uint8)


def surface_to_grayscale(
    surface: cairo.ImageSurface,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Convert a Cairo ARGB32 surface to a grayscale array with alpha handling.

    Performs proper unpremultiplication of alpha and blends to white
    background for grayscale calculation.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.

    Returns:
        Tuple of (grayscale_array, alpha_array) as numpy arrays.
        grayscale_array: uint8 array with values 0-255.
        alpha_array: float32 array with values 0.0-1.0.
    """
    width_px = surface.get_width()
    height_px = surface.get_height()
    stride = surface.get_stride()
    buf = surface.get_data()
    data_with_padding = numpy.ndarray(
        shape=(height_px, stride // 4, 4), dtype=numpy.uint8, buffer=buf
    )
    data = data_with_padding[:, :width_px, :]

    alpha = data[:, :, 3].astype(numpy.float32) / 255.0

    r = data[:, :, 2].astype(numpy.float32)
    g = data[:, :, 1].astype(numpy.float32)
    b = data[:, :, 0].astype(numpy.float32)

    alpha_safe = numpy.maximum(alpha, 1e-6)

    r_unpremult = r / alpha_safe
    g_unpremult = g / alpha_safe
    b_unpremult = b / alpha_safe

    r_unpremult = numpy.clip(r_unpremult, 0, 255)
    g_unpremult = numpy.clip(g_unpremult, 0, 255)
    b_unpremult = numpy.clip(b_unpremult, 0, 255)

    r_blended = 255.0 - (255.0 - r_unpremult) * alpha
    g_blended = 255.0 - (255.0 - g_unpremult) * alpha
    b_blended = 255.0 - (255.0 - b_unpremult) * alpha

    gray_image = (
        0.2989 * r_blended + 0.5870 * g_blended + 0.1140 * b_blended
    ).astype(numpy.uint8)

    return gray_image, alpha


def surface_to_binary(
    surface: cairo.ImageSurface,
    threshold: int = 128,
    invert: bool = False,
) -> numpy.ndarray:
    """
    Convert a Cairo ARGB32 surface to a binary array using thresholding.

    Converts the surface to grayscale and applies a threshold to produce
    a binary (0 or 1) output. Transparent pixels are always treated as
    white (0).

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        threshold: Brightness value (0-255) for binarization. Pixels with
            grayscale value below this threshold become black (1).
        invert: If True, invert the binarization logic. Pixels above the
            threshold become black (1).

    Returns:
        2D numpy array with values 0 (white/transparent) or 1 (black).

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    width = surface.get_width()
    height = surface.get_height()
    data = numpy.frombuffer(surface.get_data(), dtype=numpy.uint8)
    data = data.reshape((height, width, 4))

    blue = data[:, :, 0]
    green = data[:, :, 1]
    red = data[:, :, 2]
    alpha = data[:, :, 3]

    grayscale = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    if invert:
        binary = (grayscale > threshold).astype(numpy.uint8)
    else:
        binary = (grayscale < threshold).astype(numpy.uint8)

    binary[alpha == 0] = 0
    return binary


def convert_surface_to_grayscale_inplace(
    surface: cairo.ImageSurface,
) -> None:
    """
    Convert a Cairo ARGB32 surface to grayscale in place.

    Modifies the surface directly, converting RGB channels to grayscale
    while preserving the alpha channel.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    width, height = surface.get_width(), surface.get_height()
    data = surface.get_data()
    data_array = numpy.frombuffer(data, dtype=numpy.uint8).reshape(
        (height, width, 4)
    )

    gray = (
        0.299 * data_array[:, :, 2]
        + 0.587 * data_array[:, :, 1]
        + 0.114 * data_array[:, :, 0]
    ).astype(numpy.uint8)

    data_array[:, :, :3] = gray[:, :, None]
