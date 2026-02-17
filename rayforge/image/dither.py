"""Dithering algorithms for converting grayscale images to binary."""

import numpy as np
from enum import Enum


class DitherAlgorithm(Enum):
    FLOYD_STEINBERG = "floyd_steinberg"
    BAYER2 = "bayer2"
    BAYER4 = "bayer4"
    BAYER8 = "bayer8"


BAYER_MATRICES = {
    DitherAlgorithm.BAYER2: np.array([[0, 2], [3, 1]], dtype=np.float32),
    DitherAlgorithm.BAYER4: np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        dtype=np.float32,
    ),
    DitherAlgorithm.BAYER8: np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    ),
}


def apply_floyd_steinberg_dither(
    grayscale: np.ndarray, invert: bool
) -> np.ndarray:
    """
    Apply Floyd-Steinberg error diffusion dithering to a grayscale image.

    Args:
        grayscale: 2D array of grayscale values (0-255).
        invert: If True, invert the output (engrave light areas).

    Returns:
        Binary image where 1 represents areas to engrave.
    """
    height, width = grayscale.shape
    dithered = grayscale.astype(np.float32).copy()

    for y in range(height):
        for x in range(width):
            old_pixel = dithered[y, x]
            new_pixel = 0.0 if old_pixel < 128.0 else 255.0
            dithered[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < width:
                dithered[y, x + 1] += quant_error * 7.0 / 16.0
            if y + 1 < height:
                if x > 0:
                    dithered[y + 1, x - 1] += quant_error * 3.0 / 16.0
                dithered[y + 1, x] += quant_error * 5.0 / 16.0
                if x + 1 < width:
                    dithered[y + 1, x + 1] += quant_error * 1.0 / 16.0

    if invert:
        return (dithered > 128).astype(np.uint8)
    else:
        return (dithered < 128).astype(np.uint8)


def apply_minimum_run_length(
    binary: np.ndarray, min_run_length: int
) -> np.ndarray:
    """
    Ensure all runs of 1s are at least min_run_length pixels long.

    Short runs are removed (set to 0). This prevents the laser from
    attempting to engrave features smaller than its spot size.

    Args:
        binary: 2D binary array.
        min_run_length: Minimum run length in pixels.

    Returns:
        Binary array with short runs removed.
    """
    if min_run_length <= 1:
        return binary

    result = binary.copy()
    height, width = binary.shape

    for y in range(height):
        row = result[y, :]
        x = 0
        while x < width:
            if row[x] == 1:
                run_start = x
                while x < width and row[x] == 1:
                    x += 1
                run_length = x - run_start
                if run_length < min_run_length:
                    row[run_start:x] = 0
            else:
                x += 1

    return result


def apply_bayer_dither(
    grayscale: np.ndarray,
    bayer_matrix: np.ndarray,
    invert: bool,
    cell_size: int = 1,
) -> np.ndarray:
    """
    Apply ordered dithering using a Bayer matrix.

    Args:
        grayscale: 2D array of grayscale values (0-255).
        bayer_matrix: Bayer threshold matrix.
        invert: If True, invert the output (engrave light areas).
        cell_size: Size of each dither cell in pixels. Values > 1 create
            larger threshold regions suitable for coarser laser resolution.

    Returns:
        Binary image where 1 represents areas to engrave.
    """
    height, width = grayscale.shape
    matrix_size = bayer_matrix.shape[0]

    normalized_matrix = (bayer_matrix / (matrix_size * matrix_size)) * 255.0

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            cell_x = (x // cell_size) % matrix_size
            cell_y = (y // cell_size) % matrix_size
            threshold = normalized_matrix[cell_y, cell_x]
            if invert:
                result[y, x] = 1 if grayscale[y, x] > threshold else 0
            else:
                result[y, x] = 1 if grayscale[y, x] < threshold else 0

    return result


def surface_to_dithered_array(
    surface,
    dither_algorithm: DitherAlgorithm,
    invert: bool,
    min_feature_px: int = 1,
) -> np.ndarray:
    """
    Convert Cairo surface to dithered binary array.

    Args:
        surface: Cairo surface in ARGB32 format.
        dither_algorithm: The dithering algorithm to use.
        invert: If True, invert the output (engrave light areas).
        min_feature_px: Minimum feature size in pixels. Ensures dithered
            patterns have features no smaller than this size, suitable
            for the laser's spot size.

    Returns:
        Binary image where 1 represents areas to engrave.
    """
    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    buf = surface.get_data()
    data_with_padding = np.ndarray(
        shape=(height, stride // 4, 4), dtype=np.uint8, buffer=buf
    )
    data = data_with_padding[:, :width, :]

    blue = data[:, :, 0].astype(np.float32)
    green = data[:, :, 1].astype(np.float32)
    red = data[:, :, 2].astype(np.float32)
    alpha = data[:, :, 3].astype(np.float32)

    alpha_safe = np.maximum(alpha, 1e-6)

    red_unpremult = np.clip(red * 255.0 / alpha_safe, 0, 255)
    green_unpremult = np.clip(green * 255.0 / alpha_safe, 0, 255)
    blue_unpremult = np.clip(blue * 255.0 / alpha_safe, 0, 255)

    alpha_normalized = alpha / 255.0
    red_blended = 255.0 - (255.0 - red_unpremult) * alpha_normalized
    green_blended = 255.0 - (255.0 - green_unpremult) * alpha_normalized
    blue_blended = 255.0 - (255.0 - blue_unpremult) * alpha_normalized

    grayscale = (
        0.2989 * red_blended + 0.5870 * green_blended + 0.1140 * blue_blended
    ).astype(np.float32)

    if dither_algorithm == DitherAlgorithm.FLOYD_STEINBERG:
        bw_image = apply_floyd_steinberg_dither(grayscale, invert)
        bw_image = apply_minimum_run_length(bw_image, min_feature_px)
    else:
        bayer_matrix = BAYER_MATRICES[dither_algorithm]
        bw_image = apply_bayer_dither(
            grayscale, bayer_matrix, invert, cell_size=min_feature_px
        )

    bw_image[alpha == 0] = 0
    return bw_image
