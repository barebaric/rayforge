"""Dithering algorithms for converting grayscale images to binary."""

from enum import Enum

import numpy as np
from raygeo.image.convert import rgba_to_grayscale
from raygeo.image.dither import (
    apply_bayer_dither,
    apply_floyd_steinberg_dither,
    apply_minimum_run_length,
)


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
        min_feature_px: Minimum feature size in pixels.

    Returns:
        Binary image where 1 represents areas to engrave.
    """
    width = surface.get_width()
    height = surface.get_height()
    stride_px = surface.get_stride() // 4
    buf = np.frombuffer(surface.get_data(), dtype=np.uint8).copy()

    grayscale, _ = rgba_to_grayscale(buf, width, height, stride_px)

    if dither_algorithm == DitherAlgorithm.FLOYD_STEINBERG:
        bw_image = apply_floyd_steinberg_dither(grayscale, invert)
        bw_image = apply_minimum_run_length(bw_image, min_feature_px)
    else:
        bayer_matrix = BAYER_MATRICES[dither_algorithm]
        bw_image = apply_bayer_dither(
            grayscale, bayer_matrix, invert, cell_size=min_feature_px
        )

    return bw_image
