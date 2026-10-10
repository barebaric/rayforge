"""Dithering algorithms for converting grayscale images to binary."""

from enum import Enum
from gettext import gettext as _

import numpy as np
from raygeo.image.convert import rgba_to_grayscale
from raygeo.image.dither import (
    apply_bayer_dither,
    apply_error_diffusion_dither,
    apply_floyd_steinberg_dither,
    apply_halftone_dither,
    apply_minimum_run_length,
    apply_newsprint_dither,
)

DEFAULT_HALFTONE_CELL_MM = 0.5
DEFAULT_HALFTONE_ANGLE = 45.0


class DitherAlgorithm(Enum):
    FLOYD_STEINBERG = "floyd_steinberg"
    ATKINSON = "atkinson"
    STUCKI = "stucki"
    JARVIS_JUDICE_NINKE = "jarvis"
    SIERRA = "sierra"
    SIERRA_2ROW = "sierra_2row"
    SIERRA_LITE = "sierra_lite"
    BURKES = "burkes"
    BAYER2 = "bayer2"
    BAYER4 = "bayer4"
    BAYER8 = "bayer8"
    NEWSPRINT = "newsprint"
    HALFTONE = "halftone"

    @property
    def display_name(self) -> str:
        names = {
            self.FLOYD_STEINBERG: _("Floyd Steinberg"),
            self.ATKINSON: _("Atkinson"),
            self.STUCKI: _("Stucki"),
            self.JARVIS_JUDICE_NINKE: _("Jarvis, Judice & Ninke"),
            self.SIERRA: _("Sierra"),
            self.SIERRA_2ROW: _("Sierra 2-Row"),
            self.SIERRA_LITE: _("Sierra Lite"),
            self.BURKES: _("Burkes"),
            self.BAYER2: _("Bayer 2"),
            self.BAYER4: _("Bayer 4"),
            self.BAYER8: _("Bayer 8"),
            self.NEWSPRINT: _("Newsprint"),
            self.HALFTONE: _("Halftone"),
        }
        return names[self]

    @property
    def is_error_diffusion(self) -> bool:
        return self in KERNEL_NAMES


#: Raygeo kernel names of the error-diffusion algorithms. The Jarvis,
#: Judice & Ninke kernel is spelled out in raygeo while the enum keeps
#: its shorter historical value.
KERNEL_NAMES: dict[DitherAlgorithm, str] = {
    DitherAlgorithm.FLOYD_STEINBERG: "floyd_steinberg",
    DitherAlgorithm.ATKINSON: "atkinson",
    DitherAlgorithm.STUCKI: "stucki",
    DitherAlgorithm.JARVIS_JUDICE_NINKE: "jarvis_judice_ninke",
    DitherAlgorithm.SIERRA: "sierra",
    DitherAlgorithm.SIERRA_2ROW: "sierra_2row",
    DitherAlgorithm.SIERRA_LITE: "sierra_lite",
    DitherAlgorithm.BURKES: "burkes",
}


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


def grayscale_to_dithered_array(
    grayscale: np.ndarray,
    dither_algorithm: DitherAlgorithm,
    invert: bool = False,
    min_feature_px: int = 1,
    *,
    serpentine: bool = False,
    halftone_cell_mm: float = DEFAULT_HALFTONE_CELL_MM,
    halftone_angle: float = DEFAULT_HALFTONE_ANGLE,
    pixels_per_mm: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """
    Convert a grayscale array to a dithered binary array.

    Args:
        grayscale: 2D uint8 grayscale array (0-255).
        dither_algorithm: The dithering algorithm to use.
        invert: If True, invert the output (engrave light areas).
        min_feature_px: Minimum feature size in pixels.
        serpentine: Alternate the scan direction per row
            (error-diffusion algorithms only).
        halftone_cell_mm: Halftone dot spacing in mm.
        halftone_angle: Halftone screen angle in degrees.
        pixels_per_mm: ``(x, y)`` image resolution, used by the
            screens to keep their cells square in millimetres.

    Returns:
        Binary image where 1 represents areas to engrave.
    """
    if dither_algorithm == DitherAlgorithm.FLOYD_STEINBERG and not serpentine:
        bw_image = apply_floyd_steinberg_dither(grayscale, invert)
        return apply_minimum_run_length(bw_image, min_feature_px)
    if dither_algorithm in BAYER_MATRICES:
        return apply_bayer_dither(
            grayscale,
            BAYER_MATRICES[dither_algorithm],
            invert,
            cell_size=min_feature_px,
        )
    if invert:
        grayscale = 255 - grayscale
    if dither_algorithm == DitherAlgorithm.HALFTONE:
        return apply_halftone_dither(
            grayscale, halftone_cell_mm, halftone_angle, pixels_per_mm
        )
    if dither_algorithm == DitherAlgorithm.NEWSPRINT:
        return apply_newsprint_dither(grayscale, min_feature_px, pixels_per_mm)
    bw_image = apply_error_diffusion_dither(
        grayscale, KERNEL_NAMES[dither_algorithm], serpentine=serpentine
    )
    return apply_minimum_run_length(bw_image, min_feature_px)


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

    return grayscale_to_dithered_array(
        grayscale, dither_algorithm, invert, min_feature_px
    )
