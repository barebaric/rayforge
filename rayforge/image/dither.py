"""Dithering algorithms for converting grayscale images to binary."""

import math
from enum import Enum
from gettext import gettext as _

import numpy as np
from raygeo.image.convert import rgba_to_grayscale
from raygeo.image.dither import (
    apply_bayer_dither,
    apply_floyd_steinberg_dither,
    apply_minimum_run_length,
)
from raygeo.image.srgb import srgb_to_linear

DEFAULT_HALFTONE_CELL_MM = 0.5
DEFAULT_HALFTONE_ANGLE = 45.0

#: Rows processed per block when building the halftone screen, to
#: bound the memory of the per-pixel coordinate arrays.
_HALFTONE_BLOCK_ROWS = 256


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
        return self in ERROR_DIFFUSION_KERNELS


#: Error-diffusion kernels as ``(divisor, [(dx, dy, weight), ...])``.
#: ``dx`` is relative to the scan direction, so serpentine rows mirror
#: it. Atkinson deliberately diffuses only 6/8 of the error.
ERROR_DIFFUSION_KERNELS: dict[
    DitherAlgorithm, tuple[int, list[tuple[int, int, int]]]
] = {
    DitherAlgorithm.FLOYD_STEINBERG: (
        16,
        [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)],
    ),
    DitherAlgorithm.ATKINSON: (
        8,
        [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)],
    ),
    DitherAlgorithm.STUCKI: (
        42,
        [
            (1, 0, 8),
            (2, 0, 4),
            (-2, 1, 2),
            (-1, 1, 4),
            (0, 1, 8),
            (1, 1, 4),
            (2, 1, 2),
            (-2, 2, 1),
            (-1, 2, 2),
            (0, 2, 4),
            (1, 2, 2),
            (2, 2, 1),
        ],
    ),
    DitherAlgorithm.JARVIS_JUDICE_NINKE: (
        48,
        [
            (1, 0, 7),
            (2, 0, 5),
            (-2, 1, 3),
            (-1, 1, 5),
            (0, 1, 7),
            (1, 1, 5),
            (2, 1, 3),
            (-2, 2, 1),
            (-1, 2, 3),
            (0, 2, 5),
            (1, 2, 3),
            (2, 2, 1),
        ],
    ),
    DitherAlgorithm.SIERRA: (
        32,
        [
            (1, 0, 5),
            (2, 0, 3),
            (-2, 1, 2),
            (-1, 1, 4),
            (0, 1, 5),
            (1, 1, 4),
            (2, 1, 2),
            (-1, 2, 2),
            (0, 2, 3),
            (1, 2, 2),
        ],
    ),
    DitherAlgorithm.SIERRA_2ROW: (
        16,
        [
            (1, 0, 4),
            (2, 0, 3),
            (-2, 1, 1),
            (-1, 1, 2),
            (0, 1, 3),
            (1, 1, 2),
            (2, 1, 1),
        ],
    ),
    DitherAlgorithm.SIERRA_LITE: (4, [(1, 0, 2), (-1, 1, 1), (0, 1, 1)]),
    DitherAlgorithm.BURKES: (
        32,
        [
            (1, 0, 8),
            (2, 0, 4),
            (-2, 1, 2),
            (-1, 1, 4),
            (0, 1, 8),
            (1, 1, 4),
            (2, 1, 2),
        ],
    ),
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


def _spot_function(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Round-dot screen spot function in cell units, 1 at dot centres
    and 0 halfway between them."""
    return (np.cos(2.0 * np.pi * u) + np.cos(2.0 * np.pi * v)) / 4.0 + 0.5


def _build_spot_cdf(samples: int = 256) -> np.ndarray:
    """Sorted spot values over one cell, used to turn a spot value
    into the fraction of the cell it covers."""
    grid = (np.arange(samples) + 0.5) / samples
    u, v = np.meshgrid(grid, grid)
    return np.sort(_spot_function(u, v).ravel())


def _build_newsprint_matrix(size: int = 8) -> np.ndarray:
    """Clustered-dot threshold ranks with the dots on a 45° lattice."""
    grid = (np.arange(size) + 0.5) / size
    u, v = np.meshgrid(grid, grid)
    spot = _spot_function(u + v, u - v)
    order = np.argsort(-spot.ravel(), kind="stable")
    ranks = np.empty(size * size, dtype=np.int32)
    ranks[order] = np.arange(size * size)
    return ranks.reshape(size, size)


_SPOT_CDF = _build_spot_cdf()
NEWSPRINT_MATRIX = _build_newsprint_matrix()


def _diffuse_row(
    row: np.ndarray,
    forward: tuple[float, float],
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantise one row, pushing the error to the next one or two
    pixels in scan direction. Returns ``(errors, dark)`` in image
    order."""
    values = (row[::-1] if reverse else row).tolist()
    width = len(values)
    values.extend((0.0, 0.0))
    w1, w2 = forward
    errors = [0.0] * width
    dark = [0] * width
    for x in range(width):
        value = values[x]
        if value < 0.5:
            dark[x] = 1
            error = value
        else:
            error = value - 1.0
        errors[x] = error
        values[x + 1] += error * w1
        values[x + 2] += error * w2
    errors_arr = np.asarray(errors)
    dark_arr = np.asarray(dark, dtype=np.uint8)
    if reverse:
        return errors_arr[::-1], dark_arr[::-1]
    return errors_arr, dark_arr


def apply_error_diffusion(
    values: np.ndarray,
    kernel: tuple[int, list[tuple[int, int, int]]],
    serpentine: bool = False,
) -> np.ndarray:
    """
    Error-diffusion dithering with an arbitrary forward kernel.

    The error within a row is propagated pixel by pixel; the error for
    the following rows is spread with one vectorised add per kernel
    tap once the row is done.

    Args:
        values: 2D float array of intensities in [0, 1] (1 = white).
        kernel: ``(divisor, [(dx, dy, weight), ...])`` with
            ``dy == 0`` taps limited to ``dx`` 1 and 2.
        serpentine: Scan odd rows right-to-left, mirroring the kernel.

    Returns:
        Binary uint8 array where 1 marks pixels to engrave.
    """
    height, width = values.shape
    out = np.zeros((height, width), dtype=np.uint8)
    if height == 0 or width == 0:
        return out
    divisor, taps = kernel
    forward = [0.0, 0.0, 0.0]
    later = []
    for dx, dy, weight in taps:
        if dy == 0:
            forward[dx] = weight / divisor
        else:
            later.append((dx, dy, weight / divisor))
    pad = max((abs(dx) for dx, _dy, _w in later), default=0)
    extra_rows = max((dy for _dx, dy, _w in later), default=0)
    work = np.zeros((height + extra_rows, width + 2 * pad))
    work[:height, pad : pad + width] = values
    for y in range(height):
        reverse = serpentine and y % 2 == 1
        errors, dark = _diffuse_row(
            work[y, pad : pad + width], (forward[1], forward[2]), reverse
        )
        out[y] = dark
        for dx, dy, weight in later:
            start = pad + (-dx if reverse else dx)
            work[y + dy, start : start + width] += errors * weight
    return out


def apply_halftone(
    grayscale: np.ndarray,
    cell_mm: float,
    angle_deg: float,
    pixels_per_mm: tuple[float, float],
) -> np.ndarray:
    """
    Amplitude-modulated halftone screen with round dots.

    The screen is laid out in millimetres, so the dots stay round on
    non-square pixels. Dot area tracks darkness: a pixel is engraved
    when its brightness is below the share of the cell that the spot
    function ranks at or under it.

    Args:
        grayscale: 2D uint8 grayscale array (0-255).
        cell_mm: Distance between dot centres in mm.
        angle_deg: Screen angle in degrees.
        pixels_per_mm: ``(x, y)`` image resolution.

    Returns:
        Binary uint8 array where 1 marks pixels to engrave.
    """
    height, width = grayscale.shape
    ppm_x, ppm_y = pixels_per_mm
    if cell_mm <= 0:
        cell_mm = 1.0 / max(ppm_x, ppm_y)
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta) / cell_mm
    sin_t = math.sin(theta) / cell_mm
    x_mm = (np.arange(width) + 0.5) / ppm_x
    out = np.empty((height, width), dtype=np.uint8)
    for top in range(0, height, _HALFTONE_BLOCK_ROWS):
        bottom = min(height, top + _HALFTONE_BLOCK_ROWS)
        y_mm = ((np.arange(top, bottom) + 0.5) / ppm_y)[:, None]
        u = x_mm * cos_t + y_mm * sin_t
        v = y_mm * cos_t - x_mm * sin_t
        spot = _spot_function(u, v)
        coverage = (np.searchsorted(_SPOT_CDF, spot) + 0.5) / _SPOT_CDF.size
        brightness = grayscale[top:bottom] / 255.0
        out[top:bottom] = brightness < coverage
    return out


def _apply_newsprint(
    grayscale: np.ndarray,
    min_feature_px: int,
    pixels_per_mm: tuple[float, float],
) -> np.ndarray:
    """Tile the clustered-dot matrix with physically square cells."""
    height, width = grayscale.shape
    size = NEWSPRINT_MATRIX.shape[0]
    ppm_x, ppm_y = pixels_per_mm
    cell_x = max(1, min_feature_px)
    cell_y = max(1, round(cell_x * ppm_y / ppm_x))
    rows = (np.arange(height) // cell_y) % size
    cols = (np.arange(width) // cell_x) % size
    ranks = NEWSPRINT_MATRIX[rows[:, None], cols[None, :]]
    thresholds = 255.0 * (1.0 - (ranks + 0.5) / (size * size))
    return (grayscale < thresholds).astype(np.uint8)


def _error_diffusion_dither(
    grayscale: np.ndarray,
    dither_algorithm: DitherAlgorithm,
    serpentine: bool,
) -> np.ndarray:
    """Dither in linear light, like raygeo's Floyd-Steinberg."""
    linear = srgb_to_linear(np.ascontiguousarray(grayscale, dtype=np.uint8))
    return apply_error_diffusion(
        linear.astype(np.float64),
        ERROR_DIFFUSION_KERNELS[dither_algorithm],
        serpentine,
    )


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
        return apply_halftone(
            grayscale, halftone_cell_mm, halftone_angle, pixels_per_mm
        )
    if dither_algorithm == DitherAlgorithm.NEWSPRINT:
        return _apply_newsprint(grayscale, min_feature_px, pixels_per_mm)
    bw_image = _error_diffusion_dither(grayscale, dither_algorithm, serpentine)
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
