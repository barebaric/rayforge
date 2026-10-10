"""Tests for the error-diffusion and screen dithering algorithms."""

from typing import cast

import numpy as np
import pytest
from raygeo.image.dither import (
    apply_floyd_steinberg_dither,
    apply_halftone_dither,
    apply_minimum_run_length,
)
from raygeo.image.srgb import srgb_to_linear
from scipy import ndimage

from rayforge.image.dither import DitherAlgorithm, grayscale_to_dithered_array

ERROR_DIFFUSION = [
    DitherAlgorithm.FLOYD_STEINBERG,
    DitherAlgorithm.ATKINSON,
    DitherAlgorithm.STUCKI,
    DitherAlgorithm.JARVIS_JUDICE_NINKE,
    DitherAlgorithm.SIERRA,
    DitherAlgorithm.SIERRA_2ROW,
    DitherAlgorithm.SIERRA_LITE,
    DitherAlgorithm.BURKES,
]

BAYER = [
    DitherAlgorithm.BAYER2,
    DitherAlgorithm.BAYER4,
    DitherAlgorithm.BAYER8,
]

NEW_ALGORITHMS = [
    a
    for a in DitherAlgorithm
    if a is not DitherAlgorithm.FLOYD_STEINBERG and a not in BAYER
]


def _label(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected dots of a binary image and their count."""
    return cast(tuple[np.ndarray, int], ndimage.label(image))


class TestAlgorithmCatalogue:
    def test_new_algorithms_exist_with_stable_values(self):
        assert DitherAlgorithm.ATKINSON.value == "atkinson"
        assert DitherAlgorithm.STUCKI.value == "stucki"
        assert DitherAlgorithm.JARVIS_JUDICE_NINKE.value == "jarvis"
        assert DitherAlgorithm.SIERRA.value == "sierra"
        assert DitherAlgorithm.SIERRA_2ROW.value == "sierra_2row"
        assert DitherAlgorithm.SIERRA_LITE.value == "sierra_lite"
        assert DitherAlgorithm.BURKES.value == "burkes"
        assert DitherAlgorithm.NEWSPRINT.value == "newsprint"
        assert DitherAlgorithm.HALFTONE.value == "halftone"

    def test_every_algorithm_has_a_display_name(self):
        for algo in DitherAlgorithm:
            assert algo.display_name

    def test_error_diffusion_flag(self):
        for algo in DitherAlgorithm:
            assert algo.is_error_diffusion == (algo in ERROR_DIFFUSION)


class TestGrayscaleToDitheredArray:
    @pytest.mark.parametrize("algo", NEW_ALGORITHMS)
    def test_white_is_never_engraved(self, algo):
        white = np.full((24, 24), 255, dtype=np.uint8)
        result = grayscale_to_dithered_array(white, algo)
        assert result.dtype == np.uint8
        assert result.shape == white.shape
        assert not result.any()

    @pytest.mark.parametrize("algo", NEW_ALGORITHMS)
    def test_black_is_fully_engraved(self, algo):
        black = np.zeros((24, 24), dtype=np.uint8)
        result = grayscale_to_dithered_array(black, algo)
        assert result.all()

    @pytest.mark.parametrize("algo", NEW_ALGORITHMS)
    def test_output_is_binary_and_deterministic(self, algo):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (31, 47)).astype(np.uint8)
        first = grayscale_to_dithered_array(img, algo)
        second = grayscale_to_dithered_array(img.copy(), algo)
        assert set(np.unique(first)) <= {0, 1}
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("algo", NEW_ALGORITHMS)
    def test_invert_engraves_the_complement(self, algo):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, (20, 30)).astype(np.uint8)
        inverted = grayscale_to_dithered_array(img, algo, invert=True)
        expected = grayscale_to_dithered_array(255 - img, algo)
        np.testing.assert_array_equal(inverted, expected)

    @pytest.mark.parametrize(
        "algo",
        [a for a in ERROR_DIFFUSION if a is not DitherAlgorithm.ATKINSON],
    )
    @pytest.mark.parametrize("gray", [40, 128, 200])
    def test_error_diffusion_preserves_linear_tone(self, algo, gray):
        img = np.full((64, 96), gray, dtype=np.uint8)
        result = grayscale_to_dithered_array(img, algo)
        expected = 1.0 - float(
            srgb_to_linear(np.array([gray], dtype=np.uint8))[0]
        )
        assert result.mean() == pytest.approx(expected, abs=0.02)

    def test_atkinson_loses_a_quarter_of_the_error(self):
        """Atkinson only diffuses 6/8 of the error, which clips dark
        and light tones and raises contrast compared to FS."""
        img = np.full((64, 64), 230, dtype=np.uint8)
        atkinson = grayscale_to_dithered_array(img, DitherAlgorithm.ATKINSON)
        stucki = grayscale_to_dithered_array(img, DitherAlgorithm.STUCKI)
        assert atkinson.mean() < stucki.mean()

    def test_floyd_steinberg_without_serpentine_keeps_raygeo(self):
        rng = np.random.default_rng(11)
        img = rng.integers(0, 256, (40, 60)).astype(np.uint8)
        result = grayscale_to_dithered_array(
            img, DitherAlgorithm.FLOYD_STEINBERG, min_feature_px=2
        )
        expected = apply_minimum_run_length(
            apply_floyd_steinberg_dither(img, False), 2
        )
        np.testing.assert_array_equal(result, expected)

    def test_floyd_steinberg_serpentine_differs_from_raygeo(self):
        img = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (32, 1))
        plain = grayscale_to_dithered_array(
            img, DitherAlgorithm.FLOYD_STEINBERG
        )
        serpentine = grayscale_to_dithered_array(
            img, DitherAlgorithm.FLOYD_STEINBERG, serpentine=True
        )
        assert not np.array_equal(plain, serpentine)
        assert serpentine.mean() == pytest.approx(plain.mean(), abs=0.02)

    def test_enum_values_translate_to_raygeo_kernels(self):
        """The enum values are not raygeo kernel names (Jarvis, Judice
        & Ninke in particular); the dispatch must translate them."""
        img = np.full((16, 16), 100, dtype=np.uint8)
        for algo in ERROR_DIFFUSION:
            assert grayscale_to_dithered_array(img, algo).shape == (16, 16)

    @pytest.mark.parametrize("algo", ERROR_DIFFUSION[1:])
    def test_minimum_feature_size_removes_short_runs(self, algo):
        img = np.full((32, 64), 200, dtype=np.uint8)
        result = grayscale_to_dithered_array(img, algo, min_feature_px=3)
        for row in result:
            padded = np.concatenate(([0], row, [0]))
            edges = np.flatnonzero(np.diff(padded))
            runs = edges[1::2] - edges[::2]
            assert (runs >= 3).all()

    def test_serpentine_is_ignored_by_screens(self):
        img = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (32, 1))
        for algo in (DitherAlgorithm.BAYER4, DitherAlgorithm.NEWSPRINT):
            np.testing.assert_array_equal(
                grayscale_to_dithered_array(img, algo),
                grayscale_to_dithered_array(img, algo, serpentine=True),
            )


class TestHalftone:
    def _halftone(self, img, cell_mm, angle_deg, pixels_per_mm):
        return grayscale_to_dithered_array(
            img,
            DitherAlgorithm.HALFTONE,
            halftone_cell_mm=cell_mm,
            halftone_angle=angle_deg,
            pixels_per_mm=pixels_per_mm,
        )

    @pytest.mark.parametrize("gray", [30, 100, 160, 220])
    def test_coverage_follows_darkness(self, gray):
        img = np.full((200, 200), gray, dtype=np.uint8)
        result = self._halftone(img, 1.0, 45.0, (10.0, 10.0))
        assert result.mean() == pytest.approx(1.0 - gray / 255.0, abs=0.03)

    def test_dot_count_follows_cell_size(self):
        """At 0° the dot centres sit on whole cells from the origin, so
        a 20 mm square holds 21 x 21 (partly clipped) dots at 1 mm."""
        img = np.full((200, 200), 230, dtype=np.uint8)
        for cell_mm, expected in ((1.0, 441), (2.0, 121)):
            result = self._halftone(img, cell_mm, 0.0, (10.0, 10.0))
            _labels, count = _label(result)
            assert count == pytest.approx(expected, rel=0.1)

    def test_dots_are_round_on_anisotropic_pixels(self):
        """X is sampled twice as densely as Y; the dots must still be
        round in millimetres, so twice as wide in pixels."""
        img = np.full((100, 200), 220, dtype=np.uint8)
        result = self._halftone(img, 2.0, 0.0, (20.0, 10.0))
        labels, _count = _label(result)
        boxes = ndimage.find_objects(labels)
        interior = [
            b
            for b in boxes
            if b[0].start > 0
            and b[1].start > 0
            and b[0].stop < img.shape[0]
            and b[1].stop < img.shape[1]
        ]
        assert interior
        for rows, cols in interior:
            height = rows.stop - rows.start
            width = cols.stop - cols.start
            assert width == pytest.approx(2 * height, abs=2)

    def test_angle_rotates_the_screen(self):
        img = np.full((120, 120), 200, dtype=np.uint8)
        straight = self._halftone(img, 1.0, 0.0, (10.0, 10.0))
        rotated = self._halftone(img, 1.0, 45.0, (10.0, 10.0))
        assert not np.array_equal(straight, rotated)
        assert rotated.mean() == pytest.approx(straight.mean(), abs=0.03)

    def test_invalid_cell_size_falls_back_to_one_pixel_cells(self):
        img = np.full((10, 10), 0, dtype=np.uint8)
        result = self._halftone(img, 0.0, 0.0, (10.0, 10.0))
        assert result.all()

    def test_dispatch_forwards_the_screen_settings_to_raygeo(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        dispatched = self._halftone(img, 0.8, 30.0, (10.0, 10.0))
        expected = apply_halftone_dither(
            img, 0.8, 30.0, pixels_per_mm=(10.0, 10.0)
        )
        np.testing.assert_array_equal(dispatched, expected)


class TestNewsprint:
    def test_coverage_follows_darkness(self):
        img = np.full((64, 64), 128, dtype=np.uint8)
        result = grayscale_to_dithered_array(img, DitherAlgorithm.NEWSPRINT)
        assert result.mean() == pytest.approx(1.0 - 128 / 255.0, abs=0.03)

    def test_dots_are_clustered(self):
        """A clustered-dot screen forms far fewer separate dots than a
        dispersed (Bayer) screen at the same tone."""
        img = np.full((64, 64), 200, dtype=np.uint8)
        news = grayscale_to_dithered_array(img, DitherAlgorithm.NEWSPRINT)
        bayer = grayscale_to_dithered_array(img, DitherAlgorithm.BAYER8)
        _l1, news_dots = _label(news)
        _l2, bayer_dots = _label(bayer)
        assert news_dots < bayer_dots / 2

    def test_cell_follows_minimum_feature_size(self):
        img = np.full((64, 64), 200, dtype=np.uint8)
        fine = grayscale_to_dithered_array(
            img, DitherAlgorithm.NEWSPRINT, min_feature_px=1
        )
        coarse = grayscale_to_dithered_array(
            img, DitherAlgorithm.NEWSPRINT, min_feature_px=2
        )
        _l1, fine_dots = _label(fine)
        _l2, coarse_dots = _label(coarse)
        assert coarse_dots < fine_dots
