import numpy as np
import pytest

from rayforge.image.util.srgb import (
    srgb_to_linear,
    linear_to_srgb,
    _SRGB_TO_LINEAR,
    _LINEAR_TO_SRGB,
)


class TestSrgbToLinear:
    def test_black(self):
        result = srgb_to_linear(np.array([0], dtype=np.uint8))
        assert result[0] == pytest.approx(0.0)

    def test_white(self):
        result = srgb_to_linear(np.array([255], dtype=np.uint8))
        assert result[0] == pytest.approx(1.0)

    def test_midgray_srgb_128(self):
        result = srgb_to_linear(np.array([128], dtype=np.uint8))
        assert result[0] == pytest.approx(0.2158, abs=0.001)

    def test_known_value_64(self):
        s = 64 / 255.0
        expected = ((s + 0.055) / 1.055) ** 2.4
        result = srgb_to_linear(np.array([64], dtype=np.uint8))
        assert result[0] == pytest.approx(expected, abs=1e-5)

    def test_linear_segment_below_threshold(self):
        s10 = 10 / 255.0
        expected = s10 / 12.92
        result = srgb_to_linear(np.array([10], dtype=np.uint8))
        assert result[0] == pytest.approx(expected, abs=1e-5)

    def test_2d_array(self):
        arr = np.array([[0, 128, 255], [64, 192, 10]], dtype=np.uint8)
        result = srgb_to_linear(arr)
        assert result.shape == (2, 3)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(1.0)

    def test_output_dtype(self):
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = srgb_to_linear(arr)
        assert result.dtype == np.float32

    def test_clamping(self):
        arr = np.array([0, 255], dtype=np.uint8)
        result = srgb_to_linear(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_monotonicity(self):
        values = np.arange(256, dtype=np.uint8)
        result = srgb_to_linear(values)
        diffs = np.diff(result)
        assert np.all(diffs > 0)


class TestLinearToSrgb:
    def test_zero(self):
        result = linear_to_srgb(np.array([0.0]))
        assert result[0] == 0

    def test_one(self):
        result = linear_to_srgb(np.array([1.0]))
        assert result[0] == 255

    def test_output_dtype(self):
        arr = np.array([0.0, 0.5, 1.0])
        result = linear_to_srgb(arr)
        assert result.dtype == np.uint8

    def test_clamping_below(self):
        result = linear_to_srgb(np.array([-0.5]))
        assert result[0] == 0

    def test_clamping_above(self):
        result = linear_to_srgb(np.array([1.5]))
        assert result[0] == 255

    def test_2d_array(self):
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = linear_to_srgb(arr)
        assert result.shape == (1, 3)

    def test_dithering_produces_valid_output(self):
        rng = np.random.default_rng(42)
        arr = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        result = linear_to_srgb(arr, dither=True, rng=rng)
        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_dithering_preserves_endpoints(self):
        rng = np.random.default_rng(42)
        arr = np.array([0.0, 1.0])
        result = linear_to_srgb(arr, dither=True, rng=rng)
        assert result[0] == 0
        assert result[1] == 255


class TestRoundTrip:
    def test_round_trip_all_values(self):
        srgb_in = np.arange(256, dtype=np.uint8)
        linear = srgb_to_linear(srgb_in)
        srgb_out = linear_to_srgb(linear)
        np.testing.assert_array_equal(srgb_out, srgb_in)

    def test_round_trip_2d(self):
        srgb_in = np.array([[0, 50, 100, 150, 200, 255]], dtype=np.uint8)
        linear = srgb_to_linear(srgb_in)
        srgb_out = linear_to_srgb(linear)
        np.testing.assert_array_equal(srgb_out, srgb_in)


class TestLutProperties:
    def test_forward_lut_size(self):
        assert len(_SRGB_TO_LINEAR) == 256

    def test_inverse_lut_size(self):
        assert len(_LINEAR_TO_SRGB) == 32769

    def test_forward_lut_bounds(self):
        assert _SRGB_TO_LINEAR[0] == pytest.approx(0.0)
        assert _SRGB_TO_LINEAR[255] == pytest.approx(1.0)

    def test_inverse_lut_bounds(self):
        assert _LINEAR_TO_SRGB[0] == 0
        assert _LINEAR_TO_SRGB[-1] == 255
