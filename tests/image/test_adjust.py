"""Tests for the pre-engrave image adjustments."""

import numpy as np
import pytest

from rayforge.image.adjust import (
    ImageAdjustments,
    adjust_tone,
    apply_image_adjustments,
    sharpen,
)


def _ramp(width=256, height=4):
    return np.tile(np.arange(width, dtype=np.uint8), (height, 1))


class TestImageAdjustments:
    def test_defaults_are_neutral(self):
        assert ImageAdjustments().is_neutral
        assert not ImageAdjustments(gamma=1.2).is_neutral
        assert not ImageAdjustments(brightness=5).is_neutral
        assert not ImageAdjustments(contrast=-5).is_neutral
        assert not ImageAdjustments(sharpen_amount=50).is_neutral

    def test_sharpen_without_radius_is_neutral(self):
        assert ImageAdjustments(
            sharpen_amount=50, sharpen_radius_mm=0.0
        ).is_neutral


class TestAdjustTone:
    def test_neutral_returns_identical_values(self):
        img = _ramp()
        result = adjust_tone(img, ImageAdjustments())
        np.testing.assert_array_equal(result, img)
        assert result.dtype == np.uint8

    def test_does_not_modify_input(self):
        img = _ramp()
        original = img.copy()
        adjust_tone(img, ImageAdjustments(brightness=40, gamma=2.0))
        np.testing.assert_array_equal(img, original)

    def test_brightness_shifts_values(self):
        img = np.full((2, 2), 100, dtype=np.uint8)
        brighter = adjust_tone(img, ImageAdjustments(brightness=20))
        darker = adjust_tone(img, ImageAdjustments(brightness=-20))
        assert brighter[0, 0] == 151
        assert darker[0, 0] == 49

    def test_brightness_clips(self):
        img = _ramp()
        assert adjust_tone(img, ImageAdjustments(brightness=100)).min() == 255
        assert adjust_tone(img, ImageAdjustments(brightness=-100)).max() == 0

    def test_contrast_scales_around_mid_gray(self):
        img = np.array([[64, 128, 192]], dtype=np.uint8)
        more = adjust_tone(img, ImageAdjustments(contrast=50))
        less = adjust_tone(img, ImageAdjustments(contrast=-50))
        np.testing.assert_array_equal(more, [[32, 128, 224]])
        np.testing.assert_array_equal(less, [[96, 128, 160]])

    def test_full_negative_contrast_is_flat_gray(self):
        result = adjust_tone(_ramp(), ImageAdjustments(contrast=-100))
        assert set(np.unique(result)) == {128}

    def test_gamma_above_one_brightens_midtones(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        lighter = adjust_tone(img, ImageAdjustments(gamma=2.0))
        darker = adjust_tone(img, ImageAdjustments(gamma=0.5))
        assert lighter[0, 0] == 0 and lighter[0, 2] == 255
        assert darker[0, 0] == 0 and darker[0, 2] == 255
        assert lighter[0, 1] == round(255 * (128 / 255) ** 0.5)
        assert darker[0, 1] == round(255 * (128 / 255) ** 2)

    def test_tone_curve_is_monotonic(self):
        adjustments = ImageAdjustments(brightness=10, contrast=30, gamma=1.4)
        result = adjust_tone(_ramp(), adjustments)[0].astype(int)
        assert (np.diff(result) >= 0).all()


class TestSharpen:
    def _edge(self):
        img = np.full((20, 40), 80, dtype=np.uint8)
        img[:, 20:] = 180
        return img

    def test_flat_image_is_unchanged(self):
        img = np.full((10, 10), 90, dtype=np.uint8)
        result = sharpen(img, amount=2.0, sigma_px=(2.0, 2.0))
        np.testing.assert_array_equal(result, img)

    def test_edges_get_overshoot(self):
        img = self._edge()
        result = sharpen(img, amount=1.0, sigma_px=(2.0, 2.0))
        row = result[10].astype(int)
        assert row[19] < 80
        assert row[20] > 180
        assert row[0] == 80 and row[-1] == 180

    def test_sigma_is_per_axis(self):
        """A vertical edge only sharpens along X; a zero X sigma
        leaves it alone."""
        img = self._edge()
        result = sharpen(img, amount=1.0, sigma_px=(0.0, 3.0))
        np.testing.assert_array_equal(result, img)

    def test_zero_amount_is_noop(self):
        img = self._edge()
        np.testing.assert_array_equal(
            sharpen(img, amount=0.0, sigma_px=(2.0, 2.0)), img
        )


class TestApplyImageAdjustments:
    def test_neutral_returns_input_values(self):
        img = _ramp()
        result = apply_image_adjustments(
            img, ImageAdjustments(), pixels_per_mm=(10.0, 10.0)
        )
        np.testing.assert_array_equal(result, img)

    def test_transparent_pixels_stay_untouched(self):
        img = np.full((6, 6), 255, dtype=np.uint8)
        img[:, :3] = 100
        alpha = np.ones((6, 6), dtype=np.float32)
        alpha[:, 3:] = 0.0
        result = apply_image_adjustments(
            img,
            ImageAdjustments(brightness=-50),
            pixels_per_mm=(10.0, 10.0),
            alpha=alpha,
        )
        assert (result[:, 3:] == 255).all()
        assert (result[:, :3] < 100).all()

    def test_sharpen_radius_uses_pixels_per_mm(self):
        img = np.full((40, 80), 80, dtype=np.uint8)
        img[:, 40:] = 180
        narrow = apply_image_adjustments(
            img,
            ImageAdjustments(sharpen_amount=100, sharpen_radius_mm=0.1),
            pixels_per_mm=(10.0, 10.0),
        )
        wide = apply_image_adjustments(
            img,
            ImageAdjustments(sharpen_amount=100, sharpen_radius_mm=0.1),
            pixels_per_mm=(40.0, 10.0),
        )
        changed_narrow = (narrow[20] != img[20]).sum()
        changed_wide = (wide[20] != img[20]).sum()
        assert changed_wide > changed_narrow

    @pytest.mark.parametrize("brightness", [-30, 30])
    def test_order_tone_after_sharpen(self, brightness):
        img = np.full((10, 10), 128, dtype=np.uint8)
        result = apply_image_adjustments(
            img,
            ImageAdjustments(brightness=brightness, sharpen_amount=100),
            pixels_per_mm=(10.0, 10.0),
        )
        assert (result == 128 + round(brightness * 2.55)).all()
