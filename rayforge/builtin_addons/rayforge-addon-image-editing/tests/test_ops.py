import numpy as np
import pyvips
from image_editing.ops import (
    INVERT,
    InvertOp,
    SharpenOp,
    ToneOp,
    tone_curve,
)

from rayforge.core.image_edit import image_edit_registry


def _image(*bands: int) -> pyvips.Image:
    return pyvips.Image.new_from_array(np.array([[bands]], dtype=np.uint8))


def _pixel(image: pyvips.Image) -> list[float]:
    return list(image(0, 0))


def test_invert_is_registered_under_its_stable_name():
    assert INVERT.op == "invert"
    assert image_edit_registry.get("invert") is InvertOp


def test_invert_keeps_alpha():
    rgba = _image(10, 20, 30, 128).copy(interpretation="srgb")
    assert _pixel(InvertOp.apply(rgba, {})) == [245, 235, 225, 128]


def test_invert_without_alpha():
    rgb = _image(0, 100, 255).copy(interpretation="srgb")
    assert _pixel(InvertOp.apply(rgb, {})) == [255, 155, 0]


def test_invert_grayscale_with_alpha():
    gray_alpha = _image(40, 0).copy(interpretation="b-w")
    assert _pixel(InvertOp.apply(gray_alpha, {})) == [215, 0]


def _ramp(width: int = 256, height: int = 4, alpha: int | None = None):
    row = np.arange(width, dtype=np.uint8)
    pixels = np.tile(row, (height, 1))
    if alpha is None:
        return pyvips.Image.new_from_array(pixels).copy(interpretation="b-w")
    bands = np.dstack([pixels, np.full_like(pixels, alpha)])
    return pyvips.Image.new_from_array(bands).copy(interpretation="b-w")


def _gray(*values: int) -> pyvips.Image:
    pixels = np.array([values], dtype=np.uint8)
    return pyvips.Image.new_from_array(pixels).copy(interpretation="b-w")


def _row(image: pyvips.Image) -> list[int]:
    return image.numpy().reshape(image.height, -1)[0].tolist()


def _tone(image: pyvips.Image, **params) -> pyvips.Image:
    return ToneOp.apply(image, params)


def test_tone_and_sharpen_are_registered():
    assert image_edit_registry.get("tone") is ToneOp
    assert image_edit_registry.get("sharpen") is SharpenOp


def test_tone_curve_is_identity_by_default():
    np.testing.assert_array_equal(tone_curve(0, 0, 1.0), np.arange(256))
    assert tone_curve(0, 0, 1.0, 65535).dtype == np.uint16


def test_neutral_tone_keeps_the_image():
    image = _ramp()
    assert _row(_tone(image)) == _row(image)


def test_brightness_shifts_and_clips():
    assert _row(_tone(_gray(100), brightness=20)) == [151]
    assert _row(_tone(_gray(100), brightness=-20)) == [49]
    assert set(_row(_tone(_ramp(), brightness=100))) == {255}
    assert set(_row(_tone(_ramp(), brightness=-100))) == {0}


def test_contrast_scales_around_mid_gray():
    image = _gray(64, 128, 192)
    assert _row(_tone(image, contrast=50)) == [32, 128, 224]
    assert _row(_tone(image, contrast=-50)) == [96, 128, 160]
    assert set(_row(_tone(_ramp(), contrast=-100))) == {128}


def test_gamma_above_one_lightens_midtones():
    image = _gray(0, 128, 255)
    lighter = _row(_tone(image, gamma=2.0))
    darker = _row(_tone(image, gamma=0.5))
    assert lighter == [0, round(255 * (128 / 255) ** 0.5), 255]
    assert darker == [0, round(255 * (128 / 255) ** 2), 255]


def test_tone_curve_is_monotonic():
    curve = tone_curve(10, 30, 1.4).astype(int)
    assert (np.diff(curve) >= 0).all()


def test_tone_keeps_alpha_and_colour_format():
    rgba = _image(100, 100, 100, 7).copy(interpretation="srgb")
    result = _tone(rgba, brightness=20)
    assert _pixel(result) == [151, 151, 151, 7]
    assert result.format == "uchar"


def test_tone_handles_16_bit_images():
    pixels = np.array([[0, 32768, 65535]], dtype=np.uint16)
    image = pyvips.Image.new_from_array(pixels).copy(interpretation="grey16")
    result = _tone(image, contrast=-100)
    assert result.format == "ushort"
    assert set(_row(result)) == {32768}


def _edge(ppm: float = 10.0) -> pyvips.Image:
    pixels = np.zeros((20, 40), dtype=np.uint8)
    pixels[:, :20] = 60
    pixels[:, 20:] = 180
    image = pyvips.Image.new_from_array(pixels).copy(interpretation="b-w")
    return image.copy(xres=ppm, yres=ppm)


def _sharpen(image: pyvips.Image, **params) -> pyvips.Image:
    return SharpenOp.apply(image, params)


def test_sharpen_without_amount_or_radius_is_neutral():
    image = _edge()
    assert _sharpen(image, amount=0, radius_mm=0.2) is image
    assert _sharpen(image, amount=100, radius_mm=0.0) is image


def test_sharpen_increases_edge_contrast():
    result = _row(_sharpen(_edge(), amount=100, radius_mm=0.2))
    assert result[19] < 60
    assert result[20] > 180
    assert result[0] == 60 and result[-1] == 180


def test_sharpen_radius_is_in_mm():
    coarse = _row(_sharpen(_edge(ppm=10.0), amount=100, radius_mm=0.2))
    fine = _row(_sharpen(_edge(ppm=5.0), amount=100, radius_mm=0.4))
    assert coarse == fine


def test_sharpen_ignores_transparent_neighbours():
    pixels = np.zeros((10, 20, 2), dtype=np.uint8)
    pixels[:, 10:] = (120, 255)
    image = pyvips.Image.new_from_array(pixels).copy(
        interpretation="b-w", xres=10.0, yres=10.0
    )
    result = _sharpen(image, amount=200, radius_mm=0.2).numpy()
    assert set(result[:, 10:, 0].ravel()) == {120}
    assert set(result[..., 1].ravel()) == {0, 255}
