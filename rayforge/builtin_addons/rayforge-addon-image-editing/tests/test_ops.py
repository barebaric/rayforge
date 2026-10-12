import numpy as np
import pyvips
from image_editing.ops import INVERT, InvertOp

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
