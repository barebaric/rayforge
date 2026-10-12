"""
The Invert edit replaces the Engrave step's old Invert setting; these
tests pin the behaviour that setting had.
"""

import numpy as np
import pyvips
from image_editing.ops import InvertOp

from rayforge.image.dither import DitherAlgorithm
from rayforge.image.util import normalize_to_rgba, vips_rgba_to_cairo_surface
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    preprocess_raster_image,
)


def _white_left_half_on_transparent() -> pyvips.Image:
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[:, :5] = (255, 255, 255, 255)
    return pyvips.Image.new_from_array(pixels).copy(interpretation="srgb")


def _surface(image: pyvips.Image):
    return vips_rgba_to_cairo_surface(normalize_to_rgba(image))


def test_inverted_dither_engraves_light_areas_only():
    inverted = InvertOp.apply(_white_left_half_on_transparent(), {})
    image, _ = preprocess_raster_image(
        _surface(inverted),
        mode=DepthMode.DITHER,
        auto_levels=False,
        dither_algorithm=DitherAlgorithm.BAYER2,
        laser_spot_x_mm=0.1,
        pixels_per_mm_x=10.0,
    )
    assert image is not None
    assert np.all(image[:, :5] == 1)
    assert np.all(image[:, 5:] == 0)


def test_inverted_power_modulation_flips_visible_pixels_only():
    original = _white_left_half_on_transparent()
    gray, alpha = preprocess_raster_image(
        _surface(InvertOp.apply(original, {})),
        mode=DepthMode.POWER_MODULATION,
        auto_levels=False,
    )
    assert gray is not None and alpha is not None
    assert np.all(gray[:, :5] == 0)
    assert np.all(alpha[:, 5:] == 0)
