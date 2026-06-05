"""
sRGB <-> linear light conversion utilities.

Pure-array conversions delegate to raygeo.image. The remaining
functions (create_lut_from_color, resize_linear_nd) have no raygeo
equivalents and are kept in Python.
"""

from typing import Tuple

import cv2
import numpy as np
from raygeo.image import linear_to_srgb, srgb_to_linear


def create_lut_from_color(
    color: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Create a 256x4 LUT from a single color (grayscale to color gradient).

    Interpolation is performed in linear light so the gradient ramp is
    perceptually uniform.  The output values are float32 in [0, 1] sRGB
    space, matching existing consumers.
    """
    r, g, b, a = color
    rgb_uint8 = np.clip([r * 255, g * 255, b * 255], 0, 255).astype(np.uint8)
    lin = srgb_to_linear(rgb_uint8)

    t = np.linspace(0, 1, 256, dtype=np.float32)
    lut = np.zeros((256, 4), dtype=np.float32)

    for c in range(3):
        ch_linear = np.clip(lin[c] * t, 0, 1)
        ch_uint8 = linear_to_srgb(ch_linear)
        lut[:, c] = ch_uint8.astype(np.float32) / 255.0

    lut[:, 3] = a * t
    return lut


def resize_linear_nd(
    image: np.ndarray,
    size: tuple[int, int],
    interpolation: int = -1,
) -> np.ndarray:
    """
    Resize a uint8 image in linear light, channel by channel.

    Converts each channel to linear float, resizes, and converts
    back to sRGB uint8.  Channels are processed one at a time to
    minimise peak memory.

    Requires OpenCV (cv2) for the actual resize.

    Args:
        image: HxW or HxWxC uint8 numpy array (sRGB).
        size: Target (width, height) in pixels.
        interpolation: OpenCV interpolation flag.  Defaults to
            cv2.INTER_AREA.

    Returns:
        Resized uint8 numpy array with the same number of channels.
    """
    if interpolation < 0:
        interpolation = cv2.INTER_AREA

    ndim = image.ndim
    if ndim == 2:
        image = image[:, :, np.newaxis]

    n_channels = image.shape[2]
    out_h, out_w = size[1], size[0]
    result = np.empty((out_h, out_w, n_channels), dtype=np.uint8)

    for c in range(n_channels):
        ch_linear = srgb_to_linear(image[:, :, c]).astype(np.float32)
        ch_resized = cv2.resize(ch_linear, size, interpolation=interpolation)
        result[:, :, c] = linear_to_srgb(np.clip(ch_resized, 0, 1))

    return result[:, :, 0] if ndim == 2 else result
