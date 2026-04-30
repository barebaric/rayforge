"""
sRGB <-> linear light conversion utilities.

Implements the sRGB transfer function defined in IEC 61966-2-1 using
precomputed lookup tables for fast vectorized conversion with NumPy.
"""

import cv2
import numpy as np

_SRGB_TO_LINEAR = np.empty(256, dtype=np.float32)
for _i in range(256):
    _s = _i / 255.0
    if _s <= 0.04045:
        _SRGB_TO_LINEAR[_i] = _s / 12.92
    else:
        _SRGB_TO_LINEAR[_i] = ((_s + 0.055) / 1.055) ** 2.4

_FRAC_BITS = 15
_SCALE = 1 << _FRAC_BITS
_INV_TABLE_SIZE = _SCALE + 1

_LINEAR_TO_SRGB = np.empty(_INV_TABLE_SIZE, dtype=np.uint8)
for _i in range(_INV_TABLE_SIZE):
    _lin = _i / _SCALE
    if _lin <= 0.0031308:
        _s = 12.92 * _lin
    else:
        _s = 1.055 * (_lin ** (1.0 / 2.4)) - 0.055
    _LINEAR_TO_SRGB[_i] = min(255, max(0, round(_s * 255.0)))


def srgb_to_linear(array: np.ndarray) -> np.ndarray:
    """
    Convert sRGB uint8 values to linear float [0.0, 1.0].

    Uses a 256-entry lookup table for vectorized conversion.

    Args:
        array: uint8 numpy array with sRGB values (0-255).

    Returns:
        float32 numpy array with linear values in [0.0, 1.0].
    """
    return _SRGB_TO_LINEAR[np.clip(array, 0, 255).astype(np.uint8)]


def linear_to_srgb(
    array: np.ndarray,
    dither: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Convert linear float values to sRGB uint8.

    Uses an inverse lookup table indexed by fixed-point linear values
    (15 fractional bits) for fast conversion.

    Args:
        array: float numpy array with linear values in [0.0, 1.0].
            Values outside this range are clamped.
        dither: If True, add uniform noise before quantization to
            reduce banding in gradients.
        rng: Optional NumPy random generator instance. If None and
            dither is True, a default generator is created.

    Returns:
        uint8 numpy array with sRGB values (0-255).
    """
    clipped = np.clip(array, 0.0, 1.0)
    fixed = clipped * _SCALE

    if dither:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.uniform(-0.5, 0.5, fixed.shape)
        fixed = np.clip(fixed + noise, 0, _SCALE)

    indices = np.round(fixed).astype(np.intp)
    return _LINEAR_TO_SRGB[indices]


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
