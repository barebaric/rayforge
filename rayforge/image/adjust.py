"""Tone and sharpness adjustments applied to an image before engraving."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ImageAdjustments:
    """User adjustments for a raster image.

    ``brightness`` and ``contrast`` run from -100 to 100, ``gamma``
    above 1 lightens the midtones and ``sharpen_amount`` is the
    unsharp-mask strength in percent, with ``sharpen_radius_mm`` as
    the blur radius (Gaussian sigma).
    """

    brightness: float = 0.0
    contrast: float = 0.0
    gamma: float = 1.0
    sharpen_amount: float = 0.0
    sharpen_radius_mm: float = 0.2

    @property
    def has_tone(self) -> bool:
        return (
            self.brightness != 0.0 or self.contrast != 0.0 or self.gamma != 1.0
        )

    @property
    def has_sharpen(self) -> bool:
        return self.sharpen_amount > 0.0 and self.sharpen_radius_mm > 0.0

    @property
    def is_neutral(self) -> bool:
        return not (self.has_tone or self.has_sharpen)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _tone_lut(adjustments: ImageAdjustments) -> np.ndarray:
    """Lookup table for brightness, then contrast, then gamma."""
    values = np.arange(256, dtype=np.float64)
    values = values + adjustments.brightness * 2.55
    factor = max(0.0, 1.0 + adjustments.contrast / 100.0)
    values = (values - 128.0) * factor + 128.0
    values = np.clip(values, 0.0, 255.0)
    gamma = max(adjustments.gamma, 1e-3)
    values = 255.0 * (values / 255.0) ** (1.0 / gamma)
    return _to_uint8(values)


def adjust_tone(gray: np.ndarray, adjustments: ImageAdjustments) -> np.ndarray:
    """Apply brightness, contrast and gamma to a uint8 image."""
    if not adjustments.has_tone:
        return gray.copy()
    return _tone_lut(adjustments)[gray]


def sharpen(
    gray: np.ndarray, amount: float, sigma_px: tuple[float, float]
) -> np.ndarray:
    """Unsharp mask with a per-axis ``(x, y)`` Gaussian sigma."""
    sigma_x, sigma_y = sigma_px
    if amount <= 0.0 or (sigma_x <= 0.0 and sigma_y <= 0.0):
        return gray.copy()
    values = gray.astype(np.float64)
    blurred = gaussian_filter(values, sigma=(sigma_y, sigma_x), mode="nearest")
    return _to_uint8(values + amount * (values - blurred))


def apply_image_adjustments(
    gray: np.ndarray,
    adjustments: ImageAdjustments,
    *,
    pixels_per_mm: tuple[float, float],
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sharpen, then apply the tone adjustments to a grayscale image.

    Args:
        gray: 2D uint8 grayscale image.
        adjustments: The adjustments to apply.
        pixels_per_mm: ``(x, y)`` image resolution, used to turn the
            sharpen radius into pixels.
        alpha: Optional alpha channel; fully transparent pixels keep
            their value.

    Returns:
        The adjusted uint8 image; the input is not modified.
    """
    if adjustments.is_neutral:
        return gray.copy()
    result = gray
    if adjustments.has_sharpen:
        ppm_x, ppm_y = pixels_per_mm
        radius = adjustments.sharpen_radius_mm
        result = sharpen(
            result,
            adjustments.sharpen_amount / 100.0,
            (radius * ppm_x, radius * ppm_y),
        )
    result = adjust_tone(result, adjustments)
    if alpha is not None:
        transparent = alpha <= 0
        result[transparent] = gray[transparent]
    return result
