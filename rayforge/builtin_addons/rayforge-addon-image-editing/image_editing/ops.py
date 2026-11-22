from collections.abc import Mapping
from gettext import gettext as _
from typing import Any

import numpy as np
import pyvips

from rayforge.core.image_edit import ImageEditOp, ImageEditStep

#: Smallest blur sigma, in pixels, that still changes the image.
MIN_SIGMA_PX = 0.1


def _max_value(image: pyvips.Image) -> int:
    return 65535 if image.format == "ushort" else 255


def _split_alpha(
    image: pyvips.Image,
) -> tuple[pyvips.Image, pyvips.Image | None]:
    if not image.hasalpha():
        return image, None
    color = image.extract_band(0, n=image.bands - 1)
    return color, image.extract_band(image.bands - 1)


def _join_alpha(
    color: pyvips.Image, alpha: pyvips.Image | None
) -> pyvips.Image:
    return color if alpha is None else color.bandjoin(alpha)


class InvertOp(ImageEditOp):
    """Inverts the colour of the image and leaves its alpha alone."""

    NAME = "invert"
    LABEL = _("Invert")

    @classmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        if not image.hasalpha():
            return image.invert()
        color = image.extract_band(0, n=image.bands - 1)
        alpha = image.extract_band(image.bands - 1)
        return color.invert().bandjoin(alpha)


def tone_curve(
    brightness: float, contrast: float, gamma: float, max_value: int = 255
) -> np.ndarray:
    """
    The tone curve as a lookup table with ``max_value + 1`` entries:
    brightness, then contrast around mid-gray, then gamma. Brightness
    and contrast run from -100 to 100; a gamma above 1 lightens the
    midtones.
    """
    values = np.linspace(0.0, 1.0, max_value + 1)
    values = values + brightness / 100.0
    factor = max(0.0, 1.0 + contrast / 100.0)
    values = np.clip((values - 0.5) * factor + 0.5, 0.0, 1.0)
    values = values ** (1.0 / max(gamma, 1e-3))
    dtype = np.uint16 if max_value > 255 else np.uint8
    return np.rint(values * max_value).astype(dtype)


class ToneOp(ImageEditOp):
    """Brightness, contrast and gamma of the colour; alpha is kept."""

    NAME = "tone"
    LABEL = _("Brightness and Contrast")

    @classmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        color, alpha = _split_alpha(image)
        if color.format not in ("uchar", "ushort"):
            color = color.cast("uchar")
        lut = tone_curve(
            params.get("brightness", 0.0),
            params.get("contrast", 0.0),
            params.get("gamma", 1.0),
            _max_value(color),
        )
        table = pyvips.Image.new_from_array(lut.reshape(1, -1))
        return _join_alpha(color.maplut(table), alpha)


def _blur(image: pyvips.Image, sigma_x: float, sigma_y: float) -> pyvips.Image:
    if sigma_x >= MIN_SIGMA_PX:
        mask = pyvips.Image.gaussmat(
            sigma_x, 0.01, separable=True, precision="float"
        )
        image = image.conv(mask, precision="float")
    if sigma_y >= MIN_SIGMA_PX:
        mask = pyvips.Image.gaussmat(
            sigma_y, 0.01, separable=True, precision="float"
        )
        image = image.conv(mask.rot90(), precision="float")
    return image


def _blurred_color(
    image: pyvips.Image, sigma_x: float, sigma_y: float
) -> pyvips.Image:
    """
    The blurred colour bands. With alpha, the blur runs on premultiplied
    colour, so transparent pixels do not darken the edges.
    """
    if not image.hasalpha():
        return _blur(image.cast("float"), sigma_x, sigma_y)
    max_alpha = _max_value(image)
    premultiplied = image.premultiply(max_alpha=max_alpha)
    blurred = _blur(premultiplied, sigma_x, sigma_y)
    color = blurred.unpremultiply(max_alpha=max_alpha)
    return color.extract_band(0, n=image.bands - 1)


class SharpenOp(ImageEditOp):
    """
    Unsharp mask: ``amount`` is the strength in percent and
    ``radius_mm`` the blur radius (Gaussian sigma) in millimetres,
    turned into pixels with the image's resolution.
    """

    NAME = "sharpen"
    LABEL = _("Sharpen")

    @classmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        amount = params.get("amount", 0.0) / 100.0
        radius = params.get("radius_mm", 0.0)
        sigma_x, sigma_y = radius * image.xres, radius * image.yres
        if amount <= 0.0 or max(sigma_x, sigma_y) < MIN_SIGMA_PX:
            return image
        color, alpha = _split_alpha(image)
        values = color.cast("float")
        blurred = _blurred_color(image, sigma_x, sigma_y)
        sharpened = values + (values - blurred) * amount
        result = sharpened.rint().cast(color.format)
        return _join_alpha(result, alpha)


INVERT = ImageEditStep(InvertOp.NAME)

ALL_OPS = (InvertOp, ToneOp, SharpenOp)
