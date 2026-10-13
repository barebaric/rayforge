from collections.abc import Mapping
from gettext import gettext as _
from typing import Any

import pyvips

from rayforge.core.image_edit import ImageEditOp, ImageEditStep


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


INVERT = ImageEditStep(InvertOp.NAME)

ALL_OPS = (InvertOp,)
