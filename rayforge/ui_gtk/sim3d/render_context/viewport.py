"""Viewport-derived render context section."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import FrameInputs


class ViewportContext:
    """Grid/world transforms derived from the viewport configuration.

    The plain constructor leaves the section empty; call :meth:`update`
    each frame to populate it from the viewport config.
    """

    def __init__(
        self,
        *,
        model_matrix: np.ndarray | None = None,
        margin_shift: np.ndarray | None = None,
        wcs_offset_mm: tuple[float, float, float] | None = None,
        x_right: bool = False,
        x_negative: bool = False,
        y_negative: bool = False,
    ):
        identity = np.eye(4, dtype=np.float32)
        self.model_matrix = identity if model_matrix is None else model_matrix
        self.margin_shift = identity if margin_shift is None else margin_shift
        self.wcs_offset_mm = (
            (0.0, 0.0, 0.0) if wcs_offset_mm is None else wcs_offset_mm
        )
        self.x_right = x_right
        self.x_negative = x_negative
        self.y_negative = y_negative

    def update(self, frame: "FrameInputs") -> None:
        """Recomputes the viewport section from the current frame inputs."""
        viewport = frame.viewport
        self.model_matrix = viewport.model_matrix
        self.margin_shift = viewport.margin_shift
        self.wcs_offset_mm = viewport.wcs_offset_mm
        self.x_right = viewport.x_right
        self.x_negative = viewport.x_negative
        self.y_negative = viewport.y_negative
