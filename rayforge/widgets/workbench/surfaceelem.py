import logging
from abc import ABC
from typing import Tuple
from ..canvas import CanvasElement

logger = logging.getLogger(__name__)


class SurfaceElement(CanvasElement, ABC):
    """
    A base class for elements that need to be positioned and sized
    according to real-world dimensions on a WorkSurface.
    """

    def __init__(self, x, y, width, height, **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.width_mm = 0.0
        self.height_mm = 0.0

    def mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Converts real-world mm coordinates to canvas pixel coordinates."""
        # Canvas origin is top-left, y-down.
        # Real-world origin is typically bottom-left, y-up.
        # Need to flip y-axis and scale.
        if not self.canvas:
            return 0, 0
        x_px = x_mm * self.canvas.pixels_per_mm_x
        y_px = self.canvas.root.height - (y_mm * self.canvas.pixels_per_mm_y)
        return x_px, y_px

    def pixel_to_mm(self, x_px: int, y_px: int) -> Tuple[float, float]:
        """Converts canvas pixel coordinates to real-world mm coordinates."""
        # Canvas origin is top-left, y-down.
        # Real-world origin is typically bottom-left, y-up.
        # Need to flip y-axis and scale.
        if not self.canvas:
            return 0, 0
        x_mm = (
            x_px / self.canvas.pixels_per_mm_x
            if self.canvas.pixels_per_mm_x
            else 0
        )
        y_mm = (
            ((self.canvas.root.height - self.height - y_px) /
             self.canvas.pixels_per_mm_y)
            if self.canvas.pixels_per_mm_y
            else 0
        )
        return x_mm, y_mm

    def pos_mm(self) -> Tuple[float, float]:
        """Gets the absolute position of the element in real-world mm."""
        if not self.canvas:
            return 0.0, 0.0
        x_px, y_px = self.pos()
        return self.pixel_to_mm(x_px, y_px)

    def pos_abs_mm(self) -> Tuple[float, float]:
        """Gets the absolute position of the element in real-world mm."""
        if not self.canvas:
            return 0.0, 0.0
        x_px, y_px = self.pos_abs()
        return self.pixel_to_mm(x_px, y_px)

    def set_pos_mm(self, x_mm, y_mm):
        """Sets the position of the element in real-world mm."""
        if not self.canvas:
            return
        x_px, y_px = self.mm_to_pixel(x_mm, y_mm)
        super().set_pos(x_px, y_px)

    def rect_mm(self) -> Tuple[float, float, float, float]:
        """Gets the absolute position of the element in real-world mm."""
        if not self.canvas:
            return 0.0, 0.0, 0.0, 0.0
        x_px, y_px, width_px, height_px = self.rect()
        return *self.pixel_to_mm(x_px, y_px), *self.pixel_to_mm(
            width_px, height_px
        )
