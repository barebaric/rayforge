from __future__ import annotations

import math
import cairo
from ...canvas import CanvasElement


class RotarySurfaceElement(CanvasElement):
    """
    A non-interactive canvas element that draws a dashed rectangle
    indicating the overall rotary surface extent in rotary mode.

    Y is centered on the WCS origin (origin_y ± circumference/2).
    X starts at the WCS origin and extends away from the machine origin,
    covering the remaining bed width in that direction.
    """

    def __init__(self, **kwargs):
        super().__init__(
            x=0,
            y=0,
            width=200.0,
            height=200.0,
            selectable=False,
            draggable=False,
            clip=False,
            **kwargs,
        )
        self._color = (0.2, 0.2, 0.9, 1.0)
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._x_axis_right = False

    def set_origin(self, x: float, y: float):
        """Sets the WCS origin position in world (canvas) coordinates."""
        if self._origin_x == x and self._origin_y == y:
            return
        self._origin_x = x
        self._origin_y = y
        self._update_geometry()
        if self.canvas:
            self.canvas.queue_draw()

    def set_x_axis_right(self, x_axis_right: bool):
        """Sets whether X axis origin is on the right."""
        if self._x_axis_right == x_axis_right:
            return
        self._x_axis_right = x_axis_right
        self._update_geometry()
        if self.canvas:
            self.canvas.queue_draw()

    def _update_geometry(self):
        """Recalculates position and size from diameter and origin."""
        pass

    def update_for_diameter(self, diameter: float, bed_width: float):
        """
        Updates the element geometry for the given cylinder diameter
        and machine bed width.
        """
        circumference = math.pi * diameter
        half_circ = circumference / 2.0

        y = self._origin_y - half_circ
        height = circumference

        if self._x_axis_right:
            x = 0.0
            width = self._origin_x
        else:
            x = self._origin_x
            width = bed_width - self._origin_x

        if width < 0:
            width = 0.0

        self.set_pos(x, y)
        self.set_size(width, height)

    def draw(self, ctx: cairo.Context):
        """Renders the cylinder extent indicator as a dashed rectangle."""
        if self.width <= 0 or self.height <= 0:
            return

        ctx.save()
        ctx.set_source_rgba(*self._color)
        ctx.set_hairline(True)
        ctx.set_dash([5.0, 5.0])
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.stroke()
        ctx.restore()
