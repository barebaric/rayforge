"""A draggable crosshair marker.

The element draws a small crosshair + ring and allows the user to
drag it on the canvas.  The visual centre sits at the centre of
the hit rectangle, so a click anywhere near the drawn crosshair
registers as a hit.
The element's position (set via ``move_to``) refers to the centre,
not the bottom‑left corner.
"""

import logging
import math
from typing import Callable, Optional, Tuple

import cairo

from ...canvas import CanvasElement
from ...canvas.region import ElementRegion

logger = logging.getLogger(__name__)

# Hit-test region width & height in world mm — matches the drawn
# crosshair span (arm‑to‑arm) so the grab zone feels natural.
_HIT_SIZE = 5


class CrosshairElement(CanvasElement):
    """A small, draggable crosshair at a given world position.

    Call ``move_to(x, y)`` to position the visual centre at ``(x, y)``.
    The hit region (bounding box) is centred on the same point, so
    clicks near the drawn crosshair always register.
    """

    def __init__(
        self,
        on_drag: Optional[Callable[[Tuple[float, float]], None]] = None,
        **kwargs,
    ):
        super().__init__(
            0,
            0,
            _HIT_SIZE,
            _HIT_SIZE,
            selectable=True,
            draggable=True,
            drag_handler_controls_transform=True,
            visible=True,
            clip=False,
            background=(0, 0, 0, 0),
            show_selection_frame=False,
            pixel_perfect_hit=False,
            **kwargs,
        )
        self._on_drag = on_drag
        self._drag_origin: Optional[Tuple[float, float]] = None
        self._centre_offset = _HIT_SIZE / 2.0

    def move_to(self, x: float, y: float) -> None:
        """Places the visual centre at ``(x, y)`` world."""
        self.set_pos(x - self._centre_offset, y - self._centre_offset)

    def draw(self, ctx: cairo.Context):
        c = self._centre_offset  # 2.5 mm — centre of the hit rect
        s = c  # visual arm reaches the same distance as the hit boundary

        ctx.save()
        ctx.set_source_rgba(0.45, 0.70, 1.0, 0.6)
        ctx.arc(c, c, 1.0, 0.0, 2.0 * math.pi)
        ctx.fill()

        # Crosshair arms.
        ctx.set_line_width(1.0)
        ctx.move_to(c - s, c)
        ctx.line_to(c + s, c)
        ctx.move_to(c, c - s)
        ctx.line_to(c, c + s)
        ctx.stroke()

        # Outer ring (slightly smaller than the arms).
        ctx.arc(c, c, s - 0.3, 0.0, 2.0 * math.pi)
        ctx.stroke()
        ctx.restore()

    def check_region_hit(self, x_abs, y_abs, candidates=None) -> ElementRegion:
        world = self.get_world_transform()
        inv = world.invert()
        lx, ly = inv.transform_point((x_abs, y_abs))
        if 0 <= lx < self.width and 0 <= ly < self.height:
            return ElementRegion.BODY
        return ElementRegion.NONE

    def end_interactive_transform(self):
        self._drag_origin = None
        super().end_interactive_transform()

    def handle_drag_move(
        self, world_dx: float, world_dy: float
    ) -> Tuple[float, float]:
        if self._drag_origin is None:
            self._drag_origin = self.get_world_transform().get_translation()
        ox, oy = self._drag_origin
        new_pos = (ox + world_dx, oy + world_dy)
        self.set_pos(*new_pos)
        if self._on_drag:
            self._on_drag(
                (
                    new_pos[0] + self._centre_offset,
                    new_pos[1] + self._centre_offset,
                )
            )
        return 0.0, 0.0
