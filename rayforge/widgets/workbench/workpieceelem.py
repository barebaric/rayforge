import logging
import cairo
from typing import Optional, TYPE_CHECKING
from ...models.workpiece import WorkPiece
from ..canvas import CanvasElement

if TYPE_CHECKING:
    from .surface import WorkSurface


logger = logging.getLogger(__name__)


class WorkPieceElement(CanvasElement):
    """
    A CanvasElement that displays a WorkPiece.
    """

    def __init__(self, workpiece: WorkPiece, **kwargs):
        self.canvas: Optional["WorkSurface"]
        self.data: WorkPiece = workpiece
        self._in_update = False
        super().__init__(
            0,
            0,
            0,
            0,
            data=workpiece,
            clip=False,
            buffered=True,
            pixel_perfect_hit=True,
            **kwargs,
        )
        workpiece.size_changed.connect(self._on_workpiece_size_changed)
        workpiece.pos_changed.connect(self._on_workpiece_pos_changed)
        workpiece.angle_changed.connect(self._on_workpiece_angle_changed)

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        return self.data.renderer.render_to_pixels(
            width=width,
            height=height,
        )

    def allocate(self, force: bool = False):
        if not self.canvas or self._in_update:
            return

        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = (
            self.data.size
            or self.data.get_default_size(*self.canvas.get_size())
        )

        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1
        new_width = round(width_mm * px_per_mm_x)
        new_height = round(height_mm * px_per_mm_y)

        # The element's position is its top-left corner. The model's `pos` is
        # its bottom-left corner in a Y-up coordinate system.
        # We convert to the top-left corner in the canvas's root element's
        # coordinate space (Y-down). Pan is handled by moving the root element,
        # so we don't account for it here.
        top_left_y_mm = y_mm + height_mm

        x_px = x_mm * px_per_mm_x
        y_px = self.canvas.root.height - (top_left_y_mm * px_per_mm_y)

        # We call super().set_pos() to bypass the model update logic in this
        # class's set_pos(), as we are updating the view from the model.
        super().set_pos(round(x_px), round(y_px))
        self.set_angle(self.data.angle)  # Sync angle from model

        size_changed = self.width != new_width or self.height != new_height

        if not size_changed and not force:
            # If only position or angle changed, we don't need to re-render the
            # buffer, just return.
            return

        self.width, self.height = new_width, new_height
        super().allocate(force)

    def set_pos(self, x: int, y: int):
        super().set_pos(x, y)
        if not self.canvas or self._in_update:
            return

        # Convert the element's new top-left pixel coordinate (x, y), which
        # is relative to the canvas's root element, back to the model's
        # bottom-left millimeter coordinate.
        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1

        # The model's origin (pos) is its bottom-left corner.
        x_mm = x / px_per_mm_x
        y_mm = (self.canvas.root.height - (y + self.height)) / px_per_mm_y

        self._in_update = True
        try:
            self.data.set_pos(x_mm, y_mm)
        finally:
            self._in_update = False

    def set_size(self, width: int, height: int):
        # Update our size for immediate scaled drawing.
        self.width, self.height = int(width), int(height)
        if self.canvas:
            self.canvas.queue_draw()

        self.trigger_update()

        # Update the model.
        if not self.canvas or self._in_update:
            return

        old_x_mm, old_y_mm = self.data.pos or (0, 0)
        _, old_height_mm = self.data.size or self.data.get_default_size(
            *self.canvas.get_size()
        )

        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1
        new_width_mm = width / px_per_mm_x
        new_height_mm = height / px_per_mm_y
        new_y_mm = (old_y_mm + old_height_mm) - new_height_mm

        self._in_update = True
        try:
            self.data.set_pos(old_x_mm, new_y_mm)
            self.data.set_size(new_width_mm, new_height_mm)
        finally:
            self._in_update = False

    def set_angle(self, angle: float):
        super().set_angle(angle)
        if self._in_update:
            return
        self._in_update = True
        try:
            self.data.set_angle(angle)
        finally:
            self._in_update = False

    def _on_workpiece_size_changed(self, workpiece):
        if self._in_update:
            return
        self.allocate()

    def _on_workpiece_pos_changed(self, workpiece):
        if self._in_update or not self.parent:
            return
        # This is a cheap operation, no re-render needed.
        self.allocate()
        if self.parent:
            self.parent.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def _on_workpiece_angle_changed(self, workpiece):
        if self._in_update:
            return
        self.set_angle(workpiece.angle)
        if self.parent:
            self.parent.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()
