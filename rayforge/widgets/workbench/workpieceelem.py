import logging
import cairo
from typing import Optional, TYPE_CHECKING
from ...models.workpiece import WorkPiece
from .surfaceelem import SurfaceElement

if TYPE_CHECKING:
    from .surface import WorkSurface


logger = logging.getLogger(__name__)


class WorkPieceElement(SurfaceElement):
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
            **kwargs,
        )
        workpiece.size_changed.connect(self._on_workpiece_size_changed)
        workpiece.pos_changed.connect(self._on_workpiece_pos_changed)

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
        width_mm, height_mm = self.data.size or self.data.get_default_size()
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # The element's position is its top-left corner. The model's `pos` is
        # its bottom-left corner in a Y-up coordinate system. To find the
        # element's position on the canvas (Y-down), we first find the
        # model's top-left corner in mm and then convert it to pixels.
        top_left_y_mm = y_mm + height_mm
        x_px, y_px = self.mm_to_pixel(x_mm, top_left_y_mm)

        # We call super().set_pos() to bypass the model update logic in this
        # class's set_pos(), as we are updating the view from the model.
        super().set_pos(x_px, y_px)

        size_changed = self.width != new_width or self.height != new_height

        if not size_changed and not force:
            return

        self.width, self.height = new_width, new_height
        super().allocate(force)

    def set_pos(self, x: int, y: int):
        super().set_pos(x, y)
        if not self.canvas or self._in_update:
            return
        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1
        x_mm = x / px_per_mm_x
        y_mm = (self.canvas.root.height - self.height - y) / px_per_mm_y
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
        _, old_height_mm = self.data.size or self.data.get_default_size()
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
