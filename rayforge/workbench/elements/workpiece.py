import logging
import cairo
from typing import Optional, TYPE_CHECKING
from ...core.workpiece import WorkPiece
from ..canvas import CanvasElement

if TYPE_CHECKING:
    from ..surface import WorkSurface


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
            0.0,
            0.0,
            0.0,
            0.0,
            data=workpiece,
            clip=False,
            buffered=True,
            pixel_perfect_hit=True,
            **kwargs,
        )
        workpiece.changed.connect(self._on_model_data_changed)
        workpiece.transform_changed.connect(self._on_transform_changed)

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        return self.data.importer.render_to_pixels(
            width=width,
            height=height,
        )

    def allocate(self, force: bool = False):
        if not self.canvas or self._in_update:
            return

        # The data object (workpiece) is the source of truth.
        new_pos_px, new_size_px = (
            self.canvas.workpiece_coords_to_element_coords(self.data)
        )

        # We call super().set_pos() to bypass the model update logic in this
        # class's set_pos(), as we are updating the view from the model.
        super().set_pos(*new_pos_px)
        self.set_angle(self.data.angle)  # Sync angle from model

        new_width, new_height = new_size_px
        size_changed = self.width != new_width or self.height != new_height

        if not size_changed and not force:
            # If only position or angle changed, we don't need to re-render the
            # buffer, just return.
            return

        self.width, self.height = new_width, new_height
        super().allocate(force)

    def set_pos(self, x: float, y: float):
        super().set_pos(x, y)
        if not self.canvas or self._in_update:
            return

        # For a pure move operation, the size doesn't change. Convert the new
        # element pixel position back to the model's mm position.
        pos_px = x, y
        size_px = self.width, self.height
        new_pos_mm, _ = self.canvas.element_coords_to_workpiece_coords(
            pos_px, size_px
        )

        self._in_update = True
        try:
            self.data.pos = new_pos_mm
        finally:
            self._in_update = False

    def set_size(self, width: float, height: float):
        # Update our size for immediate scaled drawing.
        # During a resize drag, the canvas logic updates the element's x, y,
        # width, and height. This method is called after set_pos.
        self.width, self.height = width, height
        if self.canvas:
            self.canvas.queue_draw()

        self.trigger_update()

        if not self.canvas or self._in_update:
            return

        # During a resize, both the element's position and size may change.
        # We use the final element geometry (pos and size) to perform a
        # single, correct update to the model. This overwrites any
        # intermediate position set by set_pos during the same drag event,
        # ensuring the final model state is accurate.
        pos_px = self.x, self.y
        size_px = self.width, self.height
        new_pos_mm, new_size_mm = (
            self.canvas.element_coords_to_workpiece_coords(pos_px, size_px)
        )

        self._in_update = True
        try:
            # Update the model atomically by changing pos and size together.
            # We set size first, as the workpiece's `pos` property might
            # depend on its size for pivot calculations.
            self.data.set_size(*new_size_mm)
            self.data.pos = new_pos_mm
        finally:
            self._in_update = False

    def set_angle(self, angle: float):
        super().set_angle(angle)
        if self._in_update:
            return
        self._in_update = True
        try:
            self.data.angle = angle
        finally:
            self._in_update = False

    def _on_model_data_changed(self, workpiece: WorkPiece):
        """
        Handles data-only changes (like size) from the model. This is an
        expensive operation that triggers a full reallocation and re-render.
        """
        if self._in_update:
            return
        self.allocate(force=True)  # Force re-render for size changes
        if self.parent:
            self.parent.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def _on_transform_changed(self, workpiece: WorkPiece):
        """
        Handles transform-only changes (pos/angle). This is a lightweight
        update that only repositions the element without re-rendering its
        buffered surface.
        """
        if self._in_update or not self.canvas:
            return

        # Just update position and angle, do not call allocate()
        new_pos_px, _ = self.canvas.workpiece_coords_to_element_coords(
            self.data
        )
        super().set_pos(*new_pos_px)
        super().set_angle(self.data.angle)
        if self.parent:
            self.parent.mark_dirty()
        self.canvas.queue_draw()
