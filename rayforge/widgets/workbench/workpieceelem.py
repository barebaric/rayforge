import logging
from typing import Optional, TYPE_CHECKING
from ...models.workpiece import WorkPiece
from .surfaceelem import SurfaceElement
from .util import copy_surface


if TYPE_CHECKING:
    from .surface import WorkSurface


logger = logging.getLogger(__name__)


class WorkPieceElement(SurfaceElement):
    """
    A CanvasElement that displays a WorkPiece.

    It handles position and size updates based on the WorkPiece data,
    and uses _copy_surface to render the WorkPiece's surface.
    """
    def __init__(self, workpiece: WorkPiece, **kwargs):
        """
        Initializes a new WorkPieceElement.

        Args:
            workpiece: The WorkPiece to display.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        self.canvas: Optional["WorkSurface"]
        self.data: WorkPiece = workpiece
        self._in_update = False
        super().__init__(0, 0, 0, 0, data=workpiece, **kwargs)
        workpiece.size_changed.connect(self._on_workpiece_size_changed)
        workpiece.pos_changed.connect(self._on_workpiece_pos_changed)

    def _update_workpiece(self):
        """
        Updates the WorkPiece data with the element's current position and
        size.
        """
        if not self.canvas:
            return

        # Get the element's position and size in pixels.
        x, y, width, height = self.rect()

        # Convert the pixel values to mm values.
        x_mm, y_mm = self.pixel_to_mm(x, y)
        width_mm = width / self.canvas.pixels_per_mm_x
        height_mm = height / self.canvas.pixels_per_mm_y

        self._in_update = True
        try:
            self.data.set_pos(x_mm, y_mm)
            self.data.set_size(width_mm, height_mm)
        finally:
            self._in_update = False

    def allocate(self, force: bool = False):
        """
        Allocates the element's position and size based on the WorkPiece data.

        Args:
            force: Whether to force allocation, even if the position and size
                have not changed.
        """
        if not self.canvas:
            return
        if self._in_update:
            return

        # Get the position and size in mm from the WorkPiece.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        # Convert the mm values to pixel values.
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # Update the element's position and size.
        self.set_pos_mm(x_mm, y_mm+height_mm)
        self.width, self.height = new_width, new_height

        # Create the surface for the new element.
        super().allocate(force)

    def render(
        self,
        clip: tuple[int, int, int, int] | None = None,
        force: bool = False,
    ):
        """
        Renders the WorkPiece element to the canvas.

        Args:
            clip: The clipping rectangle, or None for no clipping.
            force: Whether to force rendering, even if the element is not
            dirty.
        """
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            logger.debug(
                "WorkPieceElement.render: canvas, parent, or surface is None"
            )
            return

        surface, changed = self.data.render(
            self.canvas.pixels_per_mm_x,
            self.canvas.pixels_per_mm_y,
            (self.width, self.height),
        )
        if not changed or surface is None:
            return

        self.clear_surface(clip or self.rect())
        self.surface = copy_surface(
            surface,
            self.surface,
            self.width,
            self.height,
            clip or (0, 0, self.width, self.height),
        )
        self.dirty = False

    def set_pos(self, x: int, y: int):
        """
        Sets the position of the element in pixels.

        Args:
            x: The new x-coordinate in pixels.
            y: The new y-coordinate in pixels.
        """
        super().set_pos(x, y)
        self._update_workpiece()

    def set_size(self, width: int, height: int):
        """
        Sets the size of the element in pixels.

        Args:
            width: The new width in pixels.
            height: The new height in pixels.
        """
        super().set_size(width, height)
        self._update_workpiece()

    def _on_workpiece_size_changed(self, workpiece):
        """
        Handles workpiece size changes and triggers a redraw.
        """
        self.allocate()
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def _on_workpiece_pos_changed(self, workpiece):
        """
        Handles workpiece position changes and updates the element's position.
        """
        if not self.parent:
            return
        self.allocate()
        self.parent.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()
