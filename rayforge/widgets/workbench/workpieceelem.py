import logging
from typing import Optional, Tuple, TYPE_CHECKING
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
        self._last_pos_mm: Optional[Tuple[float, float]] = None
        self._last_size_mm: Optional[Tuple[float, float]] = None
        x_mm, y_mm = workpiece.pos or (0, 0)
        width_mm, height_mm = workpiece.size or workpiece.get_default_size()
        super().__init__(0, 0, 0, 0, data=workpiece, **kwargs)
        self._last_pos_mm = (x_mm, y_mm)
        self._last_size_mm = (width_mm, height_mm)
        workpiece.size_changed.connect(self.allocate)
        workpiece.changed.connect(self._on_workpiece_changed)

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        """
        Handles changes to the WorkPiece data.

        Updates the element's position and size if they have changed
        significantly.

        Args:
            workpiece: The WorkPiece that has changed.
        """
        if not self.canvas:
            return

        # Get the new position and size in mm.
        x_mm, y_mm = workpiece.pos or (0, 0)
        width_mm, height_mm = workpiece.size or (0, 0)

        # Convert the mm values to pixel values.
        new_x, new_y = self.mm_to_pixel(x_mm, y_mm + height_mm)
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # Check if the position or size has changed significantly.
        if (
            abs(new_x - self.x) >= 1
            or abs(new_y - self.y) >= 1
            or abs(new_width - self.width) >= 1
            or abs(new_height - self.height) >= 1
        ):
            # Update the element's position and size.
            self.x, self.y = new_x, new_y
            self.width, self.height = new_width, new_height

            # Update the last known position and size in mm.
            self._last_pos_mm = (x_mm, y_mm)
            self._last_size_mm = (width_mm, height_mm)

            # Allocate the element and mark it as dirty.
            super().allocate()
            self.mark_dirty()
            self.canvas.queue_draw()


    def _update_workpiece(self):
        """
        Updates the WorkPiece data with the element's current position and size.
        """
        if not self.canvas:
            return

        # Get the element's position and size in pixels.
        x, y, width, height = self.rect_abs()

        # Convert the pixel values to mm values.
        x_mm, y_mm = self.pixel_to_mm(x, y + height)
        width_mm = width / self.canvas.pixels_per_mm_x
        height_mm = height / self.canvas.pixels_per_mm_y

        # Update the WorkPiece's position if it has changed.
        if self._last_pos_mm is None or (x_mm, y_mm) != self._last_pos_mm:
            self.data.set_pos(x_mm, y_mm)
            self._last_pos_mm = (x_mm, y_mm)

        # Update the WorkPiece's size if it has changed.
        if (
            self._last_size_mm is None
            or (width_mm, height_mm) != self._last_size_mm
        ):
            self.data.set_size(width_mm, height_mm)
            self._last_size_mm = (width_mm, height_mm)


    def allocate(self, force: bool = False):
        """
        Allocates the element's position and size based on the WorkPiece data.

        Args:
            force: Whether to force allocation, even if the position and size
                have not changed.
        """
        if not self.canvas:
            return

        # Get the position and size in mm from the WorkPiece.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        # Convert the mm values to pixel values.
        new_x, new_y = self.mm_to_pixel(x_mm, y_mm + height_mm)
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # Check if the position or size has changed significantly.
        if (
            force
            or abs(new_x - self.x) >= 1
            or abs(new_y - self.y) >= 1
            or abs(new_width - self.width) >= 1
            or abs(new_height - self.height) >= 1
        ):
            # Update the element's position and size.
            self.x, self.y = new_x, new_y
            self.width, self.height = new_width, new_height

            # Update the last known position and size in mm.
            self._last_pos_mm = (x_mm, y_mm)
            self._last_size_mm = (width_mm, height_mm)

            # Allocate the element and mark it as dirty.
            super().allocate(force)
            self.mark_dirty()
            self.canvas.queue_draw()

    def render(
        self,
        clip: tuple[float, float, float, float] | None = None,
        force: bool = False,
    ):
        """
        Renders the WorkPiece element to the canvas.

        Args:
            clip: The clipping rectangle, or None for no clipping.
            force: Whether to force rendering, even if the element is not dirty.
        """
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
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