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
        self._cached_surface: Optional[cairo.ImageSurface] = None

    def allocate(self, force: bool = False):
        """
        Allocates the element's position and size based on the WorkPiece data.
        This is the "model to view" update path.
        """
        if not self.canvas or self._in_update:
            return

        # Get the position and size in mm from the WorkPiece.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        # Convert the mm values to pixel values.
        new_width = round(width_mm * self.canvas.pixels_per_mm_x)
        new_height = round(height_mm * self.canvas.pixels_per_mm_y)

        # Update the element's position and size. The data model's Y is the
        # bottom edge, so we add height to get the top edge for positioning.
        self.set_pos_mm(x_mm, y_mm + height_mm)
        self.width, self.height = new_width, new_height

        # Create the surface for the new element.
        super().allocate(force)

    def render(
        self,
        clip: tuple[int, int, int, int] | None = None,
        force: bool = False,
    ):
        if not self.dirty and not force:
            return
        if (
            not self.canvas
            or self.width <= 0
            or self.height <= 0
            or self.surface is None
        ):
            return

        # Determine if we need a better quality cache
        needs_new_cache = False
        if self._cached_surface is None:
            needs_new_cache = True
        else:
            # Re-render if the element's size has changed by more than
            # a 5px threshold. This avoids frequent re-renders on minor
            # zoom changes but ensures the cache is updated when the size
            # difference becomes significant.
            cache_w = self._cached_surface.get_width()
            cache_h = self._cached_surface.get_height()
            if abs(self.width - cache_w) > 5 or abs(self.height - cache_h) > 5:
                needs_new_cache = True

        if needs_new_cache or force:
            logger.debug(f"Cache miss/refresh for element. Requesting new "
                         f"surface at {self.width}x{self.height}px.")
            # Request a new surface from the WorkPiece's generic renderer.
            self._cached_surface = self.data.renderer.render_to_pixels(
                self.data.data,
                width=self.width,
                height=self.height
            )

        # If we still don't have a surface, we can't draw.
        if self._cached_surface is None:
            return

        # Always clear the entire surface before drawing.
        self.clear_surface()
        ctx = cairo.Context(self.surface)

        if clip:
            ctx.rectangle(*clip)
            ctx.clip()

        source_w = self._cached_surface.get_width()
        source_h = self._cached_surface.get_height()

        scale_x = self.width / source_w if source_w > 0 else 0
        scale_y = self.height / source_h if source_h > 0 else 0

        ctx.save()
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(self._cached_surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_BEST)
        ctx.paint()
        ctx.restore()

        self.dirty = False

    def set_pos(self, x: int, y: int):
        """
        Sets the position of the element in pixels and updates the model.
        This is the "view to model" update path for position.
        """
        super().set_pos(x, y)
        if not self.canvas or self._in_update:
            return

        x_mm, y_mm = self.pixel_to_mm(x, y)
        self._in_update = True
        try:
            # Only update the position, not the size.
            self.data.set_pos(x_mm, y_mm)
        finally:
            self._in_update = False

    def set_size(self, width: int, height: int):
        """
        Sets the size of the element in pixels and updates the model,
        adjusting the model's position to keep the top-left corner fixed.
        This is the "view to model" update path for size.
        """
        super().set_size(width, height)
        if not self.canvas or self._in_update:
            return

        # Get old model state before changes
        old_x_mm, old_y_mm = self.data.pos or (0, 0)
        _, old_height_mm = self.data.size or self.data.get_default_size()

        # Calculate new model size from view pixels
        new_width_mm = width / self.canvas.pixels_per_mm_x
        new_height_mm = height / self.canvas.pixels_per_mm_y

        # The data model's y-position is its bottom edge. To keep the top
        # edge stationary during a resize, we must adjust the bottom edge's
        # position based on the change in height.
        new_y_mm = (old_y_mm + old_height_mm) - new_height_mm

        self._in_update = True
        try:
            # Atomically update both position and size in the model to reflect
            # the resize operation correctly.
            self.data.set_pos(old_x_mm, new_y_mm)
            self.data.set_size(new_width_mm, new_height_mm)
        finally:
            self._in_update = False

    def _on_workpiece_size_changed(self, workpiece):
        """
        Handles workpiece size changes from the model and triggers a redraw.
        """
        if self._in_update:
            return
        self.allocate()
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def _on_workpiece_pos_changed(self, workpiece):
        """
        Handles workpiece position changes from the model and updates the
        element.
        """
        if self._in_update:
            return
        if not self.parent:
            return
        self.allocate()
        self.parent.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()
