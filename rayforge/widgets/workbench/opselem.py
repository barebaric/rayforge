import logging
from typing import Optional, Tuple
from ...config import config
from ...opsencoder.cairoencoder import CairoEncoder
from ...models.ops import Ops
from ...models.workpiece import WorkPiece
from .surfaceelem import SurfaceElement


logger = logging.getLogger(__name__)

# This margin defines an area around the element into which Ops can draw.
# This is because the Ops path may be slightly larger than the workpiece,
# and we need to prevent it from being clipped.
OPS_MARGIN_PX = 10


class WorkPieceOpsElement(SurfaceElement):
    """Displays the generated Ops for a single WorkPiece."""

    def __init__(self, workpiece: WorkPiece, **kwargs):
        """
        Initializes a WorkPieceOpsElement.

        Args:
            workpiece: The WorkPiece data object.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        if not workpiece.size:
            raise AttributeError(
                f"attempt to add workpiece {workpiece.name} with no size"
            )
        super().__init__(
            0, 0, 0, 0, data=workpiece, selectable=False, **kwargs
        )
        self._accumulated_ops = Ops()
        workpiece.changed.connect(self.allocate)

    def allocate(self, force: bool = False):
        """
        Updates the element's position and size based on the workpiece.

        This method is optimized to distinguish between cheap position updates
        and expensive size updates. A size change requires re-allocating the
        surface and re-rendering the vector graphics, while a position change
        only requires moving the existing surface.
        """
        if not self.canvas or not self.parent:
            return

        # Get the position and size in mm from the WorkPiece.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        # Convert the mm dimensions to pixel dimensions.
        width_px = round(width_mm * self.canvas.pixels_per_mm_x)
        height_px = round(height_mm * self.canvas.pixels_per_mm_y)

        new_width = width_px + 2 * OPS_MARGIN_PX
        new_height = height_px + 2 * OPS_MARGIN_PX

        size_changed = self.width != new_width or self.height != new_height

        # Always update the element's position. This is a cheap operation.
        x_px, y_px = self.mm_to_pixel(x_mm, y_mm + height_mm)
        self.set_pos(x_px - OPS_MARGIN_PX, y_px - OPS_MARGIN_PX)

        # If the size hasn't changed, we don't need to do the expensive
        # re-rendering. We just queue a draw to show the new position.
        if not size_changed and not force:
            self.canvas.queue_draw()
            return

        # --- Expensive Operations ---
        # If the size HAS changed, we must re-allocate the surface and
        # mark the element as dirty to trigger a full content re-render.
        self.width = new_width
        self.height = new_height

        # Allocate the element's surface with the new, larger dimensions.
        super().allocate(force)
        # Mark dirty to schedule a call to the expensive render() method.
        self.mark_dirty()

        self.canvas.queue_draw()

    def clear_ops(self):
        """Clears the accumulated operations and the drawing surface."""
        self._accumulated_ops = Ops()
        self.clear_surface()
        self.mark_dirty()

    def add_ops(self, ops_chunk: Ops):
        """Adds a chunk of operations to the accumulated total."""
        if not ops_chunk:
            return
        self._accumulated_ops += ops_chunk
        self.mark_dirty()

    def render(
        self,
        clip: Optional[Tuple[int, int, int, int]] = None,
        force: bool = False,
    ):
        """
        Renders the accumulated Ops to the element's surface, translating
        them to fit within the margin. This is an expensive method and should
        only be called when the content actually changes.
        """
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            return

        # Always clear the entire surface before drawing.
        self.clear_surface()

        if not self._accumulated_ops:
            self.dirty = False  # Cleared, nothing to draw, so we are clean.
            return

        # Get pixels_per_mm from the WorkSurface (self.canvas)
        pixels_per_mm_x = self.canvas.pixels_per_mm_x
        pixels_per_mm_y = self.canvas.pixels_per_mm_y
        pixels_per_mm = (pixels_per_mm_x, pixels_per_mm_y)

        # Create a temporary copy to avoid modifying the cached Ops.
        render_ops = self._accumulated_ops.copy()

        # Convert the pixel margin to mm to perform the translation.
        margin_mm_x = OPS_MARGIN_PX / pixels_per_mm_x if pixels_per_mm_x else 0
        margin_mm_y = OPS_MARGIN_PX / pixels_per_mm_y if pixels_per_mm_y else 0

        # Translate the ops by the margin amount.
        render_ops.translate(margin_mm_x, margin_mm_y)

        encoder = CairoEncoder()
        show_travel = self.canvas.show_travel_moves if self.canvas else False

        # Encode the translated ops.
        encoder.encode(
            render_ops,
            config.machine,
            self.surface,
            pixels_per_mm,
            show_travel_moves=show_travel,
        )

        # Mark the element as clean after rendering.
        self.dirty = False
