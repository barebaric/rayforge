import logging
from ...config import config
from ...opsencoder.cairoencoder import CairoEncoder
from ...models.ops import Ops
from ...models.workpiece import WorkPiece
from ..canvas import CanvasElement


logger = logging.getLogger(__name__)


class WorkPieceOpsElement(CanvasElement):
    """Displays the generated Ops for a single WorkPiece."""
    def __init__(self, workpiece: WorkPiece, **kwargs):
        """
        Initializes a WorkPieceOpsElement.

        Args:
            workpiece: The WorkPiece data object.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        if not workpiece.size:
            raise AttributeError(f"attempt to add workpiece {workpiece.name} with no size")
        super().__init__(0,
                         0,
                         0,
                         0,
                         data=workpiece,
                         selectable=False,
                         **kwargs)
        self._accumulated_ops = Ops()
        workpiece.changed.connect(self._on_workpiece_changed)

    def allocate(self, force: bool = False):
        """Updates the element's position and size based on the workpiece."""
        if not self.canvas:
            return

        # Even though allocate() does not require the position, we update
        # it here anyway to do it as early as possible. We need to update
        # because the pixels_per_mm in the canvas may have changed, requiring
        # a re-caclulation of the positions in pixel.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or (0, 0)
        self.x, self.y = self.canvas._mm_to_canvas_pixel(x_mm, y_mm+height_mm)

        # Calculate the size of the surface.
        width_mm, height_mm = self.data.size or (0, 0)
        pixels_per_mm_x = self.canvas.pixels_per_mm_x
        pixels_per_mm_y = self.canvas.pixels_per_mm_y
        width_px = round(width_mm * pixels_per_mm_x)
        height_px = round(height_mm * pixels_per_mm_y)
        self.width, self.height = width_px, height_px

        super().allocate(force)

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        # Workpiece position and/or size in mm changed.
        if not self.canvas:
            return
        self.allocate()
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

    def render(self, clip: tuple[float, float, float, float] | None = None, force: bool = False):
        """Renders the accumulated Ops to the element's surface."""
        logger.debug(f"WorkPieceOpsElement.render: Workpiece {self.data.name}. clip={clip}, force={force}")
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            return

        # Clear the surface.
        clip = clip or self.rect()
        self.clear_surface(clip)

        if not self._accumulated_ops:
            return

        # Get pixels_per_mm from the WorkSurface (self.canvas)
        pixels_per_mm = self.canvas.pixels_per_mm_x, self.canvas.pixels_per_mm_y

        encoder = CairoEncoder()
        show_travel = self.canvas.show_travel_moves if self.canvas else False
        encoder.encode(self._accumulated_ops,
                       config.machine,
                       self.surface,
                       pixels_per_mm,
                       show_travel_moves=show_travel)