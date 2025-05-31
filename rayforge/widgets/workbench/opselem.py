import logging
from ...config import config
from ...opsencoder.cairoencoder import CairoEncoder
from ...models.ops import Ops
from ...models.workpiece import WorkPiece
from .surfaceelem import SurfaceElement


logger = logging.getLogger(__name__)


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
        super().__init__(0,
                         0,
                         0,
                         0,
                         data=workpiece,
                         selectable=False,
                         **kwargs)
        self._accumulated_ops = Ops()
        workpiece.changed.connect(self.allocate)

    def allocate(self, force: bool = False):
        """Updates the element's position and size based on the workpiece."""
        if not self.canvas or not self.parent:
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

        # Allocate the element and mark it as dirty.
        super().allocate(force)
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
        clip: tuple[float, float, float, float] | None = None,
        force: bool = False,
    ):
        """Renders the accumulated Ops to the element's surface."""
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
        pixels_per_mm = (
            self.canvas.pixels_per_mm_x,
            self.canvas.pixels_per_mm_y,
        )

        encoder = CairoEncoder()
        show_travel = self.canvas.show_travel_moves if self.canvas else False
        encoder.encode(self._accumulated_ops,
                       config.machine,
                       self.surface,
                       pixels_per_mm,
                       show_travel_moves=show_travel)
