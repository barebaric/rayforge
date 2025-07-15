import logging
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
        Updates the element's position and size based on the workpiece,
        adding a margin to prevent clipping.
        """
        if not self.canvas or not self.parent:
            return

        # Get the position and size in mm from the WorkPiece.
        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        # Convert the mm dimensions to pixel dimensions.
        width_px = round(width_mm * self.canvas.pixels_per_mm_x)
        height_px = round(height_mm * self.canvas.pixels_per_mm_y)

        # Adjust the element size by the margin
        self.width = width_px + 2 * OPS_MARGIN_PX
        self.height = height_px + 2 * OPS_MARGIN_PX

        # Adjust the element's position to center the content.
        # We need to shift the element's top-left corner up and left by
        # the margin.
        # First, get the pixel position for the workpiece's top-left corner.
        x_px, y_px = self.mm_to_pixel(x_mm, y_mm + height_mm)
        # Then, set the element's position, offsetting by the margin.
        self.set_pos(x_px - OPS_MARGIN_PX, y_px - OPS_MARGIN_PX)

        # Allocate the element's surface with the new, larger dimensions.
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
        """
        Renders the accumulated Ops to the element's surface, translating
        them to fit within the margin.
        """
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
        pixels_per_mm_x = self.canvas.pixels_per_mm_x
        pixels_per_mm_y = self.canvas.pixels_per_mm_y
        pixels_per_mm = (pixels_per_mm_x, pixels_per_mm_y)

        # --- MODIFIED: Translate Ops to draw inside the margin ---
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
