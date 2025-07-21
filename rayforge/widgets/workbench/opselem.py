import logging
import cairo
from typing import Optional
from ...config import config
from ...opsencoder.cairoencoder import CairoEncoder
from ...models.ops import Ops
from ...models.workpiece import WorkPiece
from .surfaceelem import SurfaceElement


logger = logging.getLogger(__name__)

OPS_MARGIN_PX = 10


class WorkPieceOpsElement(SurfaceElement):
    """
    Displays the generated Ops for a single WorkPiece.
    """

    def __init__(
        self, workpiece: WorkPiece, show_travel_moves: bool = False, **kwargs
    ):
        if not workpiece.size:
            raise AttributeError(
                f"attempt to add workpiece {workpiece.name} with no size"
            )
        super().__init__(
            0,
            0,
            0,
            0,
            data=workpiece,
            selectable=False,
            buffered=True,
            **kwargs,
        )
        self._accumulated_ops = Ops()
        self.show_travel_moves = show_travel_moves

        # Connect to specific signals instead of the generic 'changed' signal.
        workpiece.pos_changed.connect(self.allocate)
        workpiece.size_changed.connect(self.allocate)

    def allocate(self, force: bool = False):
        """
        Updates position and size. Triggers a re-render ONLY if the size
        has changed or if `force=True`.
        """
        if not self.canvas or not self.parent:
            return

        x_mm, y_mm = self.data.pos or (0, 0)
        width_mm, height_mm = self.data.size or self.data.get_default_size()

        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1
        width_px = round(width_mm * px_per_mm_x)
        height_px = round(height_mm * px_per_mm_y)

        new_width = width_px + 2 * OPS_MARGIN_PX
        new_height = height_px + 2 * OPS_MARGIN_PX
        size_changed = self.width != new_width or self.height != new_height

        x_mm_tl, y_mm_tl = x_mm, y_mm + height_mm

        # Convert mm (machine coords, origin bottom-left) to canvas-relative
        # pixel coordinates (origin top-left).
        content_height_px = self.canvas.root.height

        x_px = x_mm_tl * px_per_mm_x
        y_px = content_height_px - y_mm_tl * px_per_mm_y

        self.set_pos(round(x_px) - OPS_MARGIN_PX, round(y_px) - OPS_MARGIN_PX)

        if not size_changed and not force:
            return

        self.width, self.height = new_width, new_height
        super().allocate(force)

    def clear_ops(self):
        self._accumulated_ops = Ops()
        self.clear_surface()
        self.trigger_update()

    def add_ops(self, ops_chunk: Ops):
        if not ops_chunk:
            return
        self._accumulated_ops += ops_chunk
        self.trigger_update()

    def set_show_travel_moves(self, show: bool):
        """
        Sets the travel move visibility and triggers a re-render if changed.
        """
        if self.show_travel_moves != show:
            self.show_travel_moves = show
            self.trigger_update()

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the accumulated ops to a new surface. This runs in a
        background thread.
        """
        if width <= 0 or height <= 0:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(*self.background)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()

        render_ops = self._accumulated_ops.copy()
        if not render_ops or not self.canvas:
            return surface

        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1
        pixels_per_mm = px_per_mm_x, px_per_mm_y

        margin_mm_x = OPS_MARGIN_PX / px_per_mm_x
        margin_mm_y = OPS_MARGIN_PX / px_per_mm_y
        render_ops.translate(margin_mm_x, margin_mm_y)

        encoder = CairoEncoder()
        encoder.encode(
            render_ops,
            config.machine,
            ctx,
            pixels_per_mm,
            show_travel_moves=self.show_travel_moves,
        )
        return surface
