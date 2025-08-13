import logging
import cairo
from typing import Optional
from ...pipeline.encoder.cairoencoder import CairoEncoder
from ...core.ops import Ops
from ...core.matrix import Matrix
from ...core.workpiece import WorkPiece
from ..canvas import CanvasElement


logger = logging.getLogger(__name__)

OPS_MARGIN_PX = 10.0

# Cairo has a hard limit on surface dimensions, 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 30000


class WorkPieceOpsElement(CanvasElement):
    """
    Displays the generated Ops for a single WorkPiece. Its transform is
    driven by the WorkPiece model.
    """

    def __init__(
        self, workpiece: WorkPiece, show_travel_moves: bool = False, **kwargs
    ):
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
        self._ops_generation_id = -1
        self.show_travel_moves = show_travel_moves

        # Connect to the signals from the WorkPiece model
        workpiece.changed.connect(self._on_workpiece_changed)
        workpiece.transform_changed.connect(
            self._on_workpiece_transform_changed
        )

        # Set initial state
        self._on_workpiece_transform_changed(workpiece)
        self.trigger_update()

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        """
        Handles data changes (e.g., size) that require ops to be regenerated.
        We clear our current ops; the OpsGenerator will provide new ones.
        """
        self.clear_ops()

    def _on_workpiece_transform_changed(self, workpiece: WorkPiece):
        """
        A lightweight handler for transform (pos/angle/size) changes.
        It updates the element's matrix and local geometry to match the model.
        """
        if not self.canvas or not self.parent:
            return

        # 1. The ops element's transform is based on the workpiece's transform,
        #    but shifted by a margin.
        w, h = workpiece.size
        world_transform = workpiece.get_world_transform()

        # Convert the pixel margin to local (mm) units using the matrix scale.
        sx, sy = world_transform.get_scale()
        # To get the transform's scale factor, we need to divide by the local
        # geometry's size (w,h)
        scale_x_factor = sx / w if w > 0 else 1.0
        scale_y_factor = sy / h if h > 0 else 1.0

        margin_x_mm = (
            OPS_MARGIN_PX / scale_x_factor if scale_x_factor > 0 else 0
        )
        margin_y_mm = (
            OPS_MARGIN_PX / scale_y_factor if scale_y_factor > 0 else 0
        )

        # Create a local transform to apply the margin offset.
        # This translates the element left/up by the margin amount.
        offset_transform = Matrix.translation(-margin_x_mm, -margin_y_mm)

        # The final transform is the workpiece's world transform composed
        # with our local offset.
        self.set_transform(world_transform @ offset_transform)

        # 2. The element's local geometry must now include the margin.
        new_width = w + (2 * margin_x_mm)
        new_height = h + (2 * margin_y_mm)
        size_changed = self.width != new_width or self.height != new_height

        if size_changed:
            self.width, self.height = new_width, new_height
            # A size change invalidates the buffer, so we must re-render.
            self.trigger_update()

        self.canvas.queue_draw()

    def clear_ops(self, generation_id: Optional[int] = None):
        """Clears ops. If a generation_id is provided, it is stored."""
        self._accumulated_ops = Ops()
        if generation_id is not None:
            self._ops_generation_id = generation_id
        self.clear_surface()
        self.trigger_update()

    def set_ops(self, ops: Optional[Ops], generation_id: Optional[int] = None):
        """Replaces all current ops, but only if generation_id is current."""
        if (
            generation_id is not None
            and generation_id < self._ops_generation_id
        ):
            logger.debug(
                f"Ignoring stale final ops (gen {generation_id}) for "
                f"'{self.data.source_file}', "
                f"current is {self._ops_generation_id}"
            )
            return
        if generation_id is not None:
            self._ops_generation_id = generation_id
        self._accumulated_ops = ops or Ops()
        self.trigger_update()

    def add_ops(self, ops_chunk: Ops, generation_id: Optional[int] = None):
        """Adds a chunk of ops, but only if the generation_id is not stale."""
        if not ops_chunk:
            return

        # If no generation_id is provided, we can't perform staleness checks.
        # Just append the chunk.
        if generation_id is None:
            self._accumulated_ops += ops_chunk
            self.trigger_update()
            return

        # Case 1: The chunk is from a newer generation. This means we either
        # missed the `ops_generation_starting` signal or a new generation
        # started. We should clear the old ops and start accumulating for
        # the new generation.
        if generation_id > self._ops_generation_id:
            logger.debug(
                f"New generation chunk (gen {generation_id}) for "
                f"'{self.data.source_file}', replacing current (gen "
                f"{self._ops_generation_id})."
            )
            self._ops_generation_id = generation_id
            self._accumulated_ops = ops_chunk.copy()
            self.trigger_update()
            return

        # Case 2: The chunk is from a past generation. It's stale and should
        # be ignored.
        if generation_id < self._ops_generation_id:
            logger.debug(
                f"Ignoring stale ops chunk (gen {generation_id}) for "
                f"'{self.data.source_file}', "
                f"current is {self._ops_generation_id}"
            )
            return

        # Case 3: The chunk belongs to the current generation. Append it.
        # This is the expected behavior for subsequent chunks.
        # (This block executes if generation_id == self._ops_generation_id)
        self._accumulated_ops += ops_chunk
        self.trigger_update()

    def set_show_travel_moves(self, show: bool):
        """
        Sets the travel move visibility and triggers a re-render if changed.
        """
        if self.show_travel_moves != show:
            self.show_travel_moves = show
            self.trigger_update()

    def draw(self, ctx: cairo.Context):
        """
        Draws the buffered surface. Since the transform is already handled
        by the parent render() method, we just need to paint the surface.
        """
        if not self.surface:
            # Draw background if no surface, then stop.
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
            return

        source_w, source_h = (
            self.surface.get_width(),
            self.surface.get_height(),
        )
        if (
            source_w <= 0
            or source_h <= 0
            or self.width <= 0
            or self.height <= 0
        ):
            return

        # The parent render method has already applied the transform.
        # We draw into the local geometry space (0,0) to (width, height).
        ctx.save()
        scale_x = self.width / source_w
        scale_y = self.height / source_h
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(self.surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the accumulated ops to a new surface. The margin is handled
        by transforming the ops before rendering.
        """
        if width <= 0 or height <= 0:
            return None

        render_width, render_height = width, height
        scale_factor = 1.0

        if (
            render_width > CAIRO_MAX_DIMENSION
            or render_height > CAIRO_MAX_DIMENSION
        ):
            if render_width > CAIRO_MAX_DIMENSION:
                scale_factor = CAIRO_MAX_DIMENSION / render_width
            if render_height > CAIRO_MAX_DIMENSION:
                scale_factor = min(
                    scale_factor, CAIRO_MAX_DIMENSION / render_height
                )

            render_width = max(1, int(render_width * scale_factor))
            render_height = max(1, int(render_height * scale_factor))

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, render_width, render_height
        )
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(*self.background)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()

        render_ops = self._accumulated_ops.copy()
        if not render_ops or not self.canvas:
            return surface

        # The ops need to be drawn with a margin inside the buffer.
        # We use the element's current scale to convert the fixed pixel
        # margin into local (mm) units for the transformation.
        sx, sy = self.transform.get_scale()
        scale_x_factor = sx / self.width if self.width > 0 else 1.0
        scale_y_factor = sy / self.height if self.height > 0 else 1.0

        margin_x_mm = (
            OPS_MARGIN_PX / scale_x_factor if scale_x_factor > 0 else 0
        )
        margin_y_mm = (
            OPS_MARGIN_PX / scale_y_factor if scale_y_factor > 0 else 0
        )
        render_ops.translate(margin_x_mm, margin_y_mm)

        # The pixels_per_mm for the CairoEncoder needs to be the element's
        # world scale factor, adjusted by any down-scaling we did to avoid
        # the Cairo surface limit.
        scaled_pixels_per_mm = (
            scale_x_factor * scale_factor,
            scale_y_factor * scale_factor,
        )

        encoder = CairoEncoder()
        encoder.encode(
            render_ops,
            ctx,
            scaled_pixels_per_mm,
            show_travel_moves=self.show_travel_moves,
        )
        return surface
