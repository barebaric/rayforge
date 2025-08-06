import logging
import cairo
from typing import Optional
from ...pipeline.encoder.cairoencoder import CairoEncoder
from ...core.ops import Ops
from ...core.workpiece import WorkPiece
from ..canvas import CanvasElement


logger = logging.getLogger(__name__)

OPS_MARGIN_PX = 10.0

# Cairo has a hard limit on surface dimensions, 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 30000


class WorkPieceOpsElement(CanvasElement):
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
        self.width_mm = 0.0
        self.height_mm = 0.0
        self._accumulated_ops = Ops()
        self._ops_generation_id = -1
        self.show_travel_moves = show_travel_moves

        workpiece.pos_changed.connect(self._on_pos_changed)
        workpiece.size_changed.connect(self.allocate)
        workpiece.angle_changed.connect(self._on_angle_changed)

    def _on_pos_changed(self, workpiece: WorkPiece):
        """A lightweight handler for position changes. Does not re-render."""
        if not self.canvas or not self.parent:
            return

        # Recalculate pixel position based on the new mm position.
        pos_px, _ = self.canvas.workpiece_coords_to_element_coords(self.data)

        # set_pos is cheap, it just updates coordinates and marks the parent
        # dirty.
        # This is all that's needed for a move operation.
        self.set_pos(pos_px[0] - OPS_MARGIN_PX, pos_px[1] - OPS_MARGIN_PX)

    def _on_angle_changed(self, workpiece: WorkPiece):
        """A lightweight handler for angle changes. Does not re-render."""
        if not self.canvas:
            return

        # set_angle is cheap, it just updates the angle property.
        self.set_angle(self.data.angle)

    def allocate(self, force: bool = False):
        """
        Updates position and size. Triggers a re-render. If the workpiece's
        size in millimeters changes, the current ops are cleared. A simple
        canvas zoom will not clear the ops.
        """
        if not self.canvas or not self.parent:
            return

        current_mm_size = self.data.get_current_size()

        if not current_mm_size:
            self.width, self.height = 0.0, 0.0
            self.width_mm, self.height_mm = 0.0, 0.0
            self.clear_ops()
            return

        pos_px, size_px = self.canvas.workpiece_coords_to_element_coords(
            self.data
        )

        # Check if the fundamental size in mm has changed by comparing against
        # the values stored in this class.
        mm_size_changed = (self.width_mm, self.height_mm) != current_mm_size

        new_width = size_px[0] + 2 * OPS_MARGIN_PX
        new_height = size_px[1] + 2 * OPS_MARGIN_PX

        pixel_size_changed = round(self.width) != round(new_width) or round(
            self.height
        ) != round(new_height)

        # Update position and angle here as well, to ensure they are correct
        # after a size change.
        self.set_pos(pos_px[0] - OPS_MARGIN_PX, pos_px[1] - OPS_MARGIN_PX)
        self.set_angle(self.data.angle)

        if not pixel_size_changed and not force:
            return

        # If the workpiece's actual mm size changed, the existing ops are
        # invalid and must be cleared. This will NOT trigger on a canvas zoom.
        if mm_size_changed:
            self.clear_ops()

        # Update the state in this class.
        self.width_mm, self.height_mm = current_mm_size
        self.width, self.height = new_width, new_height
        super().allocate(force)

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
                f"'{self.data.name}', current is {self._ops_generation_id}"
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
                f"'{self.data.name}', replacing current (gen "
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
                f"'{self.data.name}', current is {self._ops_generation_id}"
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
        Custom draw method to handle intermediate scaling correctly.
        It preserves the pixel margin by scaling only the content area.
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

        # If the surface and element size are identical, we can use the fast
        # default drawing method.
        if round(self.width) == source_w and round(self.height) == source_h:
            super().draw(ctx)
            return

        # --- Smart scaling logic for intermediate drawing ---

        # Calculate the dimensions of the actual content, excluding the margin.
        source_content_w = source_w - 2 * OPS_MARGIN_PX
        source_content_h = source_h - 2 * OPS_MARGIN_PX
        dest_content_w = self.width - 2 * OPS_MARGIN_PX
        dest_content_h = self.height - 2 * OPS_MARGIN_PX

        if source_content_w <= 0 or source_content_h <= 0:
            return  # Cannot scale if source content has no area.

        # Calculate the scaling factor based on the content areas.
        scale_x = dest_content_w / source_content_w
        scale_y = dest_content_h / source_content_h

        ctx.save()
        # 1. Translate to the destination content's top-left corner.
        ctx.translate(OPS_MARGIN_PX, OPS_MARGIN_PX)
        # 2. Scale the context to match the content area's new size.
        ctx.scale(scale_x, scale_y)
        # 3. Set the source surface, but shift it left/up by the margin.
        #    This aligns the source content's top-left (which is at
        #    (margin, margin) on the surface) with the current origin (0,0)
        #    of our translated and scaled context.
        ctx.set_source_surface(self.surface, -OPS_MARGIN_PX, -OPS_MARGIN_PX)

        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the accumulated ops to a new surface.
        """
        if width <= 0 or height <= 0:
            return None

        render_width, render_height = width, height
        scale_factor = 1.0

        # If the requested render size exceeds Cairo's hard limit, we must
        # scale it down to prevent a crash. The UI layer will scale the
        # resulting (smaller) surface back up, resulting in pixelation,
        # which is an acceptable trade-off at extreme zoom levels.
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

            new_width = int(render_width * scale_factor)
            new_height = int(render_height * scale_factor)

            render_width = max(1, new_width)
            render_height = max(1, new_height)

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, render_width, render_height
        )
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(*self.background)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()

        render_ops = self._accumulated_ops.copy()
        if (
            not render_ops
            or not self.canvas
            or not self.canvas.pixels_per_mm_x
        ):
            return surface

        px_per_mm_x = self.canvas.pixels_per_mm_x or 1
        px_per_mm_y = self.canvas.pixels_per_mm_y or 1

        # This translation of the millimeter-based Ops data is correct for
        # the final render, as the CairoEncoder will scale this mm value
        # by the ppm, resulting in a constant pixel margin.
        margin_mm_x = OPS_MARGIN_PX / px_per_mm_x
        margin_mm_y = OPS_MARGIN_PX / px_per_mm_y
        render_ops.translate(margin_mm_x, margin_mm_y)

        # We must scale the pixels_per_mm value by the same factor we scaled
        # the surface, otherwise the toolpaths will be drawn too large.
        scaled_pixels_per_mm = (
            px_per_mm_x * scale_factor,
            px_per_mm_y * scale_factor,
        )

        encoder = CairoEncoder()
        encoder.encode(
            render_ops,
            ctx,
            scaled_pixels_per_mm,
            show_travel_moves=self.show_travel_moves,
        )
        return surface
