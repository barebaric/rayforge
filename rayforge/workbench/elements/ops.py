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
    driven by its corresponding WorkPieceElement in the view hierarchy.
    It is designed to be a perfect overlay for a WorkPieceElement.
    """

    def __init__(
        self, workpiece: WorkPiece, show_travel_moves: bool = False, **kwargs
    ):
        # An Ops element's local geometry is a fixed 1x1 unit square,
        # just like its corresponding WorkPieceElement.
        super().__init__(
            0,
            0,
            1,
            1,
            data=workpiece,
            selectable=False,
            buffered=True,
            # background=(0, 1, 0, 0.2),
            **kwargs,
        )
        self._accumulated_ops = Ops()
        self._ops_generation_id = -1
        self.show_travel_moves = show_travel_moves

        # All content transformation is handled explicitly during rendering.
        # The base class should not apply any transform when drawing the
        # buffer.
        self.content_transform = Matrix.identity()

        # NOTE: This element INTENTIONALLY does not connect to
        # workpiece.transform_changed.
        # Its transform is driven exclusively by calls to
        #  _on_workpiece_transform_changed
        # from the WorkSurface and LayerElement orchestrators.
        self.trigger_update()

    def remove(self):
        """Disconnects signals before removing the element."""
        super().remove()

    def _on_workpiece_transform_changed(
        self,
        workpiece: WorkPiece,
        transient_world_transform: Optional[Matrix] = None,
    ):
        """
        Handles transform changes by calculating the correct local matrix
        that positions the ops content in the world, accounting for margins
        and hierarchy.
        If transient_world_transform is provided, it is used instead of the
        model's transform. This is for interactive updates during a drag.
        """
        if not self.canvas or not self.parent:
            return

        # 1. Determine the desired world transform for the core ops content.
        if transient_world_transform is not None:
            model_world_transform = transient_world_transform
        else:
            model_world_transform = workpiece.get_world_transform()

        # --- FIX: Derive size from the world transform, not the local one ---
        # The absolute scale of the workpiece's world matrix gives its
        # final dimensions in world space (mm), regardless of grouping.
        w, h = model_world_transform.get_abs_scale()

        # The Ops are pre-scaled, so we remove the scale from the workpiece's
        # matrix to get a transform for just rotation and translation.
        scale_inv_matrix = Matrix.scale(
            1 / w if w > 0 else 1, 1 / h if h > 0 else 1
        )
        ops_world_transform = model_world_transform @ scale_inv_matrix

        # 2. Calculate a margin in mm to prevent stroke clipping.
        view_scale_x, view_scale_y = self.canvas.get_view_scale()
        if view_scale_x <= 1e-9 or view_scale_y <= 1e-9:
            # If the scale is invalid, we can't calculate a correct size.
            # Only queue a draw if this is a model-driven update.
            if transient_world_transform is None:
                self.canvas.queue_draw()
            return
        margin_x_mm = OPS_MARGIN_PX / view_scale_x
        margin_y_mm = OPS_MARGIN_PX / view_scale_y

        # 3. Calculate the element's total geometry and an offset transform.
        # The element's geometry must be larger than the ops to include the
        # margin.
        final_width = w + (2 * margin_x_mm)
        final_height = h + (2 * margin_y_mm)
        # To align the inner ops area correctly, we must pre-translate the
        # element's geometry by a negative margin.
        margin_offset_transform = Matrix.translation(
            -margin_x_mm, -margin_y_mm
        )

        # 4. Calculate the final desired world transform for the element
        # itself.
        desired_element_world_transform = (
            ops_world_transform @ margin_offset_transform
        )

        # 5. Convert the desired world transform to a local transform
        # relative to the parent.
        parent_inv_world = self.parent.get_world_transform().invert()
        final_element_local_transform = (
            parent_inv_world @ desired_element_world_transform
        )
        self.set_transform(final_element_local_transform)

        # 6. Update size and content orientation if changed.
        size_changed = (
            abs(self.width - final_width) > 1e-9
            or abs(self.height - final_height) > 1e-9
        )
        if size_changed:
            self.width = final_width
            self.height = final_height
            # This flips the Y-up content to display correctly in the
            # Y-down view.
            self.content_transform = Matrix.translation(
                0, self.height
            ) @ Matrix.scale(1, -1)
            self.trigger_update()

        if transient_world_transform is None:
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
        ops_len = len(ops.commands) if ops else "None"
        logger.debug(
            f"WorkPieceOpsElem for '{self.data.source_file}': set_ops called. "
            f"Received {ops_len} commands. "
            f"Incoming Gen ID: {generation_id}, "
            f"Current Gen ID: {self._ops_generation_id}"
        )

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

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the accumulated ops to a new surface.
        """
        ops_to_render_len = len(self._accumulated_ops.commands)
        logger.debug(
            f"WorkPieceOpsElem for '{self.data.source_file}': "
            f"render_to_surface called. Rendering {ops_to_render_len} "
            "commands."
        )
        if ops_to_render_len > 0:
            logger.debug(f"-> Ops Bbox: {self._accumulated_ops.rect()}")

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

        # The effective pixels-per-mm for this rendering operation is the ratio
        # of the target pixel size to the element's size in world units (mm).
        ppm_x = render_width / self.width if self.width > 0 else 1.0
        ppm_y = render_height / self.height if self.height > 0 else 1.0

        # The ops need to be drawn with a margin inside the buffer.
        margin_x_mm = OPS_MARGIN_PX / ppm_x if ppm_x > 0 else 0
        margin_y_mm = OPS_MARGIN_PX / ppm_y if ppm_y > 0 else 0
        render_ops.translate(margin_x_mm, margin_y_mm)

        # The pixels_per_mm for the CairoEncoder must account for any
        # down-scaling we did to avoid the Cairo surface limit.
        final_pixels_per_mm = (
            ppm_x * scale_factor,
            ppm_y * scale_factor,
        )
        logger.debug(
            f"WorkPieceOpsElem for '{self.data.source_file}': "
            f"render_to_surface. Re-rendering {ops_to_render_len} commands "
            f"PPMM {final_pixels_per_mm}"
        )

        encoder = CairoEncoder()
        encoder.encode(
            render_ops,
            ctx,
            final_pixels_per_mm,
            show_travel_moves=self.show_travel_moves,
        )
        return surface
