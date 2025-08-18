import logging
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast
import cairo
from concurrent.futures import Future

from ...core.workpiece import WorkPiece
from ...core.step import Step
from ...core.matrix import Matrix
from ...core.ops import Ops
from ..canvas import CanvasElement
from ...pipeline.encoder.cairoencoder import CairoEncoder

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ...pipeline.generator import OpsGenerator

logger = logging.getLogger(__name__)

# Cairo has a hard limit on surface dimensions.
CAIRO_MAX_DIMENSION = 30000
OPS_MARGIN_PX = 5


class WorkPieceView(CanvasElement):
    """A unified CanvasElement that visualizes a single WorkPiece model.

    This class customizes its rendering by overriding the `draw`
    method to correctly handle the coordinate system transform (from the
    canvas's Y-Up world to Cairo's Y-Down surfaces) for both the base
    image and all ops overlays.

    By setting `clip=False`, this element signals to the base `render`
    method that its drawing should not be clipped to its geometric bounds.
    This allows the ops margin to be drawn correctly.
    """

    def __init__(
        self, workpiece: WorkPiece, ops_generator: "OpsGenerator", **kwargs
    ):
        """Initializes the WorkPieceView.

        Args:
            workpiece: The WorkPiece data model to visualize.
            ops_generator: The generator responsible for creating ops.
            **kwargs: Additional arguments for the CanvasElement.
        """
        logger.debug(f"Initializing WorkPieceView for '{workpiece.name}'")
        self.data: WorkPiece = workpiece
        self.ops_generator = ops_generator
        self._base_image_visible = True

        self._ops_surfaces: Dict[str, Optional[cairo.ImageSurface]] = {}
        self._ops_visibility: Dict[str, bool] = {}
        self._ops_render_futures: Dict[str, Future] = {}
        self._ops_generation_ids: Dict[
            str, int
        ] = {}  # Tracks the *expected* generation ID of the *next* render.

        # The element's geometry is a 1x1 unit square.
        # The transform matrix handles all scaling and positioning.
        super().__init__(
            0.0,
            0.0,
            1.0,
            1.0,
            data=workpiece,
            # CRITICAL: clip must be False so the parent `render` method
            # does not clip the drawing, allowing margins to show.
            clip=False,
            buffered=True,
            pixel_perfect_hit=True,
            **kwargs,
        )

        self.content_transform = Matrix.translation(0, 1) @ Matrix.scale(1, -1)

        self.data.updated.connect(self._on_model_content_changed)
        self.data.transform_changed.connect(self._on_transform_changed)
        self.ops_generator.ops_generation_starting.connect(
            self._on_ops_generation_starting
        )
        self.ops_generator.ops_chunk_available.connect(
            self._on_ops_chunk_available
        )
        self.ops_generator.ops_generation_finished.connect(
            self._on_ops_generation_finished
        )
        self._on_transform_changed(self.data)
        self.trigger_update()

    def remove(self):
        """Disconnects signals and removes the element from the canvas."""
        logger.debug(f"Removing WorkPieceView for '{self.data.name}'")
        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        self.ops_generator.ops_generation_starting.disconnect(
            self._on_ops_generation_starting
        )
        self.ops_generator.ops_chunk_available.disconnect(
            self._on_ops_chunk_available
        )
        self.ops_generator.ops_generation_finished.disconnect(
            self._on_ops_generation_finished
        )
        super().remove()

    def set_base_image_visible(self, visible: bool):
        """
        Controls the visibility of the base rendered image, while leaving
        ops overlays unaffected.
        """
        if self._base_image_visible != visible:
            self._base_image_visible = visible
            if self.canvas:
                self.canvas.queue_draw()

    def set_ops_visibility(self, step_uid: str, visible: bool):
        """Sets the visibility for a specific step's ops overlay.

        Args:
            step_uid: The unique identifier of the step.
            visible: True to make the ops visible, False to hide them.
        """
        if self._ops_visibility.get(step_uid, True) != visible:
            logger.debug(
                f"Setting ops visibility for step '{step_uid}' to {visible}"
            )
            self._ops_visibility[step_uid] = visible
            if self.canvas:
                self.canvas.queue_draw()

    def clear_ops_surface(self, step_uid: str):
        """
        Cancels any pending render and removes the cached surface for a step.
        """
        logger.debug(f"Clearing ops surface for step '{step_uid}'")
        if future := self._ops_render_futures.pop(step_uid, None):
            future.cancel()
        if self._ops_surfaces.pop(step_uid, None):
            if self.canvas:
                self.canvas.queue_draw()

    def _on_model_content_changed(self, workpiece: WorkPiece):
        """Handler for when the workpiece model's content changes."""
        logger.debug(
            f"Model content changed for '{workpiece.name}', triggering update."
        )
        self.trigger_update()

    def _on_transform_changed(self, workpiece: WorkPiece):
        """Handler for when the workpiece model's transform changes."""
        if not self.canvas or self.transform == workpiece.matrix:
            return
        logger.debug(
            f"Transform changed for '{workpiece.name}', updating view."
        )
        self.set_transform(workpiece.matrix)

    def _on_ops_generation_starting(
        self, sender: Step, workpiece: WorkPiece, generation_id: int, **kwargs
    ):
        """Handler for when ops generation for a step begins."""
        if workpiece is not self.data:
            return
        step_uid = sender.uid
        self._ops_generation_ids[step_uid] = (
            generation_id  # Sets the ID when generation starts.
        )
        self.clear_ops_surface(step_uid)

    def _on_ops_generation_finished(
        self, sender: Step, workpiece: WorkPiece, generation_id: int, **kwargs
    ):
        """Handler for when ops generation for a step finishes."""
        if workpiece is not self.data:
            return
        step = sender
        logger.debug(
            f"Ops generation finished for step '{step.uid}'. "
            f"Scheduling async surface render for gen_id {generation_id}."
        )

        # Ensure the local generation ID is set for this step.
        # This handles cases where _on_ops_generation_starting might have been
        # missed (e.g., if WorkPieceView was instantiated late or re-created).
        # This makes sure that the _on_ops_surface_rendered callback will
        # have a matching generation_id to compare against.
        self._ops_generation_ids[step.uid] = generation_id

        if future := self._ops_render_futures.pop(step.uid, None):
            future.cancel()
        future = self._executor.submit(
            self._render_ops_surface_async, step, generation_id
        )
        self._ops_render_futures[step.uid] = future
        future.add_done_callback(self._on_ops_surface_rendered)

    def _render_ops_surface_async(
        self, step: Step, generation_id: int
    ) -> Optional[Tuple[str, cairo.ImageSurface, int]]:
        """
        Renders the complete, final Ops to a NEW, flicker-free surface in a
        background thread.
        """
        logger.debug(
            f"Rendering FINAL ops surface for workpiece "
            f"'{self.data.name}', step '{step.uid}', gen_id {generation_id}"
        )
        ops = self.ops_generator.get_ops(step, self.data)
        if not ops or not self.canvas:
            return None

        # --- FLICKER-FREE LOGIC: Create a new surface from scratch ---
        work_surface = cast("WorkSurface", self.canvas)
        world_w, world_h = self.data.size
        view_ppm_x, view_ppm_y = work_surface.get_view_scale()
        content_width_px = round(world_w * view_ppm_x)
        content_height_px = round(world_h * view_ppm_y)

        surface_width = content_width_px + 2 * OPS_MARGIN_PX
        surface_height = content_height_px + 2 * OPS_MARGIN_PX
        surface_width = min(surface_width, CAIRO_MAX_DIMENSION)
        surface_height = min(surface_height, CAIRO_MAX_DIMENSION)

        if (
            surface_width <= 2 * OPS_MARGIN_PX
            or surface_height <= 2 * OPS_MARGIN_PX
        ):
            return None

        # Create the new, clean surface for the final render.
        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, surface_width, surface_height
        )
        ctx = cairo.Context(surface)

        # Transform the coordinate system to be Y-UP for the CairoEncoder,
        # which expects to draw in a Y-UP world where (0,0) is bottom-left.
        ctx.translate(OPS_MARGIN_PX, surface_height + OPS_MARGIN_PX)
        ctx.scale(1, -1)

        # After clamping the surface, the actual content dimensions might have
        # changed. Recalculate them here to ensure the encoder scales the
        # ops correctly to fit the available space.
        content_width_px = surface_width - 2 * OPS_MARGIN_PX
        content_height_px = surface_height - 2 * OPS_MARGIN_PX

        # Calculate the pixels-per-millimeter needed for the encoder.
        # This scale must be based on the workpiece's dimensions
        # (world_w, world_h), not the ops' own bounding box, to ensure a
        # consistent coordinate system and prevent gaps.
        encoder_ppm_x = content_width_px / world_w if world_w > 1e-9 else 1.0
        encoder_ppm_y = content_height_px / world_h if world_h > 1e-9 else 1.0
        ppms = (encoder_ppm_x, encoder_ppm_y)

        show_travel = work_surface._show_travel_moves
        encoder = CairoEncoder()
        encoder.encode(ops, ctx, ppms, show_travel_moves=show_travel)

        return step.uid, surface, generation_id

    def _on_ops_chunk_available(
        self,
        sender: Step,
        workpiece: WorkPiece,
        chunk: "Ops",
        generation_id: int,
        **kwargs,
    ):
        """
        Handler for when a chunk of ops is ready for progressive rendering.
        """
        if workpiece is not self.data:
            return

        # STALE CHECK: Ignore chunks from a previous generation request.
        step_uid = sender.uid
        if generation_id != self._ops_generation_ids.get(step_uid):
            return

        # For chunks, we want to create OR reuse the surface.
        prepared = self._prepare_ops_surface_and_context(sender)
        if not prepared:
            return

        _, ctx, ppms = prepared

        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface._show_travel_moves

        # Encode just the chunk onto the existing surface
        encoder = CairoEncoder()
        encoder.encode(chunk, ctx, ppms, show_travel_moves=show_travel)

        # Trigger a redraw to show the progress
        if self.canvas:
            self.canvas.queue_draw()

    def _prepare_ops_surface_and_context(
        self, step: Step
    ) -> Optional[
        Tuple[cairo.ImageSurface, cairo.Context, Tuple[float, float]]
    ]:
        """
        Used by chunk rendering. Ensures an ops surface exists for a step,
        creating it if necessary. Returns the surface, a transformed Cairo
        context, and the pixels-per-mm scale.
        """
        if not self.canvas:
            return None

        step_uid = step.uid
        surface = self._ops_surfaces.get(step_uid)

        # If surface doesn't exist (e.g., first chunk), create it.
        if surface is None:
            work_surface = cast("WorkSurface", self.canvas)
            world_w, world_h = self.data.size
            view_ppm_x, view_ppm_y = work_surface.get_view_scale()
            content_width_px = round(world_w * view_ppm_x)
            content_height_px = round(world_h * view_ppm_y)

            surface_width = content_width_px + 2 * OPS_MARGIN_PX
            surface_height = content_height_px + 2 * OPS_MARGIN_PX
            surface_width = min(surface_width, CAIRO_MAX_DIMENSION)
            surface_height = min(surface_height, CAIRO_MAX_DIMENSION)

            if (
                surface_width <= 2 * OPS_MARGIN_PX
                or surface_height <= 2 * OPS_MARGIN_PX
            ):
                return None

            surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, surface_width, surface_height
            )
            self._ops_surfaces[step_uid] = surface

        ctx = cairo.Context(surface)
        surface_height = surface.get_height()

        # Transform the coordinate system to be Y-UP for the CairoEncoder.
        ctx.translate(OPS_MARGIN_PX, surface_height + OPS_MARGIN_PX)
        ctx.scale(1, -1)

        # Calculate the pixels-per-millimeter needed for the encoder.
        world_w, world_h = self.data.size
        content_width_px = surface.get_width() - 2 * OPS_MARGIN_PX
        content_height_px = surface.get_height() - 2 * OPS_MARGIN_PX
        encoder_ppm_x = content_width_px / world_w if world_w > 1e-9 else 1.0
        encoder_ppm_y = content_height_px / world_h if world_h > 1e-9 else 1.0
        ppms = (encoder_ppm_x, encoder_ppm_y)

        return surface, ctx, ppms

    def _on_ops_surface_rendered(self, future: Future):
        """Callback executed when the async ops rendering is done."""
        if future.cancelled():
            logger.debug("Ops surface render future was cancelled.")
            return
        if exc := future.exception():
            logger.error(
                f"Error rendering ops surface for '{self.data.name}': {exc}",
                exc_info=exc,
            )
            return
        result = future.result()
        if not result:
            logger.debug("Ops surface render future returned no result.")
            return

        step_uid, new_surface, received_generation_id = result

        # Ignore results from a previous generation request.
        if received_generation_id != self._ops_generation_ids.get(step_uid):
            logger.debug(
                f"Ignoring stale final render for step '{step_uid}'. "
                f"Have ID {self._ops_generation_ids.get(step_uid)}, "
                f"received {received_generation_id}."
            )
            return

        logger.debug(
            f"Applying newly rendered ops surface for step '{step_uid}'."
        )
        self._ops_surfaces[step_uid] = new_surface
        self._ops_render_futures.pop(step_uid, None)
        if self.canvas:
            self.canvas.queue_draw()

    def _start_update(self) -> bool:
        """
        Extends the base class's update starter to also trigger a re-render
        of all ops surfaces. This ensures that when a zoom-related update
        occurs, both the base image and the ops get re-rendered at the
        new resolution.
        """
        # Let the base class handle the main content surface update.
        # This will return False for the GLib timer.
        res = super()._start_update()

        # Trigger the ops re-render. This will happen inside the same
        # debounced call as the base surface update.
        self.trigger_ops_rerender()

        return res

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Renders the base workpiece content to a new surface."""
        return self.data.render_to_pixels(width=width, height=height)

    def draw(self, ctx: cairo.Context):
        """Draws the element's content and ops overlays.

        The context is already transformed into the element's local 1x1
        Y-UP space.

        Args:
            ctx: The cairo context to draw on.
        """
        if self._base_image_visible:
            super().draw(ctx)

        # Draw Ops Surfaces
        for step_uid, surface in self._ops_surfaces.items():
            if not self._ops_visibility.get(step_uid, True) or not surface:
                continue

            ctx.save()
            surface_w = surface.get_width()
            surface_h = surface.get_height()
            content_w_px = surface_w - 2 * OPS_MARGIN_PX
            content_h_px = surface_h - 2 * OPS_MARGIN_PX

            if content_w_px <= 0 or content_h_px <= 0:
                ctx.restore()
                continue

            # 1. Scale the 1x1 Y-UP space to the ops's content dimensions.
            ctx.scale(1.0 / content_w_px, 1.0 / content_h_px)

            # 2. Draw the Y-DOWN surface, offsetting by the margin.
            # The content on the ops surface was rendered "upside down"
            # into a Y-UP context. When we draw that Y-DOWN surface into our
            # Y-UP canvas context, the two flips cancel out, and it appears
            # correctly.
            ctx.set_source_surface(surface, -OPS_MARGIN_PX, -OPS_MARGIN_PX)
            ctx.get_source().set_filter(cairo.FILTER_GOOD)
            ctx.paint()
            ctx.restore()

    def push_transform_to_model(self):
        """Updates the data model's matrix with the view's transform."""
        if self.data.matrix != self.transform:
            logger.debug(
                f"Pushing view transform to model for '{self.data.name}'."
            )
            self.data.matrix = self.transform.copy()

    def trigger_ops_rerender(self):
        """Triggers a re-render of all applicable ops for this workpiece."""
        if not self.data.layer:
            return
        logger.debug(f"Triggering ops rerender for '{self.data.name}'.")
        applicable_steps = self.data.layer.workflow.steps
        for step in applicable_steps:
            if self.ops_generator.get_ops(step, self.data) is None:
                logger.debug(
                    f"Skipping ops rerender for step '{step.uid}'; "
                    "ops not yet available in cache."
                )
                continue

            # Re-use the existing generation ID to avoid race conditions.
            # This ID comes from the _ops_generation_ids, which gets
            # updated in _on_ops_generation_finished, ensuring it always holds
            # the ID of the render currently being requested by *this*
            # WorkPieceView.
            gen_id = self._ops_generation_ids.get(step.uid, 0)
            self._on_ops_generation_finished(
                step, self.data, generation_id=gen_id
            )
