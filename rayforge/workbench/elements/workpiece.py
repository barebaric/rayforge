import logging
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast, List, Generator
import cairo
from concurrent.futures import Future
from copy import deepcopy

from ...core.workpiece import WorkPiece
from ...core.step import Step
from ...core.matrix import Matrix
from ...core.ops import Ops
from ..canvas import CanvasElement
from ...pipeline.encoder.cairoencoder import CairoEncoder
from ...core.geometry import LineToCommand, ArcToCommand, MoveToCommand
from ...core.tab import Tab
from ...undo import ChangePropertyCommand
from .tab_handle import TabHandleElement
import math

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

        # Edit Mode State
        self._tab_handles: List[TabHandleElement] = []
        self._dragged_handle: Optional[TabHandleElement] = None
        self._initial_tabs_state: Optional[List[Tab]] = None
        self._tabs_visible_override = False  # For the toolbar toggle

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
            hit_distance=5,
            is_editable=True,
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
        self._create_or_update_tab_handles()
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

    def set_tabs_visible_override(self, visible: bool):
        """Controls visibility of tab handles when not in edit mode."""
        if self._tabs_visible_override != visible:
            self._tabs_visible_override = visible
            self._update_tab_handles_state()
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
        self._create_or_update_tab_handles()
        self.trigger_update()

    def _on_transform_changed(self, workpiece: WorkPiece):
        """
        Handler for when the workpiece model's transform changes.

        This is the key fix for the blurriness issue. When the transform
        changes, we check if the object's *size* has also changed. If so,
        the buffered raster image is now invalid (it would be stretched and
        blurry), so we must trigger a full update to re-render it cleanly at
        the new resolution.
        """
        if not self.canvas or self.transform == workpiece.matrix:
            return
        logger.debug(
            f"Transform changed for '{workpiece.name}', updating view."
        )

        # Get the size from the view's current (old) transform matrix.
        old_w, old_h = self.transform.get_abs_scale()

        self.set_transform(workpiece.matrix)

        # Get the size from the new transform that was just set.
        new_w, new_h = self.transform.get_abs_scale()

        # Check for a meaningful change in size to invalidate the cache.
        if abs(new_w - old_w) > 1e-6 or abs(new_h - old_h) > 1e-6:
            self.trigger_update()

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

        # Create a new surface from scratch
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
        if not self.data.layer or not self.data.layer.workflow:
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

    def _create_or_update_tab_handles(self):
        """Creates or replaces TabHandleElements based on the model."""
        # Remove old handles
        for handle in self._tab_handles:
            if handle in self.children:
                self.remove_child(handle)
        self._tab_handles.clear()

        if not self.data.tabs:
            return

        for tab in self.data.tabs:
            handle = TabHandleElement(tab_data=tab, parent=self)
            self._position_handle_from_tab(handle)
            self._tab_handles.append(handle)
            self.add(handle)

        self._update_tab_handles_state()

    def _configure_tab_handles(self, is_editing: bool):
        """
        A helper to unambiguously set the state of tab handles based on
        whether the element is in edit mode.
        """
        show_tabs = is_editing or self._tabs_visible_override
        for handle in self._tab_handles:
            handle.set_visible(show_tabs)
            handle.draggable = is_editing
            handle.opacity = 1.0 if is_editing else 0.3

    def _update_tab_handles_state(self):
        """
        Updates visibility, opacity, and interactivity of handles based on the
        current canvas edit context.
        """
        is_edit_mode = bool(self.canvas and self.canvas.edit_context is self)
        self._configure_tab_handles(is_edit_mode)

    def on_edit_mode_enter(self):
        """Activates tab handles for editing."""
        logger.debug(f"Entering edit mode for tabs on '{self.data.name}'")
        self._configure_tab_handles(True)
        if self.canvas:
            self.canvas.queue_draw()

    def on_edit_mode_leave(self):
        """Deactivates tab handles and commits any changes."""
        logger.debug(f"Leaving edit mode on '{self.data.name}'")
        self._configure_tab_handles(False)
        self._dragged_handle = None
        self._initial_tabs_state = None
        if self.canvas:
            self.canvas.queue_draw()

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws the source vector path as a guide for tab placement."""
        # Use workpiece.vectors which are already in local space.
        if not self.data.vectors or not self.canvas:
            return

        widget_width = self.canvas.get_allocated_width()
        widget_height = self.canvas.get_allocated_height()
        if widget_width <= 0 or widget_height <= 0:
            return

        temp_surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, widget_width, widget_height
        )
        temp_ctx = cairo.Context(temp_surface)

        screen_transform_matrix = (
            self.canvas.view_transform @ self.get_world_transform()
        )
        cairo_matrix = cairo.Matrix(*screen_transform_matrix.for_cairo())
        temp_ctx.transform(cairo_matrix)

        # Draw the vector path with a distinct style
        # Line width needs to be adjusted by the inverse of the scale factor
        # to appear consistent regardless of zoom.
        scale_x_screen = screen_transform_matrix.get_abs_scale()[0]
        temp_ctx.set_line_width(
            2.0 / scale_x_screen if scale_x_screen > 0 else 1.0
        )
        temp_ctx.set_source_rgba(0.1, 0.5, 0.9, 0.8)  # Bright blue

        # Convert geometry to ops and encode it
        ops = Ops.from_geometry(self.data.vectors)
        encoder = CairoEncoder()
        # Pass (1.0, 1.0) for scale because the ctx is now scaled to match
        # the workpiece's local mm space *visually*.
        encoder.encode(
            ops, temp_ctx, scale=(1.0, 1.0), show_travel_moves=False
        )

        ctx.set_source_surface(temp_surface, 0, 0)
        ctx.paint()

    def handle_edit_press(self, world_x: float, world_y: float) -> bool:
        """Handles a mouse press, checking if a tab handle was clicked."""
        # Find which handle was clicked, if any
        # Delegate to base CanvasElement's hit-test, which iterates children.
        hit_element = self.get_elem_hit(world_x, world_y)
        if isinstance(hit_element, TabHandleElement):
            self._dragged_handle = hit_element
            self._initial_tabs_state = deepcopy(self.data.tabs)
            logger.debug(f"Started dragging tab {hit_element.data.uid}")
            return True
        return False

    def handle_edit_drag(self, world_dx: float, world_dy: float):
        """Handles dragging a tab handle along the vector path."""
        if (
            not self._dragged_handle
            or not self.canvas
            or not self.data.vectors
        ):
            return

        # Get the current mouse position in world coordinates (pixel space).
        world_mouse_x, world_mouse_y = self.canvas._get_world_coords(
            self.canvas._last_mouse_x, self.canvas._last_mouse_y
        )

        # Transform world mouse pos to this element's local 1x1 unit space
        try:
            inv_world = self.get_world_transform().invert()
            local_x_norm, local_y_norm = inv_world.transform_point(
                (world_mouse_x, world_mouse_y)
            )
        except Exception:
            return

        # Un-normalize from 1x1 space to the geometry's local millimeter space.
        # This must use the geometry's "natural" size, not the current scaled
        # size of the workpiece view.
        natural_size = self.data.get_natural_size()
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            # Fallback if natural size is not available.
            natural_w, natural_h = self.data.get_local_size()

        local_x_mm = local_x_norm * natural_w
        local_y_mm = local_y_norm * natural_h

        # Find the closest point on the source geometry using mm coordinates
        result = self.data.vectors.find_closest_point(local_x_mm, local_y_mm)
        if not result:
            return

        seg_idx, t, _ = result
        tab_to_update = cast(Tab, self._dragged_handle.data)

        # Update the model directly for live feedback
        tab_to_update.segment_index = seg_idx
        tab_to_update.t = t

        # Reposition the visual handle
        self._position_handle_from_tab(self._dragged_handle)
        self.canvas.queue_draw()

    def handle_edit_release(self, world_x: float, world_y: float):
        """Finalizes a tab drag by creating an undo command."""
        if not self._dragged_handle or not self._initial_tabs_state:
            return

        # Access the Doc object through the Canvas and its editor.
        if self.canvas and hasattr(self.canvas, "editor"):
            doc = cast("WorkSurface", self.canvas).editor.doc
            if doc:
                # Create a command with the initial and final states
                cmd = ChangePropertyCommand(
                    target=self.data,
                    property_name="tabs",
                    new_value=deepcopy(self.data.tabs),
                    old_value=self._initial_tabs_state,
                    name=_("Move Tab"),
                )
                # Restore the model to its initial state before executing the
                # command, so the undo/redo stack is correct.
                self.data.tabs = self._initial_tabs_state
                doc.history_manager.execute(cmd)
        else:
            logger.warning(
                "Could not finalize tab drag: Canvas or DocEditor not found."
            )

        # Reset drag state
        self._dragged_handle = None
        self._initial_tabs_state = None
        if self.canvas:
            self.canvas.queue_draw()

    def _get_path_segments_local(
        self,
    ) -> Generator[
        Tuple[int, Tuple[float, float], Tuple[float, float]], None, None
    ]:
        """
        Yields all movable segments from the source vectors as
        (index, p_start, p_end) tuples in local coordinates.
        ArcToCommands are linearized for this purpose.
        """
        if not self.data.vectors:
            return
        last_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        for i, cmd in enumerate(self.data.vectors.commands):
            if isinstance(cmd, MoveToCommand):
                if cmd.end:
                    last_pos_3d = cmd.end
            elif isinstance(cmd, LineToCommand):
                if cmd.end:
                    yield i, last_pos_3d[:2], cmd.end[:2]
                    last_pos_3d = cmd.end
            elif isinstance(cmd, ArcToCommand):
                if cmd.end:
                    arc_segments = self.data.vectors._linearize_arc(
                        cmd, last_pos_3d
                    )
                    for p_start_arc, p_end_arc in arc_segments:
                        yield i, p_start_arc[:2], p_end_arc[:2]
                    last_pos_3d = cmd.end

    def _position_handle_from_tab(self, handle: TabHandleElement):
        """Calculates and sets a handle's transform from its tab data."""
        tab = cast(Tab, handle.data)
        if not self.data.vectors or tab.segment_index >= len(
            self.data.vectors.commands
        ):
            return

        cmd = self.data.vectors.commands[tab.segment_index]
        if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
            return

        start_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        for i in range(tab.segment_index):
            prev_cmd = self.data.vectors.commands[i]
            if prev_cmd.end:
                start_pos_3d = prev_cmd.end

        px, py, tangent_angle_rad = 0.0, 0.0, 0.0
        if isinstance(cmd, LineToCommand):
            p_start, p_end = start_pos_3d[:2], cmd.end[:2]
            px = p_start[0] + (p_end[0] - p_start[0]) * tab.t
            py = p_start[1] + (p_end[1] - p_start[1]) * tab.t
            tangent_angle_rad = math.atan2(
                p_end[1] - p_start[1], p_end[0] - p_start[0]
            )
        elif isinstance(cmd, ArcToCommand):
            p0 = start_pos_3d
            center = (
                p0[0] + cmd.center_offset[0],
                p0[1] + cmd.center_offset[1],
            )
            radius = math.dist(p0[:2], center)
            if radius == 0:
                p_start, p_end = p0[:2], cmd.end[:2]
                px = p_start[0] + (p_end[0] - p_start[0]) * tab.t
                py = p_start[1] + (p_end[1] - p_start[1]) * tab.t
                tangent_angle_rad = math.atan2(
                    p_end[1] - p_start[1], p_end[0] - p_start[0]
                )
            else:
                start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
                end_angle = math.atan2(
                    cmd.end[1] - center[1], cmd.end[0] - center[0]
                )
                angle_range = end_angle - start_angle
                if cmd.clockwise:
                    if angle_range > 0:
                        angle_range -= 2 * math.pi
                else:
                    if angle_range < 0:
                        angle_range += 2 * math.pi

                tab_angle = start_angle + angle_range * tab.t
                px = center[0] + radius * math.cos(tab_angle)
                py = center[1] + radius * math.sin(tab_angle)
                # The tangent to a circle is perpendicular to the radius vector
                tangent_angle_rad = tab_angle + (
                    math.pi / 2.0 if not cmd.clockwise else -math.pi / 2.0
                )

        natural_size = self.data.get_natural_size()
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            natural_w, natural_h = self.data.get_local_size()

        # Normalize scale and position into the parent's 1x1 local space.
        # The handle's local X-axis is its width, Y-axis is its length.
        scale_x_norm = tab.width / natural_w if natural_w > 0 else 0
        scale_y_norm = tab.length / natural_h if natural_h > 0 else 0
        pos_x_norm = px / natural_w if natural_w > 0 else 0
        pos_y_norm = py / natural_h if natural_h > 0 else 0

        # Build transform: center, scale, rotate, then translate.
        # The rotation aligns the handle's local X-axis (width) with the
        # path's tangent. The Y-axis (length) becomes perpendicular.
        transform = (
            Matrix.translation(pos_x_norm, pos_y_norm)
            @ Matrix.rotation(math.degrees(tangent_angle_rad) - 90)
            @ Matrix.scale(scale_x_norm, scale_y_norm)
            @ Matrix.translation(-0.5, -0.5)
        )
        handle.set_transform(transform)
