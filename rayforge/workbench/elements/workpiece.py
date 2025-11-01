import logging
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast, List
import cairo
import numpy as np
from gi.repository import GLib
from ...context import get_context
from ...core.workpiece import WorkPiece
from ...core.step import Step
from ...core.matrix import Matrix
from ...pipeline.artifact import (
    WorkPieceViewArtifact,
    RenderContext,
    WorkPieceArtifactHandle,
)
from ...shared.util.colors import ColorSet
from ...shared.util.gtk_color import GtkColorResolver, ColorSpecDict
from ..canvas import CanvasElement
from .tab_handle import TabHandleElement

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ...pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Cairo has a hard limit on surface dimensions.
CAIRO_MAX_DIMENSION = 30000
OPS_MARGIN_PX = 5
REC_MARGIN_MM = 0.1  # A small "safe area" margin in mm for recordings


class WorkPieceElement(CanvasElement):
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
        self,
        workpiece: WorkPiece,
        pipeline: "Pipeline",
        **kwargs,
    ):
        """Initializes the WorkPieceElement.

        Args:
            workpiece: The WorkPiece data model to visualize.
            pipeline: The generator responsible for creating ops.
            **kwargs: Additional arguments for the CanvasElement.
        """
        logger.debug(f"Initializing WorkPieceElement for '{workpiece.name}'")
        self.data: WorkPiece = workpiece
        self.pipeline = pipeline
        self._base_image_visible = True
        self._surface: Optional[cairo.ImageSurface] = None

        self._ops_visibility: Dict[str, bool] = {}
        self._ops_generation_ids: Dict[str, int] = {}

        # This now stores this element's private, safe copies of the bitmaps
        self._drawable_surfaces: Dict[
            str, List[Tuple[cairo.ImageSurface, WorkPieceViewArtifact]]
        ] = {}
        self._surface_source_sizes: Dict[str, Tuple[float, float]] = {}

        self._tab_handles: List[TabHandleElement] = []
        # Default to False; the correct state will be pulled from the surface.
        self._tabs_visible_override: bool = False

        self._color_spec: ColorSpecDict = {
            "cut": ("#ffeeff", "#ff00ff"),
            "engrave": ("#FFFFFF", "#000000"),
            "travel": ("#FF6600", 0.7),
            "zero_power": ("@accent_color", 0.5),
        }
        self._color_set: Optional[ColorSet] = None
        self._last_style_context_hash = -1

        self._last_known_width_mm: float = 0.0
        self._last_known_height_mm: float = 0.0

        # The element's geometry is a 1x1 unit square.
        # The transform matrix handles all scaling and positioning.
        super().__init__(
            0.0,
            0.0,
            1.0,
            1.0,
            data=workpiece,
            # clip must be False so the parent `render` method
            # does not clip the drawing, allowing margins to show.
            clip=False,
            buffered=True,
            pixel_perfect_hit=True,
            hit_distance=5,
            is_editable=False,
            **kwargs,
        )

        # After super().__init__, self.canvas is set. Pull the initial
        # tab visibility state from the WorkSurface, which is the state owner.
        if self.canvas:
            work_surface = cast("WorkSurface", self.canvas)
            self._tabs_visible_override = (
                work_surface.get_global_tab_visibility()
            )

        self.content_transform = Matrix.translation(0, 1) @ Matrix.scale(1, -1)

        self.data.updated.connect(self._on_model_content_changed)
        self.data.transform_changed.connect(self._on_transform_changed)
        self.pipeline.workpiece_starting.connect(
            self._on_ops_generation_starting
        )
        self.pipeline.workpiece_chunk_available.connect(
            self._on_ops_chunk_available
        )
        self.pipeline.workpiece_artifact_ready.connect(
            self._on_ops_generation_finished
        )
        self.pipeline.view_artifacts_changed.connect(
            self._on_view_artifacts_changed
        )

        self._on_transform_changed(self.data)
        self._create_or_update_tab_handles()
        self.trigger_update()

    def _on_ops_chunk_available(
        self,
        sender,
        key: Tuple[str, str],
        chunk_handle: WorkPieceArtifactHandle,
        generation_id: int,
        **kwargs,
    ):
        """
        Handler for when a raw WorkPieceArtifact chunk is available. This
        triggers a request to render it into a viewable artifact. The sender
        guarantees the handle is valid for the duration of this handler.
        """
        step_uid, workpiece_uid = key
        if workpiece_uid != self.data.uid or not self.canvas:
            return

        if generation_id != self._ops_generation_ids.get(step_uid):
            logger.debug(
                f"Ignoring stale ops CHUNK event for step '{step_uid}'."
            )
            return

        work_surface = cast("WorkSurface", self.canvas)
        ppm_x, ppm_y = work_surface.get_view_scale()
        self._resolve_colors_if_needed()
        if not self._color_set:
            logger.warning("Cannot render chunk: color set not resolved.")
            return

        context = RenderContext(
            pixels_per_mm=(ppm_x, ppm_y),
            show_travel_moves=work_surface.show_travel_moves,
            margin_px=OPS_MARGIN_PX,
            color_set_dict=self._color_set.to_dict(),
        )

        # Pass the handle to the pipeline. The view stage will acquire its own
        # reference for the duration of its background task.
        self.pipeline.request_view_render(
            step_uid,
            self.data.uid,
            context,
            source_handle=chunk_handle,
        )

    def _on_view_artifacts_changed(
        self, sender, *, step_uid: str, workpiece_uid: str, **kwargs
    ):
        """
        Handler for when the set of drawable view artifacts has changed.
        This method fetches the new artifacts, copies their bitmaps into
        safe, privately owned Cairo surfaces, and triggers a redraw.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            return

        # Get the latest list of handles from the pipeline stage.
        # This will be empty if the artifacts were just invalidated.
        handles = self.pipeline.get_view_artifacts_for_workpiece(
            step_uid, workpiece_uid
        )
        logger.debug(
            f"'{self.data.name}' received view_artifacts_changed for step "
            f"'{step_uid}'. Got {len(handles)} new handles. "
            f"Clearing local surfaces."
        )

        artifact_store = get_context().artifact_store
        new_surfaces: List[
            Tuple[cairo.ImageSurface, WorkPieceViewArtifact]
        ] = []

        for handle in handles:
            try:
                # Use a scoped reference to ensure the handle is released
                # even if errors occur during the copy.
                with artifact_store.hold(handle):
                    artifact = artifact_store.get(handle)
                    if not isinstance(artifact, WorkPieceViewArtifact):
                        continue

                    h, w, _ = artifact.bitmap_data.shape
                    if h == 0 or w == 0:
                        continue

                    # Ensure the numpy array is C-contiguous and assign it to a
                    # variable that will persist through the 'create_for_data'
                    # call.
                    contiguous_bitmap_data = np.ascontiguousarray(
                        artifact.bitmap_data
                    )

                    # Create a temporary surface that VIEWS the shared memory.
                    # This is only safe because 'contiguous_bitmap_data' is in
                    # scope.
                    shm_surface = cairo.ImageSurface.create_for_data(
                        memoryview(contiguous_bitmap_data),
                        cairo.FORMAT_ARGB32,
                        w,
                        h,
                    )

                    # Create a new, safe surface with its own private memory.
                    safe_surface = cairo.ImageSurface(
                        cairo.FORMAT_ARGB32, w, h
                    )

                    # Let Cairo perform the copy. This correctly handles stride
                    # and memory layout, preventing crashes.
                    ctx = cairo.Context(safe_surface)
                    ctx.set_source_surface(shm_surface, 0, 0)
                    ctx.paint()

                    new_surfaces.append((safe_surface, artifact))
            except Exception as e:
                logger.error(
                    f"Failed to copy view artifact for {handle.shm_name}: {e}"
                )

        # Atomically replace the old surfaces with the new ones.
        self._drawable_surfaces[step_uid] = new_surfaces
        self._surface_source_sizes[step_uid] = self.data.size
        self.trigger_update()

    def get_closest_point_on_path(
        self, world_x: float, world_y: float, threshold_px: float = 5.0
    ) -> Optional[Dict]:
        """
        Checks if a point in world coordinates is close to the workpiece's
        vector path.

        Args:
            world_x: The x-coordinate in world space (mm).
            world_y: The y-coordinate in world space (mm).
            threshold_px: The maximum distance in screen pixels to be
                          considered "close".

        Returns:
            A dictionary with location info
              `{'segment_index': int, 't': float}`
            if the point is within the threshold, otherwise None.
        """
        if not self.data.vectors or not self.canvas:
            return None

        work_surface = cast("WorkSurface", self.canvas)

        # 1. Convert pixel threshold to a world-space (mm) threshold
        ppm_x, _ = work_surface.get_view_scale()
        if ppm_x < 1e-9:
            return None
        threshold_mm = threshold_px / ppm_x

        # 2. Transform click coordinates to local, natural millimeter space
        try:
            inv_world_transform = self.get_world_transform().invert()
            local_x_norm, local_y_norm = inv_world_transform.transform_point(
                (world_x, world_y)
            )
        except Exception:
            return None  # Transform not invertible

        natural_size = self.data.get_natural_size()
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            natural_w, natural_h = self.data.get_local_size()

        if natural_w <= 1e-9 or natural_h <= 1e-9:
            return None

        local_x_mm = local_x_norm * natural_w
        local_y_mm = local_y_norm * natural_h

        # 3. Find closest point on path in local mm space
        closest = self.data.vectors.find_closest_point(local_x_mm, local_y_mm)
        if not closest:
            return None

        segment_index, t, closest_point_local_mm = closest

        # 4. Transform local closest point back to world space
        closest_point_norm_x = closest_point_local_mm[0] / natural_w
        closest_point_norm_y = closest_point_local_mm[1] / natural_h
        (
            closest_point_world_x,
            closest_point_world_y,
        ) = self.get_world_transform().transform_point(
            (closest_point_norm_x, closest_point_norm_y)
        )

        # 5. Perform distance check in world space
        dist_sq_world = (world_x - closest_point_world_x) ** 2 + (
            world_y - closest_point_world_y
        ) ** 2

        if dist_sq_world > threshold_mm**2:
            return None

        # 6. Return location info if within threshold
        return {"segment_index": segment_index, "t": t}

    def remove(self):
        """Disconnects signals and removes the element from the canvas."""
        logger.debug(f"Removing WorkPieceElement for '{self.data.name}'")
        # Invalidate view artifacts in the central stage
        self.pipeline.invalidate_view_for_workpiece(self.data.uid)

        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        self.pipeline.workpiece_starting.disconnect(
            self._on_ops_generation_starting
        )
        self.pipeline.workpiece_chunk_available.disconnect(
            self._on_ops_chunk_available
        )
        self.pipeline.workpiece_artifact_ready.disconnect(
            self._on_ops_generation_finished
        )
        self.pipeline.view_artifacts_changed.disconnect(
            self._on_view_artifacts_changed
        )
        super().remove()
        self._drawable_surfaces.clear()

    def set_base_image_visible(self, visible: bool):
        """
        Controls the visibility of the base rendered image, while leaving
        ops overlays unaffected.
        """
        if self._base_image_visible != visible:
            self._base_image_visible = visible
            self.trigger_update()

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
            self.trigger_update()

    def _resolve_colors_if_needed(self):
        """
        Creates or updates the ColorSet if the theme has changed. This
        should be called before any rendering operation.
        """
        if not self.canvas:
            return

        # A simple hash check to see if the style context has changed.
        # This is not perfect but good enough to detect theme switches.
        style_context = self.canvas.get_style_context()
        current_hash = hash(style_context)
        if (
            self._color_set is None
            or current_hash != self._last_style_context_hash
        ):
            logger.debug(
                "Resolving colors for WorkPieceElement due to theme change."
            )
            resolver = GtkColorResolver(style_context)
            self._color_set = resolver.resolve(self._color_spec)
            self._last_style_context_hash = current_hash

    def _find_step_by_uid(self, step_uid: str) -> Optional[Step]:
        """Finds a step object in this element's layer by its UID."""
        if self.data.layer and self.data.layer.workflow:
            for step in self.data.layer.workflow.steps:
                if step.uid == step_uid:
                    return step
        return None

    def _request_view_artifact_updates(
        self,
        specific_step_uid: Optional[str] = None,
        generation_id: Optional[int] = None,
    ):
        """
        Requests new, view-dependent bitmap artifacts from the pipeline for
        all applicable steps, or a single specific step.
        """
        if (
            not self.canvas
            or not self.data.layer
            or not self.data.layer.workflow
        ):
            return

        logger.debug(
            f"'{self.data.name}'._request_view_artifact_updates called for "
            f"step: {specific_step_uid or 'all'}"
        )

        work_surface = cast("WorkSurface", self.canvas)
        ppm_x, ppm_y = work_surface.get_view_scale()
        self._resolve_colors_if_needed()

        if not self._color_set:
            logger.warning(
                "Cannot request view update: color set not resolved."
            )
            return

        steps_to_process: List[Step] = []
        if specific_step_uid:
            step = self._find_step_by_uid(specific_step_uid)
            if step:
                steps_to_process.append(step)
        else:
            # Global refresh: re-render all steps with their latest gen ID
            if self.data.layer and self.data.layer.workflow:
                steps_to_process.extend(self.data.layer.workflow.steps)

        for step in steps_to_process:
            if self.pipeline.is_workpiece_generating(step.uid, self.data.uid):
                continue

            source_handle = self.pipeline.get_artifact_handle(
                step.uid, self.data.uid
            )
            if not isinstance(source_handle, WorkPieceArtifactHandle):
                continue

            context = RenderContext(
                pixels_per_mm=(ppm_x, ppm_y),
                show_travel_moves=work_surface.show_travel_moves,
                margin_px=OPS_MARGIN_PX,
                color_set_dict=self._color_set.to_dict(),
            )
            self.pipeline.request_view_render(
                step.uid,
                self.data.uid,
                context,
                source_handle=source_handle,
            )

    def _on_model_content_changed(self, workpiece: WorkPiece):
        """Handler for when the workpiece model's content changes."""
        logger.debug(
            f"Model content changed for '{workpiece.name}', triggering update."
        )
        self._create_or_update_tab_handles()
        self.trigger_update()

    def _on_transform_changed(self, workpiece: WorkPiece):
        """
        Handler for when the workpiece model's transform changes. This is
        called when the model's `transform_changed` signal is fired.
        """
        if not self.canvas:
            return

        logger.debug(f"'{workpiece.name}'._on_transform_changed started.")
        logger.debug(f"View transform BEFORE sync: {self.transform}")
        logger.debug(f"Model matrix from signal: {workpiece.matrix}")

        # Always sync the view's transform from the model when this signal
        # fires.
        self.set_transform(workpiece.matrix)

        new_w, new_h = self.transform.get_abs_scale()
        logger.debug(
            f"Scale check: old=({self._last_known_width_mm:.4f}, "
            f"{self._last_known_height_mm:.4f}), "
            f"new=({new_w:.4f}, {new_h:.4f})"
        )

        size_changed = (
            abs(new_w - self._last_known_width_mm) > 1e-6
            or abs(new_h - self._last_known_height_mm) > 1e-6
        )

        self._last_known_width_mm = new_w
        self._last_known_height_mm = new_h

        if size_changed:
            # The size has changed, which requires re-rendering all buffered
            # content (base image and ops overlays).
            logger.debug("  - Size changed, calling trigger_update().")
            self.trigger_update()
        else:
            # If only position/rotation changed, just queue a redraw without
            # invalidating buffered surfaces.
            logger.debug("  - Size NOT changed, calling canvas.queue_draw().")
            self.canvas.queue_draw()

    def _on_ops_generation_starting(
        self, sender: Step, workpiece: WorkPiece, generation_id: int, **kwargs
    ):
        """
        Handler for when ops generation for a step begins. We now only
        update the generation ID and do NOT clear surfaces to prevent flicker.
        """
        if workpiece is not self.data:
            return
        step_uid = sender.uid
        self._ops_generation_ids[step_uid] = generation_id

    def _on_ops_generation_finished(
        self, sender: Step, workpiece: WorkPiece, generation_id: int, **kwargs
    ):
        """
        Signal handler for when ops generation finishes. This runs on a
        background thread and schedules the actual work on the main thread.
        """
        GLib.idle_add(
            self._on_ops_generation_finished_main_thread,
            sender,
            workpiece,
            generation_id,
        )

    def _on_ops_generation_finished_main_thread(
        self,
        sender: Step,
        workpiece: WorkPiece,
        generation_id: int,
    ):
        """The thread-safe part of the ops generation finished handler."""
        if workpiece is not self.data:
            return

        step = sender
        if generation_id != self._ops_generation_ids.get(step.uid):
            logger.debug(
                f"Ignoring stale ops finish event for step '{step.uid}'."
            )
            return

        self._request_view_artifact_updates(
            specific_step_uid=step.uid, generation_id=generation_id
        )

    def _start_update(self) -> bool:
        """
        Extends the base class's update starter to also trigger a re-render
        of all ops surfaces. This ensures that when a zoom-related update
        occurs, both the base image and the ops get re-rendered at the
        new resolution.
        """
        logger.debug(f"'{self.data.name}'._start_update called.")
        res = super()._start_update()
        self._request_view_artifact_updates()
        return res

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Renders the base workpiece content to a new surface."""
        return self.data.render_to_pixels(width=width, height=height)

    def _draw_single_view_surface(
        self,
        ctx: cairo.Context,
        surface_tuple: Tuple[cairo.ImageSurface, WorkPieceViewArtifact],
        source_workpiece_size: Tuple[float, float],
    ):
        """Helper to draw one pre-rendered bitmap to the context."""
        surface, artifact = surface_tuple
        ops_x, ops_y, ops_w, ops_h = cast(Tuple[float, ...], artifact.bbox_mm)
        world_w, world_h = source_workpiece_size

        if world_w < 1e-9 or world_h < 1e-9 or ops_w < 1e-9 or ops_h < 1e-9:
            return

        ctx.save()
        ctx.translate(ops_x / world_w, ops_y / world_h)
        ctx.scale(ops_w / world_w, ops_h / world_h)
        ctx.translate(0, 1)
        ctx.scale(1, -1)

        surface_w_px = surface.get_width()
        surface_h_px = surface.get_height()
        content_w_px = surface_w_px - 2 * OPS_MARGIN_PX
        content_h_px = surface_h_px - 2 * OPS_MARGIN_PX

        if content_w_px <= 0 or content_h_px <= 0:
            ctx.restore()
            return

        ctx.scale(1.0 / content_w_px, 1.0 / content_h_px)
        ctx.set_source_surface(surface, -OPS_MARGIN_PX, -OPS_MARGIN_PX)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()

    def draw(self, ctx: cairo.Context):
        """Draws the element's content and ops overlays.

        The context is already transformed into the element's local 1x1
        Y-UP space.

        Args:
            ctx: The cairo context to draw on.
        """
        if self._base_image_visible:
            super().draw(ctx)

        worksurface = cast("WorkSurface", self.canvas) if self.canvas else None
        if not worksurface or worksurface.is_simulation_mode():
            return

        for step_uid, surface_list in self._drawable_surfaces.items():
            if not self._ops_visibility.get(step_uid, True):
                continue
            source_size = self._surface_source_sizes.get(step_uid)
            if not source_size:
                continue
            for surface_tuple in surface_list:
                self._draw_single_view_surface(ctx, surface_tuple, source_size)

    def push_transform_to_model(self):
        """
        Updates the data model's matrix with the view's transform. This
        is the critical method that UI tools (like resize/move handles) MUST
        call when an interaction is complete to commit the changes.
        """
        if self.data.matrix != self.transform:
            logger.debug(
                f"Pushing view transform to model for '{self.data.name}'."
            )
            self.data.matrix = self.transform.copy()

    def on_travel_visibility_changed(self):
        """
        Handles changes in travel move visibility.
        """
        logger.debug(
            "Travel visibility changed. Requesting new view artifacts."
        )
        self._request_view_artifact_updates()

    def set_tabs_visible_override(self, visible: bool):
        """Sets the global visibility override for tab handles."""
        if self._tabs_visible_override != visible:
            self._tabs_visible_override = visible
            self._update_tab_handle_visibility()

    def _update_tab_handle_visibility(self):
        """Applies the current visibility logic to all tab handles."""
        is_visible = self._tabs_visible_override and self.data.tabs_enabled
        for handle in self._tab_handles:
            handle.set_visible(is_visible)

    def _create_or_update_tab_handles(self):
        """Creates or replaces TabHandleElements based on the model."""
        for handle in self._tab_handles:
            if handle in self.children:
                self.remove_child(handle)
        self._tab_handles.clear()

        is_visible = self._tabs_visible_override and self.data.tabs_enabled

        if not self.data.tabs:
            return

        for tab in self.data.tabs:
            handle = TabHandleElement(tab_data=tab, parent=self)
            handle.update_base_geometry()
            handle.update_transform()
            handle.set_visible(is_visible)
            self._tab_handles.append(handle)
            self.add(handle)

    def update_handle_transforms(self):
        """
        Recalculates transforms for all tab handles. This is called on zoom.
        """
        for handle in self._tab_handles:
            handle.update_transform()
