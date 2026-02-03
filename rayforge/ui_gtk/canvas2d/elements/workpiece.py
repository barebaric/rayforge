import logging
import math
from concurrent.futures import Future
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast, List, Set, Any
import cairo
import numpy as np
from gi.repository import GLib
from ....context import get_context
from ....core.workpiece import WorkPiece
from ....core.step import Step
from ....core.matrix import Matrix
from ....pipeline.artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
    WorkPieceViewArtifact,
    RenderContext,
)
from ....shared.util.colors import ColorSet
from ...shared.gtk_color import GtkColorResolver, ColorSpecDict
from ...canvas import CanvasElement
from .tab_handle import TabHandleElement

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ....pipeline.pipeline import Pipeline
    from ....pipeline.artifact.base import VertexData

logger = logging.getLogger(__name__)

# Cairo has a hard limit on surface dimensions.
CAIRO_MAX_DIMENSION = 8192
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

        self._ops_surfaces: Dict[
            str, Optional[Tuple[cairo.ImageSurface, Tuple[float, ...]]]
        ] = {}
        self._ops_recordings: Dict[str, Optional[cairo.RecordingSurface]] = {}
        self._ops_visibility: Dict[str, bool] = {}
        self._ops_render_futures: Dict[str, Future] = {}
        self._ops_generation_ids: Dict[
            str, int
        ] = {}  # Tracks the *expected* generation ID of the *next* render.
        self._texture_surfaces: Dict[str, cairo.ImageSurface] = {}
        # Cached artifacts to avoid re-fetching from pipeline on every draw.
        self._artifact_cache: Dict[str, Optional[WorkPieceArtifact]] = {}

        # Unified cache for progressive view artifacts.
        # Key: step_uid
        # Value: (CairoSurface, bbox_mm, ArtifactHandle, KeepAliveBufferRef)
        self._progressive_view_surfaces: Dict[
            str,
            Tuple[
                cairo.ImageSurface, Tuple[float, ...], BaseArtifactHandle, Any
            ],
        ] = {}

        # Debouncing state for view render requests
        self._view_request_timer_id: Optional[int] = None
        self._pending_view_update_all: bool = False
        self._pending_view_update_steps: Set[str] = set()
        # Tracks steps that had view render requested during generation.
        # Used to avoid double-rendering when generation finishes.
        self._steps_with_progressive_render: Set[str] = set()

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
        self.pipeline.workpiece_artifact_ready.connect(
            self._on_ops_generation_finished
        )
        # Only process view_artifact_updated, not view_artifact_created.
        # The worker sends view_artifact_created BEFORE drawing to the bitmap,
        # so processing it would read an empty/transparent bitmap.
        # We wait for view_artifact_updated which is sent AFTER drawing
        # completes.
        self.pipeline.workpiece_view_updated.connect(
            self._on_view_artifact_updated
        )
        # When a workpiece artifact is adopted, trigger view rendering
        self.pipeline.workpiece_artifact_adopted.connect(
            self._on_workpiece_artifact_adopted
        )

        # Track the last known model size to detect size changes even when
        # the transform matrix is pre-synced (e.g. during interactive drags).
        self._last_synced_size = self.data.size

        # Attempt to hydrate visual state from the model's transient cache
        hydrated = self._hydrate_from_cache()

        self._on_transform_changed(self.data)
        self._create_or_update_tab_handles()

        # Only invalidate if we didn't recover state from the cache.
        if not hydrated:
            self.invalidate_and_rerender()
        else:
            # We recovered state, but verify if a repaint is needed
            super().trigger_update()
            # If we have artifacts, request view render to get bitmaps
            self._request_view_render()

    def _hydrate_from_cache(self) -> bool:
        """
        Restores visual state from the persistent model cache if available.
        Returns True if significant state was restored.
        """
        cache = self.data._view_cache
        if not cache:
            return False

        # Restore caches. We copy the dictionaries to avoid modification
        # issues, but the heavy objects (Surfaces) are shared references.
        self._surface = cache.get("surface")
        self._ops_surfaces = cache.get("ops_surfaces", {}).copy()
        self._ops_recordings = cache.get("ops_recordings", {}).copy()
        self._texture_surfaces = cache.get("texture_surfaces", {}).copy()
        self._artifact_cache = cache.get("artifact_cache", {}).copy()
        self._ops_generation_ids = cache.get("ops_generation_ids", {}).copy()

        # Note: We do NOT restore _progressive_view_surfaces because they rely
        # on shared memory handles which may have been released or are hard
        # to manage across view lifecycles.

        # Consider hydrated if we have a base surface or artifacts
        return (
            self._surface is not None
            or len(self._artifact_cache) > 0
            or len(self._ops_surfaces) > 0
        )

    def _update_model_view_cache(self):
        """
        Updates the persistent cache on the model with current view state.
        """
        cache = self.data._view_cache
        cache["surface"] = self._surface
        cache["ops_surfaces"] = self._ops_surfaces
        cache["ops_recordings"] = self._ops_recordings
        cache["texture_surfaces"] = self._texture_surfaces
        cache["artifact_cache"] = self._artifact_cache
        cache["ops_generation_ids"] = self._ops_generation_ids

    def invalidate_and_rerender(self):
        """
        Invalidates all cached rendering artifacts (base image and all ops)
        and schedules a full re-render. This should be called whenever the
        element's content or size changes.
        """
        logger.debug(f"Full invalidation for workpiece '{self.data.name}'")
        # Clear the local artifact cache to prevent drawing stale vectors
        self._artifact_cache.clear()

        # Clear the model cache as well, since the data is invalid
        self.data._view_cache.clear()

        if self.data.layer and self.data.layer.workflow:
            for step in self.data.layer.workflow.steps:
                self.clear_ops_surface(step.uid)
        super().trigger_update()

    def trigger_view_update(self):
        """
        Invalidates resolution-dependent caches (raster surfaces) and
        triggers a re-render. This is called on view changes like zooming.
        It preserves expensive-to-generate data like vector recordings.
        """
        logger.debug(f"View update for workpiece '{self.data.name}'")
        # 1. Invalidate the base image buffer.
        self._surface = None

        self._ops_recordings.clear()

        # Note: We do NOT clear the model cache here, as view updates
        # (like zooming) shouldn't erase the persistent data needed by
        # other views or future rebuilds.

        # 3. Trigger a re-render of everything at the new resolution.
        self.trigger_ops_rerender()
        self._request_view_render()
        super().trigger_update()  # Re-renders the base image.

    def _request_view_render(
        self, step_uid: Optional[str] = None, force: bool = False
    ):
        """
        Debounced request for background render of the workpiece view.
        If step_uid is None, requests render for all visible steps.

        Args:
            step_uid: The step to render, or None for all visible steps.
            force: If True, force re-rendering even if the context appears
                unchanged.
        """
        if step_uid is None:
            self._pending_view_update_all = True
        else:
            self._pending_view_update_steps.add(step_uid)

        # Store the force flag for the callback
        self._pending_view_force = force

        # Remove existing timer to reset the debounce window
        if self._view_request_timer_id is not None:
            GLib.source_remove(self._view_request_timer_id)

        # Schedule execution after 50ms of silence
        self._view_request_timer_id = GLib.timeout_add(
            50, self._execute_view_render_request
        )

    def _execute_view_render_request(self) -> bool:
        """
        Executes the pending view render requests. This is called by the
        debounce timer.
        """
        self._view_request_timer_id = None

        if (
            not self.canvas
            or not self.data.layer
            or not self.data.layer.workflow
        ):
            self._pending_view_update_all = False
            self._pending_view_update_steps.clear()
            return False

        self._resolve_colors_if_needed()
        if not self._color_set:
            return False

        work_surface = cast("WorkSurface", self.canvas)
        ppm_x, ppm_y = work_surface.get_view_scale()

        # Don't request render if scaled too small
        if ppm_x <= 1e-9 or ppm_y <= 1e-9:
            return False

        steps_to_process = set()

        if self._pending_view_update_all:
            # Process all visible steps
            for step in self.data.layer.workflow.steps:
                if self._ops_visibility.get(step.uid, True):
                    steps_to_process.add(step.uid)
        else:
            # Process specifically requested steps, checking visibility
            for uid in self._pending_view_update_steps:
                if self._ops_visibility.get(uid, True):
                    steps_to_process.add(uid)

        # Clear pending state
        self._pending_view_update_all = False
        self._pending_view_update_steps.clear()

        if not steps_to_process:
            return False

        context = RenderContext(
            pixels_per_mm=(ppm_x, ppm_y),
            show_travel_moves=work_surface.show_travel_moves,
            margin_px=OPS_MARGIN_PX,
            color_set_dict=self._color_set.to_dict(),
        )

        # Get the force flag that was stored when the request was made
        force = getattr(self, "_pending_view_force", False)

        for uid in steps_to_process:
            self.pipeline.request_view_render(
                uid, self.data.uid, context, force=force
            )

        return False  # Stop the timer

    def _process_view_artifact(
        self,
        step_uid: str,
        handle: BaseArtifactHandle,
    ):
        """
        Common helper to process a new or updated view artifact.
        Always processes the artifact to support progressive rendering,
        even if the handle is already cached (content may have changed).
        """
        # Check if this exact handle is already cached
        old_tuple = self._progressive_view_surfaces.get(step_uid, None)
        if old_tuple:
            _, _, old_handle, _ = old_tuple
            if old_handle.shm_name == handle.shm_name:
                # Same handle - content may have changed in shared memory.
                # Remove the old entry to force creation of a new surface.
                # This is critical for progressive rendering to work.
                del self._progressive_view_surfaces[step_uid]
            else:
                # Different handle - release the old one
                store = get_context().artifact_store
                if old_handle.shm_name in store._refcounts:
                    store.release(old_handle)

        try:
            # Retrieve the artifact from shared memory
            artifact = cast(
                WorkPieceViewArtifact, get_context().artifact_store.get(handle)
            )
            if not artifact:
                get_context().artifact_store.release(handle)
                return

            h, w, _ = artifact.bitmap_data.shape

            # Explicitly pass stride. width * 4 bytes/pixel
            stride = w * 4

            # IMPORTANT: We MUST keep a reference to the numpy array (buffer)
            # because PyCairo does not copy the data, it just points to it.
            # If artifact.bitmap_data is GC'd, the surface becomes invalid.
            # For progressive rendering, we need to ensure we get a fresh copy
            # of the data each time, not just a reference to the same buffer.
            buffer_ref = artifact.bitmap_data.copy()

            surface = cairo.ImageSurface.create_for_data(
                memoryview(buffer_ref),
                cairo.FORMAT_ARGB32,
                w,
                h,
                stride,
            )

            # Store surface, bbox, handle, AND the buffer reference
            self._progressive_view_surfaces[step_uid] = (
                surface,
                artifact.bbox_mm,
                handle,
                buffer_ref,
            )
            if self.canvas:
                # Use idle_add to ensure the redraw happens on the next
                # iteration of the main loop, which should allow progressive
                # rendering to work correctly
                GLib.idle_add(self._trigger_progressive_redraw)

        except Exception as e:
            logger.error(
                f"Failed to process view artifact for '{step_uid}': {e}"
            )
            get_context().artifact_store.release(handle)

    def _trigger_progressive_redraw(self):
        """Triggers a redraw for progressive rendering."""
        if self.canvas:
            self.canvas.queue_draw()
        return False  # Don't call again

    def _on_view_artifact_updated(
        self,
        sender,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
        **kwargs,
    ):
        """
        Handler for when the content of a progressive render artifact has
        been updated by the background worker.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            return

        self._process_view_artifact(step_uid, handle)

    def _on_workpiece_artifact_adopted(
        self, sender, *, step_uid: str, workpiece_uid: str, **kwargs
    ):
        """
        Handler for when a workpiece artifact has been adopted.
        Request view render immediately so the view artifact exists
        before chunks start arriving during generation.
        """
        if workpiece_uid != self.data.uid:
            return

        logger.debug(
            f"_on_workpiece_artifact_adopted called for step '{step_uid}'"
        )
        # Request view render immediately so the view artifact is created
        # before chunks start arriving. This enables progressive rendering
        # where chunks are drawn to the view artifact as they arrive.
        self._steps_with_progressive_render.add(step_uid)
        self._request_view_render(step_uid, force=True)

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
        if not self.data.boundaries or not self.canvas:
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

        natural_size = self.data.natural_size
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            natural_w, natural_h = self.data.get_local_size()

        if natural_w <= 1e-9 or natural_h <= 1e-9:
            return None

        local_x_mm = local_x_norm * natural_w
        local_y_mm = local_y_norm * natural_h

        # 3. Find closest point on path in local mm space
        closest = self.data.boundaries.find_closest_point(
            local_x_mm, local_y_mm
        )
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
        if self._view_request_timer_id is not None:
            GLib.source_remove(self._view_request_timer_id)
            self._view_request_timer_id = None

        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        self.pipeline.workpiece_starting.disconnect(
            self._on_ops_generation_starting
        )
        self.pipeline.workpiece_artifact_ready.disconnect(
            self._on_ops_generation_finished
        )
        self.pipeline.workpiece_view_updated.disconnect(
            self._on_view_artifact_updated
        )
        self.pipeline.workpiece_artifact_adopted.disconnect(
            self._on_workpiece_artifact_adopted
        )
        super().remove()

        for _, _, handle, _ in self._progressive_view_surfaces.values():
            get_context().artifact_store.release(handle)
        self._progressive_view_surfaces.clear()

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
            if visible:
                self._request_view_render(step_uid)
            if self.canvas:
                self.canvas.queue_draw()

    def clear_ops_surface(self, step_uid: str):
        """
        Cancels any pending render and removes the cached surface for a step.
        """
        logger.debug(f"Clearing ops surface for step '{step_uid}'")
        # Remove any cached progressive view for this step to ensure fresh
        # start
        old_tuple = self._progressive_view_surfaces.pop(step_uid, None)
        if old_tuple is not None:
            _, _, old_handle, _ = old_tuple
            get_context().artifact_store.release(old_handle)

        if self.canvas:
            self.canvas.queue_draw()

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

    def _on_model_content_changed(self, workpiece: WorkPiece):
        """Handler for when the workpiece model's content changes."""
        logger.debug(
            f"Model content changed for '{workpiece.name}', triggering update."
        )
        self._create_or_update_tab_handles()
        self.invalidate_and_rerender()

    def _on_transform_changed(self, workpiece: WorkPiece):
        """
        Handler for when the workpiece model's transform changes.

        This is the key fix for the blurriness issue. When the transform
        changes, we check if the object's *size* has also changed. If so,
        the buffered raster image is now invalid (it would be stretched and
        blurry), so we must trigger a full update to re-render it cleanly at
        the new resolution.
        """
        if not self.canvas:
            return

        # Check if the size has changed significantly since the last sync.
        # We cannot rely on comparing self.transform vs workpiece.matrix here,
        # because interactive tools update self.transform before the model
        # commits, leading to them being equal when this signal finally fires.
        # However, the cached artifacts/buffers correspond to the *old* size.
        new_w, new_h = workpiece.size
        old_w, old_h = self._last_synced_size

        if abs(new_w - old_w) > 1e-6 or abs(new_h - old_h) > 1e-6:
            self._last_synced_size = (new_w, new_h)
            # Invalidate progressive view cache since the old artifacts
            # are now at the wrong resolution and would appear stretched.
            for step_uid, (_, _, handle, _) in list(
                self._progressive_view_surfaces.items()
            ):
                get_context().artifact_store.release(handle)
            self._progressive_view_surfaces.clear()
            # Clear progressive render tracking so new view renders will be
            # requested after the size change, preventing chunks from being
            # drawn to the old-sized view artifact.
            self._steps_with_progressive_render.clear()
            # Sync the transform immediately
            self.set_transform(workpiece.matrix)
            # Note: We do NOT request view renders here. The pipeline will
            # automatically trigger view rendering when the workpiece
            # artifact is regenerated after the size change.
            super().trigger_update()
        else:
            # Size hasn't changed, just sync the transform
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
        self._steps_with_progressive_render.discard(step_uid)

    def _on_ops_generation_finished(
        self, sender: Step, workpiece: WorkPiece, generation_id: int, **kwargs
    ):
        """
        Signal handler for when ops generation finishes.
        This runs on a background thread, so it schedules the actual work
        on the main thread to prevent UI deadlocks.
        """
        if workpiece is not self.data:
            return

        artifact = self.pipeline.get_artifact(sender, workpiece)
        GLib.idle_add(
            self._on_ops_generation_finished_main_thread,
            sender,
            workpiece,
            generation_id,
            artifact,
        )

    def _on_ops_generation_finished_main_thread(
        self,
        sender: Step,
        workpiece: WorkPiece,
        generation_id: int,
        artifact: WorkPieceArtifact,
    ):
        """The thread-safe part of the ops generation finished handler."""
        logger.debug(
            f"_on_ops_generation_finished_main_thread called for step "
            f"'{sender.uid}'"
        )
        if workpiece is not self.data:
            return
        step = sender

        self._ops_generation_ids[step.uid] = generation_id

        # Fetch and cache the final artifact, making it available to all paths.
        self._artifact_cache[step.uid] = artifact
        self._update_model_view_cache()

        if logger.isEnabledFor(logging.DEBUG) and artifact and artifact.vertex_data:
            v_data = artifact.vertex_data
            counts = (
                v_data.powered_vertices.size,
                v_data.travel_vertices.size,
                v_data.zero_power_vertices.size,
            )
            bounds = None
            try:
                stacks = [
                    v
                    for v in (
                        v_data.powered_vertices,
                        v_data.travel_vertices,
                        v_data.zero_power_vertices,
                    )
                    if v.size > 0
                ]
                if stacks:
                    v_stack = np.vstack(stacks)
                    v_min = np.min(v_stack, axis=0)
                    v_max = np.max(v_stack, axis=0)
                    bounds = (v_min.tolist(), v_max.tolist())
            except Exception as exc:  # pragma: no cover - debug only
                logger.debug("Failed to compute vertex bounds: %s", exc)
            logger.debug(
                "Artifact vertices for step '%s': counts powered/travel/zero=%s, bounds=%s, gen_size=%s, source_dims=%s",
                step.uid,
                counts,
                bounds,
                artifact.generation_size,
                artifact.source_dimensions,
            )

        # Asynchronously prepare texture surface if it exists
        if artifact and artifact.texture_data:
            if future := self._ops_render_futures.pop(step.uid, None):
                future.cancel()

            logger.debug(
                f"PRE-submit _prepare_texture_surface_async for '{step.uid}'"
            )
            future = self._executor.submit(
                self._prepare_texture_surface_async, step.uid, artifact
            )
            self._ops_render_futures[step.uid] = future
            future.add_done_callback(self._on_texture_surface_prepared)
            logger.debug(
                f"POST-submit _prepare_texture_surface_async for '{step.uid}'"
            )

        # Trigger a view render for this step if progressive rendering
        # was not already done. If progressive rendering was used, the view
        # artifact was already created and chunks were drawn to it during
        # generation, so we don't need to re-render.
        if step.uid not in self._steps_with_progressive_render:
            self._request_view_render(step.uid, force=True)
        self._steps_with_progressive_render.discard(step.uid)

        if self.canvas:
            self.canvas.queue_draw()
        logger.debug(
            f"END _on_ops_generation_finished_main_thread for "
            f"step '{sender.uid}'"
        )
    def _prepare_texture_surface_async(
        self, step_uid: str, artifact: WorkPieceArtifact
    ) -> Optional[Tuple[str, cairo.ImageSurface]]:
        """
        Performs the CPU-intensive conversion of raw texture data to a themed,
        pre-multiplied Cairo ImageSurface. Designed to run in a background
        thread.
        """
        self._resolve_colors_if_needed()
        if not self._color_set or not artifact.texture_data:
            return None

        power_data = artifact.texture_data.power_texture_data
        if power_data.size == 0:
            return None

        engrave_lut = self._color_set.get_lut("engrave")
        rgba_texture = engrave_lut[power_data]

        # Manually set alpha to 0 where power is 0 for transparency
        zero_power_mask = power_data == 0
        rgba_texture[zero_power_mask, 3] = 0.0

        h, w = rgba_texture.shape[:2]
        # Create pre-multiplied BGRA data for Cairo
        alpha_ch = rgba_texture[..., 3, np.newaxis]
        rgb_ch = rgba_texture[..., :3]
        bgra_texture = np.empty((h, w, 4), dtype=np.uint8)
        # Pre-multiply RGB by Alpha, then convert to BGRA byte order
        premultiplied_rgb = rgb_ch * alpha_ch * 255
        bgra_texture[..., 0] = premultiplied_rgb[..., 2]  # B
        bgra_texture[..., 1] = premultiplied_rgb[..., 1]  # G
        bgra_texture[..., 2] = premultiplied_rgb[..., 0]  # R
        bgra_texture[..., 3] = alpha_ch.squeeze() * 255  # A

        texture_surface = cairo.ImageSurface.create_for_data(
            memoryview(np.ascontiguousarray(bgra_texture)),
            cairo.FORMAT_ARGB32,
            w,
            h,
        )
        return step_uid, texture_surface

    def _on_texture_surface_prepared(self, future: Future):
        """Callback for when the async texture preparation is complete."""
        GLib.idle_add(self._on_texture_surface_prepared_main_thread, future)

    def _on_texture_surface_prepared_main_thread(self, future: Future):
        """Thread-safe handler to cache the prepared texture and redraw."""
        if future.cancelled() or future.exception():
            return
        result = future.result()
        if not result:
            return

        step_uid, texture_surface = result
        self._texture_surfaces[step_uid] = texture_surface
        self._update_model_view_cache()

        if self.canvas:
            self.canvas.queue_draw()

    def _draw_vertices_to_context(
        self,
        vertex_data: "VertexData",
        ctx: cairo.Context,
        scale: Tuple[float, float],
        drawable_height: float,
        line_width: Optional[float] = None,
    ):
        """
        Draws vertex data to a Cairo context, handling scaling, theming,
        and Y-coordinate inversion.
        """
        if not self.canvas or not self._color_set:
            return

        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface.show_travel_moves
        view_ppm_x, view_ppm_y = work_surface.get_view_scale()
        line_width_mm = None
        if view_ppm_x > 1e-9 and view_ppm_y > 1e-9:
            line_width_mm = 1.0 / max(view_ppm_x, view_ppm_y)
        scale_x, scale_y = scale

        ctx.save()
        if line_width is None:
            ctx.set_hairline(True)
        else:
            ctx.set_hairline(False)
            ctx.set_line_width(line_width)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        # --- Draw Travel & Zero-Power Moves ---
        if show_travel:
            if vertex_data.travel_vertices.size > 0:
                travel_v = vertex_data.travel_vertices.reshape(-1, 2, 3)
                ctx.set_source_rgba(*self._color_set.get_rgba("travel"))
                for start, end in travel_v:
                    ctx.move_to(
                        start[0] * scale_x,
                        drawable_height - start[1] * scale_y,
                    )
                    ctx.line_to(
                        end[0] * scale_x, drawable_height - end[1] * scale_y
                    )
                ctx.stroke()

            if vertex_data.zero_power_vertices.size > 0:
                zero_v = vertex_data.zero_power_vertices.reshape(-1, 2, 3)
                ctx.set_source_rgba(*self._color_set.get_rgba("zero_power"))
                for start, end in zero_v:
                    ctx.move_to(
                        start[0] * scale_x,
                        drawable_height - start[1] * scale_y,
                    )
                    ctx.line_to(
                        end[0] * scale_x, drawable_height - end[1] * scale_y
                    )
                ctx.stroke()

        # --- Draw Powered Moves (Grouped by Color for performance) ---
        if vertex_data.powered_vertices.size > 0:
            powered_v = vertex_data.powered_vertices.reshape(-1, 2, 3)
            powered_c = vertex_data.powered_colors
            cut_lut = self._color_set.get_lut("cut")

            # Use power from the first vertex of each segment for color.
            power_indices = (powered_c[::2, 0] * 255.0).astype(np.uint8)
            themed_colors_per_segment = cut_lut[power_indices]

            unique_colors, inverse_indices = np.unique(
                themed_colors_per_segment, axis=0, return_inverse=True
            )

            for i, color in enumerate(unique_colors):
                ctx.set_source_rgba(*color)
                segment_indices = np.where(inverse_indices == i)[0]
                for seg_idx in segment_indices:
                    start, end = powered_v[seg_idx]
                    ctx.move_to(
                        start[0] * scale_x,
                        drawable_height - start[1] * scale_y,
                    )
                    ctx.line_to(
                        end[0] * scale_x, drawable_height - end[1] * scale_y
                    )
                ctx.stroke()
        ctx.restore()

    def _record_ops_drawing_async(
        self, step: Step, generation_id: int
    ) -> Optional[Tuple[str, cairo.RecordingSurface, int]]:
        """
        "Draws" the vector data to a RecordingSurface. This captures all vector
        commands and is done only when the data changes.
        """
        logger.debug(
            f"Recording vector data for workpiece "
            f"'{self.data.name}', step '{step.uid}'"
        )
        artifact = self._artifact_cache.get(step.uid)
        if not artifact or not artifact.vertex_data or not self.canvas:
            return None

        self._resolve_colors_if_needed()
        world_w, world_h = self.data.size
        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface.show_travel_moves

        # Calculate the union of the workpiece bounds and the vertex bounds to
        # ensure the recording surface is large enough.
        all_v = [artifact.vertex_data.powered_vertices]
        if show_travel:
            all_v.append(artifact.vertex_data.travel_vertices)
            all_v.append(artifact.vertex_data.zero_power_vertices)

        all_v_filtered = [v for v in all_v if v.size > 0]
        if not all_v_filtered:
            return None

        v_stack = np.vstack(all_v_filtered)
        v_x1, v_y1, _ = np.min(v_stack, axis=0)
        v_x2, v_y2, _ = np.max(v_stack, axis=0)

        union_x1 = min(0.0, v_x1)
        union_y1 = min(0.0, v_y1)
        union_x2 = max(world_w, v_x2)
        union_y2 = max(world_h, v_y2)

        union_w = union_x2 - union_x1
        union_h = union_y2 - union_y1

        if union_w <= 1e-9 or union_h <= 1e-9:
            return None

        # Create the recording surface with a small margin to prevent
        # strokes on the boundary from being clipped by the recording's
        # extents. The extents define the user-space coordinate system.
        extents = (
            union_x1 - REC_MARGIN_MM,
            union_y1 - REC_MARGIN_MM,
            union_w + 2 * REC_MARGIN_MM,
            union_h + 2 * REC_MARGIN_MM,
        )
        # The pycairo type stubs are incorrect for RecordingSurface; they don't
        # specify that a tuple is a valid type for `extents`. We ignore the
        # type checker here as the code is functionally correct.
        surface = cairo.RecordingSurface(
            cairo.CONTENT_COLOR_ALPHA,
            extents,  # type: ignore
        )
        ctx = cairo.Context(surface)

        # We are drawing 1:1 in mm space, so scale is 1.0. The vertex data
        # is Y-up, and so is the recording surface's coordinate system.
        # So we just pass a height that allows the y-flip to work correctly
        # relative to the content we are drawing.
        drawable_height_mm = union_y2 + union_y1
        self._draw_vertices_to_context(
            artifact.vertex_data,
            ctx,
            (1.0, 1.0),
            drawable_height_mm,
            line_width=line_width_mm,
        )

        return step.uid, surface, generation_id

    def _on_ops_drawing_recorded(self, future: Future):
        """
        Callback executed when the async ops recording is done.
        Schedules the main logic to run on the GTK thread.
        """
        GLib.idle_add(self._on_ops_drawing_recorded_main_thread, future)

    def _on_ops_drawing_recorded_main_thread(self, future: Future):
        """The thread-safe part of the drawing recorded callback."""
        if future.cancelled():
            return
        if exc := future.exception():
            logger.error(f"Error recording ops drawing: {exc}", exc_info=exc)
            return
        result = future.result()
        if not result:
            return

        step_uid, recording, received_gen_id = result

        if received_gen_id != self._ops_generation_ids.get(step_uid):
            logger.debug(
                f"Ignoring stale ops recording for step '{step_uid}'."
            )
            return

        logger.debug(f"Applying new ops recording for step '{step_uid}'.")
        self._ops_recordings[step_uid] = recording
        self._update_model_view_cache()

        # Find the Step object to trigger the initial rasterization.
        if self.data.layer and self.data.layer.workflow:
            for step_obj in self.data.layer.workflow.steps:
                if step_obj.uid == step_uid:
                    # This call is now safe because we are on the main thread.
                    self._trigger_ops_rasterization(step_obj, received_gen_id)
                    return
        logger.warning(
            "Could not find step '%s' to rasterize after recording.",
            step_uid,
        )

    def _trigger_ops_rasterization(self, step: Step, generation_id: int):
        """
        Schedules the fast async rasterization of ops using the cached
        recording.
        """
        step_uid = step.uid
        if future := self._ops_render_futures.get(step_uid):
            if not future.done():
                future.cancel()  # Cancel obsolete render.

        future = self._executor.submit(
            self._rasterize_ops_surface_async, step, generation_id
        )
        self._ops_render_futures[step_uid] = future
        future.add_done_callback(self._on_ops_surface_rendered)

    def _rasterize_ops_surface_async(
        self, step: Step, generation_id: int
    ) -> Optional[
        Tuple[str, cairo.ImageSurface, int, Tuple[float, float, float, float]]
    ]:
        """
        Renders ops to an ImageSurface, using the cached RecordingSurface
        for a huge speedup if it is available. Also returns the mm bounding
        box of the rendered content.
        """
        step_uid = step.uid
        logger.debug(
            f"Rasterizing ops surface for step '{step_uid}', "
            f"gen_id {generation_id}"
        )
        if not self.canvas:
            return None

        self._resolve_colors_if_needed()
        recording = self._ops_recordings.get(step_uid)
        world_w, world_h = self.data.size
        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface.show_travel_moves

        # Determine the millimeter dimensions and offset of the content.
        if recording:
            # FAST PATH: use extents from the recording surface.
            extents = recording.get_extents()
            if extents:
                rec_x, rec_y, rec_w, rec_h = extents
                content_x_mm = rec_x + REC_MARGIN_MM
                content_y_mm = rec_y + REC_MARGIN_MM
                content_w_mm = rec_w - 2 * REC_MARGIN_MM
                content_h_mm = rec_h - 2 * REC_MARGIN_MM
            else:
                logger.warning(f"Could not get extents for '{step_uid}'")
                return None
        else:
            # Slow fallback: calculate bounds from vertex data.
            artifact = self._artifact_cache.get(step.uid)
            if not artifact or not artifact.vertex_data:
                return None

            all_v = [artifact.vertex_data.powered_vertices]
            if show_travel:
                all_v.append(artifact.vertex_data.travel_vertices)
                all_v.append(artifact.vertex_data.zero_power_vertices)

            all_v_filtered = [v for v in all_v if v.size > 0]
            if not all_v_filtered:
                return None

            v_stack = np.vstack(all_v_filtered)
            v_x1, v_y1, _ = np.min(v_stack, axis=0)
            v_x2, v_y2, _ = np.max(v_stack, axis=0)

            union_x1 = min(0.0, v_x1)
            union_y1 = min(0.0, v_y1)
            union_x2 = max(world_w, v_x2)
            union_y2 = max(world_h, v_y2)

            content_x_mm = union_x1
            content_y_mm = union_y1
            content_w_mm = union_x2 - union_x1
            content_h_mm = union_y2 - union_y1

        bbox_mm = (content_x_mm, content_y_mm, content_w_mm, content_h_mm)
        view_ppm_x, view_ppm_y = work_surface.get_view_scale()
        content_width_px = round(content_w_mm * view_ppm_x)
        content_height_px = round(content_h_mm * view_ppm_y)

        surface_width = min(
            content_width_px + 2 * OPS_MARGIN_PX, CAIRO_MAX_DIMENSION
        )
        surface_height = min(
            content_height_px + 2 * OPS_MARGIN_PX, CAIRO_MAX_DIMENSION
        )

        if (
            surface_width <= 2 * OPS_MARGIN_PX
            or surface_height <= 2 * OPS_MARGIN_PX
        ):
            return None

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, surface_width, surface_height
        )
        ctx = cairo.Context(surface)
        ctx.translate(OPS_MARGIN_PX, OPS_MARGIN_PX)

        if recording:
            # FAST PATH: Replay the cached vector drawing commands.
            ctx.save()
            # 1. Scale context to match mm units.
            ctx.scale(view_ppm_x, view_ppm_y)
            # 2. The content area's top-left is at (content_x_mm, content_y_mm)
            #    in world space. Translate the context so that its origin (0,0)
            #    corresponds to the world's origin (0,0).
            ctx.translate(-content_x_mm, -content_y_mm)
            # 3. Set the recording as the source. Its internal coordinates
            #    are already in world mm, so we can now paint it directly.
            ctx.set_source_surface(recording, 0, 0)
            ctx.paint()
            ctx.restore()
        else:
            # SLOW FALLBACK: No recording yet, render from vertex data.
            artifact = self._artifact_cache.get(step.uid)
            if not artifact or not artifact.vertex_data:
                return None  # Should not happen as we checked above

            encoder_ppm_x = (
                content_width_px / content_w_mm if content_w_mm > 1e-9 else 1
            )
            encoder_ppm_y = (
                content_height_px / content_h_mm if content_h_mm > 1e-9 else 1
            )
            ppms = (encoder_ppm_x, encoder_ppm_y)

            # Translate context to draw the union box content correctly.
            ctx.translate(
                -content_x_mm * encoder_ppm_x, -content_y_mm * encoder_ppm_y
            )

            # Y-flip height must be workpiece height in pixels.
            drawable_h_px = world_h * encoder_ppm_y
            self._draw_vertices_to_context(
                artifact.vertex_data,
                ctx,
                ppms,
                drawable_h_px,
                line_width=1.0,
            )

        return step_uid, surface, generation_id, bbox_mm

    def _on_ops_chunk_available(
        self,
        sender: Step,
        workpiece: WorkPiece,
        chunk_handle: "BaseArtifactHandle",
        generation_id: int,
        **kwargs,
    ):
        """
        Handler for when a chunk of ops is ready for progressive rendering.
        This is called from a background thread. It schedules the expensive
        encoding work to happen in another background task.
        """
        if workpiece is not self.data:
            return

        # STALE CHECK: Ignore chunks from a previous generation request.
        step_uid = sender.uid
        if generation_id != self._ops_generation_ids.get(step_uid):
            get_context().artifact_store.release(chunk_handle)
            return

        # Offload the CPU-intensive encoding to the thread pool
        future = self._executor.submit(
            self._encode_chunk_async, sender, chunk_handle
        )
        future.add_done_callback(self._on_chunk_encoded)

    def _encode_chunk_async(
        self, step: Step, chunk_handle: BaseArtifactHandle
    ):
        """
        Does the heavy lifting of preparing a surface and encoding an ops
        chunk onto it. This is designed to be run in a thread pool.
        """
        # This function runs entirely in a background thread.
        chunk_artifact = None
        try:
            prepared = self._prepare_ops_surface_and_context(step)
            if prepared:
                chunk_artifact = cast(
                    WorkPieceArtifact,
                    get_context().artifact_store.get(chunk_handle),
                )
                if not chunk_artifact:
                    return step.uid

                _surface, ctx, ppms, content_h_px = prepared

                # --- Draw texture data from the chunk if it exists ---
                if self._color_set and chunk_artifact.texture_data:
                    power_data = chunk_artifact.texture_data.power_texture_data
                    if power_data.size > 0:
                        engrave_lut = self._color_set.get_lut("engrave")
                        rgba_texture = engrave_lut[power_data]

                        # Manually set alpha for transparency
                        zero_power_mask = power_data == 0
                        rgba_texture[zero_power_mask, 3] = 0.0

                        h, w = rgba_texture.shape[:2]
                        # Create pre-multiplied BGRA data for Cairo
                        alpha_ch = rgba_texture[..., 3, np.newaxis]
                        rgb_ch = rgba_texture[..., :3]
                        bgra_texture = np.empty((h, w, 4), dtype=np.uint8)
                        premultiplied_rgb = rgb_ch * alpha_ch * 255
                        bgra_texture[..., 0] = premultiplied_rgb[..., 2]  # B
                        bgra_texture[..., 1] = premultiplied_rgb[..., 1]  # G
                        bgra_texture[..., 2] = premultiplied_rgb[..., 0]  # R
                        bgra_texture[..., 3] = alpha_ch.squeeze() * 255  # A

                        texture_surface = cairo.ImageSurface.create_for_data(
                            memoryview(np.ascontiguousarray(bgra_texture)),
                            cairo.FORMAT_ARGB32,
                            w,
                            h,
                        )

                        # Draw the themed texture to the pixel context
                        _world_w, world_h = self.data.size
                        pos_mm = chunk_artifact.texture_data.position_mm
                        dim_mm = chunk_artifact.texture_data.dimensions_mm
                        encoder_ppm_x, encoder_ppm_y = ppms

                        dest_x_px = pos_mm[0] * encoder_ppm_x
                        dest_w_px = dim_mm[0] * encoder_ppm_x
                        dest_h_px = dim_mm[1] * encoder_ppm_y
                        dest_y_px = pos_mm[1] * encoder_ppm_y

                        tex_w_px = texture_surface.get_width()
                        tex_h_px = texture_surface.get_height()

                        if tex_w_px > 0 and tex_h_px > 0:
                            ctx.save()
                            ctx.translate(dest_x_px, dest_y_px)
                            # Add half-pixel offset for raster grid alignment
                            ctx.translate(0.5, 0.5)
                            ctx.scale(
                                dest_w_px / tex_w_px, dest_h_px / tex_h_px
                            )
                            ctx.set_source_surface(texture_surface, 0, 0)
                            ctx.get_source().set_filter(cairo.FILTER_GOOD)
                            ctx.paint()
                            ctx.restore()

                # --- Draw vertex data from the chunk if it exists ---
                if chunk_artifact.vertex_data:
                    self._draw_vertices_to_context(
                        chunk_artifact.vertex_data,
                        ctx,
                        ppms,
                        content_h_px,
                        line_width=1.0,
                    )
        finally:
            # IMPORTANT: Release the handle in the subprocess to free memory
            get_context().artifact_store.release(chunk_handle)
        return step.uid

    def _on_chunk_encoded(self, future: Future):
        """
        Callback for when a chunk has been encoded. Schedules the final
        UI update on the main thread.
        """
        GLib.idle_add(self._on_chunk_encoded_main_thread, future)

    def _on_chunk_encoded_main_thread(self, future: Future):
        """
        Thread-safe callback that triggers a redraw after a chunk is ready.
        """
        if future.cancelled() or future.exception():
            return
        # The result is just the step_uid, we don't need it, but we know
        # the surface has been updated.
        if self.canvas:
            self.canvas.queue_draw()

    def _prepare_ops_surface_and_context(
        self, step: Step
    ) -> Optional[
        Tuple[cairo.ImageSurface, cairo.Context, Tuple[float, float], float]
    ]:
        """
        Used by chunk rendering. Ensures an ops surface exists for a step,
        creating it if necessary. Returns the surface, a transformed context,
        scale, and drawable height in pixels.
        """
        if not self.canvas:
            return None

        self._resolve_colors_if_needed()
        step_uid = step.uid
        surface_tuple = self._ops_surfaces.get(step_uid)
        world_w, world_h = self.data.size

        # If surface doesn't exist (e.g., first chunk), create it.
        # Chunk rendering will be clipped to workpiece bounds for now.
        if surface_tuple is None:
            work_surface = cast("WorkSurface", self.canvas)
            view_ppm_x, view_ppm_y = work_surface.get_view_scale()
            content_width_px = round(world_w * view_ppm_x)
            content_height_px = round(world_h * view_ppm_y)

            surface_width = min(
                content_width_px + 2 * OPS_MARGIN_PX, CAIRO_MAX_DIMENSION
            )
            surface_height = min(
                content_height_px + 2 * OPS_MARGIN_PX, CAIRO_MAX_DIMENSION
            )

            if (
                surface_width <= 2 * OPS_MARGIN_PX
                or surface_height <= 2 * OPS_MARGIN_PX
            ):
                return None

            surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, surface_width, surface_height
            )
            # Store with workpiece bounds. This will be replaced by the
            # final render with the correct, larger bounds.
            workpiece_bbox = (0.0, 0.0, world_w, world_h)
            self._ops_surfaces[step_uid] = (surface, workpiece_bbox)
        else:
            surface, _ = surface_tuple

        ctx = cairo.Context(surface)
        # Set the origin to the top-left of the content area.
        ctx.translate(OPS_MARGIN_PX, OPS_MARGIN_PX)

        # Calculate the pixels-per-millimeter and content height for encoder.
        content_width_px = surface.get_width() - 2 * OPS_MARGIN_PX
        content_height_px = surface.get_height() - 2 * OPS_MARGIN_PX
        encoder_ppm_x = content_width_px / world_w if world_w > 1e-9 else 1.0
        encoder_ppm_y = content_height_px / world_h if world_h > 1e-9 else 1.0
        ppms = (encoder_ppm_x, encoder_ppm_y)

        return surface, ctx, ppms, content_height_px

    def _on_ops_surface_rendered(self, future: Future):
        """
        Callback executed when the async ops rendering is done.
        Schedules the main logic to run on the GTK thread.
        """
        # Schedule the actual handler on the main thread
        GLib.idle_add(self._on_ops_surface_rendered_main_thread, future)

    def _on_ops_surface_rendered_main_thread(self, future: Future):
        """The thread-safe part of the surface rendered callback."""
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

        step_uid, new_surface, received_generation_id, bbox_mm = result

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
        self._ops_surfaces[step_uid] = (new_surface, bbox_mm)
        self._update_model_view_cache()  # Save to model cache
        self._ops_render_futures.pop(step_uid, None)
        if self.canvas:
            # This call is now safe because we are on the main thread.
            self.canvas.queue_draw()

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
            # This handles the Y-flip for the base image and restores the
            # context, leaving it Y-UP for the next drawing operation.
            super().draw(ctx)

        # Draw Ops (hide during simulation mode)
        worksurface = cast("WorkSurface", self.canvas) if self.canvas else None
        if not worksurface or worksurface.is_simulation_mode():
            return

        if self.data.layer and self.data.layer.workflow:
            # Draw the new progressive bitmaps if available (Phase 4)
            self._draw_progressive_views(ctx)

    def _draw_progressive_views(self, ctx: cairo.Context):
        """
        Draws the new WorkPieceViewArtifact bitmaps if available.
        This overlays the generated bitmaps onto the canvas.
        """
        self._resolve_colors_if_needed()
        world_w, world_h = self.data.size

        if world_w < 1e-9 or world_h < 1e-9:
            return

        for step_uid, surface_tuple in self._progressive_view_surfaces.items():
            if (
                not self._ops_visibility.get(step_uid, True)
                or not surface_tuple
            ):
                continue

            surface, bbox_mm, _, _ = surface_tuple
            view_x, view_y, view_w, view_h = cast(Tuple[float, ...], bbox_mm)

            if view_w < 1e-9 or view_h < 1e-9:
                logger.debug(f"Skipping step '{step_uid}': bbox too small")
                continue

            ctx.save()

            surface_w_px = surface.get_width()
            surface_h_px = surface.get_height()

            # Reconstruct the scale (ppm) used to generate this image
            # The runner guarantees width = min(round(w_mm * ppm) + 2 *
            # margin, MAX) So the effective ppm might be different from the
            # requested ppm if clamped. effective_ppm = (width - 2*margin)
            # / w_mm

            # Guard against zero division for 1D objects
            ppm_x = (
                (surface_w_px - 2 * OPS_MARGIN_PX) / view_w
                if view_w > 1e-9
                else 0
            )
            ppm_y = (
                (surface_h_px - 2 * OPS_MARGIN_PX) / view_h
                if view_h > 1e-9
                else 0
            )

            if ppm_x <= 0 or ppm_y <= 0:
                ctx.restore()
                continue

            # Calculate the full world dimensions of the surface (content +
            # margins)
            surface_w_world = surface_w_px / ppm_x
            surface_h_world = surface_h_px / ppm_y

            # Calculate the world offset of the surface's bottom-left corner
            # relative to workpiece origin. The content (view_x, view_y)
            # starts at (margin, margin) inside the surface (top-left in
            # pixel, bottom-left in world) So surface origin is at
            # (view_x - margin_w, view_y - margin_h)
            margin_w_world = OPS_MARGIN_PX / ppm_x
            margin_h_world = OPS_MARGIN_PX / ppm_y

            origin_x = view_x - margin_w_world
            origin_y = view_y - margin_h_world

            # Move to the bottom-left of the IMAGE in normalized space
            ctx.translate(origin_x / world_w, origin_y / world_h)

            # Scale to the size of the IMAGE in normalized space
            ctx.scale(surface_w_world / world_w, surface_h_world / world_h)

            # Flip Y for drawing the raster image
            ctx.translate(0, 1)
            ctx.scale(1, -1)

            # Scale 1.0/px to draw the image 1:1
            ctx.scale(1.0 / surface_w_px, 1.0 / surface_h_px)

            ctx.set_source_surface(surface, 0, 0)
            ctx.paint()

            ctx.restore()

    def push_transform_to_model(self):
        """Updates the data model's matrix with the view's transform."""
        if self.data.matrix != self.transform:
            logger.debug(
                f"Pushing view transform to model for '{self.data.name}'."
            )
            self.data.matrix = self.transform.copy()

    def on_travel_visibility_changed(self):
        """
        Handles changes in travel move visibility.

        For the new rendering path, this simply clears any old-style raster
        caches and triggers a redraw, as the drawing logic is dynamic.
        """
        logger.debug(
            "Travel visibility changed. Clearing raster caches and redrawing."
        )
        # The new render path draws dynamically from vertex data,
        # respecting the visibility flag at draw time.

        # Also clear from persistent cache so they don't reappear on reload
        self._update_model_view_cache()

        self._request_view_render()

        if self.canvas:
            self.canvas.queue_draw()

    def trigger_ops_rerender(self):
        """Triggers a re-render of all applicable ops for this workpiece."""
        if not self.data.layer or not self.data.layer.workflow:
            return

        logger.debug(f"Triggering ops rerender for '{self.data.name}'.")
        applicable_steps = self.data.layer.workflow.steps
        for step in applicable_steps:
            self._ops_generation_ids.get(step.uid, 0)

    def set_tabs_visible_override(self, visible: bool):
        """Sets the global visibility override for tab handles."""
        if self._tabs_visible_override != visible:
            self._tabs_visible_override = visible
            self._update_tab_handle_visibility()

    def _update_tab_handle_visibility(self):
        """Applies the current visibility logic to all tab handles."""
        # A handle is visible if the global toggle is on AND tabs are enabled
        # on the workpiece model.
        is_visible = self._tabs_visible_override and self.data.tabs_enabled
        for handle in self._tab_handles:
            handle.set_visible(is_visible)

    def _create_or_update_tab_handles(self):
        """Creates or replaces TabHandleElements based on the model."""
        # Remove old handles
        for handle in self._tab_handles:
            if handle in self.children:
                self.remove_child(handle)
        self._tab_handles.clear()

        # Determine visibility based on the global override and the model flag
        is_visible = self._tabs_visible_override and self.data.tabs_enabled

        if not self.data.tabs:
            return

        for tab in self.data.tabs:
            handle = TabHandleElement(tab_data=tab, parent=self)
            # The handle is now responsible for its own geometry.
            handle.update_base_geometry()
            handle.update_transform()
            handle.set_visible(is_visible)
            self._tab_handles.append(handle)
            self.add(handle)

    def update_handle_transforms(self):
        """
        Recalculates transforms for all tab handles. This is called on zoom.
        """
        # This method is now only called by the WorkSurface on zoom.
        # The live resize update is handled implicitly by the render pass.
        for handle in self._tab_handles:
            handle.update_transform()
