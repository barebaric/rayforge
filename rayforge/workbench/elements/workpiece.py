import logging
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast, List
import cairo
import numpy as np
from gi.repository import GLib
import threading
from ...context import get_context
from ...core.workpiece import WorkPiece
from ...core.step import Step
from ...core.matrix import Matrix
from ...pipeline.artifact import (
    BaseArtifactHandle,
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
        self._ops_generation_ids: Dict[
            str, int
        ] = {}  # Tracks the *expected* generation ID of the *next* render.
        self._view_artifact_surfaces: Dict[
            str,
            Tuple[
                cairo.ImageSurface, WorkPieceViewArtifact, BaseArtifactHandle
            ],
        ] = {}
        self._progressive_view_surfaces: Dict[
            str,
            List[
                Tuple[
                    cairo.ImageSurface,
                    WorkPieceViewArtifact,
                    BaseArtifactHandle,
                ]
            ],
        ] = {}

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
        self.pipeline.workpiece_chunk_available.connect(
            self._on_ops_chunk_available
        )
        self.pipeline.workpiece_artifact_ready.connect(
            self._on_ops_generation_finished
        )
        self.pipeline.workpiece_view_ready.connect(
            self._on_view_render_task_finished
        )
        self.pipeline.workpiece_view_created.connect(
            self._on_view_artifact_created
        )
        self.pipeline.workpiece_view_updated.connect(
            self._on_view_artifact_updated
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
        triggers a request to render it into a viewable artifact.
        """
        step_uid, workpiece_uid = key
        if workpiece_uid != self.data.uid or not self.canvas:
            # If we don't handle it, we must release it to prevent a leak.
            get_context().artifact_store.release(chunk_handle)
            return

        # Stale check
        if generation_id != self._ops_generation_ids.get(step_uid):
            logger.debug(
                f"Ignoring stale ops CHUNK event for step '{step_uid}'."
            )
            get_context().artifact_store.release(chunk_handle)
            return

        logger.debug(
            f"[{threading.current_thread().name}] "
            f"Received raw ops chunk for step '{step_uid}'. "
            f"Requesting view render for this chunk."
        )

        work_surface = cast("WorkSurface", self.canvas)
        ppm_x, ppm_y = work_surface.get_view_scale()
        self._resolve_colors_if_needed()
        if not self._color_set:
            logger.warning("Cannot render chunk: color set not resolved.")
            get_context().artifact_store.release(chunk_handle)
            return

        context = RenderContext(
            pixels_per_mm=(ppm_x, ppm_y),
            show_travel_moves=work_surface.show_travel_moves,
            margin_px=OPS_MARGIN_PX,
            color_set_dict=self._color_set.to_dict(),
        )
        # The handle is now passed to the pipeline, which takes over ownership.
        # We must not release it here. The view stage is now responsible.
        self.pipeline.request_view_render(
            step_uid,
            self.data.uid,
            context,
            source_handle=chunk_handle,
            generation_id=generation_id,
        )

    def _on_view_artifact_created(
        self,
        sender,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
        source_shm_name: str,
        generation_id: int,
        **kwargs,
    ):
        """
        Handler for when a new view artifact is created. It decides whether
        the artifact is for a progressive chunk or the final render, and
        updates the appropriate cache. This is the only place where state
        transitions occur.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            get_context().artifact_store.release(handle)
            return

        if generation_id != self._ops_generation_ids.get(step_uid):
            logger.debug(
                f"Ignoring stale 'view_artifact_created' event for step "
                f"'{step_uid}' (gen {generation_id} != "
                f"expected {self._ops_generation_ids.get(step_uid)})."
            )
            get_context().artifact_store.release(handle)
            return

        final_source_handle = self.pipeline.get_artifact_handle(
            step_uid, self.data.uid
        )
        is_final_artifact = (
            final_source_handle is not None
            and final_source_handle.shm_name == source_shm_name
        )

        try:
            artifact = cast(
                WorkPieceViewArtifact, get_context().artifact_store.get(handle)
            )
            if not artifact:
                logger.warning(
                    f"Could not read view artifact from handle for step "
                    f"{step_uid}. It may have been released."
                )
                get_context().artifact_store.release(handle)
                return

            h, w, _ = artifact.bitmap_data.shape

            # Create a surface that directly VIEWS the shared memory buffer.
            # This is crucial for progressive updates. The data is not copied.
            # The `artifact` object MUST be kept alive alongside this surface
            # to prevent the underlying shared memory from being closed.
            surface = cairo.ImageSurface.create_for_data(
                memoryview(np.ascontiguousarray(artifact.bitmap_data)),
                cairo.FORMAT_ARGB32,
                w,
                h,
            )

        except Exception as e:
            logger.error(
                f"Failed to process view artifact for '{step_uid}': {e}"
            )
            get_context().artifact_store.release(handle)
            return

        if is_final_artifact:
            logger.debug(f"Received FINAL view artifact for step '{step_uid}'")
            # Clean up the previous FINAL artifact for this step.
            if old_tuple := self._view_artifact_surfaces.pop(step_uid, None):
                _, _, old_handle = old_tuple
                get_context().artifact_store.release(old_handle)

            # Atomically update the cache with the new final artifact.
            self._view_artifact_surfaces[step_uid] = (
                surface,
                artifact,
                handle,
            )

            # ATOMIC CLEANUP: Now that the final render is cached, clear the
            # now-obsolete progressive chunks.
            if old_list := self._progressive_view_surfaces.pop(step_uid, None):
                logger.debug(
                    f"Finalizing render for step '{step_uid}', clearing "
                    f"{len(old_list)} progressive chunks."
                )
                for _, _, old_chunk_handle in old_list:
                    get_context().artifact_store.release(old_chunk_handle)
        else:
            logger.debug(
                f"Received PROGRESSIVE view artifact chunk for "
                f"step '{step_uid}'"
            )
            # This is a progressive chunk, append it to the list.
            if step_uid not in self._progressive_view_surfaces:
                self._progressive_view_surfaces[step_uid] = []
            self._progressive_view_surfaces[step_uid].append(
                (surface, artifact, handle)
            )

        self.canvas.queue_draw()

    def _on_view_render_task_finished(
        self,
        sender,
        *,
        step_uid: str,
        workpiece_uid: str,
        source_shm_name: str,
        generation_id: int,
        **kwargs,
    ):
        """
        Informational handler for when a view render task completes.
        This handler MUST NOT modify visual state to prevent race conditions.
        """
        if workpiece_uid != self.data.uid:
            return

        if generation_id != self._ops_generation_ids.get(step_uid):
            return

        final_source_handle = self.pipeline.get_artifact_handle(
            step_uid, self.data.uid
        )
        is_final_task = (
            final_source_handle is not None
            and final_source_handle.shm_name == source_shm_name
        )

        if is_final_task:
            logger.debug(
                f"Final render task for step '{step_uid}' has completed."
            )

    def _on_view_artifact_updated(
        self,
        sender,
        *,
        step_uid: str,
        workpiece_uid: str,
        **kwargs,
    ):
        """
        Handler for when the content of a progressive render artifact has
        been updated by the background worker.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            return

        did_update = False
        # Mark any progressive surfaces for this step as dirty.
        if surface_list := self._progressive_view_surfaces.get(step_uid):
            for surface, _artifact, _handle in surface_list:
                surface.mark_dirty()
            did_update = True

        # Mark the final surface for this step as dirty, if it exists.
        if surface_tuple := self._view_artifact_surfaces.get(step_uid):
            surface, _artifact, _handle = surface_tuple
            surface.mark_dirty()
            did_update = True

        if did_update:
            logger.debug(
                f"[{threading.current_thread().name}] "
                f"View artifact for '{step_uid}' was updated, redrawing."
            )
            self.canvas.queue_draw()

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
        self.pipeline.workpiece_view_ready.disconnect(
            self._on_view_render_task_finished
        )
        self.pipeline.workpiece_view_created.disconnect(
            self._on_view_artifact_created
        )
        self.pipeline.workpiece_view_updated.disconnect(
            self._on_view_artifact_updated
        )
        super().remove()

        for surface_list in self._progressive_view_surfaces.values():
            for _, _, handle in surface_list:
                get_context().artifact_store.release(handle)
        self._progressive_view_surfaces.clear()

        for _, _, handle in self._view_artifact_surfaces.values():
            get_context().artifact_store.release(handle)
        self._view_artifact_surfaces.clear()

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
        """Removes the cached view artifact surfaces for a step."""
        logger.debug(f"Clearing ops surface for step '{step_uid}'")

        if old_tuple := self._view_artifact_surfaces.pop(step_uid, None):
            logger.debug(f"  - Cleared final artifact for step '{step_uid}'")
            _, _, old_handle = old_tuple
            get_context().artifact_store.release(old_handle)
        if old_list := self._progressive_view_surfaces.pop(step_uid, None):
            logger.debug(
                f"  - Cleared {len(old_list)} progressive artifact(s) for "
                f"step '{step_uid}'"
            )
            for _, _, old_handle in old_list:
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
            f"Requesting view artifact updates for '{self.data.name}'"
            f" (step: {specific_step_uid or 'all'})"
        )

        if specific_step_uid is None:
            # A global refresh (e.g., zoom, pan, travel visibility change)
            # invalidates all resolution-dependent progressive chunks.
            for step_uid in list(self._progressive_view_surfaces.keys()):
                if old_list := self._progressive_view_surfaces.pop(
                    step_uid, None
                ):
                    logger.debug(
                        f"Clearing {len(old_list)} stale progressive chunks "
                        f"for step '{step_uid}' due to view change."
                    )
                    for _, _, old_chunk_handle in old_list:
                        get_context().artifact_store.release(old_chunk_handle)

        work_surface = cast("WorkSurface", self.canvas)
        ppm_x, ppm_y = work_surface.get_view_scale()
        self._resolve_colors_if_needed()

        if not self._color_set:
            logger.warning(
                "Cannot request view update: color set not resolved."
            )
            return

        steps_to_process: List[Tuple[Step, int]] = []
        if specific_step_uid and generation_id is not None:
            step = self._find_step_by_uid(specific_step_uid)
            if step:
                steps_to_process.append((step, generation_id))
        elif specific_step_uid is None:
            # Global refresh: re-render all steps with their latest gen ID
            for step in self.data.layer.workflow.steps:
                latest_gen_id = self._ops_generation_ids.get(step.uid)
                if latest_gen_id is not None:
                    steps_to_process.append((step, latest_gen_id))

        for step, gen_id in steps_to_process:
            source_handle = self.pipeline.get_artifact_handle(
                step.uid, self.data.uid
            )
            if not isinstance(source_handle, WorkPieceArtifactHandle):
                logger.debug(
                    f"No source artifact for step '{step.uid}', "
                    "skipping view render request."
                )
                continue

            # We need to acquire a new reference for the pipeline, which will
            # own it until the task is done. The cache still holds its own ref.
            get_context().artifact_store.acquire(source_handle)

            context = RenderContext(
                pixels_per_mm=(ppm_x, ppm_y),
                show_travel_moves=work_surface.show_travel_moves,
                margin_px=OPS_MARGIN_PX,
                color_set_dict=self._color_set.to_dict(),
            )

            logger.debug(f"... for step '{step.uid}' (gen: {gen_id})")
            self.pipeline.request_view_render(
                step.uid,
                self.data.uid,
                context,
                source_handle=source_handle,
                generation_id=gen_id,
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
        logger.debug(
            f"START _on_ops_generation_starting for step '{sender.uid}'"
        )
        if workpiece is not self.data:
            return
        step_uid = sender.uid
        self._ops_generation_ids[step_uid] = generation_id
        self.clear_ops_surface(step_uid)
        logger.debug(
            f"END _on_ops_generation_starting for step '{sender.uid}'"
        )

    def _on_ops_generation_finished(
        self, sender: Step, workpiece: WorkPiece, generation_id: int, **kwargs
    ):
        """
        Signal handler for when ops generation finishes. This runs on a
        background thread and schedules the actual work on the main thread.
        """
        logger.debug(
            f"START _on_ops_generation_finished for step '{sender.uid}'"
        )
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
        logger.debug(
            f"_on_ops_generation_finished_main_thread called for step "
            f"'{sender.uid}'"
        )
        if workpiece is not self.data:
            return

        # Stale check: A new generation might have been requested since this
        # one finished.
        step = sender
        if generation_id != self._ops_generation_ids.get(step.uid):
            logger.debug(
                f"Ignoring stale ops finish event for step '{step.uid}'."
            )
            return

        logger.debug(
            f"Ops generation finished for step '{step.uid}'. "
            f"Requesting final view render."
        )
        # The final artifact is ready. We only need to request its view render.
        # The progressive chunks will be cleared atomically when the final
        # render *task* completes, which is handled by
        # _on_view_render_task_finished.
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
        # Let the base class handle the main content surface update.
        # This will return False for the GLib timer.
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
        surface_tuple: Tuple[
            cairo.ImageSurface, WorkPieceViewArtifact, BaseArtifactHandle
        ],
    ):
        """Helper to draw one pre-rendered bitmap to the context."""
        surface, artifact, handle = surface_tuple
        ops_x, ops_y, ops_w, ops_h = cast(Tuple[float, ...], artifact.bbox_mm)
        world_w, world_h = self.data.size

        logger.debug(
            f"  - Drawing surface: "
            f"handle={handle.shm_name.split('_')[-1][:8]}, "
            f"dims=({surface.get_width()}x{surface.get_height()}), "
            f"bbox_mm=({ops_x:.2f}, {ops_y:.2f}, {ops_w:.2f}, {ops_h:.2f}), "
            f"wp_size=({world_w:.2f}, {world_h:.2f})"
        )

        if world_w < 1e-9 or world_h < 1e-9 or ops_w < 1e-9 or ops_h < 1e-9:
            logger.debug("  - Skipping draw due to zero-size dimension.")
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

    def _draw_view_artifacts(self, ctx: cairo.Context):
        """Draws all available pre-rendered view artifact surfaces."""
        logger.debug(
            f"[{threading.current_thread().name}] "
            f"Drawing view artifacts. "
            f"Final keys: {list(self._view_artifact_surfaces.keys())}, "
            f"Progressive keys: {list(self._progressive_view_surfaces.keys())}"
        )
        # Draw final, completed artifacts first.
        for step_uid, surface_tuple in self._view_artifact_surfaces.items():
            if not self._ops_visibility.get(step_uid, True):
                continue
            logger.debug(f"  - Drawing FINAL artifact for step '{step_uid}'")
            self._draw_single_view_surface(ctx, surface_tuple)

        # Draw in-progress artifacts only if a final one doesn't exist.
        # This ensures the "last known good" render remains visible until
        # the new one is complete.
        for step_uid, surface_list in self._progressive_view_surfaces.items():
            if step_uid in self._view_artifact_surfaces:
                logger.debug(
                    f"  - Skip progressive for '{step_uid}' (final exists)."
                )
                continue
            if not self._ops_visibility.get(step_uid, True):
                continue

            logger.debug(
                f"  - Drawing {len(surface_list)} PROGRESSIVE artifact(s) "
                f"for step '{step_uid}'"
            )
            for surface_tuple in surface_list:
                self._draw_single_view_surface(ctx, surface_tuple)

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

        self._draw_view_artifacts(ctx)

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
