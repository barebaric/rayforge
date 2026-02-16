import logging
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast, List
import cairo
import numpy as np
from gi.repository import GLib
from ....core.workpiece import WorkPiece
from ....core.step import Step
from ....core.matrix import Matrix
from ....pipeline.artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
    WorkPieceViewArtifact,
)
from ....shared.util.colors import ColorSet
from ...shared.gtk_color import GtkColorResolver, ColorSpecDict
from ...canvas import CanvasElement
from .tab_handle import TabHandleElement

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ....pipeline.view import ViewManager

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
        view_manager: "ViewManager",
        **kwargs,
    ):
        """Initializes the WorkPieceElement.

        Args:
            workpiece: The WorkPiece data model to visualize.
            view_manager: The ViewManager for view rendering.
            **kwargs: Additional arguments for the CanvasElement.
        """
        logger.debug(f"Initializing WorkPieceElement for '{workpiece.name}'")
        self.data: WorkPiece = workpiece
        self.view_manager = view_manager
        self._base_image_visible = True
        self._surface: Optional[cairo.ImageSurface] = None

        self._ops_visibility: Dict[str, bool] = {}
        self._artifact_cache: Dict[str, Optional[WorkPieceArtifact]] = {}
        self._ops_surface_cache: Dict[str, cairo.ImageSurface] = {}
        self._ops_surface_data_cache: Dict[str, np.ndarray] = {}
        self._ops_metadata_cache: Dict[
            str, Tuple
        ] = {}  # (bbox_mm, workpiece_size_mm)

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
        self._rendered_ppm: float = 0.0

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
        self.view_manager.source_artifact_ready.connect(
            self._on_source_artifact_ready
        )
        self.view_manager.view_artifact_updated.connect(
            self._on_view_artifact_updated
        )
        self.view_manager.generation_finished.connect(
            self._on_view_generation_finished
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

    def _hydrate_from_cache(self) -> bool:
        """
        Restores visual state from the persistent model cache if available.
        Returns True if significant state was restored.
        """
        cache = self.data._view_cache
        if not cache:
            return False

        self._surface = cache.get("surface")
        self._artifact_cache = cache.get("artifact_cache", {}).copy()

        return self._surface is not None or len(self._artifact_cache) > 0

    def _update_model_view_cache(self):
        """
        Updates the persistent cache on the model with current view state.
        """
        cache = self.data._view_cache
        cache["surface"] = self._surface
        cache["artifact_cache"] = self._artifact_cache

    def invalidate_and_rerender(self):
        """
        Invalidates all cached rendering artifacts (base image and all ops)
        and schedules a full re-render. This should be called whenever the
        element's content or size changes.
        """
        logger.debug(f"Full invalidation for workpiece '{self.data.name}'")
        self._rendered_ppm = 0.0
        # Clear the local artifact cache to prevent drawing stale vectors
        self._artifact_cache.clear()
        self._ops_surface_cache.clear()
        self._ops_surface_data_cache.clear()
        self._ops_metadata_cache.clear()

        # Clear the model cache as well, since the data is invalid
        self.data._view_cache.clear()

        if self.data.layer and self.data.layer.workflow:
            for step in self.data.layer.workflow.steps:
                self.clear_ops_surface(step.uid)
        super().trigger_update()

    def trigger_view_update(self, ppm: float = 0.0) -> bool:
        """
        Invalidates resolution-dependent caches (raster surfaces) and
        triggers a re-render. This is called on view changes like zooming.
        It preserves expensive-to-generate data like vector recordings.

        Only re-renders if the new resolution (ppm) is higher than what
        was previously rendered, since scaling down an existing image
        doesn't require re-rendering.

        Returns True if a re-render was triggered, False if skipped.
        """
        if ppm <= self._rendered_ppm:
            return False

        logger.debug(f"View update for workpiece '{self.data.name}'")
        self._rendered_ppm = ppm

        # Note: We do NOT clear self._surface here to prevent flicker.
        # The old surface will be replaced once the new one is ready
        # in _apply_surface().

        # Note: We do NOT clear the model cache here, as view updates
        # (like zooming) shouldn't erase the persistent data needed by
        # other views or future rebuilds.

        # Trigger a re-render of everything at the new resolution.
        self.trigger_ops_rerender()
        super().trigger_update()  # Re-renders the base image.
        return True

    def _apply_surface(
        self, new_surface: Optional[cairo.ImageSurface]
    ) -> bool:
        """
        Applies the newly rendered surface from the background task.

        This override prevents flicker by only replacing the surface
        if the new one is valid. If the new surface is None, the old
        surface is kept as a fallback.

        Args:
            new_surface: The new surface to apply, or None.
        """
        if new_surface is not None:
            self.surface = new_surface
            self.mark_dirty(ancestors=True)
        if self.canvas:
            self.canvas.queue_draw()
        self._update_future = None
        return False

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
        Handler for when the content of a view artifact has been updated
        by the background worker.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            return
        self.canvas.queue_draw()

    def _on_view_generation_finished(
        self,
        sender,
        *,
        key,
        workpiece_uid: str,
        step_uid: str,
        **kwargs,
    ):
        """
        Handler for when view generation is complete.
        This is the safe time to update the ops surface cache.
        """
        if workpiece_uid != self.data.uid:
            return

        view_handle = self.view_manager.get_view_handle(
            self.data.uid, step_uid
        )
        if view_handle is None:
            return

        artifact = self.view_manager.store.get(view_handle)
        if not isinstance(artifact, WorkPieceViewArtifact):
            return

        try:
            new_data = np.copy(artifact.bitmap_data)
            height, width, _ = new_data.shape
            stride = cairo.ImageSurface.format_stride_for_width(
                cairo.FORMAT_ARGB32, width
            )
            new_surface = cairo.ImageSurface.create_for_data(
                new_data,
                cairo.FORMAT_ARGB32,
                width,
                height,
                stride,
            )
            self._ops_surface_cache[step_uid] = new_surface
            self._ops_surface_data_cache[step_uid] = new_data
            self._ops_metadata_cache[step_uid] = (
                artifact.bbox_mm,
                artifact.workpiece_size_mm,
            )
            if self.canvas:
                self.canvas.queue_draw()
        except Exception as e:
            logger.warning(
                f"Failed to update ops cache for step {step_uid}: {e}"
            )

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
        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        self.view_manager.source_artifact_ready.disconnect(
            self._on_source_artifact_ready
        )
        self.view_manager.view_artifact_updated.disconnect(
            self._on_view_artifact_updated
        )
        self.view_manager.generation_finished.disconnect(
            self._on_view_generation_finished
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

    def _on_transform_changed(
        self, workpiece: WorkPiece, *, old_matrix: Optional["Matrix"] = None
    ):
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
            # Sync the transform immediately
            self.set_transform(workpiece.matrix)
            # Note: We do NOT request view renders here. The pipeline will
            # automatically trigger view rendering when the workpiece
            # artifact is regenerated after the size change.
            super().trigger_update()
        else:
            # Size hasn't changed, just sync the transform
            self.set_transform(workpiece.matrix)

    def _on_source_artifact_ready(
        self,
        sender,
        *,
        step: Step,
        workpiece: WorkPiece,
        handle,
        **kwargs,
    ):
        """
        Signal handler for when source artifact is ready from ViewManager.
        This runs on a background thread, so it schedules the actual work
        on the main thread to prevent UI deadlocks.
        """
        if workpiece is not self.data:
            return

        artifact = self.view_manager.store.get(handle)
        GLib.idle_add(
            self._on_source_artifact_ready_main_thread,
            step,
            workpiece,
            artifact,
        )

    def _on_source_artifact_ready_main_thread(
        self,
        step: Step,
        workpiece: WorkPiece,
        artifact: WorkPieceArtifact,
    ):
        """The thread-safe part of the source artifact ready handler."""
        logger.debug(
            f"_on_source_artifact_ready_main_thread called for step "
            f"'{step.uid}'"
        )
        if workpiece is not self.data:
            return

        self._artifact_cache[step.uid] = artifact
        self._update_model_view_cache()

        if self.canvas:
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

        # Draw view artifacts (complete, pre-rendered bitmaps)
        self._resolve_colors_if_needed()
        world_w, world_h = self.data.size

        if world_w < 1e-9 or world_h < 1e-9:
            return

        if self.data.layer and self.data.layer.workflow:
            for step in self.data.layer.workflow.steps:
                step_uid = step.uid
                if not self._ops_visibility.get(step_uid, True):
                    continue

                # First try cached surface (prevents flicker during zoom)
                surface = self._ops_surface_cache.get(step_uid)
                metadata = self._ops_metadata_cache.get(step_uid)
                bbox_mm = None
                workpiece_size_mm = None

                if surface is not None and metadata is not None:
                    bbox_mm, workpiece_size_mm = metadata
                else:
                    # No cache - try live artifact for progressive rendering
                    view_handle = self.view_manager.get_view_handle(
                        self.data.uid, step.uid
                    )
                    if view_handle is not None:
                        artifact = self.view_manager.store.get(view_handle)
                        if isinstance(artifact, WorkPieceViewArtifact):
                            try:
                                new_data = np.copy(artifact.bitmap_data)
                                height, width, _ = new_data.shape
                                stride = (
                                    cairo.ImageSurface.format_stride_for_width(
                                        cairo.FORMAT_ARGB32, width
                                    )
                                )
                                surface = cairo.ImageSurface.create_for_data(
                                    new_data,
                                    cairo.FORMAT_ARGB32,
                                    width,
                                    height,
                                    stride,
                                )
                                bbox_mm = artifact.bbox_mm
                                workpiece_size_mm = artifact.workpiece_size_mm
                            except Exception:
                                pass

                if surface is None:
                    continue

                if bbox_mm is None or workpiece_size_mm is None:
                    continue

                try:
                    view_x, view_y, view_w, view_h = bbox_mm

                    if view_w < 1e-9 or view_h < 1e-9:
                        logger.debug(
                            f"Skipping step '{step_uid}': bbox too small"
                        )
                        continue

                    world_w, world_h = self.data.size
                    ref_w, ref_h = workpiece_size_mm

                    scale_x = world_w / ref_w if ref_w > 1e-9 else 1.0
                    scale_y = world_h / ref_h if ref_h > 1e-9 else 1.0

                    if not (
                        abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6
                    ):
                        view_x *= scale_x
                        view_y *= scale_y
                        view_w *= scale_x
                        view_h *= scale_y

                    surface_w_px = surface.get_width()
                    surface_h_px = surface.get_height()

                    # Reconstruct the scale (ppm) used to generate
                    # this image. The runner guarantees width =
                    # min(round(w_mm * ppm) + 2 * margin, MAX)
                    # So effective ppm might be different from requested
                    # ppm if clamped.
                    # effective_ppm = (width - 2*margin) / w_mm

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
                        continue

                    # Calculate the full world dimensions of the surface
                    # (content + margins)
                    surface_w_world = surface_w_px / ppm_x
                    surface_h_world = surface_h_px / ppm_y

                    # Calculate the world offset of the surface's
                    # bottom-left corner relative to workpiece origin.
                    # The content (view_x, view_y) starts at (margin,
                    # margin) inside the surface (top-left in pixel,
                    # bottom-left in world) So surface origin is at
                    # (view_x - margin_w, view_y - margin_h)
                    margin_w_world = OPS_MARGIN_PX / ppm_x
                    margin_h_world = OPS_MARGIN_PX / ppm_y

                    origin_x = view_x - margin_w_world
                    origin_y = view_y - margin_h_world

                    ctx.save()

                    # Move to the bottom-left of the IMAGE in normalized
                    # space
                    ctx.translate(origin_x / world_w, origin_y / world_h)

                    # Scale to the size of the IMAGE in normalized space
                    ctx.scale(
                        surface_w_world / world_w, surface_h_world / world_h
                    )

                    # Flip Y for drawing the raster image
                    ctx.translate(0, 1)
                    ctx.scale(1, -1)

                    # Scale 1.0/px to draw the image 1:1
                    ctx.scale(1.0 / surface_w_px, 1.0 / surface_h_px)

                    ctx.set_source_surface(surface, 0, 0)
                    ctx.paint()

                    ctx.restore()
                except Exception as e:
                    logger.warning(
                        f"Failed to draw view artifact for step "
                        f"{step_uid}: {e}"
                    )

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

        if self.canvas:
            self.canvas.queue_draw()

    def trigger_ops_rerender(self):
        """Triggers a re-render of all applicable ops for this workpiece."""
        if not self.data.layer or not self.data.layer.workflow:
            return

        logger.debug(f"Triggering ops rerender for '{self.data.name}'.")

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
