import logging
from typing import Optional, TYPE_CHECKING, Dict, Tuple, cast, List, Union
import cairo
from concurrent.futures import Future
import numpy as np
from ...core.workpiece import WorkPiece
from ...core.step import Step
from ...core.matrix import Matrix
from ...core.ops import Ops
from ..canvas import CanvasElement
from ...pipeline.encoder.cairoencoder import CairoEncoder
from ...shared.util.colors import ColorSet
from ...shared.util.gtk_color import GtkColorResolver, ColorSpecDict
from .tab_handle import TabHandleElement
from ...pipeline.artifact.vector import VectorArtifact
from ...pipeline.artifact.vertex import VertexArtifact
from ...pipeline.artifact.hybrid import HybridRasterArtifact

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ...pipeline.generator import OpsGenerator

logger = logging.getLogger(__name__)

# Cairo has a hard limit on surface dimensions.
CAIRO_MAX_DIMENSION = 30000
OPS_MARGIN_PX = 5
REC_MARGIN_MM = 0.1  # A small "safe area" margin in mm for recordings


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

    USE_NEW_RENDER_PATH = True

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
        self._raster_textures: Dict[str, cairo.ImageSurface] = {}
        # Cached artifacts to avoid re-fetching from generator on every draw.
        self._artifact_cache: Dict[
            str,
            Optional[
                Union[VectorArtifact, VertexArtifact, HybridRasterArtifact]
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
            # CRITICAL: clip must be False so the parent `render` method
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
        self._ops_surfaces.pop(step_uid, None)
        self._ops_recordings.pop(step_uid, None)
        self._raster_textures.pop(step_uid, None)
        self._artifact_cache.pop(step_uid, None)
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
                "Resolving colors for WorkPieceView due to theme change."
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
            f"Scheduling async recording of drawing commands."
        )
        self._ops_generation_ids[step.uid] = generation_id

        if self.USE_NEW_RENDER_PATH:
            # Fetch and cache the final artifact.
            # This is a one-time fetch after generation completes.
            artifact = self.ops_generator.get_artifact(step, self.data)
            self._artifact_cache[step.uid] = artifact

            # Clear intermediate chunk surfaces, as the final artifact is now
            # ready to be drawn in the next paint cycle.
            self._ops_surfaces.pop(step.uid, None)
            # Also clear the specific texture cache, as it may be stale
            # (e.g., artifact changed from raster to vector).
            self._raster_textures.pop(step.uid, None)
            if self.canvas:
                self.canvas.queue_draw()
            return

        if future := self._ops_render_futures.pop(step.uid, None):
            future.cancel()
        future = self._executor.submit(
            self._record_ops_drawing_async, step, generation_id
        )
        self._ops_render_futures[step.uid] = future
        future.add_done_callback(self._on_ops_drawing_recorded)

    def _encode_ops_to_context(
        self,
        ops: Ops,
        ctx: cairo.Context,
        scale: Tuple[float, float],
        drawable_height: float,
    ):
        """
        Helper method to centralize encoding Ops to a Cairo context with
        consistent colors and visibility settings.
        """
        if not self.canvas or not self._color_set:
            return

        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface.show_travel_moves

        encoder = CairoEncoder()
        encoder.encode(
            ops,
            ctx,
            scale,
            colors=self._color_set,
            show_cut_moves=True,
            show_engrave_moves=True,
            show_travel_moves=show_travel,
            show_zero_power_moves=show_travel,  # As per request
            drawable_height=drawable_height,
        )

    def _record_ops_drawing_async(
        self, step: Step, generation_id: int
    ) -> Optional[Tuple[str, cairo.RecordingSurface, int]]:
        """
        "Draws" the ops to a RecordingSurface. This captures all vector
        commands and is done only when ops data changes.
        """
        logger.debug(
            f"Recording vector ops for workpiece "
            f"'{self.data.name}', step '{step.uid}'"
        )
        ops = self.ops_generator.get_ops(step, self.data)
        if not ops or not self.canvas:
            return None

        self._resolve_colors_if_needed()
        world_w, world_h = self.data.size
        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface.show_travel_moves

        # Calculate the union of the workpiece bounds and the ops bounds to
        # ensure the recording surface is large enough.
        ops_x1, ops_y1, ops_x2, ops_y2 = ops.rect(include_travel=show_travel)
        union_x1 = min(0.0, ops_x1)
        union_y1 = min(0.0, ops_y1)
        union_x2 = max(world_w, ops_x2)
        union_y2 = max(world_h, ops_y2)

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

        # We are drawing 1:1 in mm space, so scale is 1.0.
        encoder_ppms = (1.0, 1.0)

        # Pass the workpiece height to the encoder. Ops coordinates are
        # relative to the workpiece's Y-up coordinate system, so the flip
        # must be relative to the workpiece height.
        self._encode_ops_to_context(ops, ctx, encoder_ppms, world_h)

        return step.uid, surface, generation_id

    def _on_ops_drawing_recorded(self, future: Future):
        """Callback executed when the async ops recording is done."""
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

        # Find the Step object to trigger the initial rasterization.
        found_step = None
        if self.data.layer and self.data.layer.workflow:
            for step_obj in self.data.layer.workflow.steps:
                if step_obj.uid == step_uid:
                    found_step = step_obj
                    break
        if found_step:
            self._trigger_ops_rasterization(found_step, received_gen_id)
        else:
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
                return  # A rasterization is already in progress.

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
            # Slow fallback calculate bounds from ops.
            ops = self.ops_generator.get_ops(step, self.data)
            if not ops:
                return None
            ops_x1, ops_y1, ops_x2, ops_y2 = ops.rect(
                include_travel=show_travel
            )
            union_x1 = min(0.0, ops_x1)
            union_y1 = min(0.0, ops_y1)
            union_x2 = max(world_w, ops_x2)
            union_y2 = max(world_h, ops_y2)
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
            # SLOW FALLBACK: No recording yet, render from Ops directly.
            ops = self.ops_generator.get_ops(step, self.data)
            if not ops:
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
            self._encode_ops_to_context(ops, ctx, ppms, drawable_h_px)

        return step_uid, surface, generation_id, bbox_mm

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

        _surface, ctx, ppms, content_h_px = prepared

        # Encode just the chunk onto the existing surface
        self._encode_ops_to_context(chunk, ctx, ppms, content_h_px)

        # Trigger a redraw to show the progress
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
        if not self.USE_NEW_RENDER_PATH:
            self.trigger_ops_rerender()

        return res

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Renders the base workpiece content to a new surface."""
        return self.data.render_to_pixels(width=width, height=height)

    def _draw_from_vertex_artifact(self, ctx: cairo.Context):
        """Draws ops overlays directly from pre-computed vertex artifacts."""
        self._resolve_colors_if_needed()
        if not self.canvas or not self._color_set:
            return

        world_w, world_h = self.data.size
        if world_w < 1e-9 or world_h < 1e-9:
            return

        work_surface = cast("WorkSurface", self.canvas)
        show_travel = work_surface.show_travel_moves

        # --- Aggregate artifacts and draw raster components first ---
        artifacts_to_draw = []
        if self.data.layer and self.data.layer.workflow:
            for step in self.data.layer.workflow.steps:
                if not self._ops_visibility.get(step.uid, True):
                    continue
                # Skip drawing the final artifact if an intermediate chunk
                # surface exists for this step.
                if step.uid in self._ops_surfaces:
                    continue

                # Use the local cache instead of fetching from the generator.
                artifact = self._artifact_cache.get(step.uid)
                if isinstance(
                    artifact, (VertexArtifact, HybridRasterArtifact)
                ):
                    artifacts_to_draw.append(artifact)

                    if isinstance(artifact, HybridRasterArtifact):
                        self._draw_raster_texture(ctx, step, artifact)

        if not artifacts_to_draw:
            return

        # --- Aggregate vector components from all artifacts ---
        all_powered_v = [
            a.powered_vertices
            for a in artifacts_to_draw
            if a.powered_vertices.size > 0
        ]
        all_powered_c = [
            a.powered_colors
            for a in artifacts_to_draw
            if a.powered_colors.size > 0
        ]
        all_travel_v = [
            a.travel_vertices
            for a in artifacts_to_draw
            if a.travel_vertices.size > 0
        ]
        all_zero_power_v = [
            a.zero_power_vertices
            for a in artifacts_to_draw
            if a.zero_power_vertices.size > 0
        ]

        # --- Draw all aggregated vector components ---
        ctx.save()
        # The context is for a 1x1 Y-UP space. Scale to workpiece mm space.
        ctx.scale(1.0 / world_w, 1.0 / world_h)
        ctx.set_hairline(True)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        # --- Draw Travel & Zero-Power Moves ---
        if show_travel:
            if all_travel_v:
                travel_v = np.concatenate(all_travel_v).reshape(-1, 2, 3)
                ctx.set_source_rgba(*self._color_set.get_rgba("travel"))
                for start, end in travel_v:
                    ctx.move_to(start[0], start[1])
                    ctx.line_to(end[0], end[1])
                ctx.stroke()
            if all_zero_power_v:
                zero_v = np.concatenate(all_zero_power_v).reshape(-1, 2, 3)
                ctx.set_source_rgba(*self._color_set.get_rgba("zero_power"))
                for start, end in zero_v:
                    ctx.move_to(start[0], start[1])
                    ctx.line_to(end[0], end[1])
                ctx.stroke()

        # --- Draw Powered Moves (Grouped by Color) ---
        if all_powered_v:
            powered_v = np.concatenate(all_powered_v).reshape(-1, 2, 3)
            powered_c = np.concatenate(all_powered_c)
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
                    ctx.move_to(start[0], start[1])
                    ctx.line_to(end[0], end[1])
                ctx.stroke()

        ctx.restore()

    def _draw_raster_texture(
        self,
        ctx: cairo.Context,
        step: Step,
        artifact: HybridRasterArtifact,
    ):
        """Generates, caches, and draws the themed raster texture."""
        if not self._color_set:
            return
        world_w, world_h = self.data.size

        texture = self._raster_textures.get(step.uid)
        if not texture and artifact.power_texture_data.size > 0:
            power_data = artifact.power_texture_data
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

            texture = cairo.ImageSurface.create_for_data(
                memoryview(np.ascontiguousarray(bgra_texture)),
                cairo.FORMAT_ARGB32,
                w,
                h,
            )
            self._raster_textures[step.uid] = texture

        if texture:
            ctx.save()
            pos_mm = artifact.position_mm
            dim_mm = artifact.dimensions_mm

            # Context is 1x1 Y-UP space, origin at bottom-left of workpiece.
            # Convert all mm values to normalized coordinates first.
            norm_w = dim_mm[0] / world_w
            norm_h = dim_mm[1] / world_h
            norm_x = pos_mm[0] / world_w
            norm_y_bottom = (world_h - pos_mm[1] - dim_mm[1]) / world_h

            tex_w_px = texture.get_width()
            tex_h_px = texture.get_height()

            if tex_w_px <= 0 or tex_h_px <= 0:
                ctx.restore()
                return

            # Go to the bottom-left corner of the destination rectangle.
            ctx.translate(norm_x, norm_y_bottom)

            # We are now in a local space. Flip Y and translate to prepare
            # for drawing the Y-down surface.
            ctx.translate(0, norm_h)
            ctx.scale(1, -1)

            # At this point, the origin is at the top-left of the destination
            # rectangle, and the Y-axis points down. The user space has the
            # same dimensions (in mm) as the destination rectangle.

            # Now, scale the context so that drawing the full pixel-sized
            # texture will fit exactly into this destination rectangle.
            ctx.scale(norm_w / tex_w_px, norm_h / tex_h_px)

            # Add a half-pixel offset in the new, pixel-scaled space to
            # align the raster grid with vector coordinates.
            ctx.translate(0.5, 0.5)

            ctx.set_source_surface(texture, 0, 0)
            ctx.get_source().set_filter(cairo.Filter.GOOD)
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
            # This handles the Y-flip for the base image and restores the
            # context, leaving it Y-UP for the next drawing operation.
            super().draw(ctx)

        # Draw Ops (hide during simulation mode)
        worksurface = cast("WorkSurface", self.canvas) if self.canvas else None
        if not worksurface or worksurface.is_simulation_mode():
            return

        if self.USE_NEW_RENDER_PATH:
            self._draw_from_vertex_artifact(ctx)

        # Draw intermediate chunk surfaces for progressive rendering,
        # regardless of the rendering path. They are cleared when the final
        # artifact is ready.
        self._draw_intermediate_chunks(ctx)

    def _draw_intermediate_chunks(self, ctx: cairo.Context):
        """Draws the old-style rasterized chunk surfaces."""
        # Draw each ops surface that is visible.
        self._resolve_colors_if_needed()
        for step_uid, surface_tuple in self._ops_surfaces.items():
            if (
                not self._ops_visibility.get(step_uid, True)
                or not surface_tuple
            ):
                continue

            surface, bbox_mm = surface_tuple
            ops_x, ops_y, ops_w, ops_h = cast(Tuple[float, ...], bbox_mm)
            world_w, world_h = self.data.size

            if (
                world_w < 1e-9
                or world_h < 1e-9
                or ops_w < 1e-9
                or ops_h < 1e-9
            ):
                continue

            ctx.save()

            # The drawing context is a 1x1 Y-UP space relative to the
            # workpiece.
            # We first transform to where the ops content should be placed
            # and scaled, in this normalized space.
            ctx.translate(ops_x / world_w, ops_y / world_h)
            ctx.scale(ops_w / world_w, ops_h / world_h)

            # Now we have a 1x1 Y-UP box that represents the ops bounding box.
            # Draw the Y-DOWN pixel surface into it, which requires a Y-flip.
            ctx.translate(0, 1)
            ctx.scale(1, -1)

            # Scale our 1x1 box to match the pixel dimensions of the surface.
            surface_w_px = surface.get_width()
            surface_h_px = surface.get_height()
            content_w_px = surface_w_px - 2 * OPS_MARGIN_PX
            content_h_px = surface_h_px - 2 * OPS_MARGIN_PX

            if content_w_px <= 0 or content_h_px <= 0:
                ctx.restore()
                continue

            ctx.scale(1.0 / content_w_px, 1.0 / content_h_px)

            # Paint the surface, offsetting for the margin.
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

    def on_travel_visibility_changed(self):
        """
        Invalidates cached ops recordings to reflect the new travel move
        visibility state. This should be called by the parent WorkSurface.
        """
        logger.debug(
            "Travel visibility changed, invalidating ops vector recordings."
        )
        # Clearing the recordings is the key step. The next
        # `trigger_ops_rerender`
        # will see they are missing and schedule a re-recording.
        self._ops_recordings.clear()
        self.trigger_ops_rerender()

    def trigger_ops_rerender(self):
        """Triggers a re-render of all applicable ops for this workpiece."""
        if not self.data.layer or not self.data.layer.workflow:
            return
        logger.debug(f"Triggering ops rerender for '{self.data.name}'.")
        applicable_steps = self.data.layer.workflow.steps
        for step in applicable_steps:
            gen_id = self._ops_generation_ids.get(step.uid, 0)
            if self._ops_recordings.get(step.uid):
                # FAST PATH: Recording exists, just trigger rasterization.
                self._trigger_ops_rasterization(step, generation_id=gen_id)
            else:
                # SLOW PATH: Recording is missing. We need to re-record it.
                # This logic mimics _on_ops_generation_finished.
                logger.debug(
                    f"No recording for step '{step.uid}', re-recording."
                )
                if future := self._ops_render_futures.pop(step.uid, None):
                    future.cancel()
                future = self._executor.submit(
                    self._record_ops_drawing_async, step, gen_id
                )
                self._ops_render_futures[step.uid] = future
                future.add_done_callback(self._on_ops_drawing_recorded)

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
