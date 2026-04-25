import logging
from typing import Optional, TYPE_CHECKING, Dict, Set, Tuple, cast, List
import cairo
import numpy as np
from gi.repository import Gdk, GLib
from ....core.geo import Geometry
from ....core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
    COL_I,
    COL_J,
    COL_CW,
)
from ....core.workpiece import WorkPiece
from ....core.step import Step
from ....core.matrix import Matrix
from ....pipeline.artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
    WorkPieceViewArtifact,
)
from ....core.color import ColorSet, OPS_COLOR_SPEC, ColorSpecDict
from ...canvas import CanvasElement
from ...shared.gtk_color import GtkColorResolver
from ..ops_cache_registry import registry
from .tab_handle import TabHandleElement

if TYPE_CHECKING:
    from ..surface import WorkSurface
    from ....pipeline.view import ViewManager

logger = logging.getLogger(__name__)

# Cairo has a hard limit on surface dimensions.
CAIRO_MAX_DIMENSION = 8192
OPS_MARGIN_PX = 5
REC_MARGIN_MM = 0.1  # A small "safe area" margin in mm for recordings
CONTOUR_HIT_THRESHOLD_PX = 8.0


def _segment_bbox(data: np.ndarray, idx: int) -> Optional[Tuple]:
    """Returns (min_x, min_y, max_x, max_y) for one segment row."""
    row = data[idx]
    cmd = row[COL_TYPE]
    if cmd == CMD_TYPE_MOVE:
        return None
    ex, ey = row[COL_X], row[COL_Y]
    if idx > 0:
        sx, sy = data[idx - 1, COL_X], data[idx - 1, COL_Y]
    else:
        sx, sy = 0.0, 0.0
    if cmd == CMD_TYPE_LINE:
        return (
            min(sx, ex),
            min(sy, ey),
            max(sx, ex),
            max(sy, ey),
        )
    if cmd == CMD_TYPE_BEZIER:
        c1x, c1y = row[COL_C1X], row[COL_C1Y]
        c2x, c2y = row[COL_C2X], row[COL_C2Y]
        pts_x = [sx, ex, c1x, c2x]
        pts_y = [sy, ey, c1y, c2y]
        return (min(pts_x), min(pts_y), max(pts_x), max(pts_y))
    if cmd == CMD_TYPE_ARC:
        import math

        ci, cj = row[COL_I], row[COL_J]
        r = math.hypot(ci, cj)
        cx, cy = sx + ci, sy + cj
        return (
            cx - r,
            cy - r,
            cx + r,
            cy + r,
        )
    return (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))


def _draw_segment(ctx: cairo.Context, data: np.ndarray, idx: int):
    """Draws a single segment (LINE/ARC/BEZIER) to a cairo context."""
    row = data[idx]
    cmd = row[COL_TYPE]
    if cmd == CMD_TYPE_MOVE:
        return
    ex, ey = row[COL_X], row[COL_Y]
    if idx > 0:
        sx, sy = data[idx - 1, COL_X], data[idx - 1, COL_Y]
    else:
        sx, sy = 0.0, 0.0
    if cmd == CMD_TYPE_LINE:
        ctx.move_to(sx, sy)
        ctx.line_to(ex, ey)
    elif cmd == CMD_TYPE_BEZIER:
        ctx.move_to(sx, sy)
        ctx.curve_to(
            row[COL_C1X],
            row[COL_C1Y],
            row[COL_C2X],
            row[COL_C2Y],
            ex,
            ey,
        )
    elif cmd == CMD_TYPE_ARC:
        import math

        ci, cj = row[COL_I], row[COL_J]
        r = math.hypot(ci, cj)
        cx, cy = sx + ci, sy + cj
        start_angle = math.atan2(-cj, -ci)
        end_angle = math.atan2(ey - cy, ex - cx)
        cw = bool(row[COL_CW])
        ctx.move_to(sx, sy)
        if cw:
            ctx.arc_negative(cx, cy, r, start_angle, end_angle)
        else:
            ctx.arc(cx, cy, r, start_angle, end_angle)


class VectorEditState:
    """Tracks the state of an in-progress vector segment edit session."""

    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.selected_segments: Set[int] = set()
        self.hovered_segment: Optional[int] = None
        self.frame_start: Optional[Tuple[float, float]] = None
        self.frame_end: Optional[Tuple[float, float]] = None
        self.frame_drag_start_world: Optional[Tuple[float, float]] = None


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
        self._ops_metadata_cache: Dict[str, Tuple] = {}

        # Composited ops surface: a single surface that blends all
        # visible step surfaces, rebuilt incrementally.
        self._composited_surface: Optional[cairo.ImageSurface] = None
        self._composited_data: Optional[np.ndarray] = None
        self._composited_dirty: bool = True
        self._composited_bbox_mm: Optional[Tuple] = None
        self._composited_wp_size_mm: Optional[Tuple] = None
        self._composited_bytes: int = 0

        self._tab_handles: List[TabHandleElement] = []
        # Default to False; the correct state will be pulled from the surface.
        self._tabs_visible_override: bool = False

        self._color_spec: ColorSpecDict = OPS_COLOR_SPEC
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
            is_editable=True,
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

        self._edit_state: Optional[VectorEditState] = None

        self.data.updated.connect(self._on_model_content_changed)
        self.data.transform_changed.connect(self._on_transform_changed)
        registry.register(self)

        self.view_manager.source_artifact_ready.connect(
            self._on_source_artifact_ready
        )
        self.view_manager.view_artifact_updated.connect(
            self._on_view_artifact_updated
        )
        self.view_manager.view_artifact_created.connect(
            self._on_view_artifact_created
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

        self._update_editable_state()

    def _hydrate_from_cache(self) -> bool:
        """
        Restores visual state from the persistent model cache if available.
        Returns True if significant state was restored.
        """
        cache = self.data._view_cache
        if not cache:
            return False

        self._surface = cache.get("surface")
        if self._surface is not None:
            self.surface = self._surface
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
        self._artifact_cache.clear()
        self.clear_all_ops_caches()

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
            self._surface = new_surface
            self._update_model_view_cache()
            self.mark_dirty(ancestors=True)
        if self.canvas:
            self.canvas.queue_draw()
        self._update_future = None
        return False

    def _update_ops_cache_from_handle(self, step_uid: str):
        """
        Loads the view artifact from shared memory and caches it.

        Reads the current bitmap from the live view buffer in shared
        memory, creates a Cairo surface wrapper around it, and stores
        the result in the per-step ops cache.

        Skips if the handle is absent, the artifact is not yet ready,
        or the buffer is entirely blank (no chunks written yet).

        Note: the numpy array backing the surface is a view into shared
        memory, not a heap copy.  It is therefore NOT registered in the
        ops-cache-registry (which tracks heap allocations only).
        """
        view_handle = self.view_manager.get_view_handle(
            self.data.uid, step_uid
        )
        if view_handle is None:
            return

        artifact = self.view_manager.store.get(view_handle)
        if not isinstance(artifact, WorkPieceViewArtifact):
            return

        try:
            new_data = artifact.bitmap_data
            if not np.any(new_data):
                self._remove_ops_surface(step_uid)
                self._invalidate_composited()
                return
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
            self._store_ops_surface(
                step_uid,
                new_surface,
                new_data,
                (artifact.bbox_mm, artifact.workpiece_size_mm),
            )
        except Exception as e:
            logger.warning(
                f"Failed to update ops cache for step {step_uid}: {e}"
            )

    def _store_ops_surface(
        self,
        step_uid: str,
        surface: cairo.ImageSurface,
        data: np.ndarray,
        metadata: Tuple,
    ):
        """
        Stores a step's Cairo surface, backing data, and metadata.

        *data* is a view into shared memory managed by the artifact
        store, so its byte size is not tracked in the OpsCacheRegistry.
        """
        self._ops_surface_cache[step_uid] = surface
        self._ops_surface_data_cache[step_uid] = data
        self._ops_metadata_cache[step_uid] = metadata

    def _remove_ops_surface(self, step_uid: str):
        """Removes a step's cached surface, data, and metadata."""
        self._ops_surface_cache.pop(step_uid, None)
        self._ops_surface_data_cache.pop(step_uid, None)
        self._ops_metadata_cache.pop(step_uid, None)

    def clear_all_ops_caches(self):
        """
        Removes every step cache entry and disposes the composite.

        Called by the registry's LRU eviction and during full invalidation.
        """
        for step_uid in list(self._ops_surface_cache.keys()):
            self._remove_ops_surface(step_uid)
        self._invalidate_composited()

    def _invalidate_composited(self):
        """Marks the composited surface as needing a full rebuild."""
        self._composited_dirty = True
        self._dispose_composited()

    def _dispose_composited(self):
        if self._composited_bytes:
            registry.remove(
                self.data.uid, "__composite__", self._composited_bytes
            )
        self._composited_surface = None
        self._composited_data = None
        self._composited_bbox_mm = None
        self._composited_wp_size_mm = None
        self._composited_bytes = 0

    def _rebuild_composited_surface(self):
        """
        Builds a single composited surface from all visible step caches.

        For each workflow step that is visible and has data, loads it
        from the view handle on demand (if not already cached), computes
        the union bounding box at the highest step PPM, allocates a
        single ARGB32 buffer, and blits each step into position.

        The composite buffer is reused across rebuilds when dimensions
        match, avoiding repeated large allocations.  Only the composite
        itself (a heap allocation) is tracked in the OpsCacheRegistry.

        Called from ``draw()`` when ``_composited_dirty`` is True.
        """
        if not self.data.layer or not self.data.layer.workflow:
            self._dispose_composited()
            self._composited_dirty = False
            return

        edited = self.data._edited_boundaries
        if edited is not None and edited.is_empty():
            self._dispose_composited()
            self._composited_dirty = False
            return

        world_w, world_h = self.data.size
        if world_w < 1e-9 or world_h < 1e-9:
            self._dispose_composited()
            self._composited_dirty = False
            return

        visible_steps = []
        for step in self.data.layer.workflow.steps:
            if not self._ops_visibility.get(step.uid, True):
                continue
            if step.uid not in self._ops_surface_cache:
                self._update_ops_cache_from_handle(step.uid)
            meta = self._ops_metadata_cache.get(step.uid)
            if meta is None:
                continue
            surf = self._ops_surface_cache.get(step.uid)
            if surf is None:
                continue
            visible_steps.append((step.uid, surf, meta))

        if not visible_steps:
            self._dispose_composited()
            self._composited_dirty = False
            return

        # Compute union of all scaled bboxes in workpiece-local mm space.
        union_x = float("inf")
        union_y = float("inf")
        union_r = float("-inf")
        union_t = float("-inf")
        ppm = 0.0

        for step_uid, surf, meta in visible_steps:
            bbox_mm, wp_size_mm = meta
            vx, vy, vw, vh = bbox_mm
            ref_w, ref_h = wp_size_mm
            sx = world_w / ref_w if ref_w > 1e-9 else 1.0
            sy = world_h / ref_h if ref_h > 1e-9 else 1.0
            vx *= sx
            vy *= sy
            vw *= sx
            vh *= sy
            if vw < 1e-9 or vh < 1e-9:
                continue
            w_px = surf.get_width()
            h_px = surf.get_height()
            step_ppm = (
                max(
                    (w_px - 2 * OPS_MARGIN_PX) / vw,
                    (h_px - 2 * OPS_MARGIN_PX) / vh,
                )
                if vw > 1e-9 and vh > 1e-9
                else 0
            )
            if step_ppm > ppm:
                ppm = step_ppm
            margin_w = OPS_MARGIN_PX / step_ppm if step_ppm > 0 else 0
            margin_h = margin_w
            union_x = min(union_x, vx - margin_w)
            union_y = min(union_y, vy - margin_h)
            union_r = max(union_r, vx + vw + margin_w)
            union_t = max(union_t, vy + vh + margin_h)

        if ppm <= 0:
            self._dispose_composited()
            self._composited_dirty = False
            return

        composite_w_mm = union_r - union_x
        composite_h_mm = union_t - union_y
        comp_w_px = min(int(round(composite_w_mm * ppm)), CAIRO_MAX_DIMENSION)
        comp_h_px = min(int(round(composite_h_mm * ppm)), CAIRO_MAX_DIMENSION)

        if comp_w_px <= 0 or comp_h_px <= 0:
            self._dispose_composited()
            self._composited_dirty = False
            return

        if (
            self._composited_data is not None
            and self._composited_data.shape == (comp_h_px, comp_w_px, 4)
        ):
            comp_data = self._composited_data
            comp_data[:] = 0
        else:
            comp_data = np.zeros((comp_h_px, comp_w_px, 4), dtype=np.uint8)
        stride = cairo.ImageSurface.format_stride_for_width(
            cairo.FORMAT_ARGB32, comp_w_px
        )
        comp_surf = cairo.ImageSurface.create_for_data(
            comp_data, cairo.FORMAT_ARGB32, comp_w_px, comp_h_px, stride
        )

        # Blit each step surface into the composite.
        comp_ctx = cairo.Context(comp_surf)
        for step_uid, surf, meta in visible_steps:
            bbox_mm, wp_size_mm = meta
            vx, vy, vw, vh = bbox_mm
            ref_w, ref_h = wp_size_mm
            sx = world_w / ref_w if ref_w > 1e-9 else 1.0
            sy = world_h / ref_h if ref_h > 1e-9 else 1.0
            vx *= sx
            vy *= sy
            vw *= sx
            vh *= sy
            if vw < 1e-9 or vh < 1e-9:
                continue
            step_ppm_x = (
                (surf.get_width() - 2 * OPS_MARGIN_PX) / vw if vw > 1e-9 else 0
            )
            step_ppm_y = (
                (surf.get_height() - 2 * OPS_MARGIN_PX) / vh
                if vh > 1e-9
                else 0
            )
            if step_ppm_x <= 0 or step_ppm_y <= 0:
                continue
            dest_x = (vx - OPS_MARGIN_PX / step_ppm_x - union_x) * ppm
            dest_y = (vy - OPS_MARGIN_PX / step_ppm_y - union_y) * ppm
            comp_ctx.save()
            comp_ctx.translate(dest_x, dest_y)
            scale_factor = ppm / step_ppm_x
            comp_ctx.scale(scale_factor, scale_factor)
            comp_ctx.set_source_surface(surf, 0, 0)
            comp_ctx.paint()
            comp_ctx.restore()

        self._dispose_composited()
        self._composited_surface = comp_surf
        self._composited_data = comp_data
        self._composited_bbox_mm = (
            union_x,
            union_y,
            composite_w_mm,
            composite_h_mm,
        )
        self._composited_wp_size_mm = (world_w, world_h)
        self._composited_bytes = comp_data.nbytes
        registry.add(self.data.uid, "__composite__", self._composited_bytes)
        self._composited_dirty = False

    def _composite_live_step(self, step_uid: str):
        """Incrementally blits a live (progressive) step surface into the
        existing composite without requiring a full rebuild.

        Falls back to marking dirty if the composite doesn't exist yet
        or if the step's bbox falls outside the current composite bounds.
        """
        if self._composited_surface is None or self._composited_dirty:
            self._composited_dirty = True
            return

        meta = self._ops_metadata_cache.get(step_uid)
        if meta is None:
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
            new_data = artifact.bitmap_data
            height, width, _ = new_data.shape
            stride = cairo.ImageSurface.format_stride_for_width(
                cairo.FORMAT_ARGB32, width
            )
            step_surf = cairo.ImageSurface.create_for_data(
                new_data, cairo.FORMAT_ARGB32, width, height, stride
            )
            self._store_ops_surface(
                step_uid,
                step_surf,
                new_data,
                (artifact.bbox_mm, artifact.workpiece_size_mm),
            )
        except Exception:
            self._composited_dirty = True
            return

        # Now blit into the composite.
        bbox_mm = artifact.bbox_mm
        wp_size_mm = artifact.workpiece_size_mm
        world_w, world_h = self.data.size
        ref_w, ref_h = wp_size_mm
        sx = world_w / ref_w if ref_w > 1e-9 else 1.0
        sy = world_h / ref_h if ref_h > 1e-9 else 1.0
        vx, vy, vw, vh = bbox_mm
        vx *= sx
        vy *= sy
        vw *= sx
        vh *= sy

        if vw < 1e-9 or vh < 1e-9:
            return

        step_ppm_x = (width - 2 * OPS_MARGIN_PX) / vw if vw > 1e-9 else 0
        step_ppm_y = (height - 2 * OPS_MARGIN_PX) / vh if vh > 1e-9 else 0
        if step_ppm_x <= 0 or step_ppm_y <= 0:
            return

        comp_bbox = self._composited_bbox_mm
        if comp_bbox is None:
            self._composited_dirty = True
            return

        # Reconstruct composite ppm from composite dimensions.
        comp_ppm = (
            (self._composited_surface.get_width()) / comp_bbox[2]
            if comp_bbox[2] > 1e-9
            else 0
        )
        if comp_ppm <= 0:
            self._composited_dirty = True
            return

        step_origin_x = vx - OPS_MARGIN_PX / step_ppm_x
        step_origin_y = vy - OPS_MARGIN_PX / step_ppm_y

        dest_x = (step_origin_x - comp_bbox[0]) * comp_ppm
        dest_y = (step_origin_y - comp_bbox[1]) * comp_ppm
        scale_factor = comp_ppm / step_ppm_x

        comp_ctx = cairo.Context(self._composited_surface)
        comp_ctx.save()
        comp_ctx.translate(dest_x, dest_y)
        comp_ctx.scale(scale_factor, scale_factor)
        comp_ctx.set_source_surface(step_surf, 0, 0)
        comp_ctx.paint()
        comp_ctx.restore()

    def _on_view_artifact_created(
        self,
        sender,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
        **kwargs,
    ):
        """
        Handles the creation of a new view artifact.

        Invalidates the ops surface cache for this step so that
        ``_rebuild_composited_surface`` will reload from the new
        handle on the next draw.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            return
        self._remove_ops_surface(step_uid)
        self._composited_dirty = True
        self.canvas.queue_draw()

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
        Handles progressive chunk updates from the background worker.

        Marks the composite dirty and schedules a redraw.  The actual
        data loading happens lazily in ``_rebuild_composited_surface``
        during the next ``draw()`` call, so this method is O(1).

        For invisible steps, the cache is evicted immediately so the
        composite is rebuilt without them.
        """
        if workpiece_uid != self.data.uid or not self.canvas:
            return
        if not self._ops_visibility.get(step_uid, True):
            self._remove_ops_surface(step_uid)
            self._composited_dirty = True
            self.canvas.queue_draw()
            return
        self._composited_dirty = True
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
        edited = self.data._edited_boundaries
        if edited is not None and edited.is_empty():
            self._remove_ops_surface(step_uid)
            self._invalidate_composited()
            return
        self._update_ops_cache_from_handle(step_uid)
        self._composited_dirty = True
        if self.canvas:
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

    def _update_editable_state(self):
        if self.data.geometry_provider_uid:
            self.is_editable = False
            return
        boundaries = self.data.boundaries
        self.is_editable = boundaries is not None and not boundaries.is_empty()

    def on_edit_mode_enter(self):
        boundaries = self.data.boundaries
        if boundaries is None or boundaries.is_empty():
            return
        self._edit_state = VectorEditState(boundaries)
        self.invalidate_and_rerender()

    def on_edit_mode_leave(self):
        self._edit_state = None
        if self.canvas:
            self.canvas.queue_draw()

    def draw_edit_overlay(self, ctx: cairo.Context):
        if not self._edit_state or not self.canvas:
            return
        data = self._edit_state.geometry.data
        if data is None:
            return

        screen_transform = (
            self.canvas.view_transform @ self.get_world_transform()
        )
        sx, sy = screen_transform.get_scale()
        screen_scale = max(abs(sx), abs(sy))
        if screen_scale < 1e-9:
            return

        ctx.save()
        cairo_matrix = cairo.Matrix(*screen_transform.for_cairo())
        ctx.transform(cairo_matrix)

        default_color = (0.5, 0.5, 0.5, 0.6)
        default_width = 1.5 / screen_scale
        selected_color = (1.0, 0.3, 0.3, 0.9)
        selected_width = 2.5 / screen_scale
        hovered_color = (0.4, 0.6, 1.0, 0.8)
        hovered_width = 2.0 / screen_scale

        prev_color = None
        prev_width = None

        for idx in range(len(data)):
            cmd = data[idx, COL_TYPE]
            if cmd == CMD_TYPE_MOVE:
                continue

            is_sel = idx in self._edit_state.selected_segments
            is_hov = idx == self._edit_state.hovered_segment

            if is_sel:
                color, width = selected_color, selected_width
            elif is_hov:
                color, width = hovered_color, hovered_width
            else:
                color, width = default_color, default_width

            if color != prev_color or width != prev_width:
                ctx.set_source_rgba(*color)
                ctx.set_line_width(width)
                prev_color = color
                prev_width = width

            _draw_segment(ctx, data, idx)
            ctx.stroke()

        ctx.restore()

        if self._edit_state.frame_start and self._edit_state.frame_end:
            self._draw_selection_frame(ctx, screen_transform)

    def _draw_selection_frame(self, ctx, screen_transform):
        if not self._edit_state:
            return
        fs = self._edit_state.frame_start
        fe = self._edit_state.frame_end
        if fs is None or fe is None:
            return
        s1 = screen_transform.transform_point(fs)
        s2 = screen_transform.transform_point(fe)
        ctx.save()
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.3)
        fx, fy = min(s1[0], s2[0]), min(s1[1], s2[1])
        fw, fh = abs(s2[0] - s1[0]), abs(s2[1] - s1[1])
        ctx.rectangle(fx, fy, fw, fh)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.2, 0.5, 0.8)
        ctx.set_line_width(1.0)
        ctx.set_dash((4, 4))
        ctx.stroke()
        ctx.restore()

    def _hit_test_segment(
        self, world_x: float, world_y: float
    ) -> Optional[int]:
        if not self._edit_state or not self.canvas:
            return None

        work_surface = cast("WorkSurface", self.canvas)
        ppm_x, _ = work_surface.get_view_scale()
        if ppm_x < 1e-9:
            return None

        try:
            inv_world = self.get_world_transform().invert()
            local_x, local_y = inv_world.transform_point((world_x, world_y))
        except Exception:
            return None

        world_w, world_h = self.data.size
        if world_w < 1e-9 or world_h < 1e-9:
            return None

        threshold_01 = CONTOUR_HIT_THRESHOLD_PX / (ppm_x * world_w)
        threshold_sq = threshold_01 * threshold_01

        result = self._edit_state.geometry.find_closest_point(local_x, local_y)
        if result is None:
            return None

        seg_idx, _, closest_pt = result
        dx = local_x - closest_pt[0]
        dy = local_y - closest_pt[1]
        if dx * dx + dy * dy <= threshold_sq:
            return seg_idx
        return None

    def _segments_in_frame(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Set[int]:
        if not self._edit_state:
            return set()
        data = self._edit_state.geometry.data
        if data is None:
            return set()
        fmin_x, fmin_y = min(x1, x2), min(y1, y2)
        fmax_x, fmax_y = max(x1, x2), max(y1, y2)
        result = set()
        for idx in range(len(data)):
            bbox = _segment_bbox(data, idx)
            if bbox is None:
                continue
            smin_x, smin_y, smax_x, smax_y = bbox
            if (
                smax_x >= fmin_x
                and smin_x <= fmax_x
                and smax_y >= fmin_y
                and smin_y <= fmax_y
            ):
                result.add(idx)
        return result

    def _world_to_local(
        self, wx: float, wy: float
    ) -> Optional[Tuple[float, float]]:
        try:
            inv = self.get_world_transform().invert()
            return inv.transform_point((wx, wy))
        except Exception:
            return None

    def handle_edit_press(
        self, world_x: float, world_y: float, n_press: int = 1
    ) -> bool:
        if not self._edit_state:
            return False

        if n_press >= 2:
            return False

        hit = self._hit_test_segment(world_x, world_y)

        if hit is None:
            local = self._world_to_local(world_x, world_y)
            if local is not None:
                self._edit_state.frame_start = local
                self._edit_state.frame_end = local
                self._edit_state.frame_drag_start_world = (
                    world_x,
                    world_y,
                )
            return True

        shift_pressed = self.canvas._shift_pressed if self.canvas else False

        if shift_pressed:
            if hit in self._edit_state.selected_segments:
                self._edit_state.selected_segments.discard(hit)
            else:
                self._edit_state.selected_segments.add(hit)
        else:
            self._edit_state.selected_segments = {hit}

        if self.canvas:
            self.canvas.queue_draw()
        return True

    def handle_edit_drag(self, world_dx: float, world_dy: float):
        if not self._edit_state or not self.canvas:
            return
        if self._edit_state.frame_drag_start_world is None:
            return

        swx, swy = self._edit_state.frame_drag_start_world
        cur_wx = swx + world_dx
        cur_wy = swy + world_dy

        cur_local = self._world_to_local(cur_wx, cur_wy)
        if cur_local is None:
            return

        start_local = self._world_to_local(swx, swy)
        if start_local is None:
            return

        self._edit_state.frame_start = start_local
        self._edit_state.frame_end = cur_local
        self.canvas.queue_draw()

    def handle_edit_release(self, world_x: float, world_y: float):
        if not self._edit_state:
            return
        if self._edit_state.frame_start and self._edit_state.frame_end:
            x1, y1 = self._edit_state.frame_start
            x2, y2 = self._edit_state.frame_end
            if abs(x2 - x1) > 1e-9 or abs(y2 - y1) > 1e-9:
                in_frame = self._segments_in_frame(x1, y1, x2, y2)
                shift = self.canvas._shift_pressed if self.canvas else False
                if shift:
                    self._edit_state.selected_segments ^= in_frame
                else:
                    self._edit_state.selected_segments = in_frame
            else:
                if not (self.canvas._shift_pressed if self.canvas else False):
                    self._edit_state.selected_segments.clear()
        self._edit_state.frame_start = None
        self._edit_state.frame_end = None
        self._edit_state.frame_drag_start_world = None
        if self.canvas:
            self.canvas.queue_draw()

    def handle_edit_motion(self, world_x: float, world_y: float) -> bool:
        if not self._edit_state:
            return False

        hit = self._hit_test_segment(world_x, world_y)

        if hit != self._edit_state.hovered_segment:
            self._edit_state.hovered_segment = hit
            if self.canvas:
                self.canvas.queue_draw()
        return True

    def handle_edit_key(self, keyval: int) -> bool:
        if not self._edit_state:
            return False

        if keyval in (Gdk.KEY_Delete, Gdk.KEY_BackSpace):
            if not self._edit_state.selected_segments:
                return True
            work_surface = cast("WorkSurface", self.canvas)
            work_surface.editor.edit.delete_segments(
                self.data, self._edit_state.selected_segments
            )
            boundaries = self.data.boundaries
            if boundaries is None or boundaries.is_empty():
                self.clear_all_ops_caches()
                if self.canvas:
                    self.canvas.leave_edit_mode()
                return True
            self._edit_state = VectorEditState(boundaries)
            if self.canvas:
                self.canvas.queue_draw()
            return True

        return False

    def handle_edit_select_all(self) -> bool:
        if not self._edit_state:
            return False
        data = self._edit_state.geometry.data
        if data is not None:
            self._edit_state.selected_segments = {
                i
                for i in range(len(data))
                if data[i, COL_TYPE] != CMD_TYPE_MOVE
            }
        else:
            self._edit_state.selected_segments = set()
        if self.canvas:
            self.canvas.queue_draw()
        return True

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
        self.view_manager.view_artifact_created.disconnect(
            self._on_view_artifact_created
        )
        self.view_manager.generation_finished.disconnect(
            self._on_view_generation_finished
        )
        self.clear_all_ops_caches()
        registry.unregister(self.data.uid)
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
            self._composited_dirty = True
            if self.canvas:
                self.canvas.queue_draw()

    def clear_ops_surface(self, step_uid: str):
        """
        Removes the cached ops surface for a step and schedules a
        composite rebuild.

        Called by ``LayerElement.sync_with_model`` when a step is
        deleted from the workflow.  Must mark the composite dirty so
        the next draw() rebuilds it without the removed step.
        """
        logger.debug(f"Clearing ops surface for step '{step_uid}'")
        self._remove_ops_surface(step_uid)
        self._composited_dirty = True
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
            resolver = GtkColorResolver(self.canvas)
            self._color_set = resolver.resolve(self._color_spec)
            self._last_style_context_hash = current_hash

    def _on_model_content_changed(self, workpiece: WorkPiece):
        """Handler for when the workpiece model's content changes."""
        logger.debug(
            f"Model content changed for '{workpiece.name}', triggering update."
        )
        self._create_or_update_tab_handles()
        self._update_editable_state()
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
        # Check if the workpiece depends on a hidden geometry provider
        provider_hidden = False
        if self.data.geometry_provider_uid and self.data.doc:
            provider = self.data.doc.get_asset_by_uid(
                self.data.geometry_provider_uid
            )
            if provider and provider.hidden:
                provider_hidden = True

        if self._base_image_visible and not provider_hidden:
            # This handles the Y-flip for the base image and restores the
            # context, leaving it Y-UP for the next drawing operation.
            super().draw(ctx)

        # Draw Ops (hide during interaction)
        worksurface = cast("WorkSurface", self.canvas) if self.canvas else None
        if not worksurface:
            return

        if worksurface.ops_suppressed:
            return

        # Draw view artifacts (complete, pre-rendered bitmaps)
        self._resolve_colors_if_needed()
        world_w, world_h = self.data.size

        if world_w < 1e-9 or world_h < 1e-9:
            return

        if self.data.layer and self.data.layer.workflow:
            registry.touch(self.data.uid)

            if self._composited_dirty:
                self._rebuild_composited_surface()

            comp_surf = self._composited_surface
            comp_bbox = self._composited_bbox_mm
            if comp_surf is None or comp_bbox is None:
                return

            try:
                view_x, view_y, view_w, view_h = comp_bbox

                if view_w < 1e-9 or view_h < 1e-9:
                    return

                surface_w_px = comp_surf.get_width()
                surface_h_px = comp_surf.get_height()

                ppm_x = surface_w_px / view_w if view_w > 1e-9 else 0
                ppm_y = surface_h_px / view_h if view_h > 1e-9 else 0

                if ppm_x <= 0 or ppm_y <= 0:
                    return

                ctx.save()

                ctx.translate(view_x / world_w, view_y / world_h)
                ctx.scale(
                    surface_w_px / (world_w * ppm_x),
                    surface_h_px / (world_h * ppm_y),
                )
                ctx.translate(0, 1)
                ctx.scale(1, -1)
                ctx.scale(1.0 / surface_w_px, 1.0 / surface_h_px)

                ctx.set_source_surface(comp_surf, 0, 0)
                ctx.get_source().set_filter(cairo.FILTER_NEAREST)
                ctx.paint()

                ctx.restore()
            except Exception as e:
                logger.warning(
                    f"Failed to draw composited ops for "
                    f"'{self.data.name}': {e}"
                )

    def push_transform_to_model(self):
        """Updates the data model's matrix with the view's transform."""
        if self.data.matrix != self.transform:
            logger.debug(
                f"Pushing view transform to model for '{self.data.name}'."
            )
            self.data.matrix = self.transform.copy()

    def on_travel_visibility_changed(self):
        """Handles changes in travel move visibility."""
        logger.debug("Travel visibility changed. Invalidating composite.")
        self._composited_dirty = True
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
