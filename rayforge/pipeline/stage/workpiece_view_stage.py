from __future__ import annotations
import cairo
import logging
import numpy as np
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional, cast
from blinker import Signal
from ...shared.util.colors import ColorSet
from ..artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
)
from ..artifact.key import ArtifactKey
from ..artifact.lifecycle import ArtifactLifecycle
from ..artifact.manager import make_composite_key
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
    WorkPieceViewArtifactHandle,
)
from .base import PipelineStage
from .workpiece_view_runner import (
    make_workpiece_view_artifact_in_subprocess,
    render_chunk_to_view,
)
from .workpiece_view_compute import (
    calculate_render_dimensions,
)

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.layer import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.manager import ArtifactManager


logger = logging.getLogger(__name__)

# Throttle interval for view updates (in seconds).
# 30fps = ~33ms, 60fps = ~16ms. We use 30fps for a balance between
# responsiveness and performance.
THROTTLE_INTERVAL = 0.033


class WorkPieceViewPipelineStage(PipelineStage):
    """
    An on-demand stage that generates pre-rendered bitmap artifacts
    (`WorkPieceViewArtifact`) for display in the UI.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: "Machine",
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._current_view_context: Optional[RenderContext] = None

        # Throttling for progressive chunk updates.
        # Tracks pending updates and last update time for each view.
        self._pending_updates: Dict[ArtifactKey, bool] = {}
        self._last_update_time: Dict[ArtifactKey, float] = {}
        self._throttle_timers: Dict[ArtifactKey, threading.Timer] = {}

        self.view_artifact_ready = Signal()
        self.view_artifact_created = Signal()
        self.view_artifact_updated = Signal()
        self.generation_finished = Signal()

    def reconcile(self, doc: "Doc", generation_id: int):
        """
        Synchronizes the cache with the document, cleaning up obsolete
        entries and syncing the ledger.

        View rendering is on-demand and triggered by update_view_context,
        so reconcile mainly handles cleanup and ledger synchronization.
        """
        logger.debug("WorkPieceViewPipelineStage reconciling...")

        all_current_workpieces = {
            workpiece.uid
            for layer in doc.layers
            if layer.workflow is not None
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }
        cached_keys = self._artifact_manager.get_all_workpiece_keys()
        cached_workpiece_uids = {key.id for key in cached_keys}

        for w_uid in cached_workpiece_uids - all_current_workpieces:
            logger.debug(f"Cleaning up obsolete view workpiece: {w_uid}")
            task_key = ArtifactKey.for_view(w_uid)
            self._task_manager.cancel_task(task_key)

    def update_view_context(
        self,
        context: RenderContext,
        view_gen_id: int,
        data_gen_id: int,
    ) -> None:
        """
        Updates the view context and triggers re-rendering for all cached
        workpiece views if the context has changed.

        This method iterates over all cached workpiece view keys in the
        ArtifactManager and checks if views are stale with the new context.
        If a view is stale, it triggers a new render request for that view.

        Args:
            context: The new render context to apply.
            view_gen_id: The view generation ID to use for rendering.
            data_gen_id: The data generation ID to use for retrieving
                workpiece artifacts.
        """
        logger.debug(
            f"update_view_context called with context "
            f"ppm={context.pixels_per_mm}, "
            f"show_travel_moves={context.show_travel_moves}, "
            f"view_gen_id={view_gen_id}"
        )

        self._current_view_context = context

        keys = self._artifact_manager.get_all_workpiece_keys()
        logger.debug(f"update_view_context: Found {len(keys)} workpiece keys")

        for key in keys:
            logger.debug(
                f"View context changed. Triggering re-render for {key} "
                f"with view_id={view_gen_id}"
            )
            self.request_view_render(key, context, view_gen_id, data_gen_id)

    def set_render_context(
        self,
        workpiece_uid: str,
        context: RenderContext,
    ) -> None:
        """
        Sets the render context for a specific workpiece.

        This public method allows the pipeline to pre-populate the render
        context in the ArtifactManager before chunks arrive, enabling
        progressive rendering to work correctly.

        Args:
            workpiece_uid: The unique identifier of the workpiece.
            context: The render context to use for this view.
        """
        key = ArtifactKey.for_view(workpiece_uid)
        logger.debug(
            f"Set render context for {key}: ppm={context.pixels_per_mm}"
        )

    def shutdown(self):
        """Cancels any active rendering tasks and releases held handles."""
        logger.debug("WorkPieceViewPipelineStage shutting down.")
        from ..artifact.manager import extract_base_key

        view_keys = {
            extract_base_key(ledger_key)
            for ledger_key in self._artifact_manager._ledger.keys()
            if extract_base_key(ledger_key).group == "view"
        }
        for key in view_keys:
            self._task_manager.cancel_task(key)

        # Cancel any pending throttle timers
        for timer in self._throttle_timers.values():
            if timer:
                timer.cancel()
        self._throttle_timers.clear()
        self._pending_updates.clear()
        self._last_update_time.clear()

    def _allocate_live_buffer(
        self,
        key: ArtifactKey,
        workpiece: "WorkPiece",
        step_uid: str,
        view_id: int,
        context: RenderContext,
    ) -> None:
        """
        Allocates a new blank view artifact based on the workpiece size and
        current render context, stores it in shared memory, and registers it
        as the live buffer for this generation.
        """
        workpiece_uid = key.id

        w_mm, h_mm = workpiece.size
        bbox = (0.0, 0.0, w_mm, h_mm)

        dims = calculate_render_dimensions(bbox, context)
        if dims is None:
            logger.warning(f"[{key}] Invalid dimensions for live buffer.")
            return

        width_px, height_px, _, _ = dims
        logger.debug(
            f"[{key}] Allocating live buffer: {width_px}x{height_px} px"
        )
        logger.info(
            f"[DIAGNOSTIC] _allocate_live_buffer called for key {key}, "
            f"view_id={view_id}"
        )

        try:
            # Create a blank bitmap
            bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)
            view_artifact = WorkPieceViewArtifact(
                bitmap_data=bitmap, bbox_mm=bbox
            )

            # Store in shared memory
            handle = self._artifact_manager._store.put(
                view_artifact, creator_tag="view_live_buffer"
            )
            # Type narrow for mypy
            view_handle = cast(WorkPieceViewArtifactHandle, handle)

            # Commit to Manager immediately
            self._artifact_manager.put_workpiece_view_handle(
                ArtifactKey.for_view(workpiece_uid),
                view_handle,
                view_id,
            )

            # Store render context in ledger metadata
            ledger_key = make_composite_key(
                ArtifactKey.for_view(workpiece_uid), view_id
            )
            entry = self._artifact_manager._get_ledger_entry(ledger_key)
            if entry is not None:
                entry.metadata["render_context"] = context

            logger.info(
                f"[DIAGNOSTIC] Allocated live buffer for key {key}: "
                f"{view_handle.shm_name}"
            )

            # Signal creation
            self._send_view_artifact_created_signals(
                step_uid, workpiece_uid, view_handle
            )

        except Exception as e:
            logger.error(
                f"[{key}] Failed to allocate live buffer: {e}", exc_info=True
            )

    def on_generation_starting(
        self,
        sender,
        *,
        step: "Step",
        workpiece: "WorkPiece",
        generation_id: int,
        view_id: int,
    ):
        """
        Called when workpiece generation starts.
        Pre-allocates the view buffer to enable progressive rendering.

        Args:
            step: The step being generated.
            workpiece: The workpiece being generated.
            generation_id: The data generation ID.
            view_id: The view generation ID from the pipeline.
        """
        key = ArtifactKey.for_view(workpiece.uid)

        logger.debug(
            f"View stage: Source data gen {generation_id} starting for {key}. "
            f"Allocating live buffer for view_id={view_id}."
        )

        # Cancel any active view render task from previous generations
        task_key = ArtifactKey.for_view(workpiece.uid)
        task = self._task_manager.get_task(task_key)
        if task and not task.is_final():
            self._task_manager.cancel_task(task_key)

        # Allocate the live buffer immediately
        context = self._current_view_context
        if context:
            self._allocate_live_buffer(
                key, workpiece, step.uid, view_id, context
            )
        else:
            logger.warning(
                f"[{key}] Cannot allocate live buffer: No RenderContext."
            )

    def request_view_render(
        self,
        key: ArtifactKey,
        context: RenderContext,
        view_id: int,
        data_generation_id: int,
        step_uid: Optional[str] = None,
    ):
        """
        Requests an asynchronous render of a workpiece view.

        Args:
            key: The ArtifactKey for the workpiece view.
            context: The render context to use.
            view_id: The view generation ID for this render.
            data_generation_id: The data generation ID to use for
                retrieving workpiece artifacts.
            step_uid: The unique identifier of the step (optional).
        """
        artifact_key = key
        ledger_key = make_composite_key(artifact_key, view_id)

        if step_uid is None:
            logger.warning(
                f"Cannot render view for {key}: "
                "step_uid is required but not provided."
            )
            return

        source_handle = self._artifact_manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(key.id),
            data_generation_id,
        )
        if not source_handle:
            logger.warning(
                f"Cannot render view for {key}: "
                "source WorkPieceArtifact not found."
            )
            return

        if not self._artifact_manager.is_view_stale(
            artifact_key,
            context,
            source_handle,
            view_id,
        ):
            logger.debug(f"View for {key} is still valid. Skipping render.")
            return

        self._current_view_context = context

        # If a task is already running, cancel it. This ensures that new
        # requests (e.g., from zooming) always take precedence.
        task_key = artifact_key
        task = self._task_manager.get_task(task_key)
        if task and not task.is_final():
            logger.debug(
                f"[{key}] View render already in progress. Cancelling "
                f"to start new one."
            )
            self._task_manager.cancel_task(task_key)

        # Get or create ledger entry and mark as pending
        entry = self._artifact_manager._get_ledger_entry(ledger_key)
        if entry is None:
            self._artifact_manager.register_intent(artifact_key, view_id)
            entry = self._artifact_manager._get_ledger_entry(ledger_key)
            if entry is not None:
                entry.metadata["render_context"] = context
        elif entry.state != ArtifactLifecycle.PROCESSING:
            if entry.state == ArtifactLifecycle.STALE:
                entry.state = ArtifactLifecycle.QUEUED

        # Only mark as pending if the entry is in QUEUED state.
        # If it's already PROCESSING, we've already started a render task.
        current_entry = self._artifact_manager._get_ledger_entry(ledger_key)
        if current_entry is not None and current_entry.state == (
            ArtifactLifecycle.QUEUED
        ):
            self._artifact_manager.mark_processing(
                artifact_key, view_id, source_handle
            )

        def when_done_callback(task: "Task"):
            logger.debug(
                f"[{key}] when_done_callback called, "
                f"task_status={task.get_status()}, "
                f"task_id={task.id}"
            )
            self._on_render_complete(task, key, view_id)

        # Namespaced key for TaskManager
        task_key = artifact_key

        self._task_manager.run_process(
            make_workpiece_view_artifact_in_subprocess,
            self._artifact_manager._store,
            workpiece_artifact_handle_dict=source_handle.to_dict(),
            render_context_dict=context.to_dict(),
            creator_tag="workpiece_view",
            key=task_key,
            generation_id=view_id,
            step_uid=step_uid,
            when_done=when_done_callback,
            when_event=self._on_render_event_received,
        )

    def _on_render_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles progressive rendering events from the worker process."""
        step_uid = task.kwargs.get("step_uid")
        if step_uid is None:
            logger.warning("step_uid is None in _on_render_event_received")
            return
        if not isinstance(task.key, ArtifactKey):
            raise TypeError(f"Expected ArtifactKey, got {type(task.key)}")
        key = task.key
        workpiece_uid = key.id
        view_id = task.kwargs.get("generation_id", 0)

        if event_name == "view_artifact_created":
            self._handle_view_artifact_created(
                task, key, step_uid, data, view_id
            )
        elif event_name == "view_artifact_updated":
            self._handle_view_artifact_updated(
                key, step_uid, workpiece_uid, view_id
            )

    def _handle_view_artifact_created(
        self,
        task: "Task",
        key: ArtifactKey,
        step_uid: str,
        data: dict,
        view_id: int,
    ):
        """Handles the view_artifact_created event."""
        try:
            handle = self._adopt_view_handle(key, data)
            if handle is None:
                return

            self._artifact_manager.put_workpiece_view_handle(
                key, handle, view_id
            )
            # Store render context in ledger metadata
            context = self._current_view_context
            ledger_key = make_composite_key(key, view_id)
            entry = self._artifact_manager._get_ledger_entry(ledger_key)
            if entry is not None:
                entry.metadata["render_context"] = context
            self._send_view_artifact_created_signals(step_uid, key.id, handle)
        except Exception as e:
            logger.error(
                f"Failed to process view_artifact_created: {e}", exc_info=True
            )

    def _adopt_view_handle(
        self, key: ArtifactKey, data: dict
    ) -> WorkPieceViewArtifactHandle | None:
        """Adopts a view artifact from the worker process."""
        handle_dict = data["handle_dict"]
        handle = self._artifact_manager.adopt_artifact(key, handle_dict)
        if not isinstance(handle, WorkPieceViewArtifactHandle):
            raise TypeError("Expected WorkPieceViewArtifactHandle")

        logger.debug(
            f"Adopting new view artifact: {handle.shm_name} for key {key}"
        )
        return handle

    def _replace_current_view_handle(
        self,
        key: ArtifactKey,
        handle: WorkPieceViewArtifactHandle,
        view_id: int,
    ):
        """Replaces the current view handle with a new one."""
        old_handle = self._artifact_manager.get_workpiece_view_handle(
            key, view_id
        )
        logger.info(
            f"[DIAGNOSTIC] _replace_current_view_handle called for key {key}, "
            f"old_handle={old_handle.shm_name if old_handle else None}, "
            f"new_handle={handle.shm_name}"
        )
        if old_handle:
            logger.debug(
                f"Releasing old view artifact: {old_handle.shm_name} "
                f"for key {key}"
            )
            self._artifact_manager.release_handle(old_handle)

        self._artifact_manager.retain_handle(handle)
        logger.info(
            f"[DIAGNOSTIC] Replaced live buffer for key {key}: "
            f"{handle.shm_name}"
        )
        logger.debug(
            f"Stored new view artifact: {handle.shm_name} for key {key}"
        )

    def _send_view_artifact_created_signals(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceViewArtifactHandle,
    ):
        """Sends signals when a view artifact is created."""
        logger.info(
            f"[DIAGNOSTIC] Sending view_artifact_created for "
            f"key ({step_uid}, {workpiece_uid})"
        )
        logger.info(
            f"[DIAGNOSTIC] view_artifact_created has "
            f"{len(self.view_artifact_created.receivers)} receivers"
        )
        self.view_artifact_created.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )
        logger.info(
            f"[DIAGNOSTIC] Sending view_artifact_ready for "
            f"key ({step_uid}, {workpiece_uid})"
        )
        logger.info(
            f"[DIAGNOSTIC] view_artifact_ready has "
            f"{len(self.view_artifact_ready.receivers)} receivers"
        )
        self.view_artifact_ready.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _handle_view_artifact_updated(
        self, key: ArtifactKey, step_uid: str, workpiece_uid: str, view_id: int
    ):
        """Handles the view_artifact_updated event."""
        handle = self._artifact_manager.get_workpiece_view_handle(key, view_id)
        self.view_artifact_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _on_workpiece_chunk_available(
        self,
        sender,
        *,
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
        generation_id: int,
    ):
        """
        Receives chunk data from the WorkPiecePipelineStage.
        Implements incremental rendering by drawing the chunk onto
        the live view artifact bitmap using a lightweight background thread.

        Args:
            sender: The sender of the signal.
            key: The view key (workpiece_uid).
            chunk_handle: Handle to the chunk artifact.
            generation_id: The data generation ID.
        """
        base_key = key
        logger.debug(
            f"View stage received chunk for {key}, "
            f"generation_id={generation_id}, "
            f"chunk_handle={chunk_handle.shm_name}"
        )

        # Find the ledger entry for this view key
        entry = None
        for lk, e in self._artifact_manager._ledger.items():
            if isinstance(lk, tuple) and len(lk) == 2:
                lk_base, lk_gen_id = lk
                if lk_base == base_key:
                    entry = e
                    break
        if entry is None:
            self._artifact_manager.release_handle(chunk_handle)
            return

        # Get view handle and render context from ArtifactManager
        view_handle, render_context = self._get_render_components(
            key, base_key, chunk_handle
        )
        if view_handle is None or render_context is None:
            return

        # Retain chunk handle for the duration of the stitching task
        self._artifact_manager.retain_handle(chunk_handle)

        # Also retain the view handle to ensure the buffer remains valid
        # even if replaced/released by the manager during stitching.
        self._artifact_manager.retain_handle(view_handle)

        # Launch stitching in a thread to avoid blocking the main loop
        # during SHM writes.
        self._task_manager.run_thread(
            render_chunk_to_view,
            self._artifact_manager,
            chunk_handle.to_dict(),
            view_handle.to_dict(),
            render_context.to_dict(),
            when_done=lambda t: self._on_stitch_complete(
                t, key, chunk_handle, view_handle, entry.generation_id
            ),
        )

    def _on_stitch_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
        view_handle: BaseArtifactHandle,
        view_id: int,
    ):
        """Callback for when stitching completes."""
        logger.info(
            f"[DIAGNOSTIC] Chunk stitch for key {key} completed, "
            f"status={task.get_status()}, chunk_handle={chunk_handle.shm_name}"
        )
        try:
            if task.get_status() == "completed" and task.result():
                self._schedule_throttled_update(key, view_id)
            else:
                logger.warning(f"Stitching failed for {key}")
        finally:
            # Release handles
            logger.info(
                f"[DIAGNOSTIC] Releasing chunk handle for key {key}: "
                f"{chunk_handle.shm_name}"
            )
            self._artifact_manager.release_handle(chunk_handle)
            self._artifact_manager.release_handle(view_handle)

    def _get_chunk_artifact(
        self, key: ArtifactKey, chunk_handle: BaseArtifactHandle
    ) -> WorkPieceArtifact | None:
        """Retrieves and validates the chunk artifact."""
        try:
            artifact = self._artifact_manager.get_artifact(chunk_handle)
            if artifact is None:
                logger.warning(f"Could not retrieve chunk artifact for {key}")
                return None
            if not isinstance(artifact, WorkPieceArtifact):
                logger.warning(
                    f"Chunk artifact for {key} is not a WorkPieceArtifact"
                )
                return None
            return artifact
        except Exception as e:
            logger.error(f"Error retrieving chunk artifact for {key}: {e}")
            return None

    def _get_render_components(
        self,
        key: ArtifactKey,
        base_key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
    ) -> tuple[WorkPieceViewArtifactHandle | None, RenderContext | None]:
        """Gets the view handle and render context from ArtifactManager."""
        # Find the ledger entry for this view key
        entry = None
        for lk, e in self._artifact_manager._ledger.items():
            if isinstance(lk, tuple) and len(lk) == 2:
                lk_base, lk_gen_id = lk
                if lk_base == base_key:
                    entry = e
                    break
        if entry is None:
            logger.debug(f"No ledger entry for {key}. Ignoring chunk.")
            self._artifact_manager.release_handle(chunk_handle)
            return None, None
        if entry is None:
            logger.debug(f"No ledger entry for {key}. Ignoring chunk.")
            self._artifact_manager.release_handle(chunk_handle)
            return None, None

        view_id = entry.generation_id
        view_handle = self._artifact_manager.get_workpiece_view_handle(
            base_key, view_id
        )
        if view_handle is None:
            logger.debug(f"No view handle found for {key}. Ignoring chunk.")
            self._artifact_manager.release_handle(chunk_handle)
            return None, None

        if not isinstance(view_handle, WorkPieceViewArtifactHandle):
            logger.warning(
                f"Expected WorkPieceViewArtifactHandle, "
                f"got {type(view_handle)}"
            )
            self._artifact_manager.release_handle(chunk_handle)
            return None, None

        render_context = entry.metadata.get("render_context")
        if render_context is None:
            logger.debug(f"No render context in ledger entry for {key}.")
            self._artifact_manager.release_handle(chunk_handle)
            return None, None

        return view_handle, render_context

    def _should_render_chunk(
        self,
        chunk_artifact: WorkPieceArtifact,
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
    ) -> bool:
        """Checks if the chunk should be rendered based on its Ops."""
        if not chunk_artifact.ops or chunk_artifact.ops.is_empty():
            logger.debug(f"Chunk for {key} has no Ops, skipping")
            self._artifact_manager.release_handle(chunk_handle)
            return False
        return True

    def _render_chunk_to_view(
        self,
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
        view_handle: WorkPieceViewArtifactHandle,
        render_context: RenderContext,
    ):
        """Renders the chunk to the view artifact."""
        try:
            success = render_chunk_to_view(
                self._artifact_manager,
                chunk_handle.to_dict(),
                view_handle.to_dict(),
                render_context.to_dict(),
            )

            if success:
                logger.debug(
                    f"Successfully rendered chunk for {key} to view "
                    f"artifact {view_handle.shm_name}"
                )
            else:
                logger.warning(
                    f"Failed to render chunk for {key} to view artifact"
                )
        except Exception as e:
            logger.error(
                f"Error rendering chunk for {key}: {e}", exc_info=True
            )
        finally:
            self._artifact_manager.release_handle(chunk_handle)

    def _draw_vertices(
        self,
        ctx: cairo.Context,
        vertex_data,
        color_set: ColorSet,
        show_travel: bool,
        line_width_mm: float,
    ):
        """Draws all vertex data onto the provided cairo context."""
        ctx.save()
        ctx.set_line_width(line_width_mm)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        if show_travel:
            self._draw_travel_vertices(ctx, vertex_data, color_set)
            self._draw_zero_power_vertices(ctx, vertex_data, color_set)

        self._draw_powered_vertices(ctx, vertex_data, color_set)
        ctx.restore()

    def _draw_travel_vertices(
        self, ctx: cairo.Context, vertex_data, color_set: ColorSet
    ):
        """Draws travel vertices onto the cairo context."""
        if vertex_data.travel_vertices.size > 0:
            travel_v = vertex_data.travel_vertices.reshape(-1, 2, 3)
            ctx.set_source_rgba(*color_set.get_rgba("travel"))
            for start, end in travel_v:
                ctx.move_to(start[0], start[1])
                ctx.line_to(end[0], end[1])
            ctx.stroke()

    def _draw_zero_power_vertices(
        self, ctx: cairo.Context, vertex_data, color_set: ColorSet
    ):
        """Draws zero-power vertices onto the cairo context."""
        if vertex_data.zero_power_vertices.size > 0:
            zero_v = vertex_data.zero_power_vertices.reshape(-1, 2, 3)
            ctx.set_source_rgba(*color_set.get_rgba("zero_power"))
            for start, end in zero_v:
                ctx.move_to(start[0], start[1])
                ctx.line_to(end[0], end[1])
            ctx.stroke()

    def _draw_powered_vertices(
        self, ctx: cairo.Context, vertex_data, color_set: ColorSet
    ):
        """Draws powered vertices grouped by color for performance."""
        if vertex_data.powered_vertices.size == 0:
            return

        powered_v = vertex_data.powered_vertices.reshape(-1, 2, 3)
        powered_c = vertex_data.powered_colors
        cut_lut = color_set.get_lut("cut")

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

    def _schedule_throttled_update(self, key: ArtifactKey, view_id: int):
        """
        Schedules a throttled update notification for the given view.
        If an update was recently sent, this will delay the next update
        to prevent flooding the UI.

        Args:
            key: The view key (step_uid, workpiece_uid).
            view_id: The view generation ID.
        """
        current_time = time.time()
        last_update = self._last_update_time.get(key, 0)

        # Cancel any existing timer for this key
        existing_timer = self._throttle_timers.pop(key, None)
        if existing_timer:
            existing_timer.cancel()

        # Mark that an update is pending
        self._pending_updates[key] = True

        # Calculate time until next update
        time_since_last = current_time - last_update
        time_until_next = max(0, THROTTLE_INTERVAL - time_since_last)

        def send_update():
            self._send_throttled_update(key, view_id)

        if time_until_next <= 0:
            # Send immediately if enough time has passed
            send_update()
        else:
            # Schedule delayed update
            timer = threading.Timer(time_until_next, send_update)
            self._throttle_timers[key] = timer
            timer.start()

    def _send_throttled_update(self, key: ArtifactKey, view_id: int):
        """
        Sends the view_artifact_updated signal for the given view.
        This should only be called by _schedule_throttled_update.

        Args:
            key: The view key (workpiece_uid).
            view_id: The view generation ID.
        """
        # Clear pending flag and timer
        self._pending_updates.pop(key, None)
        self._throttle_timers.pop(key, None)

        # Update last update time
        self._last_update_time[key] = time.time()

        # Get the view handle from ArtifactManager
        handle = self._artifact_manager.get_workpiece_view_handle(key, view_id)
        if not handle:
            logger.debug(f"No view handle for {key}, skipping update")
            return

        # Send the update signal
        self.view_artifact_updated.send(
            self,
            step_uid=None,
            workpiece_uid=key.id,
            handle=handle,
        )
        logger.debug(f"Sent throttled view_artifact_updated for {key}")

    def _on_render_complete(
        self, task: "Task", key: ArtifactKey, view_id: int
    ):
        """
        Callback for when a rendering task finishes. It now only handles
        cleanup and state management.

        Args:
            task: The completed task.
            key: The view key (workpiece_uid).
            view_id: The view generation ID for this render.
        """
        logger.debug(
            f"[{key}] _on_render_complete called, "
            f"task_status={task.get_status()}, "
            f"task_id={task.id}"
        )
        logger.info(
            f"[DIAGNOSTIC] View render task for key {key} completed, "
            f"status={task.get_status()}, task_id={task.id}"
        )

        status = task.get_status()

        if status == "completed":
            self._artifact_manager.complete_generation(key, view_id)
        elif status != "canceled":
            logger.error(f"View render for {key} failed with status: {status}")
            self._artifact_manager.fail_generation(
                key,
                f"Render failed with status: {status}",
                view_id,
            )

        # The old `view_artifact_ready` signal is now fired on the
        # `view_artifact_created` event to enable progressive rendering.
        # This callback now only signals the end of the task.
        self.generation_finished.send(self, key=key)
