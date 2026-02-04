from __future__ import annotations
import cairo
import logging
import numpy as np
import threading
import time
import sys
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, Tuple, cast
from blinker import Signal
from ...shared.util.colors import ColorSet
from ..artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
)
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
    WorkPieceViewArtifactHandle,
)
from .base import PipelineStage
from .workpiece_view_runner import make_workpiece_view_artifact_in_subprocess

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.layer import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.manager import ArtifactManager


logger = logging.getLogger(__name__)

# A view artifact is uniquely identified by the step and workpiece that
# produced its source data.
ViewKey = Tuple[str, str]  # (step_uid, workpiece_uid)

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
        self._active_tasks: Dict[ViewKey, "Task"] = {}
        self._last_context_cache: Dict[ViewKey, RenderContext] = {}
        self._thread_tasks: Dict[ViewKey, bool] = {}

        # Track the currently active handle for each view so we can release
        # it when it is replaced or when the stage shuts down.
        self._current_view_handles: Dict[ViewKey, BaseArtifactHandle] = {}

        # Track retained source workpiece handles for borrower pattern.
        # These are retained when a render starts and released when complete.
        self._retained_source_handles: Dict[ViewKey, BaseArtifactHandle] = {}

        # Live render context for progressive chunk rendering.
        # Tracks the view artifact being progressively built from chunks.
        self._live_render_contexts: Dict[ViewKey, Dict[str, Any]] = {}

        # Track generation IDs to detect stale chunks.
        # When a new generation starts, we increment the ID and ignore
        # chunks from the previous generation.
        self._generation_ids: Dict[ViewKey, int] = {}

        # Throttling for progressive chunk updates.
        # Tracks pending updates and last update time for each view.
        self._pending_updates: Dict[ViewKey, bool] = {}
        self._last_update_time: Dict[ViewKey, float] = {}
        self._throttle_timers: Dict[ViewKey, threading.Timer] = {}

        self.view_artifact_ready = Signal()
        self.view_artifact_created = Signal()
        self.view_artifact_updated = Signal()
        self.generation_finished = Signal()

    @property
    def is_busy(self) -> bool:
        """Returns True if the stage has any active tasks."""
        return bool(self._active_tasks)

    def reconcile(self, doc: "Doc"):
        """This is an on-demand stage, so reconcile does nothing."""
        pass

    def shutdown(self):
        """Cancels any active rendering tasks and releases held handles."""
        logger.debug("WorkPieceViewPipelineStage shutting down.")
        for key in list(self._active_tasks.keys()):
            task = self._active_tasks.pop(key, None)
            if task:
                self._task_manager.cancel_task(task.key)
            self._thread_tasks.pop(key, None)

        # Release all currently held view handles
        for handle in self._current_view_handles.values():
            self._artifact_manager.release_handle(handle)
        self._current_view_handles.clear()

        # Release all retained source handles
        for handle in self._retained_source_handles.values():
            self._artifact_manager.release_handle(handle)
        self._retained_source_handles.clear()

        # Clear live render contexts
        self._live_render_contexts.clear()

        # Cancel any pending throttle timers
        for timer in self._throttle_timers.values():
            if timer:
                timer.cancel()
        self._throttle_timers.clear()
        self._pending_updates.clear()
        self._last_update_time.clear()
        self._generation_ids.clear()

    def _on_generation_starting(
        self,
        sender,
        *,
        step: "Step",
        workpiece: "WorkPiece",
        generation_id: int,
    ):
        """
        Handler for when workpiece generation starts.
        Clears live render contexts to prevent chunks from being drawn
        to stale view artifacts from the previous generation.
        """
        key = (step.uid, workpiece.uid)
        logger.debug(
            f"View stage: Generation {generation_id} starting for {key}. "
            f"Clearing live render context."
        )
        # Clear the live render context for this view
        if key in self._live_render_contexts:
            del self._live_render_contexts[key]
        # Update the generation ID
        self._generation_ids[key] = generation_id

    def request_view_render(
        self,
        step_uid: str,
        workpiece_uid: str,
        context: RenderContext,
        force: bool = False,
    ):
        """
        Requests an asynchronous render of a workpiece view for a specific
        step.

        Args:
            step_uid: The unique identifier of the step.
            workpiece_uid: The unique identifier of the workpiece.
            context: The render context to use.
            force: If True, force re-rendering even if the context appears
                unchanged.
        """
        key = (step_uid, workpiece_uid)
        last_context = self._last_context_cache.get(key)

        if not force and last_context == context:
            logger.debug(f"View for {key} is already up-to-date.")
            return

        if key in self._active_tasks:
            logger.debug(f"View for {key} is already being generated.")
            return

        source_handle = self._artifact_manager.get_workpiece_handle(
            step_uid, workpiece_uid
        )
        if not source_handle:
            logger.warning(
                f"Cannot render view for {key}: "
                "source WorkPieceArtifact not found."
            )
            return

        # Retain the source handle to prevent premature release while
        # the render task is using it (borrower pattern)
        self._artifact_manager.retain_handle(source_handle)
        self._retained_source_handles[key] = source_handle

        self._last_context_cache[key] = context

        def when_done_callback(task: "Task"):
            self._on_render_complete(task, key)

        use_thread = sys.platform == "darwin" and hasattr(sys, "_MEIPASS")
        self._thread_tasks[key] = use_thread
        if use_thread:
            task = self._task_manager.run_thread_with_proxy(
                make_workpiece_view_artifact_in_subprocess,
                self._artifact_manager._store,
                workpiece_artifact_handle_dict=source_handle.to_dict(),
                render_context_dict=context.to_dict(),
                creator_tag="workpiece_view",
                key=key,
                when_done=when_done_callback,
                when_event=self._on_render_event_received,
            )
        else:
            task = self._task_manager.run_process(
                make_workpiece_view_artifact_in_subprocess,
                self._artifact_manager._store,
                workpiece_artifact_handle_dict=source_handle.to_dict(),
                render_context_dict=context.to_dict(),
                creator_tag="workpiece_view",
                key=key,
                when_done=when_done_callback,
                when_event=self._on_render_event_received,
            )
        self._active_tasks[key] = task

    def _on_render_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles progressive rendering events from the worker process."""
        key = task.key
        step_uid, workpiece_uid = key

        if event_name == "view_artifact_created":
            try:
                handle_dict = data["handle_dict"]
                handle = self._artifact_manager.adopt_artifact(
                    key,
                    handle_dict,
                    in_process=self._thread_tasks.get(key, False),
                )
                if not isinstance(handle, WorkPieceViewArtifactHandle):
                    raise TypeError("Expected WorkPieceViewArtifactHandle")

                logger.debug(
                    f"Adopting new view artifact: {handle.shm_name} "
                    f"for key {key}"
                )

                # Retain the handle to keep it alive while this stage
                # holds a reference to it (borrower pattern)
                self._artifact_manager.retain_handle(handle)

                # Release the previous handle for this key if one exists,
                # as it is now obsolete.
                old_handle = self._current_view_handles.get(key)
                if old_handle:
                    logger.debug(
                        f"Releasing old view artifact: {old_handle.shm_name} "
                        f"for key {key}"
                    )
                    self._artifact_manager.release_handle(old_handle)

                # Store the new handle as the current one
                self._current_view_handles[key] = handle
                logger.debug(
                    f"Stored new view artifact: {handle.shm_name} "
                    f"for key {key}"
                )

                # Initialize render context for progressive chunk rendering
                # This will be used by _on_workpiece_chunk_available
                self._live_render_contexts[key] = {
                    "handle": handle,
                    "generation_id": 0,  # Will be updated when chunks arrive
                    "render_context": self._last_context_cache.get(key),
                }
                logger.debug(f"Initialized live render context for {key}")

                self.view_artifact_created.send(
                    self,
                    step_uid=step_uid,
                    workpiece_uid=workpiece_uid,
                    handle=handle,
                )
                # Fire old signal for backward compatibility, enabling
                # progressive rendering for existing UI code.
                self.view_artifact_ready.send(
                    self,
                    step_uid=step_uid,
                    workpiece_uid=workpiece_uid,
                    handle=handle,
                )
            except Exception as e:
                logger.error(f"Failed to process view_artifact_created: {e}")

        elif event_name == "view_artifact_updated":
            handle = self._current_view_handles.get(key)
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
        key: ViewKey,
        chunk_handle: BaseArtifactHandle,
        generation_id: int,
    ):
        """
        Receives chunk data from the WorkPiecePipelineStage.
        Implements incremental rendering by drawing the chunk onto
        the live view artifact bitmap.
        """
        step_uid, workpiece_uid = key
        logger.debug(
            f"View stage received chunk for {key}, "
            f"generation_id={generation_id}, "
            f"chunk_handle={chunk_handle.shm_name}"
        )

        # Get or create live render context for this view
        live_ctx = self._live_render_contexts.get(key)
        if live_ctx is None:
            # No active live render for this view yet.
            # This could happen if chunks arrive before a view render
            # is requested, or if the previous render completed.
            # For now, we just log and return.
            # In the future, we might want to auto-start a render.
            logger.debug(f"No live render context for {key}. Ignoring chunk.")
            return

        # Update generation ID to track the current generation
        live_ctx["generation_id"] = generation_id

        # Check for stale chunks (should not happen with the above update,
        # but kept for safety if chunks arrive out of order)
        if live_ctx.get("generation_id") != generation_id:
            logger.debug(
                f"Stale chunk for {key}. "
                f"Expected gen_id={live_ctx.get('generation_id')}, "
                f"got {generation_id}"
            )
            return

        # Get the chunk artifact data
        try:
            artifact = self._artifact_manager.get_artifact(chunk_handle)
            if artifact is None:
                logger.warning(f"Could not retrieve chunk artifact for {key}")
                return
            if not isinstance(artifact, WorkPieceArtifact):
                logger.warning(
                    f"Chunk artifact for {key} is not a WorkPieceArtifact"
                )
                return
            chunk_artifact = artifact
        except Exception as e:
            logger.error(f"Error retrieving chunk artifact for {key}: {e}")
            return

        view_handle = live_ctx.get("handle")
        render_context = live_ctx.get("render_context")
        if not view_handle or not render_context:
            logger.warning(f"Missing view_handle or render_context for {key}")
            self._artifact_manager.release_handle(chunk_handle)
            return

        # Only render if the chunk has vertex data
        if not chunk_artifact.vertex_data:
            logger.debug(f"Chunk for {key} has no vertex data, skipping")
            self._artifact_manager.release_handle(chunk_handle)
            return

        shm = None
        try:
            # Open the shared memory for the view artifact
            shm = shared_memory.SharedMemory(name=view_handle.shm_name)

            # Get the view artifact to read its metadata
            view_artifact = cast(
                WorkPieceViewArtifact,
                self._artifact_manager.get_artifact(view_handle),
            )
            if not view_artifact:
                logger.warning(f"Could not retrieve view artifact for {key}")
                return

            bitmap_data = view_artifact.bitmap_data
            height_px, width_px = bitmap_data.shape[:2]

            # Create a numpy array view into the shared memory
            shm_bitmap = np.ndarray(
                shape=(height_px, width_px, 4),
                dtype=np.uint8,
                buffer=shm.buf,
            )

            # Create a cairo surface from the shared memory bitmap
            surface = cairo.ImageSurface.create_for_data(
                memoryview(shm_bitmap),
                cairo.FORMAT_ARGB32,
                width_px,
                height_px,
            )

            # Create a cairo context
            ctx = cairo.Context(surface)

            # Set up transform to match the worker's coordinate system
            # The worker uses: translate(margin, height_px - margin)
            #                   scale(effective_ppm_x, -effective_ppm_y)
            #                   translate(-x_mm, -y_mm)
            bbox_mm = view_artifact.bbox_mm
            x_mm, y_mm, w_mm, h_mm = bbox_mm
            margin = render_context.margin_px
            ppm_x, ppm_y = render_context.pixels_per_mm

            # Calculate effective PPM (may be clamped due to Cairo limits)
            effective_ppm_x = (
                (width_px - 2 * margin) / w_mm if w_mm > 0 else ppm_x
            )
            effective_ppm_y = (
                (height_px - 2 * margin) / h_mm if h_mm > 0 else ppm_y
            )

            # Set up transform (Y-down pixel space from Y-up mm space)
            ctx.translate(margin, height_px - margin)
            ctx.scale(effective_ppm_x, -effective_ppm_y)
            ctx.translate(-x_mm, -y_mm)

            # Get the color set from the render context
            color_set = ColorSet.from_dict(render_context.color_set_dict)

            # Draw the chunk's vertex data
            line_width_mm = (
                1.0 / effective_ppm_x if effective_ppm_x > 0 else 1.0
            )
            self._draw_vertices(
                ctx,
                chunk_artifact.vertex_data,
                color_set,
                render_context.show_travel_moves,
                line_width_mm,
            )

            # Flush the surface to ensure changes are written to shared memory
            surface.flush()

            logger.debug(
                f"Successfully drew chunk for {key} to shared memory "
                f"bitmap {view_handle.shm_name}"
            )

        except Exception as e:
            logger.error(
                f"Error rendering chunk for {key}: {e}", exc_info=True
            )
        finally:
            if shm:
                shm.close()
            # Release the chunk handle after processing
            self._artifact_manager.release_handle(chunk_handle)

        # Trigger throttled update notification
        self._schedule_throttled_update(key)

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

        # Draw Travel & Zero-Power Moves
        if show_travel:
            if vertex_data.travel_vertices.size > 0:
                travel_v = vertex_data.travel_vertices.reshape(-1, 2, 3)
                ctx.set_source_rgba(*color_set.get_rgba("travel"))
                for start, end in travel_v:
                    ctx.move_to(start[0], start[1])
                    ctx.line_to(end[0], end[1])
                ctx.stroke()

            if vertex_data.zero_power_vertices.size > 0:
                zero_v = vertex_data.zero_power_vertices.reshape(-1, 2, 3)
                ctx.set_source_rgba(*color_set.get_rgba("zero_power"))
                for start, end in zero_v:
                    ctx.move_to(start[0], start[1])
                    ctx.line_to(end[0], end[1])
                ctx.stroke()

        # Draw Powered Moves (Grouped by Color for performance)
        if vertex_data.powered_vertices.size > 0:
            powered_v = vertex_data.powered_vertices.reshape(-1, 2, 3)
            powered_c = vertex_data.powered_colors
            cut_lut = color_set.get_lut("cut")

            # Use power from the first vertex of each segment for color
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

    def _schedule_throttled_update(self, key: ViewKey):
        """
        Schedules a throttled update notification for the given view.
        If an update was recently sent, this will delay the next update
        to prevent flooding the UI.
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
            self._send_throttled_update(key)

        if time_until_next <= 0:
            # Send immediately if enough time has passed
            send_update()
        else:
            # Schedule delayed update
            timer = threading.Timer(time_until_next, send_update)
            self._throttle_timers[key] = timer
            timer.start()

    def _send_throttled_update(self, key: ViewKey):
        """
        Sends the view_artifact_updated signal for the given view.
        This should only be called by _schedule_throttled_update.
        """
        # Clear pending flag and timer
        self._pending_updates.pop(key, None)
        self._throttle_timers.pop(key, None)

        # Update last update time
        self._last_update_time[key] = time.time()

        # Get the live render context
        live_ctx = self._live_render_contexts.get(key)
        if not live_ctx:
            logger.debug(f"No live render context for {key}, skipping update")
            return

        step_uid, workpiece_uid = key
        handle = live_ctx.get("handle")

        # Send the update signal
        self.view_artifact_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )
        logger.debug(f"Sent throttled view_artifact_updated for {key}")

    def _on_render_complete(self, task: "Task", key: ViewKey):
        """
        Callback for when a rendering task finishes. It now only handles
        cleanup and state management.
        """
        self._active_tasks.pop(key, None)
        self._thread_tasks.pop(key, None)

        if task.get_status() != "completed":
            logger.error(
                f"View render for {key} failed with status: "
                f"{task.get_status()}"
            )

        # Release the retained source handle after task completes
        # (success, failure, or cancellation)
        source_handle = self._retained_source_handles.pop(key, None)
        if source_handle:
            self._artifact_manager.release_handle(source_handle)

        # The old `view_artifact_ready` signal is now fired on the
        # `view_artifact_created` event to enable progressive rendering.
        # This callback now only signals the end of the task.
        self.generation_finished.send(self, key=key)
