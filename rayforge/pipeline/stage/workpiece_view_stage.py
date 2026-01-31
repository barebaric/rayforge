from __future__ import annotations
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple
from blinker import Signal
from ...context import get_context
from ..artifact import (
    WorkPieceArtifact,
    create_handle_from_dict,
    BaseArtifactHandle,
)
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifactHandle,
)
from .base import PipelineStage
from .workpiece_view_runner import make_workpiece_view_artifact_in_subprocess

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache


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
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._active_tasks: Dict[ViewKey, "Task"] = {}
        self._last_context_cache: Dict[ViewKey, RenderContext] = {}

        # Track the currently active handle for each view so we can release
        # it when it is replaced or when the stage shuts down.
        self._current_view_handles: Dict[ViewKey, BaseArtifactHandle] = {}

        # Live render context for progressive chunk rendering.
        # Tracks the view artifact being progressively built from chunks.
        self._live_render_contexts: Dict[ViewKey, Dict[str, Any]] = {}

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

        # Release all currently held view handles
        for handle in self._current_view_handles.values():
            get_context().artifact_store.release(handle)
        self._current_view_handles.clear()

        # Clear live render contexts
        self._live_render_contexts.clear()

        # Cancel any pending throttle timers
        for timer in self._throttle_timers.values():
            if timer:
                timer.cancel()
        self._throttle_timers.clear()
        self._pending_updates.clear()
        self._last_update_time.clear()

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

        source_handle = self._artifact_cache.get_workpiece_handle(
            step_uid, workpiece_uid
        )
        if not source_handle:
            logger.warning(
                f"Cannot render view for {key}: "
                "source WorkPieceArtifact not found."
            )
            return

        self._last_context_cache[key] = context

        def when_done_callback(task: "Task"):
            self._on_render_complete(task, key)

        task = self._task_manager.run_process(
            make_workpiece_view_artifact_in_subprocess,
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
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, WorkPieceViewArtifactHandle):
                    raise TypeError("Expected WorkPieceViewArtifactHandle")

                logger.debug(
                    f"Adopting new view artifact: {handle.shm_name} "
                    f"for key {key}"
                )
                get_context().artifact_store.adopt(handle)

                # Release the previous handle for this key if one exists,
                # as it is now obsolete.
                old_handle = self._current_view_handles.get(key)
                if old_handle:
                    logger.debug(
                        f"Releasing old view artifact: {old_handle.shm_name} "
                        f"for key {key}"
                    )
                    get_context().artifact_store.release(old_handle)

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
            artifact = get_context().artifact_store.get(chunk_handle)
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

        # TODO: Implement actual rendering of chunk onto view bitmap
        # This requires:
        # 1. Opening the shared memory for the view artifact
        # 2. Creating a cairo surface from the bitmap
        # 3. Drawing the chunk's vertex data onto the surface
        # 4. Flushing and sending view_artifact_updated signal

        logger.debug(
            f"Chunk for {key} has {len(chunk_artifact.ops)} ops, "
            f"vertex_data: {chunk_artifact.vertex_data is not None}"
        )

        # Release the chunk handle after processing
        get_context().artifact_store.release(chunk_handle)

        # Trigger throttled update notification
        self._schedule_throttled_update(key)

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

        if task.get_status() != "completed":
            logger.error(
                f"View render for {key} failed with status: "
                f"{task.get_status()}"
            )
        # The old `view_artifact_ready` signal is now fired on the
        # `view_artifact_created` event to enable progressive rendering.
        # This callback now only signals the end of the task.
        self.generation_finished.send(self, key=key)
