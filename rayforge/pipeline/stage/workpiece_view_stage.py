from __future__ import annotations
import cairo
import logging
import numpy as np
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from blinker import Signal
from ...shared.util.colors import ColorSet
from ..artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
)
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifactHandle,
)
from .base import PipelineStage
from .workpiece_view_runner import (
    make_workpiece_view_artifact_in_subprocess,
    render_chunk_to_view,
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

        # Track the source artifact properties for cache invalidation
        self._last_source_properties: Dict[
            ViewKey, Tuple[Tuple[float, float], Optional[Tuple[float, float]]]
        ] = {}

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
        self._last_source_properties.clear()

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

        source_handle = self._artifact_manager.get_workpiece_handle(
            step_uid, workpiece_uid
        )
        if not source_handle:
            logger.warning(
                f"Cannot render view for {key}: "
                "source WorkPieceArtifact not found."
            )
            return

        logger.debug(
            f"[{key}] request_view_render: "
            f"gen_size={source_handle.generation_size}, "
            f"src_dims={source_handle.source_dimensions}, "
            f"ppm={context.pixels_per_mm}"
        )

        current_source_props = (
            source_handle.generation_size,
            source_handle.source_dimensions,
        )
        last_source_props = self._last_source_properties.get(key)

        if not force and last_context == context:
            if last_source_props == current_source_props:
                logger.debug(
                    f"View for {key} is already up-to-date. "
                    f"Context unchanged, source properties unchanged. "
                    f"source_handle={source_handle.shm_name}"
                )
                return
            else:
                logger.warning(
                    f"View for {key} has stale source artifact! "
                    f"last_source_props={last_source_props}, "
                    f"current_source_props={current_source_props}. "
                    f"Re-rendering with new source artifact."
                )

        if key in self._active_tasks:
            logger.debug(f"View for {key} is already being generated.")
            return

        # Retain the source handle to prevent premature release while
        # the render task is using it (borrower pattern)
        self._artifact_manager.retain_handle(source_handle)
        self._retained_source_handles[key] = source_handle

        self._last_context_cache[key] = context
        self._last_source_properties[key] = current_source_props

        def when_done_callback(task: "Task"):
            self._on_render_complete(task, key)

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
            self._handle_view_artifact_created(
                key, step_uid, workpiece_uid, data
            )
        elif event_name == "view_artifact_updated":
            self._handle_view_artifact_updated(key, step_uid, workpiece_uid)

    def _handle_view_artifact_created(
        self, key: ViewKey, step_uid: str, workpiece_uid: str, data: dict
    ):
        """Handles the view_artifact_created event."""
        try:
            handle = self._adopt_view_handle(key, data)
            if handle is None:
                return

            self._replace_current_view_handle(key, handle)
            self._initialize_live_render_context(key, handle)
            self._send_view_artifact_created_signals(
                step_uid, workpiece_uid, handle
            )
        except Exception as e:
            logger.error(f"Failed to process view_artifact_created: {e}")

    def _adopt_view_handle(
        self, key: ViewKey, data: dict
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
        self, key: ViewKey, handle: WorkPieceViewArtifactHandle
    ):
        """Replaces the current view handle with a new one."""
        old_handle = self._current_view_handles.get(key)
        if old_handle:
            logger.debug(
                f"Releasing old view artifact: {old_handle.shm_name} "
                f"for key {key}"
            )
            self._artifact_manager.release_handle(old_handle)

        self._artifact_manager.retain_handle(handle)
        self._current_view_handles[key] = handle
        logger.debug(
            f"Stored new view artifact: {handle.shm_name} for key {key}"
        )

    def _initialize_live_render_context(
        self, key: ViewKey, handle: WorkPieceViewArtifactHandle
    ):
        """Initializes the live render context for progressive rendering."""
        self._live_render_contexts[key] = {
            "handle": handle,
            "generation_id": 0,
            "render_context": self._last_context_cache.get(key),
        }
        logger.debug(f"Initialized live render context for {key}")

    def _send_view_artifact_created_signals(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceViewArtifactHandle,
    ):
        """Sends signals when a view artifact is created."""
        self.view_artifact_created.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )
        self.view_artifact_ready.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _handle_view_artifact_updated(
        self, key: ViewKey, step_uid: str, workpiece_uid: str
    ):
        """Handles the view_artifact_updated event."""
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

        live_ctx = self._get_live_render_context(key)
        if live_ctx is None:
            return

        if not self._validate_generation_id(live_ctx, key, generation_id):
            return

        chunk_artifact = self._get_chunk_artifact(key, chunk_handle)
        if chunk_artifact is None:
            return

        view_handle, render_context = self._get_render_components(
            live_ctx, key, chunk_handle
        )
        if view_handle is None or render_context is None:
            return

        if not self._should_render_chunk(chunk_artifact, key, chunk_handle):
            return

        self._render_chunk_to_view(
            key, chunk_handle, view_handle, render_context
        )
        self._schedule_throttled_update(key)

    def _get_live_render_context(self, key: ViewKey) -> Dict[str, Any] | None:
        """Gets the live render context for the given view key."""
        live_ctx = self._live_render_contexts.get(key)
        if live_ctx is None:
            logger.debug(f"No live render context for {key}. Ignoring chunk.")
        return live_ctx

    def _validate_generation_id(
        self, live_ctx: Dict[str, Any], key: ViewKey, generation_id: int
    ) -> bool:
        """Validates the generation ID for the chunk."""
        live_ctx["generation_id"] = generation_id
        if live_ctx.get("generation_id") != generation_id:
            logger.debug(
                f"Stale chunk for {key}. "
                f"Expected gen_id={live_ctx.get('generation_id')}, "
                f"got {generation_id}"
            )
            return False
        return True

    def _get_chunk_artifact(
        self, key: ViewKey, chunk_handle: BaseArtifactHandle
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
        live_ctx: Dict[str, Any],
        key: ViewKey,
        chunk_handle: BaseArtifactHandle,
    ) -> tuple[WorkPieceViewArtifactHandle | None, RenderContext | None]:
        """Gets the view handle and render context from live context."""
        view_handle = live_ctx.get("handle")
        render_context = live_ctx.get("render_context")
        if not view_handle or not render_context:
            logger.warning(f"Missing view_handle or render_context for {key}")
            self._artifact_manager.release_handle(chunk_handle)
            return None, None
        return view_handle, render_context

    def _should_render_chunk(
        self,
        chunk_artifact: WorkPieceArtifact,
        key: ViewKey,
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
        key: ViewKey,
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

        # Release the retained source handle after task completes
        # (success, failure, or cancellation)
        source_handle = self._retained_source_handles.pop(key, None)
        if source_handle:
            self._artifact_manager.release_handle(source_handle)

        # The old `view_artifact_ready` signal is now fired on the
        # `view_artifact_created` event to enable progressive rendering.
        # This callback now only signals the end of the task.
        self.generation_finished.send(self, key=key)
