from __future__ import annotations
import logging
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, cast

from blinker import Signal

from ..artifact import (
    BaseArtifactHandle,
    WorkPieceArtifactHandle,
)
from ..artifact.handle import create_handle_from_dict
from ..artifact.key import ArtifactKey
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
    WorkPieceViewArtifactHandle,
)
from .view_compute import calculate_render_dimensions
from .view_runner import (
    make_workpiece_view_artifact_in_subprocess,
    render_chunk_to_view,
)

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.layer import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.task import Task
    from ..artifact.store import ArtifactStore
    from ..pipeline import Pipeline


logger = logging.getLogger(__name__)

THROTTLE_INTERVAL = 0.033


@dataclass
class ViewEntry:
    """Holds state for a single view artifact."""

    handle: Optional[WorkPieceViewArtifactHandle] = None
    render_context: Optional[RenderContext] = None
    source_handle: Optional[WorkPieceArtifactHandle] = None


class ViewManager:
    """
    Manages view rendering for workpieces, decoupled from the data pipeline.

    The ViewManager is responsible for:
    - Maintaining the current RenderContext (from UI events like zoom/pan)
    - Tracking source WorkPieceArtifact handles it is displaying
    - Triggering rendering tasks when source data or render context changes
    - Managing view artifact lifecycle (retain/release handles)
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        artifact_store: "ArtifactStore",
        machine: Optional["Machine"],
    ):
        self._pipeline = pipeline
        self._store = artifact_store
        self._task_manager = pipeline.task_manager
        self._machine = machine
        self._current_view_context: Optional[RenderContext] = None
        self._view_generation_id = 0

        self._source_artifact_handles: Dict[str, WorkPieceArtifactHandle] = {}
        self._workpiece_to_step: Dict[str, str] = {}
        self._view_entries: Dict[str, ViewEntry] = {}

        self._pending_updates: Dict[ArtifactKey, bool] = {}
        self._last_update_time: Dict[ArtifactKey, float] = {}
        self._throttle_timers: Dict[ArtifactKey, threading.Timer] = {}

        self.view_artifact_ready = Signal()
        self.view_artifact_created = Signal()
        self.view_artifact_updated = Signal()
        self.generation_finished = Signal()
        self.source_artifact_ready = Signal()

        self._connect_pipeline_signals()

    def _connect_pipeline_signals(self):
        """Connect to pipeline signals."""
        self._pipeline.workpiece_artifact_ready.connect(
            self.on_workpiece_artifact_ready
        )
        self._pipeline.workpiece_starting.connect(self.on_generation_starting)

    @property
    def current_view_context(self) -> Optional[RenderContext]:
        """Returns the current render context."""
        return self._current_view_context

    @property
    def view_generation_id(self) -> int:
        """Returns the current view generation ID."""
        return self._view_generation_id

    @property
    def store(self) -> "ArtifactStore":
        """Returns the artifact store."""
        return self._store

    def _is_view_stale(
        self,
        workpiece_uid: str,
        new_context: Optional[RenderContext],
        source_handle: Optional[WorkPieceArtifactHandle],
    ) -> bool:
        """Check if a view needs re-rendering."""
        entry = self._view_entries.get(workpiece_uid)
        if entry is None or entry.handle is None:
            logger.debug(
                f"_is_view_stale[{workpiece_uid}]: no entry/handle -> STALE"
            )
            return True

        if new_context is not None:
            if entry.render_context is None:
                logger.debug(
                    f"_is_view_stale[{workpiece_uid}]: "
                    f"no render_context -> STALE"
                )
                return True
            if entry.render_context != new_context:
                logger.debug(
                    f"_is_view_stale[{workpiece_uid}]: "
                    f"context changed -> STALE"
                )
                return True

        if source_handle is not None:
            if entry.source_handle is None:
                logger.debug(
                    f"_is_view_stale[{workpiece_uid}]: "
                    f"no entry.source_handle -> STALE"
                )
                return True
            entry_gen_size = entry.source_handle.generation_size
            new_gen_size = source_handle.generation_size
            if entry_gen_size != new_gen_size:
                logger.debug(
                    f"_is_view_stale[{workpiece_uid}]: "
                    f"gen_size {entry_gen_size} -> {new_gen_size} -> STALE"
                )
                return True
            entry_src_dims = entry.source_handle.source_dimensions
            new_src_dims = source_handle.source_dimensions
            if entry_src_dims != new_src_dims:
                logger.debug(
                    f"_is_view_stale[{workpiece_uid}]: "
                    f"src_dims {entry_src_dims} -> {new_src_dims} -> STALE"
                )
                return True

        esh = entry.source_handle
        ssh = source_handle
        entry_sz = esh.generation_size if esh else None
        src_sz = ssh.generation_size if ssh else None
        logger.debug(
            f"_is_view_stale[{workpiece_uid}]: "
            f"entry.gen_size={entry_sz}, src.gen_size={src_sz} -> NOT STALE"
        )
        return False

    def update_render_context(
        self,
        context: RenderContext,
    ) -> None:
        """
        Updates the view context and triggers re-rendering for all tracked
        workpieces if the context has changed.

        Args:
            context: The new render context to apply.
        """
        if self._current_view_context == context:
            logger.debug("update_render_context: Context unchanged, skipping")
            return

        logger.debug(
            f"update_render_context called with context "
            f"ppm={context.pixels_per_mm}, "
            f"show_travel_moves={context.show_travel_moves}"
        )

        self._current_view_context = context
        self._view_generation_id += 1

        for workpiece_uid in self._source_artifact_handles:
            step_uid = self._workpiece_to_step.get(workpiece_uid)
            self.request_view_render(workpiece_uid, step_uid=step_uid)

    def on_workpiece_artifact_ready(
        self,
        sender,
        *,
        step: "Step",
        workpiece: "WorkPiece",
        handle: BaseArtifactHandle,
        **kwargs,
    ) -> None:
        """
        Handler for the pipeline.workpiece_artifact_ready signal.

        This method manages the source artifact handles:
        - Releases any old handle for this workpiece
        - Retains the new handle
        - Triggers a view render

        Args:
            sender: The signal sender.
            step: The step for which the artifact is ready.
            workpiece: The workpiece whose artifact is ready.
            handle: The artifact handle.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(handle, WorkPieceArtifactHandle):
            logger.warning(
                f"Expected WorkPieceArtifactHandle, got {type(handle)}"
            )
            return

        workpiece_uid = workpiece.uid

        old_handle = self._source_artifact_handles.get(workpiece_uid)
        if old_handle is not None:
            logger.debug(
                f"Releasing old source artifact handle for {workpiece_uid}"
            )
            self._store.release(old_handle)

        wp_handle = cast(WorkPieceArtifactHandle, handle)
        self._source_artifact_handles[workpiece_uid] = wp_handle
        self._workpiece_to_step[workpiece_uid] = step.uid
        self._store.retain(wp_handle)
        logger.debug(
            f"Retained new source artifact handle for {workpiece_uid}"
        )

        self.request_view_render(workpiece_uid, step_uid=step.uid)

        self.source_artifact_ready.send(
            self,
            step=step,
            workpiece=workpiece,
            handle=wp_handle,
        )

    def request_view_render(
        self,
        workpiece_uid: str,
        step_uid: Optional[str] = None,
    ) -> None:
        """
        Requests an asynchronous render of a workpiece view.

        Args:
            workpiece_uid: The unique identifier of the workpiece.
            step_uid: The unique identifier of the step (optional).
        """
        if self._current_view_context is None:
            logger.debug(
                f"Cannot render view for {workpiece_uid}: "
                "No render context set."
            )
            return

        context = self._current_view_context
        view_id = self._view_generation_id
        key = ArtifactKey.for_view(workpiece_uid)

        source_handle = self._source_artifact_handles.get(workpiece_uid)
        if source_handle is None:
            logger.warning(
                f"Cannot render view for {workpiece_uid}: "
                "No source artifact handle tracked."
            )
            return

        self._request_view_render_internal(
            key,
            context,
            view_id,
            source_handle,
            step_uid,
        )

    def _request_view_render_internal(
        self,
        key: ArtifactKey,
        context: RenderContext,
        view_id: int,
        source_handle: WorkPieceArtifactHandle,
        step_uid: Optional[str] = None,
    ):
        """
        Internal method to request a view render.

        Args:
            key: The ArtifactKey for the workpiece view.
            context: The render context to use.
            view_id: The view generation ID for this render.
            source_handle: The source WorkPieceArtifact handle.
            step_uid: The unique identifier of the step (optional).
        """
        workpiece_uid = key.id

        if step_uid is None:
            logger.warning(
                f"Cannot render view for {key}: "
                "step_uid is required but not provided."
            )
            return

        if not self._is_view_stale(workpiece_uid, context, source_handle):
            logger.debug(f"View for {key} is still valid. Skipping render.")
            return

        self._current_view_context = context

        task_key = key
        task = self._task_manager.get_task(task_key)
        if task and not task.is_final():
            logger.debug(
                f"[{key}] View render already in progress. Cancelling "
                f"to start new one."
            )
            self._task_manager.cancel_task(task_key)

        entry = self._view_entries.get(workpiece_uid)
        if entry is None:
            entry = ViewEntry()
            self._view_entries[workpiece_uid] = entry
        entry.render_context = context
        entry.source_handle = source_handle

        def when_done_callback(task: "Task"):
            logger.debug(
                f"[{key}] when_done_callback called, "
                f"task_status={task.get_status()}, "
                f"task_id={task.id}"
            )
            self._on_render_complete(task, key, view_id)

        self._task_manager.run_process(
            make_workpiece_view_artifact_in_subprocess,
            self._store,
            workpiece_artifact_handle_dict=source_handle.to_dict(),
            render_context_dict=context.to_dict(),
            creator_tag="workpiece_view",
            key=task_key,
            generation_id=view_id,
            step_uid=step_uid,
            when_done=when_done_callback,
            when_event=self._on_render_event_received,
        )

    def shutdown(self):
        """Cancels any active rendering tasks and releases held handles."""
        logger.debug("ViewManager shutting down.")

        for workpiece_uid, entry in self._view_entries.items():
            task_key = ArtifactKey.for_view(workpiece_uid)
            self._task_manager.cancel_task(task_key)
            if entry.handle is not None:
                self._store.release(entry.handle)

        self._view_entries.clear()

        for timer in self._throttle_timers.values():
            if timer:
                timer.cancel()
        self._throttle_timers.clear()
        self._pending_updates.clear()
        self._last_update_time.clear()

        for handle in self._source_artifact_handles.values():
            self._store.release(handle)
        self._source_artifact_handles.clear()

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

            workpiece_uid = key.id
            entry = self._view_entries.get(workpiece_uid)
            if entry is None:
                entry = ViewEntry()
                self._view_entries[workpiece_uid] = entry
            entry.handle = handle

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
        handle = create_handle_from_dict(handle_dict)
        self._store.adopt(handle)
        if not isinstance(handle, WorkPieceViewArtifactHandle):
            raise TypeError("Expected WorkPieceViewArtifactHandle")

        logger.debug(
            f"Adopting new view artifact: {handle.shm_name} for key {key}"
        )
        return handle

    def _send_view_artifact_created_signals(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceViewArtifactHandle,
    ):
        """Sends signals when a view artifact is created."""
        logger.debug(
            f"Sending view_artifact_created for "
            f"key ({step_uid}, {workpiece_uid})"
        )
        self.view_artifact_created.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )
        logger.debug(
            f"Sending view_artifact_ready for "
            f"key ({step_uid}, {workpiece_uid})"
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
        entry = self._view_entries.get(workpiece_uid)
        handle = entry.handle if entry else None
        self.view_artifact_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _on_render_complete(
        self, task: "Task", key: ArtifactKey, view_id: int
    ):
        """
        Callback for when a rendering task finishes.

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

        self.generation_finished.send(self, key=key)

    def on_chunk_available(
        self,
        sender,
        *,
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
        generation_id: int,
    ):
        """
        Receives chunk data from the pipeline.
        Implements incremental rendering by drawing the chunk onto
        the live view artifact bitmap.

        Args:
            sender: The sender of the signal.
            key: The view key (workpiece_uid).
            chunk_handle: Handle to the chunk artifact.
            generation_id: The data generation ID.
        """
        workpiece_uid = key.id
        logger.debug(
            f"ViewManager received chunk for {key}, "
            f"generation_id={generation_id}, "
            f"chunk_handle={chunk_handle.shm_name}"
        )

        entry = self._view_entries.get(workpiece_uid)
        if entry is None:
            self._store.release(chunk_handle)
            return

        view_handle, render_context = self._get_render_components(
            key, entry, chunk_handle
        )
        if view_handle is None or render_context is None:
            return

        self._store.retain(chunk_handle)
        self._store.retain(view_handle)

        self._task_manager.run_thread(
            render_chunk_to_view,
            self._store,
            chunk_handle.to_dict(),
            view_handle.to_dict(),
            render_context.to_dict(),
            when_done=lambda t: self._on_stitch_complete(
                t, key, chunk_handle, view_handle, workpiece_uid
            ),
        )

    def _on_stitch_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
        view_handle: BaseArtifactHandle,
        workpiece_uid: str,
    ):
        """Callback for when stitching completes."""
        logger.debug(
            f"Chunk stitch for key {key} completed, "
            f"status={task.get_status()}, chunk_handle={chunk_handle.shm_name}"
        )
        try:
            if task.get_status() == "completed" and task.result():
                view_id = self._view_generation_id
                self._schedule_throttled_update(key, view_id)
            else:
                logger.warning(f"Stitching failed for {key}")
        finally:
            self._store.release(chunk_handle)
            self._store.release(view_handle)

    def _get_render_components(
        self,
        key: ArtifactKey,
        entry: ViewEntry,
        chunk_handle: BaseArtifactHandle,
    ) -> tuple[WorkPieceViewArtifactHandle | None, RenderContext | None]:
        """Gets the view handle and render context."""
        if entry.handle is None:
            logger.debug(f"No view handle for {key}. Ignoring chunk.")
            self._store.release(chunk_handle)
            return None, None

        if not isinstance(entry.handle, WorkPieceViewArtifactHandle):
            logger.warning(
                f"Expected WorkPieceViewArtifactHandle, "
                f"got {type(entry.handle)}"
            )
            self._store.release(chunk_handle)
            return None, None

        render_context = entry.render_context
        if render_context is None:
            logger.debug(f"No render context for {key}.")
            self._store.release(chunk_handle)
            return None, None

        return entry.handle, render_context

    def _schedule_throttled_update(self, key: ArtifactKey, view_id: int):
        """
        Schedules a throttled update notification for the given view.
        """
        current_time = time.time()
        last_update = self._last_update_time.get(key, 0)

        existing_timer = self._throttle_timers.pop(key, None)
        if existing_timer:
            existing_timer.cancel()

        self._pending_updates[key] = True

        time_since_last = current_time - last_update
        time_until_next = max(0, THROTTLE_INTERVAL - time_since_last)

        def send_update():
            self._send_throttled_update(key, view_id)

        if time_until_next <= 0:
            send_update()
        else:
            timer = threading.Timer(time_until_next, send_update)
            self._throttle_timers[key] = timer
            timer.start()

    def _send_throttled_update(self, key: ArtifactKey, view_id: int):
        """
        Sends the view_artifact_updated signal for the given view.
        """
        self._pending_updates.pop(key, None)
        self._throttle_timers.pop(key, None)
        self._last_update_time[key] = time.time()

        workpiece_uid = key.id
        entry = self._view_entries.get(workpiece_uid)
        handle = entry.handle if entry else None

        if not handle:
            logger.debug(f"No view handle for {key}, skipping update")
            return

        step_uid = self._workpiece_to_step.get(workpiece_uid)
        self.view_artifact_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )
        logger.debug(f"Sent throttled view_artifact_updated for {key}")

    def allocate_live_buffer(
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

        try:
            bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)
            view_artifact = WorkPieceViewArtifact(
                bitmap_data=bitmap, bbox_mm=bbox
            )

            handle = self._store.put(
                view_artifact, creator_tag="view_live_buffer"
            )
            view_handle = cast(WorkPieceViewArtifactHandle, handle)

            entry = self._view_entries.get(workpiece_uid)
            if entry is None:
                entry = ViewEntry()
                self._view_entries[workpiece_uid] = entry
            entry.handle = view_handle
            entry.render_context = context

            logger.debug(
                f"Allocated live buffer for key {key}: {view_handle.shm_name}"
            )

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
        workpiece: "WorkPiece",
        generation_id: int,
    ):
        """
        Called when workpiece generation starts.
        Pre-allocates the view buffer to enable progressive rendering.

        Only allocates a new buffer if the source artifact is actually
        being regenerated (i.e., we have a new source handle coming).

        Args:
            sender: The step being generated (passed as sender).
            workpiece: The workpiece being generated.
            generation_id: The data generation ID.
        """
        step = sender
        key = ArtifactKey.for_view(workpiece.uid)

        entry = self._view_entries.get(workpiece.uid)
        existing_handle = entry.handle if entry else None

        logger.debug(
            f"ViewManager.on_generation_starting: Source data gen "
            f"{generation_id} starting for {key}, existing view handle: "
            f"{existing_handle.shm_name if existing_handle else None}"
        )

        task_key = ArtifactKey.for_view(workpiece.uid)
        task = self._task_manager.get_task(task_key)
        if task and not task.is_final():
            logger.debug(f"Cancelling existing task for {task_key}")
            self._task_manager.cancel_task(task_key)

        context = self._current_view_context
        if not context:
            logger.warning(
                f"[{key}] Cannot allocate live buffer: No RenderContext."
            )
            return

        if existing_handle is not None:
            logger.debug(
                f"Not allocating live buffer for {key}: "
                "existing view handle will be reused"
            )
            return

        logger.debug(f"Allocating live buffer for {key} (no existing handle)")
        self.allocate_live_buffer(
            key, workpiece, step.uid, self._view_generation_id, context
        )

    def reconcile(self, doc: "Doc", generation_id: int):
        """
        Synchronizes the cache with the document, cleaning up obsolete
        entries and syncing the ledger.
        """
        logger.debug("ViewManager reconciling...")

        all_current_workpieces = {
            workpiece.uid
            for layer in doc.layers
            if layer.workflow is not None
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }

        tracked_uids = set(self._source_artifact_handles.keys())
        obsolete_uids = tracked_uids - all_current_workpieces

        for w_uid in obsolete_uids:
            logger.debug(f"Cleaning up obsolete view workpiece: {w_uid}")
            handle = self._source_artifact_handles.pop(w_uid, None)
            if handle:
                self._store.release(handle)
            task_key = ArtifactKey.for_view(w_uid)
            self._task_manager.cancel_task(task_key)

            entry = self._view_entries.pop(w_uid, None)
            if entry and entry.handle:
                self._store.release(entry.handle)

        self._workpiece_to_step = {
            w_uid: step_uid
            for w_uid, step_uid in self._workpiece_to_step.items()
            if w_uid in all_current_workpieces
        }

    def get_view_handle(
        self, workpiece_uid: str
    ) -> Optional[WorkPieceViewArtifactHandle]:
        """Get the view handle for a workpiece."""
        entry = self._view_entries.get(workpiece_uid)
        return entry.handle if entry else None
