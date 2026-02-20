from __future__ import annotations
import logging
import numpy as np
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, cast, Tuple

from blinker import Signal

from ..artifact import (
    BaseArtifactHandle,
    WorkPieceArtifactHandle,
)
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
    from ...core.step import Step
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

    It indexes views by (workpiece_uid, step_uid) to support visualizing
    intermediate states of a workpiece across multiple steps.

    Shared Memory Lifecycle
    -----------------------
    The ViewManager retains handles for two purposes:

    1. Source artifact tracking: Handles in _source_artifact_handles are
       retained when stored and released in shutdown() or reconcile().

    2. Async task execution: Handles are retained before launching a task
       and released in the task's completion callback (e.g., when_done).

    Every retain() is paired with a release() to prevent memory leaks.
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
        self._is_shutdown = False

        # Keys are (workpiece_uid, step_uid)
        self._source_artifact_handles: Dict[
            Tuple[str, str], WorkPieceArtifactHandle
        ] = {}
        self._view_entries: Dict[Tuple[str, str], ViewEntry] = {}

        # Mapping from (workpiece_uid, step_uid) to a stable ArtifactKey
        # used for task management (deduplication/cancellation).
        self._view_task_keys: Dict[Tuple[str, str], ArtifactKey] = {}

        # Throttling state keyed by the composite key (workpiece_uid, step_uid)
        self._pending_updates: Dict[Tuple[str, str], bool] = {}
        self._last_update_time: Dict[Tuple[str, str], float] = {}
        self._throttle_timers: Dict[Tuple[str, str], threading.Timer] = {}

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
        self._pipeline.step_assembly_starting.connect(
            self.on_workpiece_artifact_ready
        )
        self._pipeline.visual_chunk_available.connect(self.on_chunk_available)

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

    def _get_task_key(self, workpiece_uid: str, step_uid: str) -> ArtifactKey:
        """
        Retrieves or creates a stable ArtifactKey for managing tasks associated
        with a specific (workpiece, step) view.
        """
        composite_id = (workpiece_uid, step_uid)
        if composite_id not in self._view_task_keys:
            # Create a unique, random ID for this specific view slot.
            # We don't embed semantics in the ID string to avoid "hacks".
            # The ViewManager maintains the semantic mapping.
            self._view_task_keys[composite_id] = ArtifactKey(
                id=str(uuid.uuid4()), group="view"
            )
        return self._view_task_keys[composite_id]

    def _is_view_stale(
        self,
        workpiece_uid: str,
        step_uid: str,
        new_context: Optional[RenderContext],
        source_handle: Optional[WorkPieceArtifactHandle],
    ) -> bool:
        """Check if a view needs re-rendering."""
        composite_id = (workpiece_uid, step_uid)
        entry = self._view_entries.get(composite_id)

        if entry is None or entry.handle is None:
            logger.debug(f"_is_view_stale[{composite_id}]: no entry -> STALE")
            return True

        if new_context is not None:
            if entry.render_context is None:
                logger.debug(
                    f"_is_view_stale[{composite_id}]: no context -> STALE"
                )
                return True
            if entry.render_context != new_context:
                logger.debug(
                    f"_is_view_stale[{composite_id}]: context changed -> STALE"
                )
                return True

        if source_handle is not None:
            if entry.source_handle is None:
                logger.debug(
                    f"_is_view_stale[{composite_id}]: no src handle -> STALE"
                )
                return True
            if entry.source_handle.shm_name != source_handle.shm_name:
                logger.debug(
                    f"_is_view_stale[{composite_id}]: "
                    f"shm_name changed -> STALE"
                )
                return True
            entry_gen_size = entry.source_handle.generation_size
            new_gen_size = source_handle.generation_size
            if entry_gen_size != new_gen_size:
                logger.debug(
                    f"_is_view_stale[{composite_id}]: "
                    f"gen_size {entry_gen_size} -> {new_gen_size} -> STALE"
                )
                return True
            entry_src_dims = entry.source_handle.source_dimensions
            new_src_dims = source_handle.source_dimensions
            if entry_src_dims != new_src_dims:
                logger.debug(
                    f"_is_view_stale[{composite_id}]: "
                    f"src_dims {entry_src_dims} -> {new_src_dims} -> STALE"
                )
                return True

        return False

    def update_render_context(
        self,
        context: RenderContext,
    ) -> None:
        """
        Updates the view context and triggers re-rendering for tracked
        workpieces if the context has changed significantly.

        Args:
            context: The new render context to apply.
        """
        if self._current_view_context == context:
            logger.debug("update_render_context: Context unchanged, skipping")
            return

        new_ppm = context.pixels_per_mm[0]

        logger.debug(
            f"update_render_context called with context "
            f"ppm={context.pixels_per_mm}, "
            f"show_travel_moves={context.show_travel_moves}"
        )

        self._current_view_context = context
        self._view_generation_id += 1

        # Re-render each view only if its rendered ppm differs from the
        # requested ppm by more than 25%. This avoids frequent re-renders
        # during small zoom adjustments.
        for key, entry in self._view_entries.items():
            old_ppm = 0.0
            old_show_travel = False
            if entry.render_context is not None:
                old_ppm = entry.render_context.pixels_per_mm[0]
                old_show_travel = entry.render_context.show_travel_moves

            ppm_changed = (
                old_ppm <= 0 or abs(new_ppm - old_ppm) / old_ppm > 0.25
            )
            travel_changed = old_show_travel != context.show_travel_moves
            logger.debug(
                f"update_render_context: key={key}, old_ppm={old_ppm:.2f}, "
                f"new_ppm={new_ppm:.2f}, ppm_changed={ppm_changed}, "
                f"travel_changed={travel_changed}"
            )
            if ppm_changed or travel_changed:
                self.request_view_render(key[0], key[1])

    def _handles_represent_same_artifact(
        self,
        handle1: Optional[WorkPieceArtifactHandle],
        handle2: Optional[WorkPieceArtifactHandle],
    ) -> bool:
        """
        Check if two handles represent the same artifact.

        Two handles represent the same artifact if they point to the same
        shared memory (shm_name), have the same generation_size, and the
        same source_dimensions.

        Returns True if both handles are None, or if they represent the
        same artifact.
        """
        if handle1 is None and handle2 is None:
            return True
        if handle1 is None or handle2 is None:
            return False
        return (
            handle1.shm_name == handle2.shm_name
            and handle1.generation_size == handle2.generation_size
            and handle1.source_dimensions == handle2.source_dimensions
        )

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
        - Releases any old handle for this workpiece (if not in use by tasks)
        - Retains the new handle
        - Triggers a view render

        If the handle represents the same artifact as the existing one
        (e.g., when step_assembly_starting is emitted during a
        position-only transform change), no signal is emitted to avoid
        unnecessary UI redraws.

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

        composite_id = (workpiece.uid, step.uid)

        old_handle = self._source_artifact_handles.get(composite_id)
        wp_handle = cast(WorkPieceArtifactHandle, handle)

        same_artifact = self._handles_represent_same_artifact(
            old_handle, wp_handle
        )

        logger.debug(
            f"on_workpiece_artifact_ready: composite_id={composite_id}, "
            f"old_handle={old_handle.shm_name if old_handle else None}, "
            f"new_handle={wp_handle.shm_name}, "
            f"same_artifact={same_artifact}"
        )

        if not same_artifact:
            if old_handle is not None:
                logger.debug(
                    f"Releasing old source artifact handle for {composite_id}"
                )
                self._store.release(old_handle)

            self._source_artifact_handles[composite_id] = wp_handle
            self._store.retain(wp_handle)
            logger.debug(
                f"Retained new source artifact handle for {composite_id}"
            )

            self.request_view_render(workpiece.uid, step_uid=step.uid)

            self.source_artifact_ready.send(
                self,
                step=step,
                workpiece=workpiece,
                handle=wp_handle,
            )
        else:
            logger.debug(
                f"Same artifact already tracked for {composite_id}, "
                "skipping signal emission"
            )

    def request_view_render(
        self,
        workpiece_uid: str,
        step_uid: str,
    ) -> None:
        """
        Requests an asynchronous render of a workpiece view for a specific
        step.

        Args:
            workpiece_uid: The unique identifier of the workpiece.
            step_uid: The unique identifier of the step (optional).
        """
        if self._current_view_context is None:
            logger.debug(
                f"Cannot render view for ({workpiece_uid}, {step_uid}): "
                "No render context set."
            )
            return

        context = self._current_view_context
        view_id = self._view_generation_id

        # Get the unique task key for this specific view slot
        task_key = self._get_task_key(workpiece_uid, step_uid)

        source_handle = self._source_artifact_handles.get(
            (workpiece_uid, step_uid)
        )
        if source_handle is None:
            logger.warning(
                f"Cannot render view for ({workpiece_uid}, {step_uid}): "
                "No source artifact handle tracked."
            )
            return

        self._request_view_render_internal(
            task_key,
            context,
            view_id,
            source_handle,
            step_uid,
            workpiece_uid,
        )

    def _request_view_render_internal(
        self,
        key: ArtifactKey,
        context: RenderContext,
        view_id: int,
        source_handle: WorkPieceArtifactHandle,
        step_uid: str,
        workpiece_uid: str,
    ):
        """
        Internal method to request a view render.

        Args:
            key: The ArtifactKey for the workpiece view.
            context: The render context to use.
            view_id: The view generation ID for this render.
            source_handle: The source WorkPieceArtifact handle.
            step_uid: The unique identifier of the step (optional).
            workpiece_uid: The unique identifier of the workpiece.
        """
        logger.debug(
            f"_request_view_render_internal: workpiece_uid={workpiece_uid}, "
            f"step_uid={step_uid}, source_shm={source_handle.shm_name}"
        )
        if not self._is_view_stale(
            workpiece_uid, step_uid, context, source_handle
        ):
            logger.debug(f"View for ({workpiece_uid}, {step_uid}) is valid.")
            return

        self._current_view_context = context

        task = self._task_manager.get_task(key)
        if task and not task.is_final():
            logger.debug(
                f"[{key}] View render already in progress. Cancelling."
            )
            self._task_manager.cancel_task(key)

        composite_id = (workpiece_uid, step_uid)
        entry = self._view_entries.get(composite_id)
        if entry is None:
            entry = ViewEntry()
            self._view_entries[composite_id] = entry
        entry.render_context = context
        entry.source_handle = source_handle

        # Retain source handle for the duration of this task
        self._store.retain(source_handle)
        task_source_handle = source_handle

        def when_done_callback(task: "Task"):
            logger.debug(
                f"[{key}] when_done_callback called, "
                f"task_status={task.get_status()}"
            )
            self._on_render_complete(
                task, key, view_id, workpiece_uid, step_uid
            )
            self._store.release(task_source_handle)

        self._task_manager.run_process(
            make_workpiece_view_artifact_in_subprocess,
            self._store,
            workpiece_artifact_handle_dict=source_handle.to_dict(),
            render_context_dict=context.to_dict(),
            creator_tag="wp_view",
            key=key,
            generation_id=view_id,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            when_done=when_done_callback,
            when_event=self._on_render_event_received,
        )

    def shutdown(self):
        """Cancels any active rendering tasks and releases held handles."""
        logger.debug("ViewManager shutting down.")
        self._is_shutdown = True

        # Cancel all managed tasks
        for key in self._view_task_keys.values():
            self._task_manager.cancel_task(key)
        self._view_task_keys.clear()

        for entry in self._view_entries.values():
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
        workpiece_uid = task.kwargs.get("workpiece_uid")

        if step_uid is None or workpiece_uid is None:
            logger.warning("Missing uids in _on_render_event_received")
            return

        key = task.key
        view_id = task.kwargs.get("generation_id", 0)

        if event_name == "view_artifact_created":
            self._handle_view_artifact_created(
                task, key, step_uid, workpiece_uid, data, view_id
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
        workpiece_uid: str,
        data: dict,
        view_id: int,
    ):
        """Handles the view_artifact_created event."""
        try:
            with self._store.safe_adoption(data["handle_dict"]) as handle:
                if not isinstance(handle, WorkPieceViewArtifactHandle):
                    raise TypeError("Expected WorkPieceViewArtifactHandle")

                logger.debug(
                    f"Adopting new view artifact: {handle.shm_name} "
                    f"for task {key}"
                )

                composite_id = (workpiece_uid, step_uid)
                entry = self._view_entries.get(composite_id)
                if entry is None:
                    entry = ViewEntry()
                    self._view_entries[composite_id] = entry
                if entry.handle is not None:
                    self._store.release(entry.handle)
                entry.handle = handle

                self._send_view_artifact_created_signals(
                    step_uid, workpiece_uid, handle
                )
        except Exception as e:
            logger.error(
                f"Failed to process view_artifact_created: {e}", exc_info=True
            )

    def _send_view_artifact_created_signals(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceViewArtifactHandle,
    ):
        """Sends signals when a view artifact is created."""
        logger.debug(
            f"_send_view_artifact_created_signals: step_uid={step_uid}, "
            f"workpiece_uid={workpiece_uid}"
        )
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
        self, key: ArtifactKey, step_uid: str, workpiece_uid: str, view_id: int
    ):
        """Handles the view_artifact_updated event."""
        logger.debug(
            f"_handle_view_artifact_updated: step_uid={step_uid}, "
            f"workpiece_uid={workpiece_uid}, view_id={view_id}"
        )
        composite_id = (workpiece_uid, step_uid)
        entry = self._view_entries.get(composite_id)
        handle = entry.handle if entry else None
        self.view_artifact_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _on_render_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        view_id: int,
        workpiece_uid: str,
        step_uid: str,
    ):
        """Callback for when a rendering task finishes."""
        self.generation_finished.send(
            self,
            key=key,
            workpiece_uid=workpiece_uid,
            step_uid=step_uid,
        )

    def on_chunk_available(
        self,
        sender,
        *,
        key: ArtifactKey,
        chunk_handle: BaseArtifactHandle,
        generation_id: int,
        step_uid: Optional[str] = None,
        **kwargs,
    ):
        """
        Receives chunk data from the pipeline.
        Implements incremental rendering by drawing the chunk onto
        the live view artifact bitmap.
        """
        if ":" in key.id:
            workpiece_uid, _ = key.id.split(":", 1)
        else:
            workpiece_uid = key.id

        if step_uid is None:
            logger.debug(
                f"Chunk available for {workpiece_uid} but no step_uid "
                "provided. Skipping live update."
            )
            self._store.release(chunk_handle)
            return

        logger.debug(
            f"ViewManager received chunk for ({workpiece_uid}, {step_uid}), "
            f"generation_id={generation_id}"
        )

        composite_id = (workpiece_uid, step_uid)
        entry = self._view_entries.get(composite_id)

        if entry is None:
            self._store.release(chunk_handle)
            return

        view_handle, render_context = self._get_render_components(
            composite_id, entry, chunk_handle
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
                t, composite_id, chunk_handle, view_handle
            ),
        )

    def _on_stitch_complete(
        self,
        task: "Task",
        composite_id: Tuple[str, str],
        chunk_handle: BaseArtifactHandle,
        view_handle: BaseArtifactHandle,
    ):
        """Callback for when stitching completes."""
        if self._is_shutdown:
            # If shutdown occurred while task was running, force clean up
            self._store.release(chunk_handle)
            self._store.release(view_handle)
            return

        try:
            if task.get_status() == "completed" and task.result():
                view_id = self._view_generation_id
                self._schedule_throttled_update(composite_id, view_id)
            else:
                logger.warning(f"Stitching failed for {composite_id}")
        finally:
            self._store.release(chunk_handle)
            self._store.release(view_handle)

    def _get_render_components(
        self,
        composite_id: Tuple[str, str],
        entry: ViewEntry,
        chunk_handle: BaseArtifactHandle,
    ) -> tuple[WorkPieceViewArtifactHandle | None, RenderContext | None]:
        """Gets the view handle and render context."""
        if entry.handle is None:
            logger.debug(f"No view handle for {composite_id}. Ignoring chunk.")
            self._store.release(chunk_handle)
            return None, None

        if not isinstance(entry.handle, WorkPieceViewArtifactHandle):
            logger.warning(
                f"Expected WorkPieceViewArtifactHandle, "
                f"got {type(entry.handle)}"
            )
            self._store.release(chunk_handle)
            return None, None

        render_context = entry.render_context or self._current_view_context
        if render_context is None:
            logger.debug(f"No render context for {composite_id}.")
            self._store.release(chunk_handle)
            return None, None

        return entry.handle, render_context

    def _schedule_throttled_update(
        self, composite_id: Tuple[str, str], view_id: int
    ):
        """Schedules a throttled update notification."""
        current_time = time.time()
        last_update = self._last_update_time.get(composite_id, 0)

        existing_timer = self._throttle_timers.pop(composite_id, None)
        if existing_timer:
            existing_timer.cancel()

        self._pending_updates[composite_id] = True

        time_since_last = current_time - last_update
        time_until_next = max(0, THROTTLE_INTERVAL - time_since_last)

        def send_update():
            self._send_throttled_update(composite_id, view_id)

        if time_until_next <= 0:
            send_update()
        else:
            timer = threading.Timer(time_until_next, send_update)
            self._throttle_timers[composite_id] = timer
            timer.start()

    def _send_throttled_update(
        self, composite_id: Tuple[str, str], view_id: int
    ):
        """Sends the view_artifact_updated signal."""
        self._pending_updates.pop(composite_id, None)
        self._throttle_timers.pop(composite_id, None)
        self._last_update_time[composite_id] = time.time()

        workpiece_uid, step_uid = composite_id
        entry = self._view_entries.get(composite_id)
        handle = entry.handle if entry else None

        if not handle:
            return

        self.view_artifact_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def allocate_live_buffer(
        self,
        workpiece: "WorkPiece",
        step_uid: str,
        view_id: int,
        context: RenderContext,
    ) -> None:
        """
        Allocates a new blank view artifact based on the workpiece size and
        registers it as the live buffer for this generation.
        """
        workpiece_uid = workpiece.uid
        composite_id = (workpiece_uid, step_uid)

        w_mm, h_mm = workpiece.size
        bbox = (0.0, 0.0, w_mm, h_mm)

        dims = calculate_render_dimensions(bbox, context)
        if dims is None:
            return

        width_px, height_px, _, _ = dims
        logger.debug(
            f"[{composite_id}] Allocating live buffer: "
            f"{width_px}x{height_px} px"
        )

        try:
            bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)
            view_artifact = WorkPieceViewArtifact(
                bitmap_data=bitmap,
                bbox_mm=bbox,
                workpiece_size_mm=(w_mm, h_mm),
            )

            handle = self._store.put(view_artifact, creator_tag="view_live")
            view_handle = cast(WorkPieceViewArtifactHandle, handle)

            entry = self._view_entries.get(composite_id)
            if entry is None:
                entry = ViewEntry()
                self._view_entries[composite_id] = entry
            if entry.handle is not None:
                self._store.release(entry.handle)
            entry.handle = view_handle
            entry.render_context = context

            self._send_view_artifact_created_signals(
                step_uid, workpiece_uid, view_handle
            )

        except Exception as e:
            logger.error(
                f"[{composite_id}] Failed to allocate live buffer: {e}",
                exc_info=True,
            )

    def on_generation_starting(
        self,
        sender,
        *,
        step: "Step",
        workpiece: "WorkPiece",
        generation_id: int,
    ):
        """
        Called when workpiece generation starts.
        Pre-allocates the view buffer to enable progressive rendering.
        """
        composite_id = (workpiece.uid, step.uid)
        entry = self._view_entries.get(composite_id)
        existing_handle = entry.handle if entry else None

        task_key = self._get_task_key(workpiece.uid, step.uid)
        task = self._task_manager.get_task(task_key)
        if task and not task.is_final():
            self._task_manager.cancel_task(task_key)

        context = self._current_view_context
        if not context:
            return

        need_new_buffer = False
        if existing_handle is None:
            need_new_buffer = True
        elif entry is not None and entry.render_context != context:
            need_new_buffer = True
        else:
            w_mm, h_mm = workpiece.size
            if existing_handle.workpiece_size_mm != (w_mm, h_mm):
                need_new_buffer = True

        if need_new_buffer:
            self.allocate_live_buffer(
                workpiece, step.uid, self._view_generation_id, context
            )

    def reconcile(self, doc: "Doc", generation_id: int):
        """
        Synchronizes the cache with the document, cleaning up obsolete
        entries and syncing the ledger.
        """
        logger.debug("ViewManager reconciling...")

        all_current_pairs = set()
        for layer in doc.layers:
            if layer.workflow and layer.workflow.steps:
                for step in layer.workflow.steps:
                    for workpiece in layer.all_workpieces:
                        all_current_pairs.add((workpiece.uid, step.uid))

        tracked_pairs = set(self._source_artifact_handles.keys())
        obsolete_pairs = tracked_pairs - all_current_pairs

        for composite_id in obsolete_pairs:
            logger.debug(f"Cleaning up obsolete view pair: {composite_id}")
            handle = self._source_artifact_handles.pop(composite_id, None)
            if handle:
                self._store.release(handle)

            task_key = self._view_task_keys.pop(composite_id, None)
            if task_key:
                self._task_manager.cancel_task(task_key)

            entry = self._view_entries.pop(composite_id, None)
            if entry and entry.handle:
                self._store.release(entry.handle)

            self._pending_updates.pop(composite_id, None)
            timer = self._throttle_timers.pop(composite_id, None)
            if timer:
                timer.cancel()
            self._last_update_time.pop(composite_id, None)

    def get_view_handle(
        self, workpiece_uid: str, step_uid: str
    ) -> Optional[WorkPieceViewArtifactHandle]:
        """Get the view handle for a specific workpiece and step."""
        composite_id = (workpiece_uid, step_uid)
        entry = self._view_entries.get(composite_id)
        return entry.handle if entry else None
