from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Tuple, Optional

from blinker import Signal
from ...context import get_context
from ..artifact import create_handle_from_dict
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifactHandle,
)
from ..artifact import (
    BaseArtifactHandle,
    WorkPieceArtifactHandle,
)
from .base import PipelineStage
from .workpiece_view_runner import make_workpiece_view_artifact_in_subprocess

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache


logger = logging.getLogger(__name__)

# A view artifact is uniquely identified by the step, workpiece, AND
# the specific source artifact being rendered (as chunks have different
# handles).
ViewKey = Tuple[str, str, str, int]  # (step_uid, wp_uid, shm_name, gen_id)


class WorkPieceViewPipelineStage(PipelineStage):
    """
    An on-demand stage that generates pre-rendered bitmap artifacts
    (`WorkPieceViewArtifact`) for display in the UI.
    """

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        # Store task, the source handle it's working on, and the final
        # view handle
        self._active_tasks: Dict[
            ViewKey,
            Tuple[
                "Task",
                BaseArtifactHandle,
                Optional[WorkPieceViewArtifactHandle],
            ],
        ] = {}
        # The last_context_cache is less critical now but can stay for now.
        # Its keying should ideally also be updated if we want to be strict.
        self._last_context_cache: Dict[
            Tuple[str, str], Tuple[RenderContext, str]
        ] = {}
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
        """Cancels any active rendering tasks and cleans up temp artifacts."""
        logger.debug("WorkPieceViewGeneratorStage shutting down.")
        for key in list(self._active_tasks.keys()):
            task_tuple = self._active_tasks.pop(key, None)
            if task_tuple:
                task, source_handle, view_handle = task_tuple
                self._task_manager.cancel_task(task.key)
                # The source handle was passed to us, we are responsible for it
                get_context().artifact_store.release(source_handle)
                # Also release the view handle if it was created
                if view_handle:
                    get_context().artifact_store.release(view_handle)

    def request_view_render(
        self,
        step_uid: str,
        workpiece_uid: str,
        context: RenderContext,
        source_handle: WorkPieceArtifactHandle,
        generation_id: int,
    ):
        """
        Requests an asynchronous render of a workpiece view for a specific
        step. This stage now takes ownership of the source_handle and is
        responsible for releasing it when done.
        """
        key: ViewKey = (
            step_uid,
            workpiece_uid,
            source_handle.shm_name,
            generation_id,
        )
        simple_key = (step_uid, workpiece_uid)

        # The context check can remain on the simple key for now to prevent
        # re-rendering the same final artifact repeatedly.
        full_context_tuple = (context, source_handle.shm_name)
        if self._last_context_cache.get(simple_key) == full_context_tuple:
            logger.debug(f"View for {simple_key} is already up-to-date.")
            # We decided not to render, so we must release the handle.
            get_context().artifact_store.release(source_handle)
            return

        if key in self._active_tasks:
            logger.debug(
                f"View render for specific chunk {key} is already in-flight. "
                "Ignoring duplicate request."
            )
            # We decided not to render, so we must release the handle.
            get_context().artifact_store.release(source_handle)
            return

        # Only update the last context for the *final* artifact render,
        # not for every chunk. We can identify the final one because its source
        # handle will be in the artifact cache. Chunks are transient.
        if (
            self._artifact_cache.get_workpiece_handle(step_uid, workpiece_uid)
            == source_handle
        ):
            self._last_context_cache[simple_key] = full_context_tuple

        def when_done_callback(task: "Task"):
            self._on_render_complete(task, key)

        task = self._task_manager.run_process(
            make_workpiece_view_artifact_in_subprocess,
            # The worker receives the handle to the original chunk/artifact.
            workpiece_artifact_handle_dict=source_handle.to_dict(),
            render_context_dict=context.to_dict(),
            generation_id=generation_id,
            creator_tag="workpiece_view",
            key=key,
            when_done=when_done_callback,
            when_event=self._on_render_event_received,
        )
        # Track active task, the source handle (for later release),
        # and a slot for the view handle.
        self._active_tasks[key] = (task, source_handle, None)

    def _on_render_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles progressive rendering events from the worker process."""
        key = task.key
        step_uid, workpiece_uid, source_shm_name, _ = key

        if event_name == "view_artifact_created":
            try:
                handle_dict = data["handle_dict"]
                generation_id = data["generation_id"]
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, WorkPieceViewArtifactHandle):
                    raise TypeError("Expected WorkPieceViewArtifactHandle")

                get_context().artifact_store.adopt(handle)

                # Store the new view handle against the active task
                if key in self._active_tasks:
                    task_obj, source_handle, _ = self._active_tasks[key]
                    self._active_tasks[key] = (task_obj, source_handle, handle)

                self.view_artifact_created.send(
                    self,
                    step_uid=step_uid,
                    workpiece_uid=workpiece_uid,
                    handle=handle,
                    source_shm_name=source_shm_name,
                    generation_id=generation_id,
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.error(f"Failed to process view_artifact_created: {e}")

        elif event_name == "view_artifact_updated":
            self.view_artifact_updated.send(
                self, step_uid=step_uid, workpiece_uid=workpiece_uid
            )

    def _on_render_complete(self, task: "Task", key: ViewKey):
        """
        Callback for when a rendering task finishes. It signals completion
        and cleans up the original source artifact handle it was given.
        """
        task_tuple = self._active_tasks.pop(key, None)

        step_uid, workpiece_uid, source_shm_name, generation_id = key

        # This signal is now informational.
        if task.get_status() == "completed":
            self.view_artifact_ready.send(
                self,
                step_uid=step_uid,
                workpiece_uid=workpiece_uid,
                source_shm_name=source_shm_name,
                generation_id=generation_id,
            )
        elif task.get_status() != "canceled":
            logger.error(
                f"View render for {key} failed with status: "
                f"{task.get_status()}"
            )

        self.generation_finished.send(self, key=(step_uid, workpiece_uid))

        if task_tuple:
            _task_obj, source_handle, view_handle = task_tuple
            # We are done with the source artifact (chunk or final),
            # release it.
            get_context().artifact_store.release(source_handle)
            # Also release the view handle if it was created
            if view_handle:
                get_context().artifact_store.release(view_handle)
            logger.debug(f"Released source artifact for view render {key}")
