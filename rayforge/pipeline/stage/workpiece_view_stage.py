from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Tuple
from blinker import Signal
from ...context import get_context
from ..artifact import create_handle_from_dict, BaseArtifactHandle
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
