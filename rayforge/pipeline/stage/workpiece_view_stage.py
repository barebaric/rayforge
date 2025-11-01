from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Tuple, List

from blinker import Signal
from ...context import get_context
from ..artifact import create_handle_from_dict
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifactHandle,
)
from ..artifact import (
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

# A view artifact is uniquely identified by the source artifact being rendered.
ViewKey = str  # source_handle.shm_name
WorkPieceKey = Tuple[str, str]  # (step_uid, workpiece_uid)


class WorkPieceViewPipelineStage(PipelineStage):
    """
    An on-demand stage that generates and MANAGES pre-rendered bitmap
    artifacts (`WorkPieceViewArtifact`) for display in the UI.

    This stage is the single source of truth for view artifacts. It caches
    the final artifact for a (step, workpiece) pair, as well as any
    in-progress "chunk" artifacts for progressive rendering.
    """

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._active_tasks: Dict[ViewKey, "Task"] = {}
        self._last_context_cache: Dict[WorkPieceKey, RenderContext] = {}

        # State management for view artifacts
        self._final_artifacts: Dict[
            WorkPieceKey, WorkPieceViewArtifactHandle
        ] = {}
        self._progressive_artifacts: Dict[
            WorkPieceKey, List[WorkPieceViewArtifactHandle]
        ] = {}

        self.view_artifacts_changed = Signal()

    @property
    def is_busy(self) -> bool:
        return bool(self._active_tasks)

    def reconcile(self, doc: "Doc"):
        pass

    def shutdown(self):
        logger.debug("WorkPieceViewGeneratorStage shutting down.")
        for key in list(self._active_tasks.keys()):
            task = self._active_tasks.pop(key, None)
            if task:
                self._task_manager.cancel_task(task.key)
        self._clear_all_artifacts()

    def invalidate_for_step(self, step_uid: str):
        """
        Invalidates all view artifacts associated with a step and notifies
        listeners to clear their displays.
        """
        logger.debug(
            f"ViewStage: Invalidating ALL view artifacts for step '{step_uid}'"
        )
        keys_to_clear = [
            k for k in self._final_artifacts if k[0] == step_uid
        ] + [k for k in self._progressive_artifacts if k[0] == step_uid]
        for key in set(keys_to_clear):
            workpiece_uid = key[1]
            self._clear_artifacts_for_key(key)
            # Send a notification so the UI element knows to clear
            # its surfaces.
            logger.debug(
                f"ViewStage: Notifying UI of invalidation for "
                f"step='{step_uid}', "
                f"wp='{workpiece_uid}'"
            )
            self.view_artifacts_changed.send(
                self, step_uid=step_uid, workpiece_uid=workpiece_uid
            )

    def invalidate_for_workpiece(self, workpiece_uid: str):
        """Invalidates all view artifacts associated with a workpiece."""
        logger.debug(
            f"ViewStage: Invalidating ALL view artifacts for "
            f"workpiece '{workpiece_uid}'"
        )
        keys_to_clear = [
            k for k in self._final_artifacts if k[1] == workpiece_uid
        ] + [k for k in self._progressive_artifacts if k[1] == workpiece_uid]
        for key in set(keys_to_clear):
            self._clear_artifacts_for_key(key)

    def _clear_artifacts_for_key(self, key: WorkPieceKey):
        """Releases all memory for a specific (step, workpiece) pair."""
        logger.debug(f"ViewStage: Clearing artifacts for key {key}")
        if handle := self._final_artifacts.pop(key, None):
            get_context().artifact_store.release(handle)
        if handles := self._progressive_artifacts.pop(key, None):
            for handle in handles:
                get_context().artifact_store.release(handle)

    def _clear_all_artifacts(self):
        """Releases all cached artifact memory."""
        for handle in self._final_artifacts.values():
            get_context().artifact_store.release(handle)
        self._final_artifacts.clear()
        for handle_list in self._progressive_artifacts.values():
            for handle in handle_list:
                get_context().artifact_store.release(handle)
        self._progressive_artifacts.clear()

    def get_view_artifacts(
        self, step_uid: str, workpiece_uid: str
    ) -> List[WorkPieceViewArtifactHandle]:
        """
        Returns the list of view artifacts to be drawn for a workpiece.
        Returns the single final artifact if available, otherwise returns the
        list of progressive chunks.
        """
        key = (step_uid, workpiece_uid)
        if final_handle := self._final_artifacts.get(key):
            return [final_handle]
        return self._progressive_artifacts.get(key, [])

    def request_view_render(
        self,
        step_uid: str,
        workpiece_uid: str,
        context: RenderContext,
        source_handle: WorkPieceArtifactHandle,
    ):
        """
        Requests an asynchronous render of a workpiece view. This stage
        acquires its own reference to the source_handle to protect it
        for the duration of the background task.
        """
        key: ViewKey = source_handle.shm_name
        wp_key: WorkPieceKey = (step_uid, workpiece_uid)
        logger.debug(
            f"ViewStage: Received view render request for {wp_key}, "
            f"source='{key}', ppm={context.pixels_per_mm}"
        )

        # Check if render is redundant
        last_context = self._last_context_cache.get(wp_key)
        if last_context == context:
            # If the render context is the same, and we already have a final
            # artifact, there's nothing to do.
            if wp_key in self._final_artifacts:
                logger.debug(
                    f"SKIP: View for {wp_key} is already up-to-date with "
                    "identical context."
                )
                return
        else:
            logger.debug(f"Context changed. Old={last_context}, New={context}")

        if key in self._active_tasks:
            logger.debug(
                f"SKIP: View render for source '{key}' is already in-flight."
            )
            return

        # Acquire the handle for the duration of the task. If it fails, abort.
        if not get_context().artifact_store.acquire(source_handle):
            logger.warning(
                f"ABORT: Could not acquire source handle "
                f"'{source_handle.shm_name}' "
                f"for view render. It may have been invalidated."
            )
            return

        self._last_context_cache[wp_key] = context

        # Determine if this source is the "final" one for the workpiece
        final_src_handle = self._artifact_cache.get_workpiece_handle(
            step_uid, workpiece_uid
        )
        final_src_name = (
            final_src_handle.shm_name if final_src_handle else "None"
        )
        is_final_source = source_handle == final_src_handle
        logger.debug(
            f"Source is_final_source: {is_final_source} "
            f"(source={source_handle.shm_name}, "
            f"final={final_src_name})"
        )

        def when_done_callback(task: "Task"):
            self._active_tasks.pop(key, None)
            get_context().artifact_store.release(source_handle)

        logger.debug(f"Creating new render task for key '{key}'.")
        task = self._task_manager.run_process(
            make_workpiece_view_artifact_in_subprocess,
            workpiece_artifact_handle_dict=source_handle.to_dict(),
            render_context_dict=context.to_dict(),
            is_final_source=is_final_source,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            creator_tag="workpiece_view",
            key=key,
            when_done=when_done_callback,
            when_event=self._on_render_event_received,
        )
        self._active_tasks[key] = task

    def _on_render_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        if event_name != "view_artifact_created":
            return

        try:
            handle = create_handle_from_dict(data["handle_dict"])
            if not isinstance(handle, WorkPieceViewArtifactHandle):
                raise TypeError("Expected WorkPieceViewArtifactHandle")

            is_final = data["is_final_source"]
            step_uid = data["step_uid"]
            workpiece_uid = data["workpiece_uid"]
            wp_key: WorkPieceKey = (step_uid, workpiece_uid)
            logger.debug(
                f"Received view_artifact_created event for {wp_key}. "
                f"is_final={is_final}, handle='{handle.shm_name}'"
            )

            get_context().artifact_store.adopt(handle)

            if is_final:
                # This is the final artifact. Clear all progressives.
                if prog_handles := self._progressive_artifacts.pop(wp_key, []):
                    logger.debug(
                        f"Clearing {len(prog_handles)} progressive "
                        f"artifacts for {wp_key}."
                    )
                    for h in prog_handles:
                        get_context().artifact_store.release(h)
                # Store the new final handle, releasing the old one.
                if old_final := self._final_artifacts.pop(wp_key, None):
                    logger.debug(f"Replacing old final artifact for {wp_key}.")
                    get_context().artifact_store.release(old_final)
                self._final_artifacts[wp_key] = handle
            else:
                # This is a progressive chunk. Add it to the list.
                if wp_key not in self._progressive_artifacts:
                    self._progressive_artifacts[wp_key] = []
                self._progressive_artifacts[wp_key].append(handle)
                logger.debug(
                    f"Appended progressive chunk. Total chunks for "
                    f"{wp_key} is {len(self._progressive_artifacts[wp_key])}."
                )

            # Notify listeners that the set of drawable artifacts has changed.
            logger.debug(
                f"Sending view_artifacts_changed signal for {wp_key}."
            )
            self.view_artifacts_changed.send(
                self, step_uid=step_uid, workpiece_uid=workpiece_uid
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to process view_artifact_created: {e}")
