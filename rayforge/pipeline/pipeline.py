"""
Defines the Pipeline, the central orchestrator for the data
pipeline.

This module contains the Pipeline class, which acts as a bridge
between the pure data models in the `core` module (Doc, Layer, Step,
WorkPiece) and the execution logic of the pipeline. Its primary responsibility
is to listen for changes in the document and delegate tasks to the
appropriate pipeline stages.
"""

from __future__ import annotations
import logging
import asyncio
from typing import (
    Optional,
    TYPE_CHECKING,
    Generator,
    Tuple,
    Union,
    Any,
    Callable,
)
from blinker import Signal
from contextlib import contextmanager
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.step import Step
from ..core.workpiece import WorkPiece
from ..core.group import Group
from ..core.item import DocItem
from ..core.ops import Ops
from ..core.matrix import Matrix
from .artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    JobArtifactHandle,
    ArtifactCache,
    RenderContext,
    WorkPieceArtifactHandle,
)
from .stage import (
    WorkpieceGeneratorStage,
    StepGeneratorStage,
    JobGeneratorStage,
    WorkPieceViewGeneratorStage,
)


if TYPE_CHECKING:
    from ..shared.tasker.manager import TaskManager

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Listens to a Doc model and orchestrates the artifact generation.

    This class acts as a "conductor" for the data pipeline. It connects to the
    document's signals and delegates invalidation and regeneration tasks to a
    set of specialized pipeline stages. It is the central point of control,
    but it contains no complex generation logic itself.

    Attributes:
        doc (Doc): The document model this pipeline is observing.
        ops_generation_starting (Signal): Fired when generation begins for a
            (Step, WorkPiece) pair.
        ops_chunk_available (Signal): Fired as chunks of Ops become available
            from a background process.
        ops_generation_finished (Signal): Fired when generation is complete
            for a (Step, WorkPiece) pair.
        step_generation_finished (Signal): Fired when a step's visual artifact
            is fully assembled and ready for rendering. This fires before
            the full task (e.g., time estimation) is complete.
        job_generation_finished (Signal): Fired when the final job artifact
            is ready.
        time_estimation_updated (Signal): Fired when a time estimate is
            updated.
        processing_state_changed (Signal): Fired when the busy state of the
            entire pipeline changes.
    """

    def __init__(self, doc: Optional["Doc"], task_manager: "TaskManager"):
        """
        Initializes the Pipeline.

        Args:
            doc: The top-level Doc object to monitor for changes.
            task_manager: The TaskManager instance for background jobs.
        """
        logger.debug(f"{self.__class__.__name__}.__init__[{id(self)}] called")
        self._doc: Optional[Doc] = doc
        self._task_manager = task_manager
        self._pause_count = 0
        self._last_known_busy_state = False

        # Signals for notifying the UI of generation progress
        self.processing_state_changed = Signal()
        self.workpiece_starting = Signal()
        self.workpiece_chunk_available = Signal()
        self.workpiece_artifact_ready = Signal()
        self.step_render_ready = Signal()
        self.step_time_updated = Signal()
        self.job_time_updated = Signal()
        self.job_ready = Signal()
        self.workpiece_view_ready = Signal()
        self.workpiece_view_created = Signal()
        self.workpiece_view_updated = Signal()

        # Initialize stages and connect signals ONE time during construction.
        self._initialize_stages_and_connections()

        if self._doc:
            self._connect_signals()
            self.reconcile_all()

    def _initialize_stages_and_connections(self):
        """A new helper method to contain all stage setup logic."""
        logger.debug(f"[{id(self)}] Initializing stages and connections.")
        self._artifact_cache = ArtifactCache()
        self._last_known_busy_state = False

        # Stages
        self._workpiece_stage = WorkpieceGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._step_stage = StepGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._job_stage = JobGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._workpiece_view_stage = WorkPieceViewGeneratorStage(
            self._task_manager, self._artifact_cache
        )

        # Connect signals from stages
        self._workpiece_stage.generation_starting.connect(
            self._on_workpiece_generation_starting
        )
        self._workpiece_stage.workpiece_chunk_available.connect(
            self._on_workpiece_chunk_available
        )
        self._workpiece_stage.generation_finished.connect(
            self._on_workpiece_generation_finished
        )
        self._step_stage.generation_finished.connect(
            self._on_step_task_completed
        )
        self._step_stage.render_artifact_ready.connect(
            self._on_step_render_artifact_ready
        )
        self._step_stage.time_estimate_ready.connect(
            self._on_step_time_estimate_ready
        )
        self._job_stage.generation_finished.connect(
            self._on_job_generation_finished
        )
        self._job_stage.generation_failed.connect(
            self._on_job_generation_failed
        )
        self._workpiece_view_stage.view_artifact_ready.connect(
            self._on_workpiece_view_ready
        )
        self._workpiece_view_stage.view_artifact_created.connect(
            self._on_workpiece_view_created
        )
        self._workpiece_view_stage.view_artifact_updated.connect(
            self._on_workpiece_view_updated
        )
        self._workpiece_view_stage.generation_finished.connect(
            self._on_workpiece_view_generation_finished
        )

    def shutdown(self) -> None:
        """
        Releases all shared memory resources held in the cache. This must be
        called before application exit to prevent memory leaks.
        """
        logger.debug(f"[{id(self)}] Pipeline shutdown called")
        logger.info("Pipeline shutting down...")
        self._artifact_cache.shutdown()
        self._workpiece_stage.shutdown()
        self._step_stage.shutdown()
        self._job_stage.shutdown()
        self._workpiece_view_stage.shutdown()
        logger.info("All pipeline resources released.")
        logger.debug(f"[{id(self)}] Pipeline shutdown finished")

    @property
    def doc(self) -> Optional[Doc]:
        """The document model this pipeline is observing."""
        return self._doc

    @doc.setter
    def doc(self, new_doc: Optional[Doc]):
        """Sets the document and manages signal connections."""
        if self._doc is new_doc:
            return
        logger.debug(f"[{id(self)}] new doc received.")

        # Teardown the old doc and resources cleanly
        if self._doc:
            logger.debug(f"[{id(self)}] Old doc exists, shutting down.")
            self._disconnect_signals()

        # Shut down the *stages* before replacing them.
        self.shutdown()

        # Set the new doc and re-initialize everything from scratch
        self._doc = new_doc
        logger.debug(f"[{id(self)}] Re-initializing stages.")
        self._initialize_stages_and_connections()  # Re-use the helper

        if self._doc:
            self._connect_signals()
            self.reconcile_all()

    @property
    def is_busy(self) -> bool:
        """Returns True if any pipeline stage is currently running tasks."""
        return (
            self._workpiece_stage.is_busy
            or self._step_stage.is_busy
            or self._job_stage.is_busy
            or self._workpiece_view_stage.is_busy
        )

    def _check_and_update_processing_state(self) -> None:
        """
        Deferred check of the pipeline's busy state. This is scheduled on
        the main thread to run after the current event chain has completed,
        avoiding race conditions.
        """
        current_busy_state = self.is_busy
        if self._last_known_busy_state != current_busy_state:
            self.processing_state_changed.send(
                self, is_processing=current_busy_state
            )
            self._last_known_busy_state = current_busy_state

    def _connect_signals(self) -> None:
        """Connects to the document's signals."""
        if not self.doc:
            return
        self.doc.descendant_added.connect(self._on_descendant_added)
        self.doc.descendant_removed.connect(self._on_descendant_removed)
        self.doc.descendant_updated.connect(self._on_descendant_updated)
        self.doc.descendant_transform_changed.connect(
            self._on_descendant_transform_changed
        )
        self.doc.job_assembly_invalidated.connect(
            self._on_job_assembly_invalidated
        )

    def _disconnect_signals(self) -> None:
        """Disconnects from the document's signals."""
        if not self.doc:
            return
        self.doc.descendant_added.disconnect(self._on_descendant_added)
        self.doc.descendant_removed.disconnect(self._on_descendant_removed)
        self.doc.descendant_updated.disconnect(self._on_descendant_updated)
        self.doc.descendant_transform_changed.disconnect(
            self._on_descendant_transform_changed
        )
        self.doc.job_assembly_invalidated.disconnect(
            self._on_job_assembly_invalidated
        )

    def pause(self) -> None:
        """
        Increments the pause counter. The pipeline is paused if the
        counter is > 0.
        """
        if self._pause_count == 0:
            logger.debug("Pipeline paused.")
        self._pause_count += 1

    def resume(self) -> None:
        """
        Decrements the pause counter. If it reaches 0, the pipeline is
        resumed and reconciles all changes.
        """
        if self._pause_count == 0:
            return
        self._pause_count -= 1
        if self._pause_count == 0:
            logger.debug("Pipeline resumed.")
            self.reconcile_all()

    @contextmanager
    def paused(self) -> Generator[None, None, None]:
        """A context manager to safely pause and resume the pipeline."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    @property
    def is_paused(self) -> bool:
        """Returns True if the pipeline is currently paused."""
        return self._pause_count > 0

    def _find_step_by_uid(self, uid: str) -> Optional[Step]:
        """Finds a step anywhere in the document by its UID."""
        if not self.doc:
            return None
        for layer in self.doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    if step.uid == uid:
                        return step
        return None

    def _find_workpiece_by_uid(self, uid: str) -> Optional[WorkPiece]:
        """Finds a workpiece anywhere in the document by its UID."""
        if not self.doc:
            return None
        for wp in self.doc.all_workpieces:
            if wp.uid == uid:
                return wp
        return None

    def _on_descendant_added(self, sender: Any, *, origin: DocItem) -> None:
        """Handles the addition of a new model object."""
        if self.is_paused:
            return
        self.reconcile_all()

    def _on_descendant_removed(self, sender: Any, *, origin: DocItem) -> None:
        """Handles the removal of a model object."""
        if self.is_paused:
            return
        self.reconcile_all()

    def _on_descendant_updated(
        self, sender: Any, *, origin: Union[Step, WorkPiece]
    ) -> None:
        """Handles non-transform updates that require regeneration."""
        if self.is_paused:
            return
        if isinstance(origin, Step):
            # A fundamental property of a Step has changed. This invalidates
            # all artifacts that depend on it, starting from the bottom.
            self._workpiece_stage.invalidate_for_step(origin.uid)
            self._step_stage.invalidate(origin.uid)
            self.reconcile_all()
        elif isinstance(origin, WorkPiece):
            self._workpiece_stage.invalidate_for_workpiece(origin.uid)
            self.reconcile_all()

    def _on_descendant_transform_changed(
        self, sender: Any, *, origin: Union[WorkPiece, Group, Layer]
    ) -> None:
        """Handles transform changes by invalidating downstream artifacts."""
        if self.is_paused:
            return

        workpieces_to_check = []
        if isinstance(origin, WorkPiece):
            workpieces_to_check.append(origin)
        elif isinstance(origin, (Group, Layer)):
            workpieces_to_check.extend(
                origin.get_descendants(of_type=WorkPiece)
            )

        for wp in workpieces_to_check:
            self._workpiece_stage.on_workpiece_transform_changed(wp)
            if wp.layer and wp.layer.workflow:
                for step in wp.layer.workflow.steps:
                    self._step_stage.mark_stale_and_trigger(step)

        self.reconcile_all()

    def _on_job_assembly_invalidated(self, sender: Any) -> None:
        """
        Handles the signal sent when per-step transformers change.
        """
        if self.is_paused:
            return
        logger.debug(
            "Per-step transformers changed. Invalidating step artifacts."
        )
        if self.doc:
            for layer in self.doc.layers:
                if layer.workflow:
                    for step in layer.workflow.steps:
                        self._step_stage.mark_stale_and_trigger(step)
        self.reconcile_all()

    def _on_workpiece_generation_starting(
        self,
        sender: WorkpieceGeneratorStage,
        *,
        step: Step,
        workpiece: WorkPiece,
        generation_id: int,
    ) -> None:
        """Relays signal from the workpiece stage."""
        self.workpiece_starting.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_workpiece_chunk_available(
        self,
        sender: WorkpieceGeneratorStage,
        *,
        key: Tuple[str, str],
        chunk_handle: BaseArtifactHandle,
        generation_id: int,
    ) -> None:
        """Relays chunk signal, finding the model objects first."""
        step_uid, workpiece_uid = key
        workpiece = self._find_workpiece_by_uid(workpiece_uid)
        step = self._find_step_by_uid(step_uid)
        if workpiece and step:
            self.workpiece_chunk_available.send(
                step,
                key=key,
                chunk_handle=chunk_handle,
                generation_id=generation_id,
            )

    def _on_workpiece_generation_finished(
        self,
        sender: WorkpieceGeneratorStage,
        *,
        step: Step,
        workpiece: WorkPiece,
        generation_id: int,
    ) -> None:
        """
        Relays signal and triggers downstream step assembly.
        """
        self.workpiece_artifact_ready.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        self._step_stage.mark_stale_and_trigger(step)
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_step_render_artifact_ready(
        self, sender: StepGeneratorStage, *, step: Step
    ) -> None:
        """
        Handles the signal that a step's visual data is ready.
        This now fires the public `step_generation_finished` signal,
        triggering fast UI updates.
        """
        self.step_render_ready.send(self, step=step, generation_id=0)

    def _on_step_task_completed(
        self, sender: StepGeneratorStage, *, step: Step, generation_id: int
    ) -> None:
        """
        Handles the signal that the entire step task (including time) is done.
        This is now only used for internal state updates, like checking the
        pipeline's busy state.
        """
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_step_time_estimate_ready(
        self, sender: StepGeneratorStage, *, step: Step, time: float
    ) -> None:
        """Handles the new, accurate time estimate from the step stage."""
        self.step_time_updated.send(self)
        self._task_manager.schedule_on_main_thread(
            self._update_and_emit_preview_time
        )
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _update_and_emit_preview_time(self) -> None:
        """
        Calculates the total estimated preview time by summing all valid
        per-step estimates and emits the preview_time_updated signal.
        """
        if not self.doc:
            return

        total_time = 0.0
        is_calculating = False

        for layer in self.doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                # Use the new, accurate, per-step time source
                estimate = self._step_stage.get_estimate(step.uid)

                if estimate is None:
                    # A value of None means it's pending calculation
                    is_calculating = True
                elif estimate > 0:  # -1 indicates an error, 0 is valid
                    total_time += estimate

        if is_calculating:
            # Send a special signal to indicate calculation is in progress
            self.job_time_updated.send(self, total_seconds=None)
        else:
            self.job_time_updated.send(self, total_seconds=total_time)

    def _on_job_generation_finished(
        self,
        sender: JobGeneratorStage,
        *,
        handle: Optional[BaseArtifactHandle],
        task_status: str,
    ) -> None:
        """Relays signal from the job stage for successful completion."""
        self.job_ready.send(self, handle=handle)
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_job_generation_failed(
        self,
        sender: JobGeneratorStage,
        *,
        error: Optional[Exception],
        task_status: str,
    ) -> None:
        """Relays signal from the job stage for failed completion."""
        # For now, a failure is treated like a completion with no handle.
        # Future UI could use the error for notifications.
        self.job_ready.send(self, handle=None)
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_workpiece_view_created(
        self,
        sender: WorkPieceViewGeneratorStage,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
        source_shm_name: str,
        generation_id: int,
    ):
        """Relays signal that a new view bitmap artifact has been created."""
        self.workpiece_view_created.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
            source_shm_name=source_shm_name,
            generation_id=generation_id,
        )

    def _on_workpiece_view_updated(
        self,
        sender: WorkPieceViewGeneratorStage,
        *,
        step_uid: str,
        workpiece_uid: str,
    ):
        """Relays signal that a view artifact has been updated."""
        self.workpiece_view_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
        )

    def _on_workpiece_view_ready(
        self,
        sender: WorkPieceViewGeneratorStage,
        *,
        step_uid: str,
        workpiece_uid: str,
        source_shm_name: str,
        generation_id: int,
    ):
        """Relays signal that a new view bitmap artifact is ready."""
        self.workpiece_view_ready.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            source_shm_name=source_shm_name,
            generation_id=generation_id,
        )

    def _on_workpiece_view_generation_finished(
        self, sender, *, key: Tuple[str, str]
    ):
        """
        Handles completion of a view render task to update the overall busy
        state.
        """
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def reconcile_all(self) -> None:
        """Synchronizes all stages with the document."""
        if self.is_paused or not self.doc:
            return
        logger.debug(f"{self.__class__.__name__}.reconcile_all called")

        # Immediately notify UI that estimates are now stale and recalculating.
        self.job_time_updated.send(self, total_seconds=None)

        self._workpiece_stage.reconcile(self.doc)
        self._step_stage.reconcile(self.doc)
        self._job_stage.reconcile(self.doc)
        self._update_and_emit_preview_time()
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def get_estimated_time(
        self, step: Step, workpiece: WorkPiece
    ) -> Optional[float]:
        """
        Retrieves a cached time estimate.
        NOTE: As part of the time estimation refactor, we no longer
        provide per-workpiece estimates, only per-step. This method will
        return None.
        """
        return None

    def get_ops(self, step: Step, workpiece: WorkPiece) -> Optional[Ops]:
        """
        [Compatibility Method] Retrieves ops by wrapping get_scaled_ops.
        """
        return self.get_scaled_ops(
            step.uid, workpiece.uid, workpiece.get_world_transform()
        )

    def get_artifact_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[BaseArtifactHandle]:
        """Retrieves the handle for a generated artifact from the cache."""
        return self._artifact_cache.get_workpiece_handle(
            step_uid, workpiece_uid
        )

    def get_step_render_artifact_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        """
        Retrieves the handle for a generated step render artifact. This is
        the lightweight artifact intended for UI consumption.
        """
        return self._artifact_cache.get_step_render_handle(step_uid)

    def get_step_ops_artifact_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        """
        Retrieves the handle for a generated step ops artifact. This is
        intended for the job assembly process.
        """
        return self._artifact_cache.get_step_ops_handle(step_uid)

    def get_scaled_ops(
        self, step_uid: str, workpiece_uid: str, world_transform: Matrix
    ) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache and scales them to the
        final world size.
        """
        return self._workpiece_stage.get_scaled_ops(
            step_uid, workpiece_uid, world_transform
        )

    def get_artifact(
        self, step: Step, workpiece: WorkPiece
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete artifact from the cache on-demand."""
        return self._workpiece_stage.get_artifact(
            step.uid, workpiece.uid, workpiece.size
        )

    def generate_job(self) -> None:
        """Triggers the final job generation process."""
        if self.doc:
            # This method now just calls the stage's method without a callback
            self._job_stage.generate_job(self.doc)

    def generate_job_artifact(
        self,
        when_done: Callable[
            [Optional[JobArtifactHandle], Optional[Exception]], None
        ],
    ):
        """
        Asynchronously generates the final job artifact and calls a
        callback with the result. This is the correct public API for
        requesting a job artifact, abstracting away the underlying stages.

        Args:
            when_done: A callback executed upon completion. It receives
                       an ArtifactHandle on success, or (None, error) on
                       failure.
        """
        # The logic is now much simpler. We just pass the callback along.
        if self.doc:
            self._job_stage.generate_job(self.doc, on_done=when_done)
        else:
            when_done(None, RuntimeError("No document is loaded."))

    async def generate_job_artifact_async(
        self,
    ) -> Optional[JobArtifactHandle]:
        """
        Asynchronously generates and returns the final job artifact.
        This awaitable method is the preferred way to get a job artifact
        in an async context.

        Returns:
            The JobArtifactHandle on success, or None if the job was empty.

        Raises:
            RuntimeError: If job generation is already in progress.
            Exception: Propagates exceptions that occur during generation.
        """
        # This method requires no changes, as it builds on top of the
        # now-corrected generate_job_artifact method.
        logger.debug(f"[{id(self)}] Starting asynchronous job generation.")
        future = asyncio.get_running_loop().create_future()

        def _when_done_callback(
            handle: Optional[JobArtifactHandle], error: Optional[Exception]
        ):
            if not future.done():
                if error:
                    future.set_exception(error)
                else:
                    future.set_result(handle)

        self.generate_job_artifact(when_done=_when_done_callback)
        result = await future
        logger.debug(f"[{id(self)}] Await returned with result: {result}.")
        return result

    def request_view_render(
        self,
        step_uid: str,
        workpiece_uid: str,
        context: RenderContext,
        source_handle: WorkPieceArtifactHandle,
        generation_id: int,
    ):
        """
        Forwards a request to the view generator stage to render a new
        bitmap for a workpiece view.
        """
        self._workpiece_view_stage.request_view_render(
            step_uid, workpiece_uid, context, source_handle, generation_id
        )
