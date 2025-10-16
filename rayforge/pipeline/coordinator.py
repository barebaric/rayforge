"""
Defines the PipelineCoordinator, the central orchestrator for the data
pipeline.

This module contains the PipelineCoordinator class, which acts as a bridge
between the pure data models in the `core` module (Doc, Layer, Step,
WorkPiece) and the execution logic of the pipeline. Its primary responsibility
is to listen for changes in the document and delegate tasks to the
appropriate pipeline stages.
"""

from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING
from blinker import Signal
from contextlib import contextmanager
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.step import Step
from ..core.workpiece import WorkPiece
from ..core.group import Group
from ..core.ops import Ops
from ..core.matrix import Matrix
from .artifact import (
    WorkPieceArtifact,
    BaseArtifactHandle,
    StepArtifactHandle,
    ArtifactCache,
)
from .stage import (
    TimeEstimatorStage,
    WorkpieceGeneratorStage,
    StepGeneratorStage,
    JobGeneratorStage,
)


if TYPE_CHECKING:
    from ..shared.tasker.manager import TaskManager

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    """
    Listens to a Doc model and orchestrates the artifact generation pipeline.

    This class acts as a "conductor" for the data pipeline. It connects to the
    document's signals and delegates invalidation and regeneration tasks to a
    set of specialized pipeline stages. It is the central point of control,
    but it contains no complex generation logic itself.

    Attributes:
        doc (Doc): The document model this coordinator is observing.
        ops_generation_starting (Signal): Fired when generation begins for a
            (Step, WorkPiece) pair.
        ops_chunk_available (Signal): Fired as chunks of Ops become available
            from a background process.
        ops_generation_finished (Signal): Fired when generation is complete
            for a (Step, WorkPiece) pair.
        step_generation_finished (Signal): Fired when a step artifact is
            fully assembled.
        job_generation_finished (Signal): Fired when the final job artifact
            is ready.
        time_estimation_updated (Signal): Fired when a time estimate is
            updated.
        processing_state_changed (Signal): Fired when the busy state of the
            entire pipeline changes.
    """

    def __init__(self, doc: "Doc", task_manager: "TaskManager"):
        """
        Initializes the PipelineCoordinator.

        Args:
            doc: The top-level Doc object to monitor for changes.
            task_manager: The TaskManager instance for background jobs.
        """
        logger.debug(f"{self.__class__.__name__}.__init__ called")
        self._doc: Doc = Doc()
        self._task_manager = task_manager
        self._artifact_cache = ArtifactCache()
        self._pause_count = 0

        # Stages
        self._workpiece_stage = WorkpieceGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._step_stage = StepGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._time_estimator_stage = TimeEstimatorStage(
            self._task_manager, self._artifact_cache
        )
        self._job_stage = JobGeneratorStage(
            self._task_manager, self._artifact_cache
        )

        # Signals for notifying the UI of generation progress
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()
        self.step_generation_finished = Signal()
        self.processing_state_changed = Signal()
        self.time_estimation_updated = Signal()
        self.job_generation_finished = Signal()

        # Connect signals from stages
        self._workpiece_stage.generation_starting.connect(
            self._on_workpiece_generation_starting
        )
        self._workpiece_stage.chunk_available.connect(
            self._on_workpiece_chunk_available
        )
        self._workpiece_stage.generation_finished.connect(
            self._on_workpiece_generation_finished
        )
        self._step_stage.generation_finished.connect(
            self._on_step_generation_finished
        )
        self._time_estimator_stage.estimation_updated.connect(
            self._on_time_estimation_updated
        )
        self._job_stage.generation_finished.connect(
            self._on_job_generation_finished
        )

        self.doc = doc

    def shutdown(self):
        """
        Releases all shared memory resources held in the cache. This must be
        called before application exit to prevent memory leaks.
        """
        logger.info("PipelineCoordinator shutting down...")
        self._artifact_cache.shutdown()
        self._workpiece_stage.shutdown()
        self._step_stage.shutdown()
        self._time_estimator_stage.shutdown()
        self._job_stage.shutdown()
        logger.info("All pipeline resources released.")

    @property
    def doc(self) -> Doc:
        """The document model this coordinator is observing."""
        return self._doc

    @doc.setter
    def doc(self, new_doc: Doc):
        """Sets the document and manages signal connections."""
        if self._doc is new_doc:
            return

        if self._doc:
            self._disconnect_signals()
            self.shutdown()

        self._doc = new_doc
        self._artifact_cache = ArtifactCache()

        # Re-initialize stages with the new, empty artifact cache
        self._workpiece_stage = WorkpieceGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._step_stage = StepGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._time_estimator_stage = TimeEstimatorStage(
            self._task_manager, self._artifact_cache
        )
        self._job_stage = JobGeneratorStage(
            self._task_manager, self._artifact_cache
        )
        self._workpiece_stage.generation_starting.connect(
            self._on_workpiece_generation_starting
        )
        self._workpiece_stage.chunk_available.connect(
            self._on_workpiece_chunk_available
        )
        self._workpiece_stage.generation_finished.connect(
            self._on_workpiece_generation_finished
        )
        self._step_stage.generation_finished.connect(
            self._on_step_generation_finished
        )
        self._time_estimator_stage.estimation_updated.connect(
            self._on_time_estimation_updated
        )
        self._job_stage.generation_finished.connect(
            self._on_job_generation_finished
        )

        if self._doc:
            self._connect_signals()
            self.reconcile_all()

    @property
    def is_busy(self) -> bool:
        """Returns True if any pipeline stage is currently running tasks."""
        return (
            self._workpiece_stage.is_busy
            or self._step_stage.is_busy
            or self._time_estimator_stage.is_busy
            or self._job_stage.is_busy
        )

    def _connect_signals(self):
        """Connects to the document's signals."""
        self.doc.descendant_added.connect(self._on_descendant_added)
        self.doc.descendant_removed.connect(self._on_descendant_removed)
        self.doc.descendant_updated.connect(self._on_descendant_updated)
        self.doc.descendant_transform_changed.connect(
            self._on_descendant_transform_changed
        )
        self.doc.job_assembly_invalidated.connect(
            self._on_job_assembly_invalidated
        )

    def _disconnect_signals(self):
        """Disconnects from the document's signals."""
        self.doc.descendant_added.disconnect(self._on_descendant_added)
        self.doc.descendant_removed.disconnect(self._on_descendant_removed)
        self.doc.descendant_updated.disconnect(self._on_descendant_updated)
        self.doc.descendant_transform_changed.disconnect(
            self._on_descendant_transform_changed
        )
        self.doc.job_assembly_invalidated.disconnect(
            self._on_job_assembly_invalidated
        )

    def pause(self):
        """
        Increments the pause counter. The coordinator is paused if the
        counter is > 0.
        """
        if self._pause_count == 0:
            logger.debug("PipelineCoordinator paused.")
        self._pause_count += 1

    def resume(self):
        """
        Decrements the pause counter. If it reaches 0, the coordinator is
        resumed and reconciles all changes.
        """
        if self._pause_count == 0:
            return
        self._pause_count -= 1
        if self._pause_count == 0:
            logger.debug("PipelineCoordinator resumed.")
            self.reconcile_all()

    @contextmanager
    def paused(self):
        """A context manager to safely pause and resume the coordinator."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    @property
    def is_paused(self) -> bool:
        """Returns True if the coordinator is currently paused."""
        return self._pause_count > 0

    def _find_step_by_uid(self, uid: str) -> Optional[Step]:
        """Finds a step anywhere in the document by its UID."""
        for layer in self.doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    if step.uid == uid:
                        return step
        return None

    def _find_workpiece_by_uid(self, uid: str) -> Optional[WorkPiece]:
        """Finds a workpiece anywhere in the document by its UID."""
        for wp in self.doc.all_workpieces:
            if wp.uid == uid:
                return wp
        return None

    def _on_descendant_added(self, sender, *, origin):
        """Handles the addition of a new model object."""
        if self.is_paused:
            return
        self.reconcile_all()

    def _on_descendant_removed(self, sender, *, origin):
        """Handles the removal of a model object."""
        if self.is_paused:
            return
        self.reconcile_all()

    def _on_descendant_updated(self, sender, *, origin):
        """Handles non-transform updates that require regeneration."""
        if self.is_paused:
            return
        if isinstance(origin, Step):
            self._artifact_cache.invalidate_for_step(origin.uid)
            self.reconcile_all()
        elif isinstance(origin, WorkPiece):
            self.reconcile_all()

    def _on_descendant_transform_changed(self, sender, *, origin):
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
                    self._step_stage.invalidate(step.uid)

        self.reconcile_all()

    def _on_job_assembly_invalidated(self, sender):
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
                        self._step_stage.invalidate(step.uid)
        self.reconcile_all()

    def _on_workpiece_generation_starting(
        self, sender, *, step, workpiece, generation_id
    ):
        """Relays signal from the workpiece stage."""
        was_busy = self.is_busy
        self.ops_generation_starting.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        if not was_busy and self.is_busy:
            self.processing_state_changed.send(self, is_processing=True)

    def _on_workpiece_chunk_available(
        self, sender, *, key, chunk, generation_id
    ):
        """Relays chunk signal, finding the model objects first."""
        step_uid, workpiece_uid = key
        workpiece = self._find_workpiece_by_uid(workpiece_uid)
        step = self._find_step_by_uid(step_uid)
        if workpiece and step:
            self.ops_chunk_available.send(
                step,
                workpiece=workpiece,
                chunk=chunk,
                generation_id=generation_id,
            )

    def _on_workpiece_generation_finished(
        self, sender, *, step, workpiece, generation_id
    ):
        """
        Relays signal and triggers downstream step assembly and time
        estimation.
        """
        was_busy = self.is_busy
        self.ops_generation_finished.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        self._step_stage.mark_stale_and_trigger(step)
        self._time_estimator_stage.generate_estimate(step, workpiece)

        if was_busy and not self.is_busy:
            self.processing_state_changed.send(self, is_processing=False)

    def _on_step_generation_finished(self, sender, *, step, generation_id):
        """Relays signal from the step stage."""
        was_busy = self.is_busy
        self.step_generation_finished.send(step, generation_id=generation_id)
        if was_busy and not self.is_busy:
            self.processing_state_changed.send(self, is_processing=False)

    def _on_time_estimation_updated(self, sender):
        """Relays the signal from the time estimator stage."""
        was_busy = self.is_busy
        self.time_estimation_updated.send(self)
        if was_busy and not self.is_busy:
            self.processing_state_changed.send(self, is_processing=False)

    def _on_job_generation_finished(self, sender, *, handle):
        """Relays signal from the job stage."""
        was_busy = self.is_busy
        self.job_generation_finished.send(self, handle=handle)
        if was_busy and not self.is_busy:
            self.processing_state_changed.send(self, is_processing=False)

    def reconcile_all(self):
        """Synchronizes all stages with the document."""
        if self.is_paused:
            return
        logger.debug(f"{self.__class__.__name__}.reconcile_all called")
        was_busy = self.is_busy
        self._workpiece_stage.reconcile(self.doc)
        self._step_stage.reconcile(self.doc)
        self._time_estimator_stage.reconcile(self.doc)
        self._job_stage.reconcile(self.doc)
        if was_busy and not self.is_busy:
            self.processing_state_changed.send(self, is_processing=False)

    def get_estimated_time(
        self, step: Step, workpiece: WorkPiece
    ) -> Optional[float]:
        """Retrieves a cached time estimate if available."""
        return self._time_estimator_stage.get_estimate(step, workpiece)

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

    def get_step_artifact_handle(
        self, step_uid: str
    ) -> Optional[StepArtifactHandle]:
        """
        Retrieves the handle for a generated step artifact from the cache.
        """
        return self._artifact_cache.get_step_handle(step_uid)

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

    def generate_job(self):
        """Triggers the final job generation process."""
        if self.doc:
            self._job_stage.generate_job(self.doc)
