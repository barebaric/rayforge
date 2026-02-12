from __future__ import annotations
import logging
import asyncio
import threading
from typing import (
    Optional,
    TYPE_CHECKING,
    Generator,
    Tuple,
    Union,
    Any,
    Callable,
    Dict,
)
from blinker import Signal
from contextlib import contextmanager
from ..core.doc import Doc
from ..core.group import Group
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.matrix import Matrix
from ..core.ops import Ops
from ..core.step import Step
from ..core.workpiece import WorkPiece
from ..shared.util.colors import ColorSet
from .artifact import (
    ArtifactManager,
    BaseArtifactHandle,
    JobArtifactHandle,
    RenderContext,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    WorkPieceArtifact,
)
from .artifact.key import ArtifactKey
from .artifact.manager import make_composite_key
from .artifact.lifecycle import ArtifactLifecycle
from .stage import (
    JobPipelineStage,
    StepPipelineStage,
    WorkPiecePipelineStage,
    WorkPieceViewPipelineStage,
)
from .stage.job_runner import JobDescription


if TYPE_CHECKING:
    from ..machine.models.machine import Machine
    from ..pipeline.artifact.store import ArtifactStore
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
        _data_generation_id (int): The current data generation ID.
        _view_generation_id (int): The current view generation ID.
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

    RECONCILIATION_DELAY_MS = 200

    def __init__(
        self,
        doc: Optional["Doc"],
        task_manager: "TaskManager",
        artifact_store: "ArtifactStore",
        machine: Optional["Machine"],
    ):
        """
        Initializes the Pipeline.

        Args:
            doc: The top-level Doc object to monitor for changes.
            task_manager: The TaskManager instance for background jobs.
            artifact_store: The ArtifactStore for managing artifacts.
            machine: The Machine instance for laser operations.
        """
        if machine is None:
            raise RuntimeError("Machine is not configured in context")
        logger.debug(f"{self.__class__.__name__}.__init__[{id(self)}] called")
        self._doc: Optional[Doc] = doc
        self._task_manager = task_manager
        self._artifact_store = artifact_store
        self._machine = machine
        self._data_generation_id = 0
        self._view_generation_id = 0
        self._docitem_to_artifact_key: Dict[DocItem, ArtifactKey] = {}
        self._job_key = ArtifactKey.for_job()
        self._pause_count = 0
        self._last_known_busy_state = False
        self._reconciliation_timer: Optional[threading.Timer] = None
        self._step_invalidation_timer: Optional[threading.Timer] = None
        self._pending_step_invalidations: set[str] = set()
        self._current_view_context: Optional[RenderContext] = None

        # Signals for notifying the UI of generation progress
        self.processing_state_changed = Signal()
        self.workpiece_starting = Signal()
        self.workpiece_artifact_ready = Signal()
        self.workpiece_artifact_adopted = Signal()
        self.step_render_ready = Signal()
        self.step_time_updated = Signal()
        self.job_time_updated = Signal()
        self.job_ready = Signal()
        self.workpiece_view_ready = Signal()
        self.workpiece_view_created = Signal()
        self.workpiece_view_updated = Signal()
        self.workpiece_view_generation_starting = Signal()

        # Initialize stages and connect signals ONE time during construction.
        self._initialize_stages_and_connections()

        if self._doc:
            self._connect_signals()
            self.reconcile_data()

    def _initialize_stages_and_connections(self):
        """A new helper method to contain all stage setup logic."""
        logger.debug(f"[{id(self)}] Initializing stages and connections.")
        self._artifact_manager = ArtifactManager(self._artifact_store)
        self._last_known_busy_state = False

        # Stages
        self._workpiece_stage = WorkPiecePipelineStage(
            self._task_manager, self._artifact_manager, self._machine
        )
        self._step_stage = StepPipelineStage(
            self._task_manager, self._artifact_manager, self._machine
        )
        self._job_stage = JobPipelineStage(
            self._task_manager, self._artifact_manager, self._machine
        )
        self._workpiece_view_stage = WorkPieceViewPipelineStage(
            self._task_manager, self._artifact_manager, self._machine
        )

        # Connect signals from stages
        self._workpiece_stage.generation_starting.connect(
            self._on_workpiece_generation_starting
        )
        self._workpiece_stage.visual_chunk_available.connect(
            self._workpiece_view_stage._on_workpiece_chunk_available
        )

        # Connect pipeline signals to stages
        self.workpiece_view_generation_starting.connect(
            self._workpiece_view_stage.on_generation_starting
        )
        self._workpiece_stage.generation_finished.connect(
            self._on_workpiece_generation_finished
        )
        self._workpiece_stage.workpiece_artifact_adopted.connect(
            self._on_workpiece_artifact_adopted
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
        if self._reconciliation_timer:
            self._reconciliation_timer.cancel()
            self._reconciliation_timer = None
        if self._step_invalidation_timer:
            self._step_invalidation_timer.cancel()
            self._step_invalidation_timer = None
        logger.info("Pipeline shutting down...")
        # Shutdown stages first to release retained handles
        self._workpiece_stage.shutdown()
        self._step_stage.shutdown()
        self._job_stage.shutdown()
        self._workpiece_view_stage.shutdown()
        self._artifact_manager.shutdown()
        logger.info("All pipeline resources released.")
        logger.debug(f"[{id(self)}] Pipeline shutdown finished")

    @property
    def artifact_manager(self) -> ArtifactManager:
        """Returns the artifact manager used by this pipeline."""
        return self._artifact_manager

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
            self.reconcile_data()

    def set_view_context(
        self,
        pixels_per_mm: Tuple[float, float],
        show_travel_moves: bool,
        margin_px: int,
        color_set: ColorSet,
    ) -> None:
        """
        Sets the global view context for rendering workpiece views.

        When the view context changes (e.g., zoom level, theme), this method
        triggers re-rendering of all cached workpiece views.

        Args:
            pixels_per_mm: The current zoom level in pixels per mm.
            show_travel_moves: Whether to show travel moves.
            margin_px: The margin in pixels for rendering.
            color_set: The color set for rendering.
        """
        context = RenderContext(
            pixels_per_mm=pixels_per_mm,
            show_travel_moves=show_travel_moves,
            margin_px=margin_px,
            color_set_dict=color_set.to_dict(),
        )

        if self._current_view_context == context:
            logger.debug("set_view_context: Context unchanged, skipping")
            return

        self._current_view_context = context
        self.reconcile_view()

    def _request_step_assembly(self, step_uid: str) -> None:
        """
        Schedules a debounced assembly trigger for the given step.
        """
        self._pending_step_invalidations.add(step_uid)
        if self.is_paused:
            return
        if self._step_invalidation_timer is None:
            self._step_invalidation_timer = threading.Timer(
                0.05, self._on_step_invalidation_timer
            )
            self._step_invalidation_timer.start()

    def _on_step_invalidation_timer(self) -> None:
        self._task_manager.schedule_on_main_thread(
            self._execute_pending_step_assemblies
        )

    def _execute_pending_step_assemblies(self) -> None:
        self._step_invalidation_timer = None
        if not self._doc:
            return
        uids_to_process = list(self._pending_step_invalidations)
        self._pending_step_invalidations.clear()
        for uid in uids_to_process:
            step = self._find_step_by_uid(uid)
            if step:
                self._step_stage.mark_stale_and_trigger(
                    step, self._data_generation_id
                )

    @property
    def is_busy(self) -> bool:
        """
        Returns True if the pipeline is currently processing or has pending
        work.
        """
        if self._reconciliation_timer is not None:
            return True
        if self._step_invalidation_timer is not None:
            return True
        return not self._artifact_manager.is_finished()

    def _check_and_update_processing_state(self) -> None:
        """
        Deferred check of the pipeline's busy state. This is scheduled on
        the main thread to run after the current event chain has completed,
        avoiding race conditions.
        """
        current_busy_state = self.is_busy
        logger.debug(
            f"_check_and_update_processing_state: "
            f"current_busy={current_busy_state}, "
            f"last_known={self._last_known_busy_state}, "
            f"reconciliation_timer={self._reconciliation_timer is not None}, "
            f"ledger_size={len(self._artifact_manager._ledger)}"
        )
        # Note: INITIAL entries are not considered "work in progress" since
        # they may never get processed if there's no associated step.

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
        resumed and schedules a reconciliation of all changes.
        """
        if self._pause_count == 0:
            return
        self._pause_count -= 1
        if self._pause_count == 0:
            logger.debug("Pipeline resumed.")
            if self._pending_step_invalidations:
                self._execute_pending_step_assemblies()
            self._schedule_reconciliation()

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

    def _schedule_reconciliation(self) -> None:
        """Schedules a debounced call to the reconciliation logic."""
        if self.is_paused:
            return

        if self._reconciliation_timer:
            self._reconciliation_timer.cancel()

        if not self._reconciliation_timer:
            # If there was no timer, we are transitioning from idle to busy.
            # Immediately update the state, but only if we're not already busy
            # (to avoid spurious state changes during rapid invalidations).
            self._task_manager.schedule_on_main_thread(
                self._check_and_update_processing_state
            )

        self._reconciliation_timer = threading.Timer(
            self.RECONCILIATION_DELAY_MS / 1000.0,
            self._trigger_main_thread_reconciliation,
        )
        self._reconciliation_timer.start()

    def _trigger_main_thread_reconciliation(self) -> None:
        """
        This is called by the threading.Timer from a background thread.
        It uses the task manager to run the actual logic on the main thread.
        """
        self._task_manager.schedule_on_main_thread(
            self._execute_reconciliation
        )

    def _execute_reconciliation(self) -> None:
        """The debounced method that actually runs reconciliation."""
        self._reconciliation_timer = None
        self.reconcile_data()

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

    def _on_descendant_added(
        self, sender: Any, *, origin: DocItem, parent_of_origin: DocItem
    ) -> None:
        """Handles the addition of a new model object."""
        self._schedule_reconciliation()

    def _on_descendant_removed(
        self, sender: Any, *, origin: DocItem, parent_of_origin: DocItem
    ) -> None:
        """Handles the removal of a model object."""
        if isinstance(origin, WorkPiece):
            # The parent_of_origin is the direct parent (Layer or Group)
            # from which the item was removed. We traverse up the tree from
            # there to robustly find the containing layer.
            layer: Optional[Layer] = None
            current_item: Optional[DocItem] = parent_of_origin
            while current_item:
                if isinstance(current_item, Layer):
                    layer = current_item
                    break
                current_item = current_item.parent

            if layer and layer.workflow:
                logger.debug(
                    f"Workpiece '{origin.name}' removed from layer "
                    f"'{layer.name}'. Invalidating old step artifacts."
                )
                for step in layer.workflow.steps:
                    self._step_stage.invalidate(step.uid)

        self._schedule_reconciliation()

    def _on_descendant_updated(
        self,
        sender: Any,
        *,
        origin: Union[Step, WorkPiece],
        parent_of_origin: DocItem,
    ) -> None:
        """Handles non-transform updates that require regeneration."""
        logger.debug(
            f"_on_descendant_updated called with "
            f"origin={type(origin).__name__}, "
            f"origin.uid={origin.uid}"
        )
        if isinstance(origin, Step):
            logger.debug(
                f"Pipeline: Invalidating workpieces for step {origin.uid}"
            )
            self._workpiece_stage.invalidate_for_step(origin.uid)
            logger.debug(f"Pipeline: Invalidating step {origin.uid}")
            self._step_stage.invalidate(origin.uid)
        elif isinstance(origin, WorkPiece):
            logger.debug(f"Pipeline: Invalidating workpiece {origin.uid}")
            self._workpiece_stage.invalidate_for_workpiece(origin.uid)
            logger.debug(f"Pipeline: Invalidating step {origin.uid}")
            self._step_stage.invalidate(origin.uid)

        self._schedule_reconciliation()

    def _on_descendant_transform_changed(
        self,
        sender: Any,
        *,
        origin: Union[WorkPiece, Group, Layer],
        parent_of_origin: DocItem,
        old_matrix: Optional["Matrix"] = None,
    ) -> None:
        """Handles transform changes by invalidating downstream artifacts."""
        workpieces_to_check = []
        if isinstance(origin, WorkPiece):
            workpieces_to_check.append(origin)
        elif isinstance(origin, (Group, Layer)):
            workpieces_to_check.extend(
                origin.get_descendants(of_type=WorkPiece)
            )

        # Always request step assembly to ensure downstream artifacts are
        # marked stale. The request itself handles the pause state.
        for wp in workpieces_to_check:
            if wp.layer and wp.layer.workflow:
                for step in wp.layer.workflow.steps:
                    self._request_step_assembly(step.uid)

        # Check for size changes which require full workpiece regeneration.
        if isinstance(origin, WorkPiece):
            size_changed = False
            if old_matrix is not None:
                _, _, _, old_sx, old_sy, _ = old_matrix.decompose()
                _, _, _, new_sx, new_sy, _ = origin.matrix.decompose()
                size_changed = (
                    abs(old_sx - new_sx) > 1e-9 or abs(old_sy - new_sy) > 1e-9
                )
            else:
                size_changed = True

            if size_changed and origin.layer and origin.layer.workflow:
                for step in origin.layer.workflow.steps:
                    self._workpiece_stage.invalidate_for_step(step.uid)

        # Schedule a single reconciliation for any transform change.
        # The scheduler itself will handle the pause state.
        self._schedule_reconciliation()

    def _on_job_assembly_invalidated(self, sender: Any) -> None:
        """
        Handles the signal sent when per-step transformers change.
        """
        logger.debug(
            "Per-step transformers changed. Invalidating step artifacts."
        )
        if self.doc:
            for layer in self.doc.layers:
                if layer.workflow:
                    for step in layer.workflow.steps:
                        self._request_step_assembly(step.uid)
        self._schedule_reconciliation()

    def _on_workpiece_generation_starting(
        self,
        sender: WorkPiecePipelineStage,
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

        self.workpiece_view_generation_starting.send(
            step=step,
            workpiece=workpiece,
            generation_id=generation_id,
            view_id=self._view_generation_id,
        )

    def _on_workpiece_generation_finished(
        self,
        sender: WorkPiecePipelineStage,
        *,
        step: Step,
        workpiece: WorkPiece,
        generation_id: int,
        task_status: str = "completed",
    ) -> None:
        """
        Relays signal and triggers downstream step assembly.
        """
        # Only send signal and trigger assembly if task was not canceled
        # (to avoid spurious updates from stale generations)
        if task_status != "canceled":
            self.workpiece_artifact_ready.send(
                step, workpiece=workpiece, generation_id=generation_id
            )
            # Trigger step assembly now that workpiece is ready
            self._request_step_assembly(step.uid)

        # Only check processing state if we're not already busy
        # (to avoid spurious state changes during rapid invalidations)
        # Also skip if reconciliation is in progress to avoid state changes
        # when a canceled task finishes while a new task is being started
        if not self.is_busy and self._reconciliation_timer is None:
            self._task_manager.schedule_on_main_thread(
                self._check_and_update_processing_state
            )

    def _on_workpiece_artifact_adopted(
        self,
        sender: WorkPiecePipelineStage,
        *,
        step_uid: str,
        workpiece_uid: str,
    ) -> None:
        """
        Handles the signal that a workpiece artifact has been adopted.
        Relays the signal to notify workpiece elements.
        Also triggers view rendering if a view context is available.
        """
        key = (step_uid, workpiece_uid)
        logger.debug(
            f"_on_workpiece_artifact_adopted: Workpiece artifact adopted "
            f"for {key}"
        )
        self.workpiece_artifact_adopted.send(
            self, step_uid=step_uid, workpiece_uid=workpiece_uid
        )

        if self._current_view_context is not None:
            logger.debug(
                f"_on_workpiece_artifact_adopted: Requesting view render "
                f"for ({step_uid}, {workpiece_uid})"
            )

            key = ArtifactKey.for_view(workpiece_uid)
            self._workpiece_view_stage.request_view_render(
                key,
                self._current_view_context,
                self._view_generation_id,
                self._data_generation_id,
                step_uid,
            )
        else:
            logger.warning(
                f"_on_workpiece_artifact_adopted: No view context available, "
                f"cannot request view render for ({step_uid}, {workpiece_uid})"
            )

    def _on_step_render_artifact_ready(
        self, sender: StepPipelineStage, *, step: Step
    ) -> None:
        """
        Handles the signal that a step's visual data is ready.
        This now fires the public `step_generation_finished` signal,
        triggering fast UI updates.
        """
        self.step_render_ready.send(self, step=step, generation_id=0)

    def _on_step_task_completed(
        self, sender: StepPipelineStage, *, step: Step, generation_id: int
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
        self, sender: StepPipelineStage, *, step: Step, time: float
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
        sender: JobPipelineStage,
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
        sender: JobPipelineStage,
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
        sender: WorkPieceViewPipelineStage,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
    ):
        """Relays signal that a new view bitmap artifact has been created."""
        self.workpiece_view_created.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _on_workpiece_view_updated(
        self,
        sender: WorkPieceViewPipelineStage,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
    ):
        """Relays signal that a view artifact has been updated."""
        self.workpiece_view_updated.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
        )

    def _on_workpiece_view_ready(
        self,
        sender: WorkPieceViewPipelineStage,
        *,
        step_uid: str,
        workpiece_uid: str,
        handle: BaseArtifactHandle,
    ):
        """Relays signal that a new view bitmap artifact is ready."""
        self.workpiece_view_ready.send(
            self,
            step_uid=step_uid,
            workpiece_uid=workpiece_uid,
            handle=handle,
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

    def reconcile_data(self) -> None:
        """
        Synchronizes data stages with the document by declaring the full
        set of required artifacts for the new data generation.
        """
        if self.is_paused or not self.doc:
            return
        logger.debug(f"{self.__class__.__name__}.reconcile_data called")

        self._data_generation_id += 1
        data_gen_id = self._data_generation_id

        self._declare_data_artifacts(data_gen_id)

        self._register_data_dependencies()

        self.job_time_updated.send(self, total_seconds=None)

        self._workpiece_stage.reconcile(self.doc, data_gen_id)
        self._step_stage.reconcile(self.doc, data_gen_id)

        step_uids = self._get_step_uids(self.doc)
        if not step_uids:
            self._artifact_manager.mark_done(self._job_key, data_gen_id)

        self._update_and_emit_preview_time()
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _declare_data_artifacts(self, gen_id: int) -> None:
        """Declares all data artifacts for a generation."""
        if not self.doc:
            return
        all_keys: set[ArtifactKey] = set()

        for layer in self.doc.layers:
            if layer.workflow and layer.workflow.steps:
                for step in layer.workflow.steps:
                    all_keys.add(ArtifactKey.for_step(step.uid))
                for workpiece in layer.all_workpieces:
                    all_keys.add(ArtifactKey.for_workpiece(workpiece.uid))

        all_keys.add(self._job_key)

        self._artifact_manager.declare_generation(all_keys, gen_id)

    def _register_data_dependencies(self) -> None:
        """Registers dependencies between data artifacts."""
        if not self.doc:
            return
        for layer in self.doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    step_key = ArtifactKey.for_step(step.uid)
                    for workpiece in layer.all_workpieces:
                        wp_key = ArtifactKey.for_workpiece(workpiece.uid)
                        self._artifact_manager.register_dependency(
                            child_key=wp_key, parent_key=step_key
                        )

        step_uids = self._get_step_uids(self.doc)
        for step_uid in step_uids:
            step_key = ArtifactKey.for_step(step_uid)
            self._artifact_manager.register_dependency(
                child_key=step_key, parent_key=self._job_key
            )

    def reconcile_view(self) -> None:
        """
        Synchronizes view stage by triggering re-rendering for
        the current view generation.
        """
        if self.is_paused or not self.doc:
            return
        logger.debug(f"{self.__class__.__name__}.reconcile_view called")

        self._view_generation_id += 1
        view_gen_id = self._view_generation_id

        if not self._current_view_context:
            return

        self._workpiece_view_stage.update_view_context(
            self._current_view_context,
            view_gen_id,
            self._data_generation_id,
        )

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
            step.uid,
            workpiece.uid,
            workpiece.get_world_transform(),
        )

    def _get_or_create_artifact_key(
        self, item: DocItem, group: str
    ) -> ArtifactKey:
        key = self._docitem_to_artifact_key.get(item)
        if key is None:
            key = ArtifactKey(id=item.uid, group=group)
            self._docitem_to_artifact_key[item] = key
        return key

    def get_artifact_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[BaseArtifactHandle]:
        """Retrieves the handle for a generated artifact from the cache."""
        key = ArtifactKey.for_workpiece(workpiece_uid)
        return self._artifact_manager.get_workpiece_handle(
            key, self._data_generation_id
        )

    def get_step_render_artifact_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        """
        Retrieves the handle for a generated step render artifact. This is
        the lightweight artifact intended for UI consumption.
        """
        return self._artifact_manager.get_step_render_handle(step_uid)

    def get_step_ops_artifact_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        """
        Retrieves the handle for a generated step ops artifact. This is
        intended for the job assembly process.
        """
        key = ArtifactKey.for_step(step_uid)
        return self._artifact_manager.get_step_ops_handle(
            key, self._data_generation_id
        )

    def get_scaled_ops(
        self, step_uid: str, workpiece_uid: str, world_transform: Matrix
    ) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache and scales them to the
        final world size.
        """
        return self._workpiece_stage.get_scaled_ops(
            step_uid, workpiece_uid, world_transform, self._data_generation_id
        )

    def get_artifact(
        self, step: Step, workpiece: WorkPiece
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete artifact from the cache on-demand."""
        return self._workpiece_stage.get_artifact(
            step.uid, workpiece.uid, workpiece.size, self._data_generation_id
        )

    def generate_job(self) -> None:
        """Triggers the final job generation process as fire-and-forget."""

        def no_op_callback(
            handle: Optional[JobArtifactHandle], error: Optional[Exception]
        ):
            if error:
                logger.error(f"Fire-and-forget job generation failed: {error}")

        self.generate_job_artifact(when_done=no_op_callback)

    def _validate_job_generation_state(self) -> Optional[RuntimeError]:
        """
        Validates the state before job generation.

        Returns:
            An error if validation fails, None otherwise.
        """
        # Check if a job generation is already pending or running
        composite_key = make_composite_key(
            self._job_key, self._data_generation_id
        )
        entry = self._artifact_manager._get_ledger_entry(composite_key)
        if entry and entry.state == ArtifactLifecycle.PROCESSING:
            msg = "Job generation is already in progress."
            logger.warning(msg)
            return RuntimeError(msg)

        if not self._machine:
            msg = "Cannot generate job: No machine is configured."
            logger.error(msg)
            return RuntimeError(msg)

        return None

    def _get_step_uids(self, doc: Optional["Doc"]) -> list:
        """
        Collects step UIDs from document layers.

        Returns:
            List of step UIDs.
        """
        if not doc:
            return []
        step_uids = []
        for layer in doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                step_uids.append(step.uid)
        return step_uids

    def _create_job_description(
        self,
        step_handles: dict,
        machine: "Machine",
        doc: "Doc",
    ) -> JobDescription:
        """
        Creates a job description from the provided components.

        Returns:
            A JobDescription object.
        """
        return JobDescription(
            step_artifact_handles_by_uid=step_handles,
            machine_dict=machine.to_dict(),
            doc_dict=doc.to_dict(),
        )

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
        if not self.doc:
            when_done(None, RuntimeError("No document is loaded."))
            return

        # 1. Validation
        validation_error = self._validate_job_generation_state()
        if validation_error:
            when_done(None, validation_error)
            return

        # 2. Check if all step artifacts are ready by trying to check them out.
        try:
            dep_handles = self._artifact_manager.checkout_dependencies(
                self._job_key, self._data_generation_id
            )
        except AssertionError:
            # This means some dependencies are not DONE.
            # Trigger generation of missing dependencies.
            self.reconcile_data()
            when_done(
                None,
                RuntimeError(
                    "Job dependencies are not ready. "
                    "Please wait and try again."
                ),
            )
            return

        # 3. Assemble the JobDescription
        step_handles = {}
        for key, handle in dep_handles.items():
            # key is ArtifactKey, so key.id is the step_uid
            step_handles[key.id] = handle.to_dict()

        self._machine.hydrate()
        job_desc = self._create_job_description(
            step_handles, self._machine, self.doc
        )

        # 4. Call the simplified stage method
        self._job_stage.generate_job(
            job_desc,
            on_done=when_done,
            generation_id=self._data_generation_id,
            job_key=self._job_key,
        )

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
        workpiece_uid: str,
        context: RenderContext,
        step_uid: str,
        force: bool = False,
    ):
        """
        Forwards a request to the view generator stage to render a new
        bitmap for a workpiece view.

        Args:
            workpiece_uid: The unique identifier of the workpiece.
            context: The render context to use.
            step_uid: The unique identifier of the step.
            force: If True, force re-rendering even if the context
            appears unchanged.
        """
        key = ArtifactKey.for_view(workpiece_uid)
        self._workpiece_view_stage.request_view_render(
            key,
            context,
            self._view_generation_id,
            self._data_generation_id,
            step_uid,
        )
