from __future__ import annotations
import concurrent.futures
import logging
import asyncio
from typing import (
    Optional,
    TYPE_CHECKING,
    Generator,
    Union,
    Any,
    Callable,
    Dict,
    List,
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
from ..core.stock import StockItem
from ..core.workflow import Workflow
from ..core.workpiece import WorkPiece
from .artifact import (
    ArtifactManager,
    BaseArtifactHandle,
    JobArtifactHandle,
    StepOpsArtifactHandle,
    WorkPieceArtifact,
)
from .artifact.key import ArtifactKey
from .context import GenerationContext
from .dag.scheduler import DagScheduler
from .stage import (
    JobPipelineStage,
    StepPipelineStage,
    WorkPiecePipelineStage,
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
        processing_state_changed (Signal): Fired when the busy state of the
            entire pipeline changes.
        workpiece_starting (Signal): Fired when workpiece generation starts.
        workpiece_artifact_ready (Signal): Fired when a workpiece artifact is
            ready.
        workpiece_artifact_adopted (Signal): Fired when a workpiece artifact
            has been adopted.
        job_time_updated (Signal): Fired when the job time estimate is
            updated.
        job_generation_finished (Signal): Fired when job generation completes
            successfully.
    """

    RECONCILIATION_DELAY_MS = 200
    REMOVAL_DEBOUNCE_DELAY_MS = 50

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
        self._contexts: Dict[int, GenerationContext] = {}
        self._active_context: Optional[GenerationContext] = None
        self._docitem_to_artifact_key: Dict[DocItem, ArtifactKey] = {}
        self._pause_count = 0
        self._last_known_busy_state = False
        self._reconciliation_timer: Optional[concurrent.futures.Future] = None
        self._removal_timer: Optional[concurrent.futures.Future] = None
        self._pending_workpiece_removals: List[tuple] = []
        self._pending_step_removals: List[Step] = []
        self._is_shutting_down = False
        self._data_stale_flag: bool = False
        self._auto_pipeline: bool = True

        # Signals for notifying the UI of generation progress
        self.processing_state_changed = Signal()
        self.workpiece_starting = Signal()
        self.workpiece_artifact_ready = Signal()
        self.workpiece_artifact_adopted = Signal()
        self.step_assembly_starting = Signal()
        self.job_generation_finished = Signal()
        self.job_time_updated = Signal()
        self.visual_chunk_available = Signal()
        self.data_stale = Signal()

        # Initialize stages and connect signals ONE time during construction.
        self._initialize_stages_and_connections()

        if self._doc:
            self._connect_signals()
            if self._auto_pipeline:
                self.reconcile_data()

    def _initialize_stages_and_connections(self):
        """A new helper method to contain all stage setup logic."""
        logger.debug(f"[{id(self)}] Initializing stages and connections.")
        self._last_known_busy_state = False
        self._is_shutting_down = False

        # Initialize artifact manager
        self._artifact_manager = ArtifactManager(self._artifact_store)

        # Initialize DAG scheduler with required dependencies
        self._scheduler = DagScheduler(
            self._task_manager, self._artifact_manager, self._machine
        )

        # Stages - no scheduler reference needed, they emit signals
        self._workpiece_stage = WorkPiecePipelineStage(
            self._task_manager,
            self._artifact_manager,
            self._machine,
        )
        self._step_stage = StepPipelineStage(
            self._task_manager,
            self._artifact_manager,
            self._machine,
        )
        self._job_stage = JobPipelineStage(
            self._task_manager,
            self._artifact_manager,
            self._machine,
        )

        # Inject stages into scheduler (scheduler subscribes to stage signals)
        self._scheduler.set_workpiece_stage(self._workpiece_stage)
        self._scheduler.set_step_stage(self._step_stage)
        self._scheduler.set_job_stage(self._job_stage)

        # Connect signals from stages for pipeline-level handling
        self._workpiece_stage.generation_starting.connect(
            self._on_workpiece_generation_starting
        )
        self._workpiece_stage.generation_finished.connect(
            self._on_workpiece_generation_finished
        )
        self._workpiece_stage.workpiece_artifact_adopted.connect(
            self._on_workpiece_artifact_adopted
        )
        self._workpiece_stage.visual_chunk_available.connect(
            self._on_visual_chunk_available
        )
        self._step_stage.assembly_starting.connect(
            self._on_step_assembly_starting
        )
        self._step_stage.generation_finished.connect(
            self._on_step_task_completed
        )
        self._step_stage.time_estimate_ready.connect(
            self._on_step_time_estimate_ready
        )
        self._job_stage.job_generation_finished.connect(
            self._on_job_generation_finished
        )
        self._job_stage.job_generation_failed.connect(
            self._on_job_generation_failed
        )

    def shutdown(self) -> None:
        """
        Releases all shared memory resources held in the cache. This must be
        called before application exit to prevent memory leaks.
        """
        logger.debug(f"[{id(self)}] Pipeline shutdown called")
        self._is_shutting_down = True
        if self._reconciliation_timer:
            self._reconciliation_timer.cancel()
            self._reconciliation_timer = None
        if self._removal_timer:
            self._removal_timer.cancel()
            self._removal_timer = None
        logger.info("Pipeline shutting down...")

        for task in list(self._task_manager):
            logger.debug(f"Cancelling task '{task.key}' during shutdown")
            self._task_manager.cancel_task(task.key)

        self._workpiece_stage.shutdown()
        self._step_stage.shutdown()
        self._job_stage.shutdown()
        self._artifact_manager.shutdown()
        logger.info("All pipeline resources released.")
        logger.debug(f"[{id(self)}] Pipeline shutdown finished")

    @property
    def artifact_manager(self) -> ArtifactManager:
        """Returns the artifact manager used by this pipeline."""
        return self._artifact_manager

    @property
    def task_manager(self) -> "TaskManager":
        """Returns the task manager used by this pipeline."""
        return self._task_manager

    @property
    def data_generation_id(self) -> int:
        """Returns the current data generation ID."""
        return self._data_generation_id

    @property
    def last_completed_handle(
        self,
    ) -> Optional[JobArtifactHandle]:
        """Returns the last completed job artifact handle, if any."""
        return self._job_stage.last_completed_handle

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
            if self._auto_pipeline:
                self.reconcile_data()

    @property
    def is_busy(self) -> bool:
        """
        Returns True if the pipeline is currently processing or has pending
        work.

        The pipeline is busy if:
        1. A reconciliation timer is pending, OR
        2. A removal timer is pending, OR
        3. The scheduler has pending work (PROCESSING nodes), OR
        4. Any context (active or inactive) has active tasks
        """
        if self._reconciliation_timer is not None:
            return True
        if self._removal_timer is not None:
            return True
        if self._scheduler.has_pending_work():
            return True
        for ctx in self._contexts.values():
            if ctx.has_active_tasks():
                return True
        return False

    def _check_and_update_processing_state(self) -> None:
        """
        Deferred check of the pipeline's busy state. This is scheduled on
        the main thread to run after the current event chain has completed,
        avoiding race conditions.
        """
        if self._is_shutting_down:
            return

        current_busy_state = self.is_busy
        logger.debug(
            f"_check_and_update_processing_state: "
            f"current_busy={current_busy_state}, "
            f"last_known={self._last_known_busy_state}, "
            f"reconciliation_timer={self._reconciliation_timer is not None}, "
            f"graph_nodes={len(self._scheduler.graph.get_all_nodes())}, "
            f"data_gen_id={self._data_generation_id}"
        )

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

    @property
    def is_data_stale(self) -> bool:
        """Returns True if data is stale and needs recalculation."""
        return self._data_stale_flag

    @property
    def auto_pipeline(self) -> bool:
        """Returns True if the pipeline recalculates automatically."""
        return self._auto_pipeline

    @auto_pipeline.setter
    def auto_pipeline(self, value: bool) -> None:
        """Sets whether the pipeline recalculates automatically.

        When switching from manual to auto mode with stale data,
        a reconciliation is triggered immediately.
        """
        if self._auto_pipeline == value:
            return
        self._auto_pipeline = value
        if value and self._data_stale_flag:
            self._data_stale_flag = False
            self.reconcile_data()

    def recalculate(self, force: bool = False) -> None:
        """Triggers a recalculation of the pipeline.

        This is the manual trigger used when ``auto_pipeline`` is disabled.
        It clears the stale flag and runs ``reconcile_data()``.

        Args:
            force: If True, invalidates all existing artifacts first
                   so every node is re-processed.
        """
        self._data_stale_flag = False
        if force and self.doc:
            for layer in self.doc.layers:
                if layer.workflow:
                    for step in layer.workflow.steps:
                        step_key = ArtifactKey.for_step(step.uid)
                        self._artifact_manager.invalidate_all_for_key(step_key)
                        for wp in layer.all_workpieces:
                            wp_key = ArtifactKey.for_workpiece(
                                wp.uid, step.uid
                            )
                            self._artifact_manager.invalidate_all_for_key(
                                wp_key
                            )
                            self._artifact_manager.invalidate_for_workpiece(
                                wp_key
                            )
        self.reconcile_data()

    def _schedule_reconciliation(self) -> None:
        """Schedules a debounced call to the reconciliation logic."""
        if self.is_paused:
            return

        if not self._auto_pipeline:
            if not self._data_stale_flag:
                self._data_stale_flag = True
                self.data_stale.send(self)
            return

        if self._reconciliation_timer:
            self._reconciliation_timer.cancel()

        if not self._reconciliation_timer:
            self._task_manager.schedule_on_main_thread(
                self._check_and_update_processing_state
            )

        self._reconciliation_timer = (
            self._task_manager.schedule_delayed_on_main_thread(
                self.RECONCILIATION_DELAY_MS,
                self._execute_reconciliation,
            )
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

    def _should_invalidate_workpiece_for_transform(
        self,
        workpiece: WorkPiece,
        step: Step,
        size_changed: bool,
    ) -> bool:
        """
        Determine if workpiece artifact should be invalidated for a transform.

        Returns True if:
        - Size changed (always invalidate)
        - Position-sensitive transformers enabled (position may affect output)

        Returns False if only position changed and step is not
        position-sensitive.
        """
        if size_changed:
            logger.debug(
                f"_should_invalidate: size_changed=True for {workpiece.uid}, "
                f"invalidating"
            )
            return True

        if step.is_position_sensitive():
            logger.debug(
                f"_should_invalidate: step is position-sensitive for "
                f"{workpiece.uid}, invalidating"
            )
            return True

        logger.debug(
            f"_should_invalidate: step is not position-sensitive for "
            f"{workpiece.uid}, NOT invalidating"
        )
        return False

    def _invalidate_node(self, key: ArtifactKey) -> None:
        """
        Centralized invalidation using the DAG.

        This method marks a node as dirty in the DAG, which automatically
        cascades to all dependents. It also cancels any running tasks
        and cleans up artifact manager entries.

        Args:
            key: The ArtifactKey of the node to invalidate.
        """
        logger.debug(f"_invalidate_node: Invalidating node {key}")

        task = self._task_manager.get_task(key)
        if task and task.is_running():
            logger.debug(f"_invalidate_node: Canceling running task for {key}")
            self._task_manager.cancel_task(key)

        if key.group == "workpiece":
            self._artifact_manager.invalidate_for_workpiece(key)
        elif key.group == "step":
            self._artifact_manager.invalidate_for_step(key)

        self._scheduler.mark_node_dirty(key)

    def _on_descendant_added(
        self, sender: Any, *, origin: DocItem, parent_of_origin: DocItem
    ) -> None:
        """Handles the addition of a new model object."""
        self._schedule_reconciliation()

    def _on_descendant_removed(
        self, sender: Any, *, origin: DocItem, parent_of_origin: DocItem
    ) -> None:
        """Handles removal of a model object with debounced processing."""
        if isinstance(origin, WorkPiece):
            layer: Optional[Layer] = None
            current_item: Optional[DocItem] = parent_of_origin
            while current_item:
                if isinstance(current_item, Layer):
                    layer = current_item
                    break
                current_item = current_item.parent

            self._pending_workpiece_removals.append((origin, layer))
            self._schedule_removal_processing()

        elif isinstance(origin, Step):
            self._pending_step_removals.append(origin)
            self._schedule_removal_processing()

        self._schedule_reconciliation()

    def _schedule_removal_processing(self) -> None:
        """Schedules a debounced batch processing of pending removals."""
        if self._removal_timer:
            self._removal_timer.cancel()

        self._removal_timer = (
            self._task_manager.schedule_delayed_on_main_thread(
                self.REMOVAL_DEBOUNCE_DELAY_MS,
                self._process_pending_removals,
            )
        )

    def _process_pending_removals(self) -> None:
        """Process all pending workpiece and step removals in a batch."""
        self._removal_timer = None

        workpiece_removals = self._pending_workpiece_removals[:]
        self._pending_workpiece_removals.clear()
        step_removals = self._pending_step_removals[:]
        self._pending_step_removals.clear()

        affected_layers: set[Layer] = set()

        for workpiece, layer in workpiece_removals:
            wp_key = ArtifactKey.for_workpiece(workpiece.uid)
            self._artifact_manager.remove_for_workpiece(wp_key)
            self._scheduler.mark_node_dirty(wp_key)

            if layer:
                affected_layers.add(layer)

        for layer in affected_layers:
            if layer.workflow:
                logger.debug(
                    f"Processing batched removals for layer '{layer.name}'. "
                    f"Invalidating step artifacts."
                )
                for step in layer.workflow.steps:
                    step_key = ArtifactKey.for_step(step.uid)
                    self._invalidate_node(step_key)

        for step in step_removals:
            logger.debug(
                f"Step '{step.name}' removed. Invalidating artifacts."
            )
            step_key = ArtifactKey.for_step(step.uid)

            if self.doc:
                for wp in self.doc.all_workpieces:
                    wp_step_key = ArtifactKey.for_workpiece(wp.uid, step.uid)
                    self._invalidate_node(wp_step_key)

            self._artifact_manager.remove_for_step(step_key)
            self._invalidate_node(step_key)

    def _collect_affected_workpieces(self, origin: DocItem) -> List[WorkPiece]:
        """
        Collects all workpieces affected by a transform change.

        Handles WorkPiece, Group, and Layer cases uniformly by collecting
        either the single workpiece or all descendant workpieces.

        Args:
            origin: The DocItem whose transform changed (WorkPiece, Group,
                    or Layer).

        Returns:
            A list of WorkPiece instances affected by the transform change.
        """
        if isinstance(origin, WorkPiece):
            return [origin]
        elif isinstance(origin, (Group, Layer)):
            return list(origin.get_descendants(of_type=WorkPiece))
        return []

    def _on_descendant_updated(
        self,
        sender: Any,
        *,
        origin: Union[Step, WorkPiece, Workflow],
        parent_of_origin: DocItem,
    ) -> None:
        """
        Handles property changes that require artifact regeneration.

        This includes geometry changes, parameter updates, and other
        non-transform modifications. These changes require full
        re-production of the workpiece artifact.

        Uses DAG-based invalidation with FULL_REPRODUCTION scope:
        - Invalidating a workpiece-step pair automatically cascades to steps
        - Steps cascade to the job
        - This ensures all dependent artifacts are regenerated

        Args:
            sender: The signal sender.
            origin: The Step, WorkPiece, or Workflow that was updated.
            parent_of_origin: The parent of the updated item.
        """
        logger.debug(
            f"_on_descendant_updated called with "
            f"origin={type(origin).__name__}, "
            f"origin.uid={origin.uid}"
        )
        if isinstance(origin, Step):
            step_key = ArtifactKey.for_step(origin.uid)
            for wp in origin.layer.all_workpieces if origin.layer else []:
                wp_step_key = ArtifactKey.for_workpiece(wp.uid, origin.uid)
                self._invalidate_node(wp_step_key)
            self._invalidate_node(step_key)
        elif isinstance(origin, WorkPiece):
            if origin.layer and origin.layer.workflow:
                for step in origin.layer.workflow.steps:
                    wp_step_key = ArtifactKey.for_workpiece(
                        origin.uid, step.uid
                    )
                    self._invalidate_node(wp_step_key)
        elif isinstance(origin, Workflow):
            layer = origin.parent if isinstance(origin.parent, Layer) else None
            if layer:
                for step in origin.steps:
                    step_key = ArtifactKey.for_step(step.uid)
                    self._invalidate_node(step_key)
                    for wp in layer.all_workpieces:
                        wp_step_key = ArtifactKey.for_workpiece(
                            wp.uid, step.uid
                        )
                        self._invalidate_node(wp_step_key)

        self._schedule_reconciliation()

    def _on_stock_transform_changed(self, stock_item: "StockItem") -> None:
        """
        Handles stock item transform changes.

        When a stock item moves, check all workpieces that have the
        boundary filter enabled and invalidate only those whose
        intersection with the stock has changed.
        """
        if not self.doc:
            return

        logger.debug(
            f"_on_stock_transform_changed: stock={stock_item.uid} moved"
        )

        for layer in self.doc.layers:
            if not layer.workflow:
                continue

            for wp in layer.all_workpieces:
                for step in layer.workflow.steps:
                    if not step.is_position_sensitive():
                        continue

                    should_invalidate = (
                        self._should_invalidate_workpiece_for_transform(
                            wp, step, size_changed=False
                        )
                    )
                    logger.debug(
                        f"_on_stock_transform_changed: "
                        f"should_invalidate={should_invalidate} "
                        f"for wp={wp.uid}"
                    )
                    if should_invalidate:
                        wp_step_key = ArtifactKey.for_workpiece(
                            wp.uid, step.uid
                        )
                        self._invalidate_node(wp_step_key)
                    step_key = ArtifactKey.for_step(step.uid)
                    self._invalidate_node(step_key)

        self._schedule_reconciliation()

    def _on_descendant_transform_changed(
        self,
        sender: Any,
        *,
        origin: Union[WorkPiece, Group, Layer, "StockItem"],
        parent_of_origin: DocItem,
        old_matrix: Optional["Matrix"] = None,
    ) -> None:
        """
        Handles transform changes with selective invalidation.

        Different transform changes require different invalidation scopes:
        - Position/rotation only: Use STEP_ONLY scope (skip workpiece
          regeneration since geometry is unchanged)
        - Size change: Use FULL_REPRODUCTION scope (workpiece geometry
          changed, requiring regeneration)
        - Invalidate workpiece if the anything moved if a position-sensitive
          step is present (e.g., crop-to-stock)

        This selective approach optimizes performance by avoiding
        unnecessary workpiece regeneration for pure position/rotation
        changes that don't affect crop-to-stock.

        Args:
            sender: The signal sender.
            origin: The WorkPiece, Group, Layer, or StockItem whose transform
                    changed.
            parent_of_origin: The parent of the transformed item.
            old_matrix: The previous transform matrix, used to detect size
                         changes. Only provided for WorkPiece origins.
        """

        if isinstance(origin, StockItem):
            self._on_stock_transform_changed(origin)
            return

        workpieces_to_check = self._collect_affected_workpieces(origin)

        for wp in workpieces_to_check:
            size_changed = False
            if (
                wp is origin
                and isinstance(origin, WorkPiece)
                and old_matrix is not None
            ):
                _, _, _, old_sx, old_sy, _ = old_matrix.decompose()
                _, _, _, new_sx, new_sy, _ = origin.matrix.decompose()
                size_changed = (
                    abs(old_sx - new_sx) > 1e-9 or abs(old_sy - new_sy) > 1e-9
                )

            if wp.layer and wp.layer.workflow:
                for step in wp.layer.workflow.steps:
                    should_invalidate_wp = (
                        self._should_invalidate_workpiece_for_transform(
                            wp, step, size_changed
                        )
                    )
                    if should_invalidate_wp:
                        wp_step_key = ArtifactKey.for_workpiece(
                            wp.uid, step.uid
                        )
                        self._invalidate_node(wp_step_key)
                    step_key = ArtifactKey.for_step(step.uid)
                    self._invalidate_node(step_key)

        self._schedule_reconciliation()

    def _on_job_assembly_invalidated(self, sender: Any) -> None:
        """
        Handles the signal sent when per-step transformers change.
        Uses DAG-based invalidation for all step artifacts.
        """
        logger.debug(
            "Per-step transformers changed. Invalidating step artifacts."
        )
        if self.doc:
            for layer in self.doc.layers:
                if layer.workflow:
                    for step in layer.workflow.steps:
                        step_key = ArtifactKey.for_step(step.uid)
                        self._invalidate_node(step_key)
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
            self,
            step=step,
            workpiece=workpiece,
            generation_id=generation_id,
        )
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_workpiece_generation_finished(
        self,
        sender: WorkPiecePipelineStage,
        *,
        step: Step,
        workpiece: WorkPiece,
        handle: Optional[BaseArtifactHandle],
        generation_id: int,
        task_status: str = "completed",
    ) -> None:
        """
        Relays signal and triggers downstream step assembly via DAG.
        """
        if self._is_shutting_down:
            return

        if handle is not None:
            self.workpiece_artifact_ready.send(
                self,
                step=step,
                workpiece=workpiece,
                handle=handle,
                generation_id=generation_id,
            )

        if task_status != "canceled":
            self._scheduler.process_graph()

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
        """
        key = (step_uid, workpiece_uid)
        logger.debug(
            f"_on_workpiece_artifact_adopted: Workpiece artifact adopted "
            f"for {key}"
        )
        self.workpiece_artifact_adopted.send(
            self, step_uid=step_uid, workpiece_uid=workpiece_uid
        )

    def _on_visual_chunk_available(
        self,
        sender,
        *,
        key,
        chunk_handle: BaseArtifactHandle,
        generation_id: int,
        step_uid: str,
    ) -> None:
        """
        Relays visual chunk availability from the scheduler.

        The chunk_handle is passed directly to listeners. Listeners that
        want to keep the handle must retain it.
        """
        self.visual_chunk_available.send(
            self,
            key=key,
            chunk_handle=chunk_handle,
            generation_id=generation_id,
            step_uid=step_uid,
        )

    def _on_step_assembly_starting(
        self,
        sender,
        *,
        step: Step,
        workpiece: WorkPiece,
        handle: BaseArtifactHandle,
    ) -> None:
        """
        Relays signal when step assembly starts with valid workpiece deps.

        This is called when a step is about to be assembled and its workpiece
        dependencies already have valid handles (e.g., when adding a new step
        to an existing workpiece).
        """
        self.step_assembly_starting.send(
            self, step=step, workpiece=workpiece, handle=handle
        )

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
        sender,
        *,
        handle: Optional[BaseArtifactHandle],
        task_status: str,
    ) -> None:
        """Relays signal from the scheduler for successful job completion."""
        self.job_generation_finished.send(
            self, handle=handle, task_status=task_status
        )
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def _on_job_generation_failed(
        self,
        sender,
        *,
        error: Optional[Exception],
        task_status: str,
    ) -> None:
        """Relays signal from the scheduler for failed job completion."""
        self._task_manager.schedule_on_main_thread(
            self._check_and_update_processing_state
        )

    def reconcile_data(self) -> None:
        """
        Synchronizes data stages with the document by declaring the full
        set of required artifacts for the new data generation.
        """
        if self.is_paused or not self.doc or self._is_shutting_down:
            return
        logger.debug(
            f"{self.__class__.__name__}.reconcile_data called "
            f"(data_gen_id will be {self._data_generation_id + 1})"
        )

        self._data_generation_id += 1
        data_gen_id = self._data_generation_id

        old_context = self._active_context
        if old_context is not None:
            old_context.mark_superseded()

        ctx = GenerationContext(
            generation_id=data_gen_id,
            release_callback=self._artifact_manager.release_handle,
        )
        self._contexts[data_gen_id] = ctx
        self._active_context = ctx
        self._scheduler.set_context(ctx)

        self._declare_data_artifacts(data_gen_id)

        self._register_data_dependencies()

        # Build the DAG and set scheduler context before stage reconciliation
        self._scheduler.set_doc(self.doc)
        self._scheduler.graph.build_from_doc(
            self.doc, data_gen_id, self._artifact_manager
        )
        self._scheduler.set_generation_id(data_gen_id)
        self._scheduler.sync_graph_with_artifact_manager()

        self.job_time_updated.send(self, total_seconds=None)

        self._scheduler.process_graph()

        self._update_and_emit_preview_time()

        # Prune old generations to keep the ledger clean and is_busy accurate
        processing_gen_ids = self._scheduler.get_processing_generation_ids()
        self._artifact_manager.prune(
            active_data_gen_ids={self._data_generation_id},
            processing_data_gen_ids=processing_gen_ids,
        )

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
                        all_keys.add(
                            ArtifactKey.for_workpiece(workpiece.uid, step.uid)
                        )

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
                        wp_step_key = ArtifactKey.for_workpiece(
                            workpiece.uid, step.uid
                        )
                        self._artifact_manager.register_dependency(
                            child_key=wp_step_key, parent_key=step_key
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
        key = ArtifactKey.for_workpiece(workpiece_uid, step_uid)
        return self._artifact_manager.get_workpiece_handle(
            key, self._data_generation_id
        )

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
        if self._scheduler.is_job_running:
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
            if not layer.all_workpieces:
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

        validation_error = self._validate_job_generation_state()
        if validation_error:
            when_done(None, validation_error)
            return

        step_uids = self._get_step_uids(self.doc)
        if not step_uids:
            logger.warning(
                "generate_job_artifact called with no visible steps. "
                "Job is empty."
            )
            when_done(None, None)
            self.job_generation_finished.send(
                self, handle=None, task_status="completed"
            )
            return

        all_deps_ready = True
        for uid in step_uids:
            step_key = ArtifactKey.for_step(uid)
            handle = self._artifact_manager.get_step_ops_handle(
                step_key, self._data_generation_id
            )
            if handle is None:
                all_deps_ready = False
                break

        if not all_deps_ready:
            when_done(
                None,
                RuntimeError(
                    "Job dependencies are not ready. "
                    "Please wait and try again."
                ),
            )
            return

        job_key = ArtifactKey.for_job()
        self._artifact_manager.declare_generation(
            {job_key}, self._data_generation_id
        )

        self._machine.hydrate()

        self._scheduler.generate_job(
            step_uids=step_uids,
            on_done=when_done,
            job_key=job_key,
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

    def get_existing_job_handle(
        self,
    ) -> Optional[JobArtifactHandle]:
        """
        Returns the most recent job artifact handle if one exists.

        This method checks for existing job artifacts from previous
        generations and returns the handle without triggering a new
        generation. Useful for UI operations that need to display
        existing data without reprocessing.

        Returns:
            The JobArtifactHandle if a valid one exists, None otherwise.
        """
        job_nodes = self._scheduler.graph.get_nodes_by_group("job")
        for node in job_nodes:
            handle = self._artifact_manager.get_job_handle(
                node.key, self._data_generation_id
            )
            if handle is None and self._data_generation_id > 1:
                handle = self._artifact_manager.get_job_handle(
                    node.key, self._data_generation_id - 1
                )
            if handle is not None:
                return handle
        return None
