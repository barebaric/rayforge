"""
Defines the Layer class, a central component for organizing and processing
workpieces within a document.
"""

from __future__ import annotations
import uuid
import logging
from typing import List, TYPE_CHECKING, Dict, Tuple, Optional
from copy import deepcopy
from blinker import Signal

from ..config import task_mgr
from ..tasker.task import Task
from .step import Step
from .ops import Ops
from .workflow import Workflow

if TYPE_CHECKING:
    from .workpiece import WorkPiece
    from .doc import Doc

logger = logging.getLogger(__name__)


class Layer:
    """
    Represents a group of workpieces processed by a single workflow.

    A Layer acts as a container for `WorkPiece` objects and owns a
    `Workflow`. It is responsible for triggering, managing, and caching the
    generation of machine operations (`Ops`) for each workpiece based on the
    steps in its workflow.
    """

    # Type alias for the structure of the operations cache.
    OpsCacheType = Dict[
        Tuple[str, str], Tuple[Optional[Ops], Optional[Tuple[int, int]]]
    ]

    def __init__(self, doc: "Doc", name: str):
        """Initializes a Layer instance.

        Args:
            doc: The parent document object.
            name: The user-facing name of the layer.
        """
        self.uid: str = str(uuid.uuid4())
        self.doc: "Doc" = doc
        self.name: str = name
        self.workpieces: List[WorkPiece] = []
        # Reference for static analysis tools to detect class relations.
        self._workpiece_ref_for_pyreverse: WorkPiece

        self.workflow: Workflow = Workflow(self, f"{name} Workflow")
        # Reference for static analysis tools to detect class relations.
        self._workflow_ref_for_pyreverse: Workflow

        self.visible: bool = True

        # Cache for generated operations.
        # Key: (step_uid, workpiece_uid)
        # Value: (Ops object, pixel_dimensions_tuple)
        self._ops_cache: Layer.OpsCacheType = {}

        # Tracks the latest generation request ID for a given
        # (step, workpiece) pair to prevent race conditions from stale
        # async results. Key: (step_uid, workpiece_uid), Value: int
        self._generation_id_map: Dict[Tuple[str, str], int] = {}

        # Signals for notifying other parts of the application of changes.
        self.changed = Signal()
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()

        # Connect to signals from child objects.
        self.workflow.changed.connect(self._on_workflow_changed)

    @property
    def active(self) -> bool:
        """
        Returns True if this layer is the currently active layer in the
        document.
        """
        # A layer is active if it is the one currently designated as active
        # in its parent document.
        return self.doc.active_layer is self

    def _on_workflow_changed(
        self, sender, step: Optional[Step] = None, **kwargs
    ):
        """
        Handles the 'changed' signal from the Workflow.

        If a specific step is provided, this indicates that only that step's
        parameters were modified, so only its operations are regenerated.
        If no step is provided, the workflow's structure (e.g., step order,
        add/remove) has changed, triggering a full regeneration for the layer.
        """
        if step:
            logger.debug(
                f"Layer '{self.name}': Workplan changed for specific step "
                f"'{step.name}'. Updating ops for that step only."
            )
            self._update_ops_for_step(step)
        else:
            self._update_ops_for_all_workpieces()
        self.changed.send(self)

    def _on_step_ops_generation_starting(self, step: Step, **kwargs):
        """Bubbles up the signal from a step to this layer."""
        self.ops_generation_starting.send(self, step=step, **kwargs)

    def _on_step_ops_chunk_available(self, step: Step, **kwargs):
        """Bubbles up a chunk availability signal from a step."""
        logger.debug(
            f"Layer '{self.name}': Received ops_chunk_available from step "
            f"'{step.name}'. Bubbling up."
        )
        self.ops_chunk_available.send(self, step=step, **kwargs)

    def _on_step_ops_generation_finished(self, step: Step, **kwargs):
        """Bubbles up the signal from a step to this layer."""
        self.ops_generation_finished.send(self, step=step, **kwargs)

    def _on_workpiece_size_changed(self, workpiece: WorkPiece, **kwargs):
        """Handles workpiece size changes by regenerating its ops."""
        self._update_ops_for_workpiece(workpiece)

    def set_name(self, name: str):
        """Sets the name of the layer.

        Args:
            name: The new name for the layer.
        """
        if self.name == name:
            return
        self.name = name
        self.workflow.name = f"{name} Workflow"
        self.changed.send(self)

    def set_visible(self, visible: bool):
        """Sets the visibility of the layer.

        Args:
            visible: The new visibility state.
        """
        if self.visible == visible:
            return
        self.visible = visible
        self.changed.send(self)

    def add_workpiece(self, workpiece: "WorkPiece"):
        """Adds a single workpiece to the layer.

        Args:
            workpiece: The workpiece to add.
        """
        if workpiece not in self.workpieces:
            workpiece.layer = self
            self.workpieces.append(workpiece)
            workpiece.size_changed.connect(self._on_workpiece_size_changed)
            self._update_ops_for_workpiece(workpiece)
            self.changed.send(self)

    def remove_workpiece(self, workpiece: "WorkPiece"):
        """Removes a single workpiece from the layer.

        Args:
            workpiece: The workpiece to remove.
        """
        if workpiece in self.workpieces:
            workpiece.layer = None
            try:
                workpiece.size_changed.disconnect(
                    self._on_workpiece_size_changed
                )
            except TypeError:
                # This can happen if the signal was already disconnected.
                pass
            self.workpieces.remove(workpiece)
            self._cleanup_workpiece_ops(workpiece)
            self.changed.send(self)

    def set_workpieces(self, workpieces: List["WorkPiece"]):
        """
        Sets the layer's workpieces to a new list.

        This method efficiently updates workpieces by comparing the new
        list with the old one, handling connections and cache cleanup.

        Args:
            workpieces: A list of WorkPiece objects to associate with
                this layer.
        """
        old_set = set(self.workpieces)
        new_set = set(workpieces)

        # Disconnect and clean up workpieces that are being removed.
        for wp in old_set - new_set:
            wp.layer = None
            try:
                wp.size_changed.disconnect(self._on_workpiece_size_changed)
            except TypeError:
                pass
            self._cleanup_workpiece_ops(wp)

        # Connect new workpieces.
        for wp in new_set - old_set:
            wp.layer = self
            wp.size_changed.connect(self._on_workpiece_size_changed)

        self.workpieces = list(workpieces)
        self._update_ops_for_all_workpieces()
        self.changed.send(self)

    def _get_workpiece_by_uid(self, uid: str) -> Optional[WorkPiece]:
        """Finds a workpiece in this layer by its UID."""
        return next((wp for wp in self.workpieces if wp.uid == uid), None)

    def _get_step_by_uid(self, uid: str) -> Optional[Step]:
        """Finds a step in this layer's workflow by its UID."""
        return next((s for s in self.workflow.steps if s.uid == uid), None)

    def _cleanup_workpiece_ops(self, workpiece: WorkPiece):
        """Removes all cached ops and metadata for a workpiece."""
        w_uid = workpiece.uid
        keys_to_del = [key for key in self._ops_cache if key[1] == w_uid]
        for key in keys_to_del:
            self._ops_cache.pop(key, None)
            self._generation_id_map.pop(key, None)

        # Also cancel any running tasks for this workpiece.
        for step in self.workflow.steps:
            # The key for a task is (step.uid, workpiece.uid)
            task_key = (step.uid, w_uid)
            task_mgr.cancel_task(task_key)

    def _update_ops_for_workpiece(self, workpiece: WorkPiece):
        """Triggers ops generation for all steps for one workpiece."""
        for step in self.workflow.steps:
            self._trigger_ops_generation(step, workpiece)

    def _update_ops_for_all_workpieces(self):
        """Triggers ops generation for all steps and all workpieces."""
        for workpiece in self.workpieces:
            self._update_ops_for_workpiece(workpiece)

    def _update_ops_for_step(self, step: Step):
        """Triggers ops generation for a single step across all workpieces."""
        for workpiece in self.workpieces:
            self._trigger_ops_generation(step, workpiece)

    def _trigger_ops_generation(self, step: Step, workpiece: WorkPiece):
        """
        Starts an asynchronous task to generate operations by delegating
        to the Step.

        This method manages generation IDs to prevent race conditions and
        tells the Step to start its generation task. It provides a
        callback for the Layer to handle the final result.

        Args:
            step: The Step to be applied.
            workpiece: The WorkPiece to process.
        """
        size = workpiece.get_current_size()
        if not size or None in size:
            return

        # Increment generation ID to invalidate any ongoing, older tasks.
        key = step.uid, workpiece.uid
        self._generation_id_map[key] = self._generation_id_map.get(key, 0) + 1
        current_gen_id = self._generation_id_map[key]

        # Notify listeners that generation is starting.
        step.ops_generation_starting.send(
            step, workpiece=workpiece, generation_id=current_gen_id
        )

        # Pre-emptively clear the cache to ensure stale data isn't served.
        self._ops_cache[key] = (None, None)

        # This callback executes in the main process when the task ends.
        s_uid, w_uid = step.uid, workpiece.uid

        def when_done_callback(task: Task):
            self._on_generation_complete(task, s_uid, w_uid, current_gen_id)

        # Delegate the actual task creation to the step.
        step.start_generation_task(
            workpiece,
            current_gen_id,
            when_done_callback,
        )

    def _on_generation_complete(
        self, task: Task, s_uid: str, w_uid: str, task_generation_id: int
    ):
        """
        Callback for when an ops generation task finishes.

        Validates that the result is not stale, updates the ops cache,
        and signals that the process is finished.

        Args:
            task: The completed Task object.
            s_uid: The UID of the Step.
            w_uid: The UID of the WorkPiece.
            task_generation_id: The generation ID of the completed task.
        """
        key = (s_uid, w_uid)

        # Prevents a race condition. If a new task was started after this
        # one, the generation ID will be higher, and this result is now
        # stale and must be ignored.
        if (
            key not in self._generation_id_map
            or self._generation_id_map[key] != task_generation_id
        ):
            logger.debug(
                f"Ignoring stale ops result for Step '{s_uid}' / "
                f"WP '{w_uid}' (gen {task_generation_id})."
            )
            return

        workpiece = self._get_workpiece_by_uid(w_uid)
        step = self._get_step_by_uid(s_uid)
        if not workpiece or not step:
            # Item was deleted since the task was started.
            return

        if task.get_status() == "completed":
            self._handle_completed_task(task, key, step, workpiece)
        else:
            logger.warning(
                f"Ops generation for '{step.name}' / '{workpiece.name}' "
                f"failed. Status: {task.get_status()}."
            )
            self._ops_cache[key] = (None, None)

        step.ops_generation_finished.send(
            step, workpiece=workpiece, generation_id=task_generation_id
        )

    def _handle_completed_task(
        self,
        task: Task,
        key: Tuple[str, str],
        step: Step,
        workpiece: WorkPiece,
    ):
        """Processes the result of a successfully completed task."""
        try:
            result = task.result()
            ops, px_size = result if result else (None, None)
            self._ops_cache[key] = (ops, px_size)
            ops_len = len(ops) if ops else 0
            logger.info(
                f"CACHED ops for '{step.name}' / '{workpiece.name}'. "
                f"Ops count: {ops_len}, Px size: {px_size}"
            )
        except Exception as e:
            logger.error(
                f"Error getting result for '{step.name}' on "
                f"'{workpiece.name}': {e}",
                exc_info=True,
            )
            self._ops_cache[key] = (None, None)

    def get_ops(self, step: Step, workpiece: WorkPiece) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache.

        This returns a deep copy of the cached Ops. If the ops were
        generated from a source with a specific pixel size (e.g., an
        image), this method scales them to the workpiece's current size.

        Args:
            step: The Step for which to retrieve operations.
            workpiece: The WorkPiece for which to retrieve operations.

        Returns:
            A deep copy of the scaled Ops object, or None if no
            operations are available in the cache.
        """
        key = (step.uid, workpiece.uid)
        if not workpiece.get_current_size():
            return None

        raw_ops, pixel_size = self._ops_cache.get(key, (None, None))
        if raw_ops is None:
            return None

        ops = deepcopy(raw_ops)

        # If ops were generated from a raster source, they were created
        # in pixel units and must be scaled to the workpiece's final size.
        if pixel_size:
            self._scale_ops_to_workpiece_size(ops, pixel_size, workpiece)

        return ops

    def _scale_ops_to_workpiece_size(
        self, ops: Ops, px_size: Tuple[int, int], workpiece: "WorkPiece"
    ):
        """
        Scales an Ops object from pixel size to the workpiece size in mm.
        """
        traced_width_px, traced_height_px = px_size
        size = workpiece.get_current_size()

        # Should already be checked by caller, but acts as a safeguard.
        if not size:
            return

        final_width_mm, final_height_mm = size

        if traced_width_px > 0 and traced_height_px > 0:
            scale_x = final_width_mm / traced_width_px
            scale_y = final_height_mm / traced_height_px
            ops.scale(scale_x, scale_y)

    def get_renderable_items(self) -> List[Tuple[Step, WorkPiece]]:
        """
        Gets a list of all visible step/workpiece pairs for rendering.

        Returns:
            A list of (Step, WorkPiece) tuples that are currently
            visible and have valid geometry for rendering.
        """
        if not self.visible:
            return []
        items = []
        for workpiece in self.workpieces:
            if not workpiece.pos or not workpiece.size:
                continue
            for step in self.workflow.steps:
                if step.visible:
                    items.append((step, workpiece))
        return items
