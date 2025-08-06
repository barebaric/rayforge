"""
Defines the OpsGenerator, the central orchestrator for the data pipeline.

This module contains the OpsGenerator class, which acts as a bridge between the
pure data models in the `core` module (Doc, Layer, Step, WorkPiece) and the
execution logic of the pipeline. Its primary responsibility is to listen for
changes in the document, trigger asynchronous generation of machine operations
(Ops), and maintain a cache of the results for quick retrieval by the UI and
the final job assembler.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from copy import deepcopy
from blinker import Signal
from ..shared.tasker import task_mgr
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.step import Step
from ..core.workpiece import WorkPiece
from ..core.ops import Ops
from .steprunner import run_step_in_subprocess

if TYPE_CHECKING:
    from ..shared.tasker.task import Task

logger = logging.getLogger(__name__)


class OpsGenerator:
    """
    Listens to a Doc model and orchestrates the generation of Ops.

    This class acts as a "conductor" for the data pipeline. It connects to the
    `doc.changed` signal and intelligently determines what needs to be
    regenerated based on model changes. It manages a cache of generated
    operations, where each entry corresponds to a specific (Step, WorkPiece)
    pair.

    The generated Ops are "pure" and workpiece-local; they are not yet
    positioned, rotated, or clipped to the machine work area. That final
    assembly is handled by the `job.generate_job_ops` function.

    Attributes:
        doc (Doc): The document model this generator is observing.
        ops_generation_starting (Signal): Fired when generation begins for a
            (Step, WorkPiece) pair. UI components listen to this to clear old
            visuals.
        ops_chunk_available (Signal): Fired as chunks of Ops become available
            from a background process, allowing for progressive UI updates,
            especially for long raster operations.
        ops_generation_finished (Signal): Fired when generation is complete
            for a (Step, WorkPiece) pair. UI components listen to this to
            request the final, complete Ops for rendering.
    """

    # Type alias for the structure of the operations cache.
    # Key: (step_uid, workpiece_uid)
    # Value: (Ops object, pixel_dimensions_tuple)
    OpsCacheType = Dict[
        Tuple[str, str], Tuple[Optional[Ops], Optional[Tuple[int, int]]]
    ]

    def __init__(self, doc: "Doc"):
        """
        Initializes the OpsGenerator.

        Args:
            doc: The top-level Doc object to monitor for changes.
        """
        self.doc = doc
        self._ops_cache: OpsGenerator.OpsCacheType = {}
        self._generation_id_map: Dict[Tuple[str, str], int] = {}
        self._active_tasks: Dict[Tuple[str, str], "Task"] = {}
        self._is_paused = False

        # Signals for notifying the UI of generation progress
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()

        self._connect_signals()
        self.reconcile_all()

    def _connect_signals(self):
        """Connects to the document's signals."""
        self.doc.descendant_added.connect(self._on_descendant_added)
        self.doc.descendant_removed.connect(self._on_descendant_removed)
        self.doc.descendant_updated.connect(self._on_descendant_updated)

    def _disconnect_signals(self):
        """Disconnects from the document's signals."""
        # Blinker's disconnect is safe to call even if not connected.
        self.doc.descendant_added.disconnect(self._on_descendant_added)
        self.doc.descendant_removed.disconnect(self._on_descendant_removed)
        self.doc.descendant_updated.disconnect(self._on_descendant_updated)

    def pause(self):
        """
        Temporarily stops listening to model changes.

        This is used to prevent storms of regeneration events during
        continuous UI operations like resizing a workpiece, where the model
        changes rapidly. The UI is responsible for calling `resume()` or
        `resume_and_reconcile()` when the operation is complete.
        """
        if self._is_paused:
            return
        logger.debug("OpsGenerator paused.")
        self._disconnect_signals()
        self._is_paused = True

    def resume(self):
        """
        Resumes listening to model changes without an immediate update.

        This is used after operations that do not invalidate the cached Ops,
        such as moving or rotating a workpiece.
        """
        if not self._is_paused:
            return
        logger.debug("OpsGenerator resumed.")
        self._is_paused = False
        self._connect_signals()
        self.reconcile_all()

    def _find_step_by_uid(self, uid: str) -> Optional[Step]:
        """Finds a step anywhere in the document by its UID."""
        for layer in self.doc.layers:
            for step in layer.workflow.steps:
                if step.uid == uid:
                    return step
        return None

    def _find_workpiece_by_uid(self, uid: str) -> Optional[WorkPiece]:
        """Finds a workpiece anywhere in the document by its UID."""
        for layer in self.doc.layers:
            for wp in layer.workpieces:
                if wp.uid == uid:
                    return wp
        return None

    def _on_descendant_added(self, sender, *, origin):
        """Handles the addition of a new model object."""
        if self._is_paused:
            return
        logger.debug(
            f"OpsGenerator: Noticed added {origin.__class__.__name__}"
        )
        if isinstance(origin, Step):
            self._update_ops_for_step(origin)
        elif isinstance(origin, WorkPiece):
            self._update_ops_for_workpiece(origin)
        elif isinstance(origin, Layer):
            for step in origin.workflow:
                self._update_ops_for_step(step)

    def _on_descendant_removed(self, sender, *, origin):
        """Handles the removal of a model object."""
        if self._is_paused:
            return
        logger.debug(
            f"OpsGenerator: Noticed removed {origin.__class__.__name__}"
        )
        uids_to_remove = set()

        if isinstance(origin, Step):
            uids_to_remove.add(origin.uid)
            keys_to_clean = [
                k for k in self._ops_cache if k[0] in uids_to_remove
            ]
        elif isinstance(origin, WorkPiece):
            uids_to_remove.add(origin.uid)
            keys_to_clean = [
                k for k in self._ops_cache if k[1] in uids_to_remove
            ]
        elif isinstance(origin, Layer):
            step_uids = {s.uid for s in origin.workflow}
            keys_to_clean = [k for k in self._ops_cache if k[0] in step_uids]
        else:
            return

        for key in keys_to_clean:
            self._cleanup_key(key)

    def _on_descendant_updated(self, sender, *, origin):
        """Handles updates to an existing model object's data."""
        if self._is_paused:
            return
        logger.debug(
            f"OpsGenerator: Noticed updated {origin.__class__.__name__}"
        )
        if isinstance(origin, Step):
            self._update_ops_for_step(origin)
        elif isinstance(origin, WorkPiece):
            self._update_ops_for_workpiece(origin)

    def _cleanup_key(self, key: Tuple[str, str]):
        """Removes a cache entry and cancels any associated task."""
        logger.debug(f"OpsGenerator: Cleaning up key {key}.")
        self._ops_cache.pop(key, None)
        self._generation_id_map.pop(key, None)
        self._active_tasks.pop(key, None)
        task_mgr.cancel_task(key)

    def reconcile_all(self):
        """
        Synchronizes the generator's state with the document.

        This method compares the complete set of (Step, WorkPiece) pairs in
        the document with its internal cache and running tasks. It starts
        generation for any new or modified items and cancels/cleans up any
        tasks or cache entries that are no longer present in the document.
        """
        if self._is_paused:
            return

        all_current_pairs = set()
        for layer in self.doc.layers:
            for step in layer.workflow.steps:
                for workpiece in layer.workpieces:
                    all_current_pairs.add((step.uid, workpiece.uid))

        cached_pairs = set(self._ops_cache.keys())

        # Clean up obsolete items
        for s_uid, w_uid in cached_pairs - all_current_pairs:
            self._cleanup_key((s_uid, w_uid))

        # Trigger generation for all current items
        for layer in self.doc.layers:
            for step in layer.workflow.steps:
                for workpiece in layer.workpieces:
                    self._trigger_ops_generation(step, workpiece)

    def _update_ops_for_step(self, step: Step):
        """Triggers ops generation for a single step across all workpieces."""
        if step.workflow and step.workflow.layer:
            for workpiece in step.workflow.layer.workpieces:
                self._trigger_ops_generation(step, workpiece)

    def _update_ops_for_workpiece(self, workpiece: WorkPiece):
        """Triggers ops generation for a single workpiece across all steps."""
        if workpiece.layer:
            for step in workpiece.layer.workflow:
                self._trigger_ops_generation(step, workpiece)

    def _trigger_ops_generation(self, step: Step, workpiece: WorkPiece):
        """
        Starts the asynchronous task to generate operations.

        This method manages generation IDs to prevent race conditions from
        stale async results. It serializes the necessary model data into
        dictionaries and passes them to the `run_step_in_subprocess` function
        via the TaskManager.

        Args:
            step: The Step configuration to apply.
            workpiece: The WorkPiece to process.
        """
        if not workpiece.get_current_size():
            return

        key = (step.uid, workpiece.uid)
        generation_id = self._generation_id_map.get(key, 0) + 1
        self._generation_id_map[key] = generation_id

        self.ops_generation_starting.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        self._ops_cache[key] = (None, None)

        s_uid, w_uid = step.uid, workpiece.uid

        def when_done_callback(task: "Task"):
            self._on_generation_complete(task, s_uid, w_uid, generation_id)

        settings = {
            "power": step.power,
            "cut_speed": step.cut_speed,
            "travel_speed": step.travel_speed,
            "air_assist": step.air_assist,
            "pixels_per_mm": step.pixels_per_mm,
        }

        if not all(
            [
                step.opsproducer_dict,
                step.laser_dict,
            ]
        ):
            logger.error(
                f"Step '{step.name}' is not fully configured. Skipping."
            )
            return

        task = task_mgr.run_process(
            run_step_in_subprocess,
            workpiece.to_dict(),
            step.opsproducer_dict,
            step.modifiers_dicts,
            step.opstransformers_dicts,
            step.laser_dict,
            settings,
            generation_id,
            key=key,
            when_done=when_done_callback,
            when_event=self._on_task_event_received,
        )
        self._active_tasks[key] = task

    def _on_task_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles `ops_chunk` events from a background task."""
        if event_name != "ops_chunk":
            return

        step_uid, workpiece_uid = task.key
        workpiece = self._find_workpiece_by_uid(workpiece_uid)
        step = self._find_step_by_uid(step_uid)
        if not workpiece or not step:
            return

        chunk = data.get("chunk")
        generation_id = data.get("generation_id")
        if not chunk or generation_id is None:
            return

        self.ops_chunk_available.send(
            step,
            workpiece=workpiece,
            chunk=chunk,
            generation_id=generation_id,
        )

    def _on_generation_complete(
        self, task: "Task", s_uid: str, w_uid: str, task_generation_id: int
    ):
        """
        Callback for when an ops generation task finishes.

        It validates that the result is not from a stale task, updates the
        ops cache with the final result, and fires the
        `ops_generation_finished` signal.
        """
        key = (s_uid, w_uid)
        self._active_tasks.pop(key, None)

        if (
            key not in self._generation_id_map
            or self._generation_id_map[key] != task_generation_id
        ):
            logger.debug(
                f"Ignoring stale ops result for {key} "
                f"(gen {task_generation_id})."
            )
            return

        workpiece = self._find_workpiece_by_uid(w_uid)
        step = self._find_step_by_uid(s_uid)
        if not workpiece or not step:
            return

        if task.get_status() == "completed":
            self._handle_completed_task(task, key, step, workpiece)
        else:
            logger.warning(
                f"Ops generation for '{step.name}' / '{workpiece.name}' "
                f"failed. Status: {task.get_status()}."
            )
            self._ops_cache[key] = (None, None)

        self.ops_generation_finished.send(
            step, workpiece=workpiece, generation_id=task_generation_id
        )

    def _handle_completed_task(
        self,
        task: "Task",
        key: Tuple[str, str],
        step: Step,
        workpiece: WorkPiece,
    ):
        """Processes the result of a successfully completed task."""
        try:
            result = task.result()
            ops, px_size = result if result else (None, None)
            self._ops_cache[key] = (ops, px_size)
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

        This is the primary method for consumers (like the UI or the job
        assembler) to get the result of the pipeline. It returns a deep copy
        of the cached Ops. If the ops were generated from a source with a
        specific pixel size (e.g., a vector trace of an SVG), this method
        scales them to the workpiece's current physical size in millimeters.

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

        if pixel_size:
            self._scale_ops_to_workpiece_size(ops, pixel_size, workpiece)

        return ops

    def _scale_ops_to_workpiece_size(
        self, ops: Ops, px_size: Tuple[int, int], workpiece: "WorkPiece"
    ):
        """
        Scales an Ops object from its generated pixel size to the workpiece's
        current physical size in millimeters.
        """
        traced_width_px, traced_height_px = px_size
        size = workpiece.get_current_size()
        if not size:
            return

        final_width_mm, final_height_mm = size

        if traced_width_px > 0 and traced_height_px > 0:
            scale_x = final_width_mm / traced_width_px
            scale_y = final_height_mm / traced_height_px
            ops.scale(scale_x, scale_y)
