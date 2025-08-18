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
from contextlib import contextmanager
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
        logger.debug(f"{self.__class__.__name__}.__init__ called")
        self._doc: Doc = Doc()
        self._ops_cache: OpsGenerator.OpsCacheType = {}
        self._world_size_cache: Dict[str, Tuple[float, float]] = {}
        self._generation_id_map: Dict[Tuple[str, str], int] = {}
        self._active_tasks: Dict[Tuple[str, str], "Task"] = {}
        self._pause_count = 0

        # Signals for notifying the UI of generation progress
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()

        # This will trigger the setter, which connects signals and runs the
        # initial reconciliation.
        self.doc = doc

    @property
    def doc(self) -> Doc:
        """The document model this generator is observing."""
        logger.debug(f"{self.__class__.__name__}.doc (getter) called")
        return self._doc

    @doc.setter
    def doc(self, new_doc: Doc):
        """Sets the document and manages signal connections."""
        logger.debug(f"{self.__class__.__name__}.doc (setter) called")
        if self._doc is new_doc:
            return  # No change

        # Disconnect from the old document if it exists
        if self._doc:
            self._disconnect_signals()
            # Clean up state related to the old document
            # Cancel any running tasks associated with the old doc and clear
            # caches
            for key in list(self._active_tasks.keys()):
                self._cleanup_key(key)
            # Ensure cache is fully cleared in case some items had no active
            # tasks
            self._ops_cache.clear()
            self._world_size_cache.clear()
            self._generation_id_map.clear()

        self._doc = new_doc

        # Connect to the new document and reconcile its state
        if self._doc:
            logger.debug(
                f"OpsGenerator assigned new document: {id(self._doc)}"
            )
            self._connect_signals()
            self.reconcile_all()

    def _connect_signals(self):
        """Connects to the document's signals."""
        logger.debug(f"{self.__class__.__name__}._connect_signals called")
        logger.debug(f"OpsGenerator connecting signals for doc {id(self.doc)}")
        self.doc.descendant_added.connect(self._on_descendant_added)
        self.doc.descendant_removed.connect(self._on_descendant_removed)
        self.doc.descendant_updated.connect(self._on_descendant_updated)
        self.doc.descendant_transform_changed.connect(
            self._on_descendant_transform_changed
        )

    def _disconnect_signals(self):
        """Disconnects from the document's signals."""
        logger.debug(f"{self.__class__.__name__}._disconnect_signals called")
        logger.debug(
            f"OpsGenerator disconnecting signals for doc {id(self.doc)}"
        )
        # Blinker's disconnect is safe to call even if not connected.
        self.doc.descendant_added.disconnect(self._on_descendant_added)
        self.doc.descendant_removed.disconnect(self._on_descendant_removed)
        self.doc.descendant_updated.disconnect(self._on_descendant_updated)
        self.doc.descendant_transform_changed.disconnect(
            self._on_descendant_transform_changed
        )

    def pause(self):
        """
        Increments the pause counter. The generator is paused if the
        counter is > 0.
        """
        logger.debug(f"{self.__class__.__name__}.pause called")
        if self._pause_count == 0:
            logger.debug("OpsGenerator paused.")
        self._pause_count += 1

    def resume(self):
        """
        Decrements the pause counter. If the counter reaches 0, the
        generator is resumed and reconciles all changes.
        """
        logger.debug(f"{self.__class__.__name__}.resume called")
        if self._pause_count == 0:
            return
        self._pause_count -= 1
        if self._pause_count == 0:
            logger.debug("OpsGenerator resumed.")
            self.reconcile_all()

    @contextmanager
    def paused(self):
        """
        A reentrant context manager to safely pause and resume the generator.
        """
        logger.debug(f"{self.__class__.__name__}.paused called")
        self.pause()
        try:
            yield
        finally:
            self.resume()

    @property
    def is_paused(self) -> bool:
        """Returns True if the generator is currently paused."""
        logger.debug(f"{self.__class__.__name__}.is_paused called")
        return self._pause_count > 0

    def _find_step_by_uid(self, uid: str) -> Optional[Step]:
        """Finds a step anywhere in the document by its UID."""
        logger.debug(f"{self.__class__.__name__}._find_step_by_uid called")
        for layer in self.doc.layers:
            for step in layer.workflow.steps:
                if step.uid == uid:
                    return step
        return None

    def _find_workpiece_by_uid(self, uid: str) -> Optional[WorkPiece]:
        """Finds a workpiece anywhere in the document by its UID."""
        logger.debug(
            f"{self.__class__.__name__}._find_workpiece_by_uid called"
        )
        # Use the recursive helper on the doc itself.
        for wp in self.doc.all_workpieces:
            if wp.uid == uid:
                return wp
        return None

    def _on_descendant_added(self, sender, *, origin):
        """Handles the addition of a new model object."""
        logger.debug(f"{self.__class__.__name__}._on_descendant_added called")
        if self.is_paused:
            return
        logger.debug(
            f"OpsGenerator: Noticed added {origin.__class__.__name__}"
        )
        if isinstance(origin, Step):
            self._update_ops_for_step(origin)
        elif isinstance(origin, WorkPiece):
            workpiece = origin
            workpiece_layer = workpiece.layer
            if workpiece_layer:
                for step in workpiece_layer.workflow:
                    key = (step.uid, workpiece.uid)
                    if self._ops_cache.get(key, (None, None))[0] is None:
                        logger.debug(
                            f"Ops for {key} not found in cache after 'add'. "
                            "Triggering generation."
                        )
                        self._trigger_ops_generation(step, workpiece)
                    else:
                        logger.debug(
                            f"Ops for {key} found in cache after 'add'. "
                            "Skipping regeneration."
                        )
        elif isinstance(origin, Layer):
            for step in origin.workflow:
                self._update_ops_for_step(step)

    def _on_descendant_removed(self, sender, *, origin):
        """Handles the removal of a model object."""
        logger.debug(
            f"{self.__class__.__name__}._on_descendant_removed called"
        )
        if self.is_paused:
            return
        logger.debug(
            f"OpsGenerator: Noticed removed {origin.__class__.__name__}"
        )
        if isinstance(origin, Step):
            keys_to_clean = [k for k in self._ops_cache if k[0] == origin.uid]
        elif isinstance(origin, Layer):
            step_uids = {s.uid for s in origin.workflow}
            keys_to_clean = [k for k in self._ops_cache if k[0] in step_uids]
        else:
            return

        for key in keys_to_clean:
            self._cleanup_key(key)

    def _on_descendant_updated(self, sender, *, origin):
        """
        Handles non-transform updates that require regeneration.
        """
        logger.debug(
            f"{self.__class__.__name__}._on_descendant_updated called"
        )
        if self.is_paused:
            return  # All changes will be caught by reconcile_all on resume.
        logger.debug(
            f"OpsGenerator: Noticed updated {origin.__class__.__name__}"
        )
        # A Step's properties (power, speed, etc.) changed, so regenerate
        if isinstance(origin, Step):
            self._update_ops_for_step(origin)
        # A generic 'updated'
        # signal on a workpiece now means its *content* (source image) has
        # changed, which requires regeneration. Transform changes are handled
        # separately.
        elif isinstance(origin, WorkPiece):
            self._update_ops_for_workpiece(origin)

    def _on_descendant_transform_changed(self, sender, *, origin):
        """
        Smartly handles transform changes. If the world size changes, the cache
        is invalidated, and regeneration is triggered. This only runs when the
        generator is not paused.
        """
        logger.debug(
            f"{self.__class__.__name__}._on_descendant_transform_changed"
        )
        if self.is_paused:
            return  # Changes are handled by reconcile_all() on resume.

        if not isinstance(origin, WorkPiece):
            return

        wp = origin
        new_size = wp.size
        # Read the size of the *currently cached* ops.
        # Do NOT update the cache here.
        old_size = self._world_size_cache.get(wp.uid)

        if old_size is None:
            # If we've never generated for this workpiece before, we must
            # generate.
            size_changed = True
        else:
            # Compare the new size to the previously cached size.
            size_changed = not (
                abs(old_size[0] - new_size[0]) < 1e-9
                and abs(old_size[1] - new_size[1]) < 1e-9
            )

        if size_changed:
            logger.info(
                f"World size for '{wp.name}' changed live. "
                f"Old: {old_size}, New: {new_size}. Regenerating ops."
            )
            self._update_ops_for_workpiece(wp)
        else:
            logger.debug(
                f"Transform for '{wp.name}' changed but world size is "
                "stable (translation/rotation only). Skipping regeneration."
            )

    def _cleanup_key(self, key: Tuple[str, str]):
        """Removes a cache entry and cancels any associated task."""
        logger.debug(f"{self.__class__.__name__}._cleanup_key called")
        logger.debug(f"OpsGenerator: Cleaning up key {key}.")
        self._ops_cache.pop(key, None)
        self._world_size_cache.pop(key[1], None)
        self._generation_id_map.pop(key, None)
        self._active_tasks.pop(key, None)
        task_mgr.cancel_task(key)

    def reconcile_all(self):
        """
        Synchronizes the generator's state with the document.

        This method compares the complete set of (Step, WorkPiece) pairs in
        the document with its internal cache. It starts generation for any new
        or invalidated items (based on world size) and cleans up tasks/cache
        for obsolete items.
        """
        logger.debug(f"{self.__class__.__name__}.reconcile_all called")
        if self.is_paused:
            return
        all_current_pairs = {
            (step.uid, workpiece.uid)
            for layer in self.doc.layers
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }
        cached_pairs = set(self._ops_cache.keys())

        # Clean up obsolete items
        for s_uid, w_uid in cached_pairs - all_current_pairs:
            self._cleanup_key((s_uid, w_uid))

        # Trigger generation for new or invalidated items.
        for layer in self.doc.layers:
            for step in layer.workflow.steps:
                for workpiece in layer.all_workpieces:
                    key = (step.uid, workpiece.uid)
                    is_valid = False
                    if key in self._ops_cache:
                        ops, _ = self._ops_cache[key]
                        if ops is not None:
                            # Perform the critical validity check.
                            old_size = self._world_size_cache.get(
                                workpiece.uid
                            )
                            new_size = workpiece.size
                            if old_size is not None:
                                size_changed = not (
                                    abs(old_size[0] - new_size[0]) < 1e-9
                                    and abs(old_size[1] - new_size[1]) < 1e-9
                                )
                                if not size_changed:
                                    is_valid = True

                    if not is_valid:
                        self._trigger_ops_generation(step, workpiece)

    def _update_ops_for_step(self, step: Step):
        """Triggers ops generation for a single step across all workpieces."""
        logger.debug(f"{self.__class__.__name__}._update_ops_for_step called")
        workflow = step.workflow
        if workflow and isinstance(workflow.parent, Layer):
            for workpiece in workflow.parent.all_workpieces:
                self._trigger_ops_generation(step, workpiece)

    def _update_ops_for_workpiece(self, workpiece: WorkPiece):
        """Triggers ops generation for a single workpiece across all steps."""
        logger.debug(
            f"{self.__class__.__name__}._update_ops_for_workpiece called"
        )
        workpiece_layer = workpiece.layer
        if workpiece_layer:
            for step in workpiece_layer.workflow:
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
        logger.debug(
            f"{self.__class__.__name__}._trigger_ops_generation called"
        )
        if any(s <= 0 for s in workpiece.size):
            return

        key = step.uid, workpiece.uid
        generation_id = self._generation_id_map.get(key, 0) + 1
        self._generation_id_map[key] = generation_id

        self.ops_generation_starting.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        self._ops_cache[key] = (None, None)

        s_uid, w_uid = step.uid, workpiece.uid

        # Capture the size we are generating for and pass it to the callback.
        generation_size = workpiece.size

        def when_done_callback(task: "Task"):
            self._on_generation_complete(
                task, s_uid, w_uid, generation_id, generation_size
            )

        settings = {
            "power": step.power,
            "cut_speed": step.cut_speed,
            "travel_speed": step.travel_speed,
            "air_assist": step.air_assist,
            "pixels_per_mm": step.pixels_per_mm,
        }
        if not all([step.opsproducer_dict, step.laser_dict]):
            logger.error(
                f"Step '{step.name}' is not fully configured. Skipping."
            )
            return

        # To ensure the workpiece renders at the correct resolution in the
        # subprocess (where it has no parent hierarchy), we create a new
        # instance with its world-transform "baked in" as its local matrix.
        world_workpiece = workpiece.in_world()

        task = task_mgr.run_process(
            run_step_in_subprocess,
            world_workpiece.to_dict(),
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
        logger.debug(
            f"{self.__class__.__name__}._on_task_event_received called"
        )
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
        self,
        task: "Task",
        s_uid: str,
        w_uid: str,
        task_generation_id: int,
        generation_size: Tuple[float, float],
    ):
        """
        Callback for when an ops generation task finishes.

        It validates that the result is not from a stale task, updates the
        ops cache with the final result, and fires the
        `ops_generation_finished` signal.
        """
        logger.debug(
            f"{self.__class__.__name__}._on_generation_complete called"
        )
        key = s_uid, w_uid
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
            self._handle_completed_task(
                task, key, step, workpiece, generation_size
            )
        else:
            logger.warning(
                f"Ops generation for '{step.name}' / "
                f"'{workpiece.source_file}' failed. "
                f"Status: {task.get_status()}."
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
        generation_size: Tuple[float, float],
    ):
        """Processes the result of a successfully completed task."""
        logger.debug(
            f"{self.__class__.__name__}._handle_completed_task called"
        )
        try:
            result = task.result()
            ops, px_size = result if result else (None, None)
            # Atomically update both caches upon success.
            self._ops_cache[key] = (ops, px_size)
            if ops is not None:
                self._world_size_cache[workpiece.uid] = generation_size
        except Exception as e:
            logger.error(
                f"Error getting result for '{step.name}' on "
                f"'{workpiece.source_file}': {e}",
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
        logger.debug(f"{self.__class__.__name__}.get_ops called")
        key = step.uid, workpiece.uid
        if any(s <= 0 for s in workpiece.size):
            return None

        raw_ops, pixel_size = self._ops_cache.get(key, (None, None))

        if raw_ops is None:
            logger.debug(f"get_ops for {key}: No ops found in cache.")
            return None
        else:
            logger.debug(
                f"get_ops for {key}: Found raw_ops with "
                f"{len(raw_ops.commands)} "
                f"commands. Pixel size: {pixel_size}."
            )

        ops = deepcopy(raw_ops)

        if pixel_size:
            self._scale_ops_to_workpiece_size(ops, pixel_size, workpiece)

        logger.debug(
            f"get_ops for {key}: Returning final ops with {len(ops.commands)} "
            f"commands. Bbox: {ops.rect()}"
        )
        return ops

    def _scale_ops_to_workpiece_size(
        self, ops: Ops, px_size: Tuple[int, int], workpiece: "WorkPiece"
    ):
        """
        Scales an Ops object from its generated pixel size to the workpiece's
        current physical size in millimeters.
        """
        logger.debug(
            f"{self.__class__.__name__}._scale_ops_to_workpiece_size called"
        )
        traced_width_px, traced_height_px = px_size
        size = workpiece.size
        if not size:
            return

        final_width_mm, final_height_mm = size

        if traced_width_px > 0 and traced_height_px > 0:
            scale_x = final_width_mm / traced_width_px
            scale_y = final_height_mm / traced_height_px
            logger.debug(f"Scaling ops by ({scale_x}, {scale_y})")
            ops.scale(scale_x, scale_y)
