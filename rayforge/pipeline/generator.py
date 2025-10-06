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
import math
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING, cast
from copy import deepcopy
from blinker import Signal
from contextlib import contextmanager
from .. import config
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.step import Step
from ..core.workpiece import WorkPiece
from ..core.ops import Ops, ScanLinePowerCommand
from .steprunner import run_step_in_subprocess
from ..core.group import Group
from .artifact.vector import VectorArtifact
from .artifact.hybrid import HybridRasterArtifact
from .artifact.handle import ArtifactHandle
from .artifact.store import ArtifactStore


if TYPE_CHECKING:
    from ..shared.tasker.task import Task
    from ..shared.tasker.manager import TaskManager

logger = logging.getLogger(__name__)


class OpsGenerator:
    """
    Listens to a Doc model and orchestrates the generation of Ops.

    This class acts as a "conductor" for the data pipeline. It connects to the
    `doc.changed` signal and intelligently determines what needs to be
    regenerated based on model changes. It manages a cache of lightweight
    `ArtifactHandle` objects, where each handle points to a full `Artifact`
    (containing Ops data) stored in shared memory. Each cache entry
    corresponds to a specific (Step, WorkPiece) pair.

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
    # Value: ArtifactHandle or HybridRasterArtifact
    OpsCacheType = Dict[
        Tuple[str, str],
        Optional[ArtifactHandle],
    ]

    def __init__(self, doc: "Doc", task_manager: "TaskManager"):
        """
        Initializes the OpsGenerator.

        Args:
            doc: The top-level Doc object to monitor for changes.
            task_manager: The TaskManager instance to use for background jobs.
        """
        logger.debug(f"{self.__class__.__name__}.__init__ called")
        self._doc: Doc = Doc()
        self._task_manager = task_manager
        self._ops_cache: OpsGenerator.OpsCacheType = {}
        self._generation_id_map: Dict[Tuple[str, str], int] = {}
        self._active_tasks: Dict[Tuple[str, str], "Task"] = {}
        self._pause_count = 0

        # Signals for notifying the UI of generation progress
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()
        # Fired when is_busy changes (0->N tasks or N->0 tasks)
        self.processing_state_changed = Signal()

        # This will trigger the setter, which connects signals and runs the
        # initial reconciliation.
        self.doc = doc

    def shutdown(self):
        """
        Releases all shared memory resources held in the cache. This must be
        called before application exit to prevent memory leaks.
        """
        logger.info("OpsGenerator shutting down and releasing artifacts...")
        for key in list(self._ops_cache.keys()):
            handle = self._ops_cache.pop(key, None)
            if handle:
                ArtifactStore.release(handle)
        logger.info("All artifacts released.")

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
            self._generation_id_map.clear()

        self._doc = new_doc

        # Connect to the new document and reconcile its state
        if self._doc:
            logger.debug(
                f"OpsGenerator assigned new document: {id(self._doc)}"
            )
            self._connect_signals()
            self.reconcile_all()

    @property
    def is_busy(self) -> bool:
        """Returns True if any ops generation tasks are currently running."""
        return bool(self._active_tasks)

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
            if layer.workflow:
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
        elif isinstance(origin, (WorkPiece, Group, Layer)):
            # If a single workpiece is added, this list will have one item.
            # If a group or layer is added, this will find all workpieces
            # inside it that need ops generation.
            workpieces_to_process = []
            if isinstance(origin, WorkPiece):
                workpieces_to_process.append(origin)
            else:  # Group or Layer
                workpieces_to_process.extend(
                    origin.get_descendants(of_type=WorkPiece)
                )

            for workpiece in workpieces_to_process:
                workpiece_layer = workpiece.layer
                if workpiece_layer and workpiece_layer.workflow:
                    for step in workpiece_layer.workflow:
                        key = (step.uid, workpiece.uid)
                        if self._ops_cache.get(key) is None:
                            logger.debug(
                                f"Ops for {key} not found in cache after "
                                "'add'. Triggering generation."
                            )
                            self._trigger_ops_generation(step, workpiece)
                        else:
                            logger.debug(
                                f"Ops for {key} found in cache after 'add'. "
                                "Skipping regeneration."
                            )
        elif isinstance(origin, Layer) and origin.workflow:
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
        elif isinstance(origin, WorkPiece):
            keys_to_clean = [k for k in self._ops_cache if k[1] == origin.uid]
        elif isinstance(origin, (Group, Layer)):
            # Find all workpieces that were inside the removed container
            removed_wp_uids = {
                wp.uid for wp in origin.get_descendants(of_type=WorkPiece)
            }
            if isinstance(origin, Layer) and origin.workflow:
                # If a whole layer is removed, also clean up its steps
                step_uids = {s.uid for s in origin.workflow}
                keys_to_clean = [
                    k
                    for k in self._ops_cache
                    if k[0] in step_uids or k[1] in removed_wp_uids
                ]
            else:
                keys_to_clean = [
                    k for k in self._ops_cache if k[1] in removed_wp_uids
                ]
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
        Smartly handles transform changes. For scalable ops, it sends an
        update notification. For non-scalable ops, it defers regeneration
        if paused.
        """
        logger.debug(
            f"{self.__class__.__name__}._on_descendant_transform_changed"
        )

        # If a group's transform changes, we need to check all workpieces
        # inside it.
        workpieces_to_check = []
        if isinstance(origin, WorkPiece):
            workpieces_to_check.append(origin)
        elif isinstance(origin, (Group, Layer)):
            workpieces_to_check.extend(
                origin.get_descendants(of_type=WorkPiece)
            )

        for wp in workpieces_to_check:
            wp_layer = wp.layer
            if not wp_layer or not wp_layer.workflow:
                continue

            for step in wp_layer.workflow:
                key = (step.uid, wp.uid)
                handle = self._ops_cache.get(key)

                # Determine if the current state is valid. An artifact is
                # valid if it exists AND (it's scalable OR (it's not
                # scalable AND its generation size matches the current
                # workpiece size)).
                is_valid = False
                if handle:
                    is_valid = (
                        handle.is_scalable or handle.generation_size == wp.size
                    )

                if is_valid:
                    # If valid and scalable, we can send a cheap notification
                    # for live UI updates (e.g., when moving vector ops).
                    # Non-scalable ops don't need this for pure translation.
                    if handle and handle.is_scalable:
                        generation_id = self._generation_id_map.get(key, 0)
                        self.ops_generation_finished.send(
                            step, workpiece=wp, generation_id=generation_id
                        )
                else:
                    # The artifact is invalid (or missing). We must regenerate.
                    if self.is_paused:
                        # Defer the regeneration until resume(), which will
                        # call reconcile_all().
                        continue

                    old_size_str = (
                        f"Old size: {handle.generation_size}"
                        if handle
                        else "Artifact missing"
                    )
                    logger.info(
                        f"Ops for '{wp.name}' are invalid due to transform. "
                        f"({old_size_str}, New size: {wp.size}). Regenerating."
                    )
                    self._trigger_ops_generation(step, wp)

    def _cleanup_key(self, key: Tuple[str, str]):
        """
        Removes a cache entry, releases its shared memory, and cancels any
        associated task.
        """
        logger.debug(f"{self.__class__.__name__}._cleanup_key called")
        logger.debug(f"OpsGenerator: Cleaning up key {key}.")
        handle = self._ops_cache.pop(key, None)
        if handle:
            ArtifactStore.release(handle)
        self._generation_id_map.pop(key, None)
        if key in self._active_tasks:
            self._task_manager.cancel_task(key)
            self._active_tasks.pop(key, None)

    def reconcile_all(self):
        """
        Synchronizes the generator's state with the document.

        This method compares the complete set of (Step, WorkPiece) pairs in
        the document with its internal cache. It starts generation for any new
        or invalidated items and cleans up tasks/cache for obsolete items.
        """
        logger.debug(f"{self.__class__.__name__}.reconcile_all called")
        if self.is_paused:
            return
        all_current_pairs = {
            (step.uid, workpiece.uid)
            for layer in self.doc.layers
            if layer.workflow is not None
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }
        cached_pairs = set(self._ops_cache.keys())

        # Clean up obsolete items
        for s_uid, w_uid in cached_pairs - all_current_pairs:
            self._cleanup_key((s_uid, w_uid))

        # Trigger generation for new or invalidated items.
        for layer in self.doc.layers:
            if layer.workflow is None:
                continue
            for step in layer.workflow.steps:
                for workpiece in layer.all_workpieces:
                    key = (step.uid, workpiece.uid)
                    is_valid = False
                    handle = self._ops_cache.get(key)
                    if handle:
                        is_valid = (
                            handle.is_scalable
                            or handle.generation_size == workpiece.size
                        )

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
        if workpiece_layer and workpiece_layer.workflow:
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
            # Clear cache for invalid size to prevent stale ops from showing.
            self._cleanup_key((step.uid, workpiece.uid))
            return

        key = step.uid, workpiece.uid

        # If a task is already running for this key, cancel it before starting
        # a new one.
        if key in self._active_tasks:
            logger.debug(f"Cancelling existing task for key {key}.")
            self._task_manager.cancel_task(key)
            self._active_tasks.pop(key, None)

        generation_id = self._generation_id_map.get(key, 0) + 1
        self._generation_id_map[key] = generation_id

        self.ops_generation_starting.send(
            step, workpiece=workpiece, generation_id=generation_id
        )
        self._ops_cache[key] = None

        was_busy = self.is_busy
        s_uid, w_uid = step.uid, workpiece.uid

        # Capture the size we are generating for and pass it to the subprocess.
        generation_size = workpiece.size

        def when_done_callback(task: "Task"):
            self._on_generation_complete(task, s_uid, w_uid, generation_id)

        settings = step.get_settings()

        if not config.config.machine:
            logger.error("Cannot generate ops: No machine is configured.")
            return

        try:
            selected_laser = step.get_selected_laser(config.config.machine)
            laser_dict_for_subprocess = selected_laser.to_dict()
        except ValueError as e:
            logger.error(f"Cannot select laser for step '{step.name}': {e}")
            return

        if not all([step.opsproducer_dict, laser_dict_for_subprocess]):
            logger.error(
                f"Step '{step.name}' is not fully configured. Skipping."
            )
            return

        # To ensure the workpiece renders at the correct resolution in the
        # subprocess (where it has no parent hierarchy), we create a new
        # instance with its world-transform "baked in" as its local matrix.
        world_workpiece = workpiece.in_world()
        workpiece_dict = world_workpiece.to_dict()

        # Hydrate the dictionary with the data needed for rendering in the
        # isolated subprocess.
        renderer = workpiece.renderer
        if renderer:
            workpiece_dict["data"] = workpiece.data
            workpiece_dict["renderer_name"] = renderer.__class__.__name__

        task = self._task_manager.run_process(
            run_step_in_subprocess,
            workpiece_dict,
            step.opsproducer_dict,
            step.modifiers_dicts,
            step.opstransformers_dicts,
            laser_dict_for_subprocess,
            settings,
            generation_id,
            generation_size,
            key=key,
            when_done=when_done_callback,
            when_event=self._on_task_event_received,
        )
        self._active_tasks[key] = task
        if not was_busy and self.is_busy:
            self.processing_state_changed.send(self, is_processing=True)

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
    ):
        """
        Callback for when an ops generation task finishes.

        It validates that the result is not from a stale task, deserializes
        the incoming `ArtifactHandle` dictionary, updates the cache with this
        new handle, and fires the `ops_generation_finished` signal. No large
        data is processed on this callback, ensuring the main thread remains
        responsive.
        """
        logger.debug(
            f"{self.__class__.__name__}._on_generation_complete called"
        )
        key = s_uid, w_uid
        was_busy = self.is_busy
        self._active_tasks.pop(key, None)

        # This first check is a fast path to discard callbacks for tasks
        # that were superseded before they even started finishing.
        if (
            key not in self._generation_id_map
            or self._generation_id_map[key] != task_generation_id
        ):
            logger.debug(
                f"Ignoring stale ops callback for {key} "
                f"(gen {task_generation_id})."
            )
            return

        workpiece = self._find_workpiece_by_uid(w_uid)
        step = self._find_step_by_uid(s_uid)
        if not workpiece or not step:
            return

        if task.get_status() == "completed":
            # This method will now handle cache update and signal sending
            self._handle_completed_task(task, key, step, workpiece)
        else:
            # Handle failed or cancelled tasks
            wp_name = (
                workpiece.source_file.name
                if workpiece.source_file
                else workpiece.name
            )
            logger.warning(
                f"Ops generation for '{step.name}' / "
                f"'{wp_name}' failed. "
                f"Status: {task.get_status()}."
            )
            self._ops_cache[key] = None
            # Still notify the view so it can stop showing a "loading" state
            self.ops_generation_finished.send(
                step, workpiece=workpiece, generation_id=task_generation_id
            )

        is_busy_now = self.is_busy
        if was_busy and not is_busy_now:
            self.processing_state_changed.send(self, is_processing=False)

    def _handle_completed_task(
        self,
        task: "Task",
        key: Tuple[str, str],
        step: Step,
        workpiece: WorkPiece,
    ):
        """
        Processes the result of a successfully completed task, validates it,
        updates the cache, and sends the final notification.
        """
        logger.debug(
            f"{self.__class__.__name__}._handle_completed_task called"
        )
        try:
            result = task.result()
            if not isinstance(result, tuple) or len(result) != 2:
                logger.error(
                    f"Task for {key} returned unexpected format: "
                    f"{type(result)}"
                )
                self._ops_cache[key] = None
                return

            result_handle_dict, result_gen_id = result

            # Ensure that the result we just received from the subprocess
            # is for the MOST RECENT request we made.
            # If another request was made while this one was in-flight,
            # this check will fail, and we will discard this stale result.
            if self._generation_id_map.get(key) != result_gen_id:
                logger.warning(
                    f"Ignoring stale subprocess result for {key}. "
                    f"Expected gen {self._generation_id_map.get(key)}, "
                    f"got {result_gen_id}."
                )
                # DO NOT update cache or send signal for this stale result.
                # The signal for the newer task will arrive later.
                return

            # --- Validation passed, now update state and notify ---
            if result_handle_dict:
                handle = ArtifactHandle.from_dict(result_handle_dict)
                self._ops_cache[key] = handle
            else:
                self._ops_cache[key] = None

            # Now that the cache is guaranteed to be up-to-date with a valid
            # result, we can safely notify the view to re-render.
            self.ops_generation_finished.send(
                step, workpiece=workpiece, generation_id=result_gen_id
            )

        except Exception as e:
            # Check source_file before using it in the log message
            wp_name = (
                workpiece.source_file.name
                if workpiece.source_file
                else workpiece.name
            )
            logger.error(
                f"Error getting result for '{step.name}' on '{wp_name}': {e}",
                exc_info=True,
            )
            self._ops_cache[key] = None

    def get_ops(self, step: Step, workpiece: WorkPiece) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache on-demand.

        This is the primary method for consumers (like the UI or the job
        assembler) to get the result of the pipeline. It retrieves the
        `ArtifactHandle` from the cache, reconstructs the full `Artifact`
        object from shared memory, and returns a deep copy of its `Ops`. If
        the ops are scalable, this method also scales them to the workpiece's
        current physical size in millimeters.

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

        handle = self._ops_cache.get(key)

        if handle is None:
            logger.debug(f"get_ops for {key}: No artifact found in cache.")
            return None

        # Check for stale non-scalable artifacts.
        if not handle.is_scalable and handle.generation_size != workpiece.size:
            logger.debug(
                f"get_ops for {key}: Found stale artifact. "
                f"Cache size: {handle.generation_size}, "
                f"WP size: {workpiece.size}. Returning None."
            )
            return None

        logger.debug(
            f"get_ops for {key}: Found artifact. "
            f"Scalable: {handle.is_scalable}."
        )

        artifact = ArtifactStore.get(handle)
        ops = deepcopy(artifact.ops)

        if isinstance(artifact, VectorArtifact) and artifact.is_scalable:
            self._scale_ops_to_workpiece_size(ops, artifact, workpiece)

        # --- VALIDATION LOGGING ---
        scanline_count = sum(
            1 for cmd in ops.commands if isinstance(cmd, ScanLinePowerCommand)
        )
        logger.debug(
            f"OpsGenerator.get_ops: Returning final ops for key {key} with "
            f"{scanline_count} ScanLinePowerCommands."
        )
        # --- END LOGGING ---
        return ops

    def get_artifact(
        self, step: Step, workpiece: WorkPiece
    ) -> Optional[Union[VectorArtifact, HybridRasterArtifact]]:
        """
        Retrieves the complete artifact from the cache on-demand.

        This method retrieves the `ArtifactHandle` from the cache and uses the
        `ArtifactStore` to reconstruct the full artifact object (e.g.,
        `VectorArtifact` or `HybridRasterArtifact`) from shared memory. This
        is useful for components that need access to additional data in the
        artifact, such as texture data for raster rendering.

        Args:
            step: The Step for which to retrieve the artifact.
            workpiece: The WorkPiece for which to retrieve the artifact.

        Returns:
            The cached artifact object, or None if no artifact is available.
        """
        logger.debug(f"{self.__class__.__name__}.get_artifact called")
        key = step.uid, workpiece.uid
        if any(s <= 0 for s in workpiece.size):
            return None

        handle = self._ops_cache.get(key)

        if handle is None:
            logger.debug(
                f"get_artifact for {key}: No artifact found in cache."
            )
            return None

        # Check for stale non-scalable artifacts.
        if not handle.is_scalable and handle.generation_size != workpiece.size:
            logger.debug(
                f"get_artifact for {key}: Found stale artifact. "
                f"Cache size: {handle.generation_size}, "
                f"WP size: {workpiece.size}. Returning None."
            )
            return None

        logger.debug(
            f"get_artifact for {key}: Found artifact. "
            f"Scalable: {handle.is_scalable}."
        )
        # Return a deep copy to prevent consumers from mutating the cache.
        artifact = cast(
            Union[VectorArtifact, HybridRasterArtifact],
            ArtifactStore.get(handle),
        )
        return deepcopy(artifact)

    def _scale_ops_to_workpiece_size(
        self,
        ops: Ops,
        artifact: VectorArtifact,
        workpiece: "WorkPiece",
    ):
        """
        Scales an Ops object from its source coordinate system to the
        workpiece's current physical size in millimeters.
        """
        logger.debug(
            f"{self.__class__.__name__}._scale_ops_to_workpiece_size called"
        )
        if not artifact.source_dimensions:
            logger.warning(
                "Cannot scale ops: artifact is missing source size."
            )
            return

        source_width, source_height = artifact.source_dimensions
        size = workpiece.size
        if not size:
            return

        final_width_mm, final_height_mm = size

        scale_x = 1.0
        if source_width > 1e-9:
            scale_x = final_width_mm / source_width

        scale_y = 1.0
        if source_height > 1e-9:
            scale_y = final_height_mm / source_height

        # Only apply scaling if it's not a no-op
        if not (math.isclose(scale_x, 1.0) and math.isclose(scale_y, 1.0)):
            logger.debug(f"Scaling ops by ({scale_x}, {scale_y})")
            ops.scale(scale_x, scale_y)
