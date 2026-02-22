from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from copy import deepcopy
from blinker import Signal
from ...core.ops import Ops, ScanLinePowerCommand
from ...shared.tasker.task import Task
from ...shared.util.size import sizes_are_close
from ..artifact import WorkPieceArtifact
from ..artifact.key import ArtifactKey
from ..artifact.manager import StaleGenerationError
from ..artifact.store import SharedMemoryNotFoundError
from ..dag.node import NodeState
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.matrix import Matrix
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager
    from ..context import GenerationContext

logger = logging.getLogger(__name__)


class WorkPiecePipelineStage(PipelineStage):
    """
    Provides access to workpiece artifacts and handles invalidation.
    Task launching creation, and completion handling reside here.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self.generation_starting = Signal()
        self.generation_finished = Signal()
        self.workpiece_artifact_adopted = Signal()
        self.visual_chunk_available = Signal()

    def invalidate_for_step(self, step_uid: str):
        """Invalidates all workpiece artifacts associated with a step."""
        logger.debug(f"Invalidating workpiece artifacts for step '{step_uid}'")
        step_key = ArtifactKey.for_step(step_uid)
        keys_to_clean = self._artifact_manager.get_dependents(step_key)
        logger.debug(
            f"WorkPiecePipelineStage: keys_to_clean for step '{step_uid}': "
            f"{keys_to_clean}"
        )
        for key in keys_to_clean:
            self._cleanup_entry(key)
        self._artifact_manager.invalidate_for_step(step_key)
        self._emit_node_state(step_key, NodeState.DIRTY)

    def invalidate_for_workpiece(self, workpiece_uid: str):
        """Invalidates all artifacts for a workpiece across all steps."""
        logger.debug(f"Invalidating artifacts for workpiece '{workpiece_uid}'")
        keys_to_clean = [
            k
            for k in self._artifact_manager.get_all_workpiece_keys()
            if k.id == workpiece_uid
        ]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def _cleanup_entry(self, key: ArtifactKey):
        """
        Removes a workpiece cache entry, releases its resources, and
        requests cancellation of its task.
        """
        logger.debug(f"WorkPiecePipelineStage: Cleaning up entry {key}.")
        task = self._task_manager.get_task(key)
        if task and task.is_running():
            logger.debug(f"Task {key} is running, canceling it")
            self._task_manager.cancel_task(key)
        self._artifact_manager.invalidate_for_workpiece(key)

    def validate_for_launch(
        self, key: ArtifactKey, workpiece: "WorkPiece"
    ) -> bool:
        """
        Validates workpiece size before launching.
        Returns True if launch should proceed, False otherwise.
        """
        if any(s <= 0 for s in workpiece.size):
            logger.warning(
                f"Skipping launch for {key}: invalid size {workpiece.size}"
            )
            self._cleanup_entry(key)
            return False
        return True

    def prepare_task_settings(
        self, step: "Step"
    ) -> Optional[Tuple[Dict, Optional[Dict]]]:
        """
        Prepares settings with machine parameters and selects laser.
        Returns (settings_dict, laser_dict) or None if preparation fails.
        """
        if not self._machine:
            logger.error("Cannot generate ops: No machine is configured.")
            return None

        settings = step.get_settings()
        settings["machine_supports_arcs"] = self._machine.supports_arcs
        settings["arc_tolerance"] = self._machine.arc_tolerance

        try:
            selected_laser = step.get_selected_laser(self._machine)
        except ValueError as e:
            logger.error(f"Cannot select laser for step '{step.name}': {e}")
            return None

        return settings, selected_laser.to_dict()

    def prepare_workpiece_dict(self, workpiece: "WorkPiece") -> Dict:
        """
        Prepares the fully-hydrated, serializable WorkPiece dictionary.
        """
        world_workpiece = workpiece.in_world()
        return world_workpiece.to_dict()

    def _on_task_event(
        self,
        task: "Task",
        event_name: str,
        data: dict,
        step: "Step",
        workpiece: "WorkPiece",
    ):
        """Handles events from a background task."""
        key = task.key
        step_uid = step.uid

        handle_dict = data.get("handle_dict")
        generation_id = data.get("generation_id")

        if generation_id is None or not handle_dict:
            logger.warning(
                f"Received malformed event '{event_name}' without "
                f"generation_id or handle_dict"
            )
            return

        try:
            if event_name == "visual_chunk_ready":
                with self._artifact_manager.transient_adoption(
                    key, handle_dict
                ) as chunk_handle:
                    self.visual_chunk_available.send(
                        self,
                        key=key,
                        chunk_handle=chunk_handle,
                        generation_id=generation_id,
                        step_uid=step_uid,
                    )
            elif event_name == "artifact_created":
                with self._artifact_manager.safe_adoption(
                    key, handle_dict
                ) as handle:
                    if self._artifact_manager.cache_handle(
                        key, handle, generation_id
                    ):
                        self._emit_node_state(key, NodeState.VALID)
                        self.workpiece_artifact_adopted.send(
                            self, step_uid=step_uid, workpiece_uid=key.id
                        )
        except StaleGenerationError as e:
            logger.debug(f"Discarding stale artifact event for {key}: {e}")
        except SharedMemoryNotFoundError as e:
            logger.debug(
                f"Shared memory not found for {key}, event may be from "
                f"terminated worker: {e}"
            )
        except Exception as e:
            logger.error(
                f"Failed to process event '{event_name}': {e}",
                exc_info=True,
            )

    def _on_task_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        generation_id: int,
        step: "Step",
        workpiece: "WorkPiece",
        context: Optional["GenerationContext"],
    ):
        """Callback for when an ops generation task finishes."""
        if context is not None:
            context.task_did_finish(key)

        task_status = task.get_status()
        logger.debug(f"[{key}] Task status is '{task_status}'.")

        if task_status == "canceled":
            with self._artifact_manager.report_cancellation(
                key, generation_id
            ) as handle:
                self._emit_node_state(key, NodeState.DIRTY)
                self.generation_finished.send(
                    self,
                    step=step,
                    workpiece=workpiece,
                    handle=handle,
                    generation_id=generation_id,
                    task_status="canceled",
                )
        elif task_status == "completed":
            try:
                task.result()
            except Exception as e:
                logger.error(f"[{key}] Error processing result: {e}")

            with self._artifact_manager.report_completion(
                key, generation_id
            ) as handle:
                if handle is None:
                    logger.debug(
                        f"[{key}] Task completed with no handle "
                        f"(empty workpiece)."
                    )
                self._emit_node_state(key, NodeState.VALID)
                self.generation_finished.send(
                    self,
                    step=step,
                    workpiece=workpiece,
                    handle=handle,
                    generation_id=generation_id,
                    task_status="completed",
                )
        else:
            wp_name = workpiece.name
            error_msg = (
                f"Ops generation for '{step.name}' on '{wp_name}' failed."
            )
            logger.warning(f"[{key}] {error_msg}")
            with self._artifact_manager.report_failure(
                key, generation_id
            ) as handle:
                self._emit_node_state(key, NodeState.ERROR)
                self.generation_finished.send(
                    self,
                    step=step,
                    workpiece=workpiece,
                    handle=handle,
                    generation_id=generation_id,
                    task_status="failed",
                )

    def launch_task(
        self,
        step: "Step",
        workpiece: "WorkPiece",
        generation_id: int,
        context: Optional["GenerationContext"],
    ):
        """Starts the asynchronous task to generate operations."""
        key = ArtifactKey.for_workpiece(workpiece.uid, step.uid)

        if not self.validate_for_launch(key, workpiece):
            logger.debug(
                f"Validation failed for launch "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return

        self.generation_starting.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=generation_id,
        )

        prep_result = self.prepare_task_settings(step)
        if prep_result is None:
            logger.debug(
                f"prep_result is None for "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return
        settings, laser_dict = prep_result

        workpiece_dict = self.prepare_workpiece_dict(workpiece)

        self._emit_node_state(key, NodeState.PROCESSING)

        if context is not None:
            context.add_task(key)

        self._create_and_register_task(
            key,
            workpiece_dict,
            step,
            workpiece,
            settings,
            laser_dict,
            generation_id,
            workpiece.size,
            context,
        )

    def _create_and_register_task(
        self,
        key: ArtifactKey,
        workpiece_dict: Dict,
        step: "Step",
        workpiece: "WorkPiece",
        settings: Dict,
        laser_dict: Optional[Dict],
        generation_id: int,
        workpiece_size: Tuple[float, float],
        context: Optional["GenerationContext"],
    ):
        """
        Creates subprocess task and registers it in active_tasks.
        """
        from .workpiece_runner import make_workpiece_artifact_in_subprocess

        self._task_manager.run_process(
            make_workpiece_artifact_in_subprocess,
            self._artifact_manager.get_store(),
            workpiece_dict,
            step.opsproducer_dict,
            step.per_workpiece_transformers_dicts,
            laser_dict,
            settings,
            generation_id,
            workpiece_size,
            "wp",
            key=key,
            when_done=lambda t: self._on_task_complete(
                t,
                key,
                generation_id,
                step,
                workpiece,
                context,
            ),
            when_event=lambda task, event_name, data: self._on_task_event(
                task, event_name, data, step, workpiece
            ),
        )

    def get_artifact(
        self,
        step_uid: str,
        workpiece_uid: str,
        workpiece_size: Tuple[float, float],
        generation_id: int,
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete, validated artifact from the cache."""
        handle = self._artifact_manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(workpiece_uid, step_uid),
            generation_id,
        )
        if handle is None:
            return None

        if not handle.is_scalable:
            if not sizes_are_close(handle.generation_size, workpiece_size):
                return None

        artifact = self._artifact_manager.get_artifact(handle)
        return artifact if isinstance(artifact, WorkPieceArtifact) else None

    def get_scaled_ops(
        self,
        step_uid: str,
        workpiece_uid: str,
        world_transform: "Matrix",
        generation_id: int,
    ) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache and scales them to the
        final world size.
        """
        final_size = world_transform.get_abs_scale()
        if any(s <= 0 for s in final_size):
            return None

        artifact = self.get_artifact(
            step_uid, workpiece_uid, final_size, generation_id
        )
        if artifact is None:
            return None

        ops = deepcopy(artifact.ops)
        if artifact.is_scalable:
            self._scale_ops_to_final_size(ops, artifact, final_size)

        scanline_count = sum(
            1 for cmd in ops.commands if isinstance(cmd, ScanLinePowerCommand)
        )
        logger.debug(
            f"Returning final ops for workpiece '{workpiece_uid}' with "
            f"{scanline_count} ScanLinePowerCommands."
        )
        return ops

    def _scale_ops_to_final_size(
        self,
        ops: Ops,
        artifact: WorkPieceArtifact,
        final_size_mm: Tuple[float, float],
    ):
        """
        Scales an Ops object from its source coordinate system to the
        provided final physical size in millimeters.
        """
        if not artifact.source_dimensions:
            logger.warning(
                "Cannot scale ops: artifact is missing source size."
            )
            return

        source_width, source_height = artifact.source_dimensions
        final_width_mm, final_height_mm = final_size_mm
        scale_x = final_width_mm / source_width if source_width > 1e-9 else 1.0
        scale_y = (
            final_height_mm / source_height if source_height > 1e-9 else 1.0
        )

        if not (math.isclose(scale_x, 1.0) and math.isclose(scale_y, 1.0)):
            logger.debug(f"Scaling ops by ({scale_x}, {scale_y})")
            ops.scale(scale_x, scale_y)
