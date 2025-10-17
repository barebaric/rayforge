from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
from .handle import BaseArtifactHandle
from .workpiece import WorkPieceArtifactHandle
from .step import StepArtifactHandle
from .job import JobArtifactHandle
from .store import ArtifactStore

logger = logging.getLogger(__name__)

# Type aliases for cache keys
WorkPieceKey = Tuple[str, str]  # (step_uid, workpiece_uid)
StepKey = str  # step_uid
JobKey = str  # A constant, as there's only one job artifact at a time


class ArtifactCache:
    """
    A centralized, stateful manager for artifact handles that understands the
    dependency graph between different artifact types.

    This class is the single source of truth for cached artifact handles, and
    it manages their lifecycle, including cascading invalidation and the
    release of underlying shared memory resources.
    """

    JOB_KEY: JobKey = "final_job"

    def __init__(self):
        self._workpiece_handles: Dict[
            WorkPieceKey, WorkPieceArtifactHandle
        ] = {}
        self._step_handles: Dict[StepKey, StepArtifactHandle] = {}
        self._job_handle: Optional[JobArtifactHandle] = None

    def get_workpiece_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[WorkPieceArtifactHandle]:
        """Retrieves a handle for a WorkPieceArtifact."""
        return self._workpiece_handles.get((step_uid, workpiece_uid))

    def put_workpiece_handle(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceArtifactHandle,
    ):
        """Stores a handle for a WorkPieceArtifact and invalidates deps."""
        key = (step_uid, workpiece_uid)
        old_handle = self._workpiece_handles.pop(key, None)
        self._release_handle(old_handle)

        self._workpiece_handles[key] = handle

        # Invalidate parent step
        step_handle = self._step_handles.pop(step_uid, None)
        self._release_handle(step_handle)

        # Invalidate grandparent job
        self.invalidate_for_job()

    def get_step_handle(self, step_uid: str) -> Optional[StepArtifactHandle]:
        """Retrieves a handle for a StepArtifact."""
        return self._step_handles.get(step_uid)

    def put_step_handle(self, step_uid: str, handle: StepArtifactHandle):
        """Stores a handle for a StepArtifact and invalidates deps."""
        old_handle = self._step_handles.pop(step_uid, None)
        self._release_handle(old_handle)

        self._step_handles[step_uid] = handle
        # A new step always invalidates the final job.
        self.invalidate_for_job()

    def get_job_handle(self) -> Optional[JobArtifactHandle]:
        """Retrieves the handle for the final JobArtifact."""
        return self._job_handle

    def put_job_handle(self, handle: JobArtifactHandle):
        """Stores the handle for the final JobArtifact."""
        if self._job_handle:
            self.invalidate_for_job()
        self._job_handle = handle

    def _release_handle(self, handle: Optional[BaseArtifactHandle]):
        """Safely releases a handle's shared memory resources."""
        if handle:
            ArtifactStore.release(handle)

    def invalidate_for_workpiece(self, step_uid: str, workpiece_uid: str):
        """
        Invalidates a specific workpiece and all downstream artifacts that
        depend on it (its parent step and the final job).
        """
        key = (step_uid, workpiece_uid)
        handle = self._workpiece_handles.pop(key, None)
        self._release_handle(handle)

        # Invalidate parent step
        step_handle = self._step_handles.pop(step_uid, None)
        self._release_handle(step_handle)

        # Invalidate grandparent job
        self.invalidate_for_job()

    def invalidate_for_step(self, step_uid: str):
        """
        Invalidates a step, all workpieces belonging to it, and the final job.
        """
        # 1. Invalidate all associated WorkPieceArtifacts first
        keys_to_remove = [
            k for k in self._workpiece_handles if k[0] == step_uid
        ]
        for key in keys_to_remove:
            handle = self._workpiece_handles.pop(key, None)
            self._release_handle(handle)

        # 2. Invalidate the StepArtifact itself
        step_handle = self._step_handles.pop(step_uid, None)
        self._release_handle(step_handle)

        # 3. Invalidate the downstream JobArtifact
        self.invalidate_for_job()

    def invalidate_for_job(self):
        """Invalidates only the final job artifact."""
        handle = self._job_handle
        self._job_handle = None
        self._release_handle(handle)

    def shutdown(self):
        """
        Clears all caches and releases all associated shared memory resources.
        This must be called on application exit to prevent memory leaks.
        """
        logger.info("ArtifactCache shutting down and releasing all artifacts.")
        for handle in self._workpiece_handles.values():
            self._release_handle(handle)
        self._workpiece_handles.clear()

        for handle in self._step_handles.values():
            self._release_handle(handle)
        self._step_handles.clear()

        self._release_handle(self._job_handle)
        self._job_handle = None
        logger.info("All cached artifacts released.")
