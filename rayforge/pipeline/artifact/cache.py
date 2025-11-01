from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional, Set
from ...context import get_context
from .handle import BaseArtifactHandle
from .job import JobArtifactHandle
from .step_ops import StepOpsArtifactHandle
from .step_render import StepRenderArtifactHandle
from .workpiece import WorkPieceArtifactHandle

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
        self._step_render_handles: Dict[StepKey, StepRenderArtifactHandle] = {}
        self._step_ops_handles: Dict[StepKey, StepOpsArtifactHandle] = {}
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
    ) -> Optional[WorkPieceArtifactHandle]:
        """
        Stores a handle for a WorkPieceArtifact, atomically invalidates
        downstream dependencies, and returns the old handle.
        """
        key = (step_uid, workpiece_uid)
        old_handle = self._workpiece_handles.pop(key, None)
        self._release_handle(old_handle)
        self._workpiece_handles[key] = handle

        # Atomically invalidate the parent step's ops artifact and the job.
        # This is the correct place for this logic.
        ops_handle = self._step_ops_handles.pop(step_uid, None)
        self._release_handle(ops_handle)
        self.invalidate_for_job()

        return old_handle

    def get_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        """Retrieves a handle for a StepRenderArtifact."""
        return self._step_render_handles.get(step_uid)

    def put_step_render_handle(
        self, step_uid: str, handle: StepRenderArtifactHandle
    ):
        """Stores a handle for a StepRenderArtifact."""
        old_handle = self._step_render_handles.pop(step_uid, None)
        self._release_handle(old_handle)
        self._step_render_handles[step_uid] = handle

    def get_step_ops_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        """Retrieves a handle for a StepOpsArtifact."""
        return self._step_ops_handles.get(step_uid)

    def put_step_ops_handle(
        self, step_uid: str, handle: StepOpsArtifactHandle
    ):
        """
        Stores a handle for a StepOpsArtifact and invalidates the final job.
        """
        old_handle = self._step_ops_handles.pop(step_uid, None)
        self._release_handle(old_handle)
        self._step_ops_handles[step_uid] = handle
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
            get_context().artifact_store.release(handle)

    def has_step_render_handle(self, step_uid: str) -> bool:
        """Checks if a StepRenderArtifact handle exists for a step UID."""
        return step_uid in self._step_render_handles

    def get_all_step_render_uids(self) -> Set[str]:
        """Returns a set of all step UIDs that have a render handle."""
        return set(self._step_render_handles.keys())

    def get_all_workpiece_keys(self) -> Set[WorkPieceKey]:
        """Returns a set of all (step_uid, workpiece_uid) keys."""
        return set(self._workpiece_handles.keys())

    def get_all_workpiece_handles_as_dict(
        self,
    ) -> Dict[WorkPieceKey, WorkPieceArtifactHandle]:
        """Returns a copy of the internal workpiece handles dictionary."""
        return self._workpiece_handles.copy()

    def pop_step_ops_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        """Removes and returns a StepOpsArtifact handle, if it exists."""
        return self._step_ops_handles.pop(step_uid, None)

    def pop_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        """Removes and returns a StepRenderArtifact handle, if it exists."""
        return self._step_render_handles.pop(step_uid, None)

    def invalidate_for_workpiece(self, step_uid: str, workpiece_uid: str):
        """
        Invalidates only a specific workpiece artifact. Does NOT cascade.
        """
        key = (step_uid, workpiece_uid)
        handle = self._workpiece_handles.pop(key, None)
        self._release_handle(handle)

    def invalidate_for_step(self, step_uid: str):
        """
        Invalidates all artifacts for a specific step and downstream.
        """
        # 1. Invalidate all associated WorkPieceArtifacts first
        keys_to_remove = [
            k for k in self._workpiece_handles if k[0] == step_uid
        ]
        for key in keys_to_remove:
            handle = self._workpiece_handles.pop(key, None)
            self._release_handle(handle)

        # 2. Invalidate the StepArtifacts themselves
        step_render_handle = self._step_render_handles.pop(step_uid, None)
        self._release_handle(step_render_handle)
        step_ops_handle = self._step_ops_handles.pop(step_uid, None)
        self._release_handle(step_ops_handle)

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

        for handle in self._step_render_handles.values():
            self._release_handle(handle)
        self._step_render_handles.clear()

        for handle in self._step_ops_handles.values():
            self._release_handle(handle)
        self._step_ops_handles.clear()

        self._release_handle(self._job_handle)
        self._job_handle = None
        logger.info("All cached artifacts released.")
