from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional, Set, Any
from .handle import BaseArtifactHandle, create_handle_from_dict
from .job import JobArtifactHandle
from .step_ops import StepOpsArtifactHandle
from .step_render import StepRenderArtifactHandle
from .workpiece import WorkPieceArtifactHandle
from .store import ArtifactStore

logger = logging.getLogger(__name__)

WorkPieceKey = Tuple[str, str]
StepKey = str
JobKey = str


class ArtifactManager:
    """
    A centralized manager for artifact handles that understands the
    dependency graph between different artifact types.
    """

    JOB_KEY: JobKey = "final_job"

    def __init__(self, store: ArtifactStore):
        self._store = store
        self._workpiece_handles: Dict[
            WorkPieceKey, WorkPieceArtifactHandle
        ] = {}
        self._step_render_handles: Dict[StepKey, StepRenderArtifactHandle] = {}
        self._step_ops_handles: Dict[StepKey, StepOpsArtifactHandle] = {}
        self._job_handle: Optional[JobArtifactHandle] = None

    def get_workpiece_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[WorkPieceArtifactHandle]:
        return self._workpiece_handles.get((step_uid, workpiece_uid))

    def get_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        return self._step_render_handles.get(step_uid)

    def get_step_ops_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        return self._step_ops_handles.get(step_uid)

    def get_job_handle(self) -> Optional[JobArtifactHandle]:
        return self._job_handle

    def put_workpiece_handle(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceArtifactHandle,
    ):
        """Stores a handle for a WorkPieceArtifact and invalidates deps."""
        key = (step_uid, workpiece_uid)
        old_handle = self._workpiece_handles.pop(key, None)
        self.release_handle(old_handle)

        self._workpiece_handles[key] = handle

        ops_handle = self._step_ops_handles.pop(step_uid, None)
        self.release_handle(ops_handle)

        self.invalidate_for_job()

    def put_step_render_handle(
        self, step_uid: str, handle: StepRenderArtifactHandle
    ):
        """Stores a handle for a StepRenderArtifact."""
        old_handle = self._step_render_handles.pop(step_uid, None)
        self.release_handle(old_handle)
        self._step_render_handles[step_uid] = handle

    def put_step_ops_handle(
        self, step_uid: str, handle: StepOpsArtifactHandle
    ):
        """Stores a handle for a StepOpsArtifact and invalidates final job."""
        old_handle = self._step_ops_handles.pop(step_uid, None)
        self.release_handle(old_handle)
        self._step_ops_handles[step_uid] = handle
        self.invalidate_for_job()

    def put_job_handle(self, handle: JobArtifactHandle):
        """Stores the handle for the final JobArtifact."""
        if self._job_handle:
            self.invalidate_for_job()
        self._job_handle = handle

    def adopt_artifact(
        self, key: WorkPieceKey | StepKey | JobKey, handle_dict: Dict[str, Any]
    ) -> BaseArtifactHandle:
        """
        Adopts an artifact from a subprocess and deserializes the handle.

        This method does NOT cache the handle. It serves as a
        factory for stages to acquire handles from raw dictionaries.
        Stages must explicitly call `put_*_handle` to treat the result
        as the canonical cached artifact.

        Args:
            key: The key context for logging (unused for logic).
            handle_dict: The serialized handle dictionary.

        Returns:
            The adopted, deserialized handle.
        """
        handle = create_handle_from_dict(handle_dict)
        self._store.adopt(handle)
        return handle

    def release_handle(self, handle: Optional[BaseArtifactHandle]):
        """Safely releases a handle's shared memory resources."""
        if handle:
            self._store.release(handle)

    def has_step_render_handle(self, step_uid: str) -> bool:
        return step_uid in self._step_render_handles

    def get_all_step_render_uids(self) -> Set[str]:
        return set(self._step_render_handles.keys())

    def get_all_workpiece_keys(self) -> Set[WorkPieceKey]:
        return set(self._workpiece_handles.keys())

    def pop_step_ops_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        return self._step_ops_handles.pop(step_uid, None)

    def pop_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        return self._step_render_handles.pop(step_uid, None)

    def invalidate_for_workpiece(self, step_uid: str, workpiece_uid: str):
        key = (step_uid, workpiece_uid)
        handle = self._workpiece_handles.pop(key, None)
        self.release_handle(handle)

        ops_handle = self._step_ops_handles.pop(step_uid, None)
        self.release_handle(ops_handle)

        self.invalidate_for_job()

    def invalidate_for_step(self, step_uid: str):
        keys_to_remove = [
            k for k in self._workpiece_handles if k[0] == step_uid
        ]
        for key in keys_to_remove:
            handle = self._workpiece_handles.pop(key, None)
            self.release_handle(handle)

        step_render_handle = self._step_render_handles.pop(step_uid, None)
        self.release_handle(step_render_handle)
        step_ops_handle = self._step_ops_handles.pop(step_uid, None)
        self.release_handle(step_ops_handle)

        self.invalidate_for_job()

    def invalidate_for_job(self):
        handle = self._job_handle
        self._job_handle = None
        self.release_handle(handle)

    def shutdown(self):
        logger.info(
            "ArtifactManager shutting down and releasing all artifacts."
        )
        for handle in self._workpiece_handles.values():
            self.release_handle(handle)
        self._workpiece_handles.clear()

        for handle in self._step_render_handles.values():
            self.release_handle(handle)
        self._step_render_handles.clear()

        for handle in self._step_ops_handles.values():
            self.release_handle(handle)
        self._step_ops_handles.clear()

        self.release_handle(self._job_handle)
        self._job_handle = None
        logger.info("All cached artifacts released.")
