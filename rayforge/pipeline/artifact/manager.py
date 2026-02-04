from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, Set, Any, Generator, Union
from .base import BaseArtifact
from .handle import BaseArtifactHandle, create_handle_from_dict
from .job import JobArtifactHandle
from .step_ops import StepOpsArtifactHandle
from .step_render import StepRenderArtifactHandle
from .store import ArtifactStore
from .workpiece import WorkPieceArtifactHandle

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
        self._ref_counts: Dict[Union[WorkPieceKey, StepKey, JobKey], int] = {}

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

    def get_artifact(self, handle: BaseArtifactHandle) -> BaseArtifact:
        """
        Retrieves an artifact from the store using its handle.

        Args:
            handle: The handle of the artifact to retrieve.

        Returns:
            The reconstructed artifact.
        """
        return self._store.get(handle)

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
        self,
        key: Union[WorkPieceKey, StepKey, JobKey],
        handle_dict: Dict[str, Any],
        in_process: bool = False,
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
        self._store.adopt(handle, increment_refcount=not in_process)
        return handle

    def retain_handle(self, handle: BaseArtifactHandle):
        """
        Retains a handle's shared memory resources.

        This increments the reference count for the handle's shared
        memory block, preventing it from being released prematurely.
        Use this when you need to hold a reference outside of a
        checkout context manager (e.g., in async callbacks).

        Args:
            handle: The handle to retain.
        """
        self._store.retain(handle)

    def release_handle(self, handle: Optional[BaseArtifactHandle]):
        """
        Safely releases a handle's shared memory resources.

        This decrements the reference count for the handle's shared
        memory block. When the count reaches zero, the memory is freed.

        Args:
            handle: The handle to release. If None, does nothing.
        """
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
        self._ref_counts.clear()
        logger.info("All cached artifacts released.")

    @contextmanager
    def checkout_handle(
        self, handle: Optional[BaseArtifactHandle]
    ) -> Generator[Optional[BaseArtifact], None, None]:
        """
        Context manager for safely using an artifact from its handle.
        This retains the handle on entry and releases it on exit.
        """
        if handle is None:
            yield None
            return

        self.retain_handle(handle)
        try:
            yield self.get_artifact(handle)
        finally:
            self.release_handle(handle)

    @contextmanager
    def checkout(
        self, key: Union[WorkPieceKey, StepKey, JobKey]
    ) -> Generator[BaseArtifactHandle, None, None]:
        """
        Context manager for checking out an artifact handle with reference
        counting.

        This method increments the reference count for the key and retains
        the underlying shared memory block. When the context exits, the
        reference count is decremented and the block is released.

        Args:
            key: The key identifying the artifact to checkout.

        Yields:
            The artifact handle.

        Raises:
            KeyError: If the key is not found in the cached handles.
        """
        handle = self._get_handle_by_key(key)
        if handle is None:
            raise KeyError(f"No handle found for key: {key}")

        self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
        self._store.retain(handle)

        try:
            yield handle
        finally:
            self._store.release(handle)
            self._ref_counts[key] -= 1
            if self._ref_counts[key] <= 0:
                del self._ref_counts[key]

    def _get_handle_by_key(
        self, key: Union[WorkPieceKey, StepKey, JobKey]
    ) -> Optional[BaseArtifactHandle]:
        """Get the handle for a given key, regardless of key type."""
        if isinstance(key, tuple):
            return self._workpiece_handles.get(key)
        elif key == self.JOB_KEY:
            return self._job_handle
        else:
            return self._step_render_handles.get(
                key
            ) or self._step_ops_handles.get(key)
