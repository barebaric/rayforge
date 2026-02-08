from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from .base import BaseArtifact
from .handle import BaseArtifactHandle, create_handle_from_dict
from .job import JobArtifactHandle
from .lifecycle import ArtifactLifecycle, LedgerEntry
from .step_ops import StepOpsArtifactHandle
from .step_render import StepRenderArtifactHandle
from .store import ArtifactStore
from .workpiece import WorkPieceArtifactHandle
from .workpiece_view import (
    WorkPieceViewArtifactHandle,
    RenderContext,
)

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
        self._step_render_handles: Dict[StepKey, StepRenderArtifactHandle] = {}
        self._workpiece_view_handles: Dict[
            WorkPieceKey, WorkPieceViewArtifactHandle
        ] = {}
        self._ref_counts: Dict[Union[WorkPieceKey, StepKey, JobKey], int] = {}
        self._ledger: Dict[Any, LedgerEntry] = {}
        self._dependencies: Dict[Any, List[Any]] = {}

    def get_workpiece_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[WorkPieceArtifactHandle]:
        key = ("workpiece", step_uid, workpiece_uid)
        entry = self._ledger.get(key)
        if entry is None or entry.state != ArtifactLifecycle.READY:
            return None
        handle = entry.handle
        if isinstance(handle, WorkPieceArtifactHandle):
            return handle
        return None

    def get_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        return self._step_render_handles.get(step_uid)

    def get_step_ops_handle(
        self, step_uid: str
    ) -> Optional[StepOpsArtifactHandle]:
        key = ("step", step_uid)
        entry = self._ledger.get(key)
        if entry is None or entry.state != ArtifactLifecycle.READY:
            return None
        handle = entry.handle
        if isinstance(handle, StepOpsArtifactHandle):
            return handle
        return None

    def get_job_handle(self) -> Optional[JobArtifactHandle]:
        key = self.JOB_KEY
        entry = self._ledger.get(key)
        if entry is None or entry.state != ArtifactLifecycle.READY:
            return None
        handle = entry.handle
        if isinstance(handle, JobArtifactHandle):
            return handle
        return None

    def get_workpiece_view_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[WorkPieceViewArtifactHandle]:
        key = ("view", step_uid, workpiece_uid)
        entry = self._ledger.get(key)
        if entry is None or entry.state != ArtifactLifecycle.READY:
            return None
        handle = entry.handle
        if isinstance(handle, WorkPieceViewArtifactHandle):
            return handle
        return None

    def is_view_stale(
        self,
        step_uid: str,
        workpiece_uid: str,
        new_context: Optional["RenderContext"],
        source_handle: Optional[WorkPieceArtifactHandle],
    ) -> bool:
        """
        Check if a workpiece view is stale and needs regeneration.

        Args:
            step_uid: The step identifier.
            workpiece_uid: The workpiece identifier.
            new_context: The new render context to compare against.
            source_handle: The source workpiece handle to compare properties.

        Returns:
            True if the view is missing, stale, or context/properties changed.
            False if the view is still valid.
        """
        key = ("view", step_uid, workpiece_uid)
        entry = self._ledger.get(key)

        if entry is None:
            return True

        if entry.state != ArtifactLifecycle.READY:
            return True

        metadata = entry.metadata

        if new_context is not None:
            stored_context = metadata.get("render_context")
            if stored_context is None:
                return True
            if not isinstance(stored_context, RenderContext):
                return True
            if stored_context != new_context:
                return True

        if source_handle is not None:
            stored_props = metadata.get("source_properties")
            if stored_props is None:
                return True
            if stored_props.get("is_scalable") != source_handle.is_scalable:
                return True
            if (
                stored_props.get("source_coordinate_system_name")
                != source_handle.source_coordinate_system_name
            ):
                return True
            if stored_props.get("generation_size") != (
                source_handle.generation_size
            ):
                return True
            if stored_props.get("source_dimensions") != (
                source_handle.source_dimensions
            ):
                return True

        return False

    def get_artifact(self, handle: BaseArtifactHandle) -> BaseArtifact:
        """
        Retrieves an artifact from the store using its handle.

        Args:
            handle: The handle of the artifact to retrieve.

        Returns:
            The reconstructed artifact.
        """
        return self._store.get(handle)

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
        """Stores a handle for a StepOpsArtifact."""
        key = ("step", step_uid)
        entry = self._ledger.get(key)
        if entry is not None and entry.handle is not None:
            self.release_handle(entry.handle)
        self._ledger[key] = LedgerEntry(
            handle=handle,
            state=ArtifactLifecycle.READY,
            generation_id=0,
            error=None,
        )

    def put_workpiece_view_handle(
        self,
        step_uid: str,
        workpiece_uid: str,
        handle: WorkPieceViewArtifactHandle,
    ):
        """Stores a handle for a WorkPieceViewArtifact."""
        key = ("view", step_uid, workpiece_uid)
        entry = self._ledger.get(key)
        if entry is not None and entry.handle is not None:
            self.release_handle(entry.handle)
        self._ledger[key] = LedgerEntry(
            handle=handle,
            state=ArtifactLifecycle.READY,
            generation_id=0,
            error=None,
        )

    def adopt_artifact(
        self,
        key: Union[WorkPieceKey, StepKey, JobKey],
        handle_dict: Dict[str, Any],
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
        return {
            (key[1], key[2])
            for key in self._ledger.keys()
            if isinstance(key, tuple) and key[0] == "workpiece"
        }

    def pop_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        return self._step_render_handles.pop(step_uid, None)

    def invalidate_for_workpiece(self, step_uid: str, workpiece_uid: str):
        ledger_key = ("workpiece", step_uid, workpiece_uid)
        entry = self._ledger.get(ledger_key)
        if entry and entry.handle is not None:
            self.release_handle(entry.handle)
        self.invalidate(ledger_key)

        # Invalidate the step ops artifact since it depends on workpieces
        step_ledger_key = ("step", step_uid)
        step_entry = self._ledger.get(step_ledger_key)
        if step_entry and step_entry.handle is not None:
            self.release_handle(step_entry.handle)
        self.invalidate(step_ledger_key)

        view_key = ("view", step_uid, workpiece_uid)
        view_entry = self._ledger.get(view_key)
        if view_entry and view_entry.handle is not None:
            self.release_handle(view_entry.handle)
        self.invalidate(view_key)

        self.invalidate_for_job()

    def invalidate_for_step(self, step_uid: str):
        step_ledger_key = ("step", step_uid)
        step_entry = self._ledger.get(step_ledger_key)
        if step_entry and step_entry.handle is not None:
            self.release_handle(step_entry.handle)
        self.invalidate(step_ledger_key)

        keys_to_invalidate = [
            key
            for key in self._ledger.keys()
            if isinstance(key, tuple)
            and key[0] == "workpiece"
            and key[1] == step_uid
        ]
        for ledger_key in keys_to_invalidate:
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                self.release_handle(entry.handle)
            self.invalidate(ledger_key)

        view_keys_to_invalidate = [
            ("view", k[0], k[1])
            for k in self._ledger.keys()
            if isinstance(k, tuple) and k[0] == "view" and k[1] == step_uid
        ]
        for view_key in view_keys_to_invalidate:
            view_entry = self._ledger.get(view_key)
            if view_entry and view_entry.handle is not None:
                self.release_handle(view_entry.handle)
            self.invalidate(view_key)

        step_render_handle = self._step_render_handles.pop(step_uid, None)
        self.release_handle(step_render_handle)
        self.invalidate_for_job()

    def invalidate_for_job(self):
        ledger_key = self.JOB_KEY
        entry = self._ledger.get(ledger_key)
        if entry and entry.handle is not None:
            self.release_handle(entry.handle)
        self.invalidate(ledger_key)

    def shutdown(self):
        logger.info(
            "ArtifactManager shutting down and releasing all artifacts."
        )
        for entry in self._ledger.values():
            if entry.handle is not None:
                self.release_handle(entry.handle)
        self._ledger.clear()

        for handle in self._step_render_handles.values():
            self.release_handle(handle)
        self._step_render_handles.clear()

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
            ledger_key = ("workpiece", key[0], key[1])
            entry = self._ledger.get(ledger_key)
            if entry and entry.state == ArtifactLifecycle.READY:
                handle = entry.handle
                if isinstance(handle, WorkPieceArtifactHandle):
                    return handle
            view_key = ("view", key[0], key[1])
            view_entry = self._ledger.get(view_key)
            if view_entry and view_entry.state == ArtifactLifecycle.READY:
                handle = view_entry.handle
                if isinstance(handle, WorkPieceViewArtifactHandle):
                    return handle
            return None
        elif key == self.JOB_KEY:
            entry = self._ledger.get(key)
            if entry and entry.state == ArtifactLifecycle.READY:
                return entry.handle
            return None
        else:
            return self._step_render_handles.get(key)

    def _get_ledger_entry(self, key: Any) -> Optional[LedgerEntry]:
        """Returns the LedgerEntry for a key, or None if not found."""
        return self._ledger.get(key)

    def _set_ledger_entry(self, key: Any, entry: LedgerEntry) -> None:
        """Sets a LedgerEntry for a key."""
        self._ledger[key] = entry

    def _register_dependency(self, child_key: Any, parent_key: Any) -> None:
        """Registers that parent_key depends on child_key."""
        if parent_key not in self._dependencies:
            self._dependencies[parent_key] = []
        if child_key not in self._dependencies[parent_key]:
            self._dependencies[parent_key].append(child_key)

    def _get_dependents(self, key: Any) -> List[Any]:
        """Returns all parent keys that depend on this key."""
        dependents: List[Any] = []
        for parent_key, children in self._dependencies.items():
            if key in children:
                dependents.append(parent_key)
        return dependents

    def _get_dependencies(self, key: Any) -> List[Any]:
        """Returns all child keys this key depends on."""
        return self._dependencies.get(key, [])

    def invalidate(self, key: Any) -> None:
        """
        Sets the entry's state to STALE.

        Recursively invalidates all parent dependents found via the
        dependency graph. If key doesn't exist in ledger, does nothing.
        """
        entry = self._get_ledger_entry(key)
        if entry is None:
            return
        entry.state = ArtifactLifecycle.STALE
        for dependent in self._get_dependents(key):
            self.invalidate(dependent)

    def has_pending_work(self) -> bool:
        """
        Returns True if any artifact in the ledger is being generated,
        is stale, or is missing.
        """
        if not self._ledger:
            return False

        for entry in self._ledger.values():
            if entry.state in (
                ArtifactLifecycle.PENDING,
                ArtifactLifecycle.STALE,
                ArtifactLifecycle.MISSING,
            ):
                return True
        return False

    def query_work_for_stage(self, stage_type: str) -> List[Any]:
        """
        Returns a list of keys that are STALE or MISSING and whose
        dependencies are all READY.

        This is the core scheduling logic. Keys are filtered by
        stage_type (either the key itself for string keys, or the
        first element of the key tuple).
        """
        work_items = []
        for key, entry in self._ledger.items():
            if entry.state not in (
                ArtifactLifecycle.STALE,
                ArtifactLifecycle.MISSING,
            ):
                continue
            if isinstance(key, tuple):
                if key[0] != stage_type:
                    continue
            elif key != stage_type:
                continue
            deps = self._get_dependencies(key)
            all_ready = True
            for dep in deps:
                dep_entry = self._get_ledger_entry(dep)
                if dep_entry is None or dep_entry.state != (
                    ArtifactLifecycle.READY
                ):
                    all_ready = False
                    break
            if all_ready:
                work_items.append(key)
        return work_items

    def mark_pending(
        self,
        key: Any,
        generation_id: int,
        source_handle: Optional[BaseArtifactHandle] = None,
    ) -> None:
        """
        Transitions state from STALE or MISSING to PENDING.

        Asserts the previous state was valid for this transition.
        Updates the generation_id and clears any error message.
        Stores the source_handle in metadata for later release.
        """
        entry = self._get_ledger_entry(key)
        assert entry is not None, f"Key {key} not found in ledger"
        assert entry.state in (
            ArtifactLifecycle.STALE,
            ArtifactLifecycle.MISSING,
        ), f"Cannot mark {entry.state} as PENDING"
        entry.state = ArtifactLifecycle.PENDING
        entry.generation_id = generation_id
        entry.error = None
        if source_handle is not None:
            entry.metadata["source_handle"] = source_handle

    def commit(
        self, key: Any, handle: BaseArtifactHandle, generation_id: int
    ) -> None:
        """
        Commits a PENDING entry to READY.

        Asserts current state is PENDING and generation_id matches.
        Adopts the new handle, releases the old handle if one existed,
        updates state to READY, and clears error message.
        Releases the source_handle from metadata if present.
        """
        entry = self._get_ledger_entry(key)
        assert entry is not None, f"Key {key} not found in ledger"
        assert entry.state == ArtifactLifecycle.PENDING, (
            f"Cannot commit {entry.state} entry"
        )
        assert entry.generation_id == generation_id, (
            f"Generation ID mismatch: {entry.generation_id} != {generation_id}"
        )
        self._store.adopt(handle)
        if entry.handle is not None:
            self._store.release(entry.handle)
        entry.handle = handle
        entry.state = ArtifactLifecycle.READY
        entry.error = None

        source_handle = entry.metadata.pop("source_handle", None)
        if source_handle is not None:
            self._store.release(source_handle)

    def mark_error(self, key: Any, error_msg: str, generation_id: int) -> None:
        """
        Marks a PENDING entry as ERROR.

        Asserts state is PENDING and generation_id matches.
        Stores the error message.
        Releases the source_handle from metadata if present.
        """
        entry = self._get_ledger_entry(key)
        assert entry is not None, f"Key {key} not found in ledger"
        assert entry.state == ArtifactLifecycle.PENDING, (
            f"Cannot mark {entry.state} as ERROR"
        )
        assert entry.generation_id == generation_id, (
            f"Generation ID mismatch: {entry.generation_id} != {generation_id}"
        )
        entry.state = ArtifactLifecycle.ERROR
        entry.error = error_msg

        source_handle = entry.metadata.pop("source_handle", None)
        if source_handle is not None:
            self._store.release(source_handle)

    def checkout_dependencies(self, key: Any) -> Dict[Any, BaseArtifactHandle]:
        """
        Returns a dict mapping dependency keys to their handles.

        Looks up all children (dependencies) of key and verifies they
        are all READY.
        """
        deps = self._get_dependencies(key)
        result = {}
        for dep in deps:
            entry = self._get_ledger_entry(dep)
            if entry is None:
                logger.error(
                    f"checkout_dependencies: Dependency {dep} not in ledger. "
                    f"Ledger keys: {list(self._ledger.keys())}"
                )
            assert entry is not None, f"Dependency {dep} not in ledger"
            assert entry.state == ArtifactLifecycle.READY, (
                f"Dependency {dep} is not READY: {entry.state}"
            )
            assert entry.handle is not None, f"Dependency {dep} has no handle"
            result[dep] = entry.handle
        return result

    def sync_keys(self, stage_type: str, valid_keys: Set[Any]) -> None:
        """
        Synchronizes the ledger with the current valid keys for a stage type.

        Adds MISSING entries for new keys and removes entries for deleted keys.
        Also removes dependency entries for deleted keys.

        Args:
            stage_type: The stage type string (e.g., 'workpiece', 'step',
              'job').
            valid_keys: A set of keys that should exist for this stage type.
        """
        for key in valid_keys:
            if key not in self._ledger:
                self._ledger[key] = LedgerEntry(
                    state=ArtifactLifecycle.MISSING
                )

        keys_to_remove = []
        for key in self._ledger.keys():
            if isinstance(key, tuple) and key[0] == stage_type:
                if key not in valid_keys:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._ledger[key]
            self._dependencies.pop(key, None)

            for parent_key, children in list(self._dependencies.items()):
                if key in children:
                    children.remove(key)
