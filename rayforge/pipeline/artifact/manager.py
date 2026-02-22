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
from ...shared.util.debug import safe_caller_stack
from ..dag.node import NodeState
from .base import BaseArtifact
from .handle import BaseArtifactHandle
from .job import JobArtifactHandle
from .key import ArtifactKey
from .lifecycle import LedgerEntry
from .step_ops import StepOpsArtifactHandle
from .step_render import StepRenderArtifactHandle
from .store import ArtifactStore
from .workpiece import WorkPieceArtifactHandle

logger = logging.getLogger(__name__)

GenerationID = int


class StaleGenerationError(Exception):
    """Raised when adopting an artifact with an outdated generation."""

    pass


def make_composite_key(
    base_key: ArtifactKey,
    generation_id: GenerationID,
) -> Tuple[ArtifactKey, GenerationID]:
    """Create a composite key (ArtifactKey, GenerationID)."""
    return (base_key, generation_id)


def extract_base_key(key: Any) -> ArtifactKey:
    """Extract the base ArtifactKey from a composite key or return base key."""
    if isinstance(key, ArtifactKey):
        return key
    if (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], ArtifactKey)
    ):
        return key[0]
    raise ValueError(f"Invalid key format: {key}")


def extract_generation_id(composite_key: Tuple) -> GenerationID:
    """Extract the generation ID from a composite key."""
    return composite_key[1]


class ArtifactManager:
    """
    A pure cache manager for artifact handles.

    This class is responsible only for caching and retrieving handles.
    State tracking is handled by the DAG scheduler via ArtifactNode.
    """

    def __init__(
        self,
        store: ArtifactStore,
    ):
        self._store = store
        self._step_render_handles: Dict[str, StepRenderArtifactHandle] = {}
        self._ref_counts: Dict[ArtifactKey, int] = {}
        self._ledger: Dict[Tuple[ArtifactKey, GenerationID], LedgerEntry] = {}
        self._dependencies: Dict[ArtifactKey, List[ArtifactKey]] = {}

    def get_workpiece_handle(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Optional[WorkPieceArtifactHandle]:
        composite_key = make_composite_key(key, generation_id)
        logger.debug(
            f"get_workpiece_handle: key={key}, "
            f"generation_id={generation_id}, "
            f"composite_key={composite_key}"
        )
        entry = self._ledger.get(composite_key)
        logger.debug(
            f"get_workpiece_handle: entry={entry}, "
            f"handle={entry.handle if entry else None}"
        )
        if entry is None or entry.handle is None:
            return None
        handle = entry.handle
        if isinstance(handle, WorkPieceArtifactHandle):
            return handle
        return None

    def get_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        return self._step_render_handles.get(step_uid)

    def is_generation_current(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> bool:
        """
        Check if a generation ID is current (not stale).

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID to check.

        Returns:
            True if the generation is current, False if the entry
            does not exist or has a different generation ID.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None:
            return False
        return entry.generation_id == generation_id

    def get_state(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> "NodeState":
        """
        Get the state of an artifact entry.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.

        Returns:
            The NodeState of the entry, or DIRTY if not found.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None:
            return NodeState.DIRTY
        return entry.state

    def set_state(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
        state: "NodeState",
    ) -> None:
        """
        Set the state of an artifact entry.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.
            state: The new NodeState to set.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is not None:
            entry.state = state

    def has_artifact(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> bool:
        """
        Check if an artifact handle exists for the given key and generation.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.

        Returns:
            True if a handle exists, False otherwise.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        return entry is not None and entry.handle is not None

    def get_step_ops_handle(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Optional[StepOpsArtifactHandle]:
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None or entry.handle is None:
            return None
        handle = entry.handle
        if isinstance(handle, StepOpsArtifactHandle):
            return handle
        return None

    def get_job_handle(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Optional[JobArtifactHandle]:
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None or entry.handle is None:
            return None
        handle = entry.handle
        if isinstance(handle, JobArtifactHandle):
            return handle
        return None

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
        """
        Stores a handle for a StepRenderArtifact.

        Releases any existing handle for this step before storing the new
        one to prevent memory leaks during rapid re-renders.
        """
        old_handle = self._step_render_handles.get(step_uid)
        if old_handle is not None:
            self.release_handle(old_handle)
        self._step_render_handles[step_uid] = handle

    def put_step_ops_handle(
        self,
        key: ArtifactKey,
        handle: StepOpsArtifactHandle,
        generation_id: GenerationID,
    ):
        """
        Stores a handle for a StepOpsArtifact.

        If an existing handle exists at this key, it is released before
        being overwritten.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is not None and entry.handle is not None:
            self._store.release(entry.handle)
        self._ledger[composite_key] = LedgerEntry(
            handle=handle,
            generation_id=generation_id,
        )

    @contextmanager
    def safe_adoption(
        self,
        key: ArtifactKey,
        handle_dict: Dict[str, Any],
    ) -> Generator[BaseArtifactHandle, None, None]:
        """
        Adopts an artifact from a dictionary with automatic rollback on error.

        If the code executing within this context manager throws an exception,
        the adopted handle is immediately released (unlinked/destroyed).
        If the block completes successfully, the handle is kept and remains
        adopted.

        Automatically checks for stale generations by verifying the handle's
        generation ID matches the current ledger entry, raising
        StaleGenerationError if the artifact is no longer valid.

        Args:
            key: The key context for logging and staleness checking.
            handle_dict: The serialized handle dictionary.

        Yields:
            The canonical BaseArtifactHandle from the store.

        Raises:
            StaleGenerationError: If the generation is stale.
        """
        handle = self._store.adopt_from_dict(handle_dict)
        try:
            if not self.is_generation_current(key, handle.generation_id):
                raise StaleGenerationError(
                    f"Generation {handle.generation_id} is stale for {key}"
                )
            yield handle
        except StaleGenerationError:
            self._store.release(handle)
            raise
        except Exception:
            self._store.release(handle)
            raise

    @contextmanager
    def transient_adoption(
        self,
        key: ArtifactKey,
        handle_dict: Dict[str, Any],
    ) -> Generator[BaseArtifactHandle, None, None]:
        """
        Context manager that adopts a handle and always releases it on exit.

        Unlike safe_adoption which only releases on exception, this manager
        releases the handle in all cases (normal exit and exception). Callers
        that want to keep the handle must call retain_handle() before the
        context exits.

        Automatically checks for stale generations by verifying the handle's
        generation ID matches the current ledger entry, raising
        StaleGenerationError if the artifact is no longer valid.

        Args:
            key: The key context for logging and staleness checking.
            handle_dict: The serialized handle dictionary.

        Yields:
            The canonical BaseArtifactHandle from the store.

        Raises:
            StaleGenerationError: If the generation is stale.
        """
        handle = self._store.adopt_from_dict(handle_dict)
        try:
            if not self.is_generation_current(key, handle.generation_id):
                raise StaleGenerationError(
                    f"Generation {handle.generation_id} is stale for {key}"
                )
            yield handle
        finally:
            self._store.release(handle)

    def adopt_artifact(
        self,
        key: ArtifactKey,
        handle_dict: Dict[str, Any],
    ) -> BaseArtifactHandle:
        """
        Adopts an artifact from a subprocess and deserializes the handle.

        Note: `safe_adoption` context manager is preferred for exception-safe
        handling, but this remains available for legacy/direct use.

        This method does NOT cache the handle. It serves as a
        factory for stages to acquire handles from raw dictionaries.
        Stages must explicitly call `put_*_handle` to treat the result
        as the canonical cached artifact.

        Args:
            key: The key context for logging (unused for logic).
            handle_dict: The serialized handle dictionary.

        Returns:
            The canonical handle from the store.
        """
        return self._store.adopt_from_dict(handle_dict)

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

    def get_all_workpiece_keys(self) -> Set[ArtifactKey]:
        return {
            extract_base_key(key)
            for key in self._ledger.keys()
            if extract_base_key(key).group == "workpiece"
        }

    def get_all_workpiece_keys_for_generation(
        self, generation_id: GenerationID
    ) -> Set[ArtifactKey]:
        return {
            extract_base_key(key)
            for key in self._ledger.keys()
            if extract_base_key(key).group == "workpiece"
            and extract_generation_id(key) == generation_id
        }

    def pop_step_render_handle(
        self, step_uid: str
    ) -> Optional[StepRenderArtifactHandle]:
        return self._step_render_handles.pop(step_uid, None)

    def invalidate_for_workpiece(self, key: ArtifactKey):
        wp_uid = key.id
        keys_to_invalidate = []
        for ledger_key in list(self._ledger.keys()):
            base_key = extract_base_key(ledger_key)
            if base_key.group == "workpiece":
                ledger_id = base_key.id
                if ledger_id == wp_uid or ledger_id.startswith(f"{wp_uid}:"):
                    keys_to_invalidate.append(ledger_key)
        for ledger_key in keys_to_invalidate:
            entry = self._ledger.get(ledger_key)
            if entry:
                if entry.handle is not None:
                    self.release_handle(entry.handle)
                    entry.handle = None
                entry.state = NodeState.DIRTY

    def remove_for_workpiece(self, key: ArtifactKey):
        """
        Remove all ledger entries for a workpiece that has been deleted.

        Unlike invalidate_for_workpiece which marks entries as DIRTY for
        re-rendering, this method permanently removes entries from the
        ledger and releases all associated handles.

        Removing the ledger entries ensures that is_generation_current()
        will return False for any in-flight tasks, preventing stale
        artifacts from being adopted.

        Args:
            key: The ArtifactKey for the workpiece to remove.
        """
        wp_uid = key.id
        logger.debug(f"remove_for_workpiece: called for wp_uid={wp_uid}")
        # The ledger keys have IDs in format "workpiece_uid:step_uid" or just
        # "workpiece_uid" for step artifacts. We need to match any key whose
        # ID starts with the workpiece UID.
        keys_to_remove = []
        for ledger_key in list(self._ledger.keys()):
            base_key = extract_base_key(ledger_key)
            if base_key.group == "workpiece":
                # Check if this entry is for the workpiece being removed
                # ID format is "workpiece_uid" or "workpiece_uid:step_uid"
                ledger_id = base_key.id
                if ledger_id == wp_uid or ledger_id.startswith(f"{wp_uid}:"):
                    keys_to_remove.append(ledger_key)
        for ledger_key in keys_to_remove:
            entry = self._ledger.get(ledger_key)
            if entry:
                if entry.handle is not None:
                    self.release_handle(entry.handle)
            del self._ledger[ledger_key]

    def remove_for_step(self, key: ArtifactKey):
        """
        Remove all ledger entries for a step that has been deleted.
        """
        step_uid = key.id
        keys_to_remove = []
        for ledger_key in list(self._ledger.keys()):
            base_key = extract_base_key(ledger_key)
            if base_key == key:
                keys_to_remove.append(ledger_key)
            elif base_key.group == "workpiece" and base_key.id.endswith(
                f":{step_uid}"
            ):
                keys_to_remove.append(ledger_key)

        for ledger_key in keys_to_remove:
            entry = self._ledger.get(ledger_key)
            if entry:
                if entry.handle is not None:
                    logger.debug(
                        f"remove_for_step: releasing {entry.handle.shm_name} "
                        f"(stack: {safe_caller_stack(15)})"
                    )
                    self.release_handle(entry.handle)
            del self._ledger[ledger_key]

        step_render_handle = self._step_render_handles.pop(step_uid, None)
        self.release_handle(step_render_handle)

    def invalidate_for_step(self, key: ArtifactKey):
        step_keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == key
        ]
        for ledger_key in step_keys_to_invalidate:
            step_entry = self._ledger.get(ledger_key)
            if step_entry:
                if step_entry.handle is not None:
                    self.release_handle(step_entry.handle)
                    step_entry.handle = None
                step_entry.state = NodeState.DIRTY

        step_render_handle = self._step_render_handles.pop(key.id, None)
        self.release_handle(step_render_handle)

    def invalidate_for_job(self, key: ArtifactKey):
        keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == key
        ]
        for ledger_key in keys_to_invalidate:
            entry = self._ledger.get(ledger_key)
            if entry:
                if entry.handle is not None:
                    self.release_handle(entry.handle)
                    entry.handle = None
                entry.state = NodeState.DIRTY

    def _remove_from_ledger(
        self, key: Union[ArtifactKey, Tuple[ArtifactKey, GenerationID]]
    ) -> None:
        """Remove an entry from the ledger."""
        if isinstance(key, tuple):
            if key in self._ledger:
                del self._ledger[key]
        else:
            base_key = extract_base_key(key)
            keys_to_remove = [
                k
                for k in self._ledger.keys()
                if extract_base_key(k) == base_key
            ]
            for k in keys_to_remove:
                del self._ledger[k]

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
        self,
        key: ArtifactKey,
        generation_id: Optional[GenerationID] = None,
    ) -> Generator[BaseArtifactHandle, None, None]:
        """
        Context manager for checking out an artifact handle with reference
        counting.

        This method increments the reference count for the key and retains
        the underlying shared memory block. When the context exits, the
        reference count is decremented and the block is released.

        Args:
            key: The key identifying the artifact to checkout.
            generation_id: The generation ID for the artifact.

        Yields:
            The artifact handle.

        Raises:
            KeyError: If the key is not found in the cached handles.
        """
        handle = self._get_handle_by_key(key, generation_id)
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
        self,
        key: ArtifactKey,
        generation_id: Optional[GenerationID] = None,
    ) -> Optional[BaseArtifactHandle]:
        """Get the handle for a given key, regardless of key type."""
        if generation_id is None:
            return None
        ledger_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(ledger_key)
        if entry and entry.handle is not None:
            return entry.handle
        return None

    def get_ledger_entry(
        self, key: Tuple[ArtifactKey, GenerationID]
    ) -> Optional[LedgerEntry]:
        """
        Returns the LedgerEntry for a composite key, or None if not found.

        Args:
            key: A composite key (ArtifactKey, GenerationID).

        Returns:
            The LedgerEntry if found, None otherwise.
        """
        return self._ledger.get(key)

    @contextmanager
    def report_completion(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
    ) -> Generator[Optional[BaseArtifactHandle], None, None]:
        """
        Marks generation as complete and yields the handle if available.

        The handle is retained for the duration of the context, allowing
        signal handlers to access it. Handlers that need to keep the handle
        must retain it themselves before the context exits.

        If the generation is stale, yields None and does nothing.

        Args:
            key: The artifact key.
            generation_id: The generation ID of this generation.

        Yields:
            The cached handle if available and current, None otherwise.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)

        if entry is None:
            logger.debug(f"[{key}] report_completion: entry missing, ignoring")
            yield None
            return

        if entry.generation_id != generation_id:
            logger.debug(
                f"[{key}] report_completion: stale generation "
                f"{generation_id}, current is {entry.generation_id}, "
                "ignoring"
            )
            yield None
            return

        entry.state = NodeState.VALID
        handle = entry.handle
        logger.debug(f"[{key}] report_completion: marked VALID")

        if handle is not None:
            self._store.retain(handle)

        try:
            yield handle
        finally:
            if handle is not None:
                self._store.release(handle)

    @contextmanager
    def report_failure(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
    ) -> Generator[Optional[BaseArtifactHandle], None, None]:
        """
        Reports that artifact generation failed.

        Failed artifacts are marked ERROR and will not be retried until
        the underlying data changes. Any cached handle is released and
        yielded for the final signal, then cleaned up.

        If the generation is stale, yields None and does nothing.

        Args:
            key: The artifact key.
            generation_id: The generation ID of this generation.

        Yields:
            The cached handle if available (for final signal), None otherwise.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)

        if entry is None:
            logger.debug(f"[{key}] report_failure: entry missing, ignoring")
            yield None
            return

        if entry.generation_id != generation_id:
            logger.debug(f"[{key}] report_failure: stale generation, ignoring")
            yield None
            return

        handle = entry.handle
        entry.handle = None
        entry.state = NodeState.ERROR
        logger.debug(f"[{key}] report_failure: marked ERROR")

        if handle is not None:
            self._store.retain(handle)

        try:
            yield handle
        finally:
            if handle is not None:
                self._store.release(handle)

    @contextmanager
    def report_cancellation(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
    ) -> Generator[Optional[BaseArtifactHandle], None, None]:
        """
        Reports that artifact generation was canceled.

        If a handle exists, it is kept and marked VALID. Otherwise, the
        entry is marked DIRTY so the scheduler can regenerate it.

        If the generation is stale, yields None and does nothing.

        Args:
            key: The artifact key.
            generation_id: The generation ID of this generation.

        Yields:
            The cached handle if available, None otherwise.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)

        if entry is None:
            logger.debug(
                f"[{key}] report_cancellation: entry missing, ignoring"
            )
            yield None
            return

        if entry.generation_id != generation_id:
            logger.debug(
                f"[{key}] report_cancellation: stale generation, ignoring"
            )
            yield None
            return

        handle = entry.handle
        if handle is not None:
            entry.state = NodeState.VALID
            logger.debug(
                f"[{key}] report_cancellation: handle exists, keeping VALID"
            )
        else:
            entry.state = NodeState.DIRTY
            logger.debug(
                f"[{key}] report_cancellation: no handle, marked DIRTY"
            )

        if handle is not None:
            self._store.retain(handle)

        try:
            yield handle
        finally:
            if handle is not None:
                self._store.release(handle)

    def get_store(self) -> ArtifactStore:
        """
        Returns the artifact store.

        This is needed for subprocess runners that need direct access
        to the shared memory store for artifact serialization.

        Returns:
            The ArtifactStore instance.
        """
        return self._store

    def _set_ledger_entry(
        self, key: Tuple[ArtifactKey, GenerationID], entry: LedgerEntry
    ) -> None:
        """Sets a LedgerEntry for a key."""
        self._ledger[key] = entry

    def register_dependency(
        self, child_key: ArtifactKey, parent_key: ArtifactKey
    ) -> None:
        """
        Registers that parent_key depends on child_key.

        Dependencies are stored as {parent: [children]}, so calling
        register_dependency(child_key, parent_key) means parent_key
        has child_key as a dependency.
        """
        if parent_key not in self._dependencies:
            self._dependencies[parent_key] = []
        if child_key not in self._dependencies[parent_key]:
            self._dependencies[parent_key].append(child_key)

    def get_dependents(self, key: ArtifactKey) -> List[ArtifactKey]:
        """
        Returns all parent keys that depend on this key.

        Dependencies are stored as {parent: [children]}, so this method
        finds all parent keys where the given key appears in their children
        list.

        Args:
            key: The ArtifactKey to find dependents for.

        Returns:
            A list of parent ArtifactKeys that depend on the given key.
        """
        dependents: List[ArtifactKey] = []
        logger.debug(
            f"get_dependents: looking for parents of {key}, "
            f"dependencies={self._dependencies}"
        )
        for parent_key, children in self._dependencies.items():
            if key in children:
                dependents.append(parent_key)
        logger.debug(f"get_dependents: found {len(dependents)} parents")
        return dependents

    def cache_handle(
        self,
        key: ArtifactKey,
        handle: BaseArtifactHandle,
        generation_id: GenerationID,
    ) -> bool:
        """
        Caches an artifact handle.

        This consumes the reference ("claim") passed in via the `handle`
        argument. It assumes the caller has already acquired this claim (e.g.
        via adoption) and is transferring ownership to the manager.

        Returns True if the handle was cached, False if the workpiece
        was deleted (entry not found in ledger).

        Args:
            key: The ArtifactKey for the entry.
            handle: The handle to cache.
            generation_id: The generation ID for the entry.
        """
        logger.debug(
            f"cache_handle: key={key}, generation_id={generation_id}, "
            f"shm_name={handle.shm_name}"
        )
        composite_key = make_composite_key(key, generation_id)
        entry = self.get_ledger_entry(composite_key)
        if entry is None:
            logger.debug(
                f"cache_handle: no entry for {composite_key}, "
                "workpiece may have been deleted, releasing handle"
            )
            self._store.release(handle)
            return False

        logger.debug(f"cache: Caching entry {composite_key}")

        if entry.handle is not None:
            self._store.release(entry.handle)
        entry.handle = handle
        entry.generation_id = generation_id
        entry.state = NodeState.VALID
        logger.debug(f"cache_handle: SUCCESS for {composite_key}")
        return True

    def mark_done(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
    ) -> None:
        """
        Marks an entry as done without providing a handle.

        This is used when an entry has nothing to do (e.g., the
        workpiece is empty or has no associated step).

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self.get_ledger_entry(composite_key)
        if entry is None:
            # Downgraded to debug to reduce noise during race conditions
            # in stress tests where workpieces are deleted rapidly.
            logger.debug(
                f"mark_done called for {key} (gen_id={generation_id}) "
                f"but no ledger entry was found. Skipping."
            )
            return

        logger.debug(f"mark_done: Marking entry {composite_key} as done")
        entry.generation_id = generation_id
        if entry.handle is not None:
            self._store.release(entry.handle)
            entry.handle = None

    def complete_generation(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
        handle: Optional[BaseArtifactHandle] = None,
    ) -> None:
        """
        Marks a generation as done.

        This is used when an entry already has a handle set (e.g., from
        put_step_ops_handle) and we just need to mark it as done, or
        when we need to create a ledger entry from a handle stored elsewhere.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.
            handle: Optional handle to set on the entry. If not provided,
                the existing handle is preserved.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None:
            if handle is None:
                logger.debug(
                    f"complete_generation: Creating new entry for {key} "
                    f"(gen_id={generation_id}) with no handle"
                )
                self._ledger[composite_key] = LedgerEntry(
                    handle=None,
                    generation_id=generation_id,
                    state=NodeState.VALID,
                )
                return
            logger.debug(
                f"complete_generation: Creating new entry for {key} "
                f"(gen_id={generation_id})"
            )
            self._ledger[composite_key] = LedgerEntry(
                handle=handle,
                generation_id=generation_id,
                state=NodeState.VALID,
            )
            return

        logger.debug(
            f"complete_generation: Marking entry {composite_key} as done"
        )
        entry.generation_id = generation_id
        if handle is not None:
            if entry.handle is not None:
                self._store.release(entry.handle)
            entry.handle = handle
        entry.state = NodeState.VALID

    def declare_generation(
        self,
        valid_keys: Set[ArtifactKey],
        generation_id: GenerationID,
    ) -> None:
        """
        Declares the expected artifacts for a specific generation.

        Creates placeholder entries in the ledger for any keys that do not yet
        exist for this generation ID. Copies handles from previous generations
        to avoid unnecessary regeneration for step and workpiece keys.

        Args:
            valid_keys: A set of ArtifactKeys that should exist.
            generation_id: The generation ID for these keys.
        """
        logger.debug(
            f"declare_generation: valid_keys={valid_keys}, "
            f"generation_id={generation_id}"
        )
        for base_key in valid_keys:
            key = make_composite_key(base_key, generation_id)
            if key not in self._ledger:
                if base_key.group in ("step", "workpiece"):
                    existing_handle = self._find_existing_handle(
                        base_key, generation_id
                    )
                    if existing_handle is not None:
                        logger.debug(
                            f"declare_generation: Copying {base_key.group} "
                            f"handle from previous generation to {key}"
                        )
                        self.retain_handle(existing_handle)
                        self._ledger[key] = LedgerEntry(
                            handle=existing_handle,
                            generation_id=generation_id,
                        )
                        continue

                logger.debug(f"declare_generation: Creating entry at {key}")
                self._ledger[key] = LedgerEntry(
                    generation_id=generation_id,
                )

    def _find_existing_handle(
        self, key: ArtifactKey, before_generation_id: GenerationID
    ) -> Optional[BaseArtifactHandle]:
        """
        Find the most recent handle for a key before a given generation.

        Searches backwards through the ledger to find the latest handle
        for the given key at a generation ID less than the specified one.

        Args:
            key: The ArtifactKey to search for.
            before_generation_id: Only consider entries with generation ID
                less than this value.

        Returns:
            The most recent handle, or None if not found.
        """
        best_gen_id = -1
        best_handle = None

        for composite_key, entry in self._ledger.items():
            if extract_base_key(composite_key) != key:
                continue
            gen_id = extract_generation_id(composite_key)
            if gen_id >= before_generation_id:
                continue
            if entry.handle is None:
                continue
            if gen_id > best_gen_id:
                best_gen_id = gen_id
                best_handle = entry.handle

        return best_handle

    def prune(
        self,
        active_data_gen_ids: Set[int],
        processing_data_gen_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Removes ledger entries that do not belong to active generations.

        This method performs garbage collection on the ledger, removing
        entries that are no longer needed. It releases the "Manager's Claim"
        (the retain() call from cache_handle) when removing entries.

        Args:
            active_data_gen_ids: Set of active data generation IDs.
            processing_data_gen_ids: Set of data generation IDs that are
                currently being processed.
        """
        if processing_data_gen_ids is None:
            processing_data_gen_ids = set()

        logger.debug(
            f"prune: active_data_gen_ids={active_data_gen_ids}, "
            f"processing_data_gen_ids={processing_data_gen_ids}, "
            f"ledger_size={len(self._ledger)}"
        )
        keys_to_remove = []
        for composite_key, entry in list(self._ledger.items()):
            generation_id = extract_generation_id(composite_key)

            should_keep = (
                generation_id in active_data_gen_ids
                or generation_id in processing_data_gen_ids
            )

            if not should_keep:
                logger.debug(
                    f"prune: Scheduling removal of {composite_key} "
                    f"(gen_id={generation_id})"
                )
                keys_to_remove.append(composite_key)

        logger.debug(f"prune: Removing {len(keys_to_remove)} entries")
        stack = safe_caller_stack(15)
        for composite_key in keys_to_remove:
            entry = self._ledger.get(composite_key)
            if entry is not None and entry.handle is not None:
                logger.debug(
                    f"prune: releasing {entry.handle.shm_name} "
                    f"(stack: {stack})"
                )
                self._store.release(entry.handle)
            del self._ledger[composite_key]
