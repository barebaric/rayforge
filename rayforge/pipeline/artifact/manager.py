from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
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
from .key import ArtifactKey
from .lifecycle import LedgerEntry
from .step_ops import StepOpsArtifactHandle
from .step_render import StepRenderArtifactHandle
from .store import ArtifactStore
from .workpiece import WorkPieceArtifactHandle
from .workpiece_view import (
    WorkPieceViewArtifactHandle,
    RenderContext,
)

logger = logging.getLogger(__name__)

GenerationID = int


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
        dag_state_callback: Optional[
            Callable[[ArtifactKey, str], None]
        ] = None,
    ):
        self._store = store
        self._step_render_handles: Dict[str, StepRenderArtifactHandle] = {}
        self._workpiece_view_handles: Dict[
            ArtifactKey, WorkPieceViewArtifactHandle
        ] = {}
        self._ref_counts: Dict[ArtifactKey, int] = {}
        self._ledger: Dict[Tuple[ArtifactKey, GenerationID], LedgerEntry] = {}
        self._dependencies: Dict[ArtifactKey, List[ArtifactKey]] = {}
        self._dag_state_callback = dag_state_callback

    def _notify_dag_state_change(self, key: ArtifactKey, state: str) -> None:
        """
        Notify the DAG scheduler of a state change.

        Args:
            key: The ArtifactKey whose state changed.
            state: The new state as a string (e.g., "valid", "processing").
        """
        if self._dag_state_callback is not None:
            self._dag_state_callback(key, state)

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

    def get_workpiece_view_handle(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Optional[WorkPieceViewArtifactHandle]:
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None or entry.handle is None:
            return None
        handle = entry.handle
        if isinstance(handle, WorkPieceViewArtifactHandle):
            return handle
        return None

    def is_view_stale(
        self,
        key: ArtifactKey,
        new_context: Optional["RenderContext"],
        source_handle: Optional[WorkPieceArtifactHandle],
        generation_id: GenerationID,
    ) -> bool:
        """
        Check if a workpiece view is stale and needs regeneration.

        Args:
            key: The ArtifactKey for the view.
            new_context: The new render context to compare against.
            source_handle: The source workpiece handle to compare properties.
            generation_id: The generation ID to check.

        Returns:
            True if the view is missing, stale, or context/properties changed.
            False if the view is still valid.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)

        if entry is None or entry.handle is None:
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
        """
        Stores a handle for a StepRenderArtifact.

        Note: This does NOT release any existing handle for this step.
        The old handle will be cleaned up when the step is invalidated
        or the pipeline is shut down. This prevents race conditions where
        a handle is released while a worker is still using it.
        """
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

    def put_workpiece_view_handle(
        self,
        key: ArtifactKey,
        handle: WorkPieceViewArtifactHandle,
        generation_id: GenerationID,
    ):
        """
        Stores a handle for a WorkPieceViewArtifact.

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

    def adopt_artifact(
        self,
        key: ArtifactKey,
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
        keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == key
        ]
        for ledger_key in keys_to_invalidate:
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                self.release_handle(entry.handle)
            self._remove_from_ledger(ledger_key)

        step_key = ArtifactKey.for_step(key.id)
        step_keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == step_key
        ]
        for ledger_key in step_keys_to_invalidate:
            step_entry = self._ledger.get(ledger_key)
            if step_entry and step_entry.handle is not None:
                self.release_handle(step_entry.handle)
            self._remove_from_ledger(ledger_key)

        view_key = ArtifactKey.for_view(key.id)
        view_keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == view_key
        ]
        for ledger_key in view_keys_to_invalidate:
            view_entry = self._ledger.get(ledger_key)
            if view_entry and view_entry.handle is not None:
                self.release_handle(view_entry.handle)
            self._remove_from_ledger(ledger_key)

        self._notify_dag_state_change(key, "stale")

    def invalidate_for_step(self, key: ArtifactKey):
        step_keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == key
        ]
        for ledger_key in step_keys_to_invalidate:
            step_entry = self._ledger.get(ledger_key)
            if step_entry and step_entry.handle is not None:
                self.release_handle(step_entry.handle)
            self._remove_from_ledger(ledger_key)

        keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key).group == "workpiece"
            and extract_base_key(ledger_key).id == key.id
        ]
        for ledger_key in keys_to_invalidate:
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                self.release_handle(entry.handle)
            self._remove_from_ledger(ledger_key)

        view_keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key).group == "view"
            and extract_base_key(ledger_key).id == key.id
        ]
        for view_key in view_keys_to_invalidate:
            view_entry = self._ledger.get(view_key)
            if view_entry and view_entry.handle is not None:
                self.release_handle(view_entry.handle)
            self._remove_from_ledger(view_key)

        step_render_handle = self._step_render_handles.pop(key.id, None)
        self.release_handle(step_render_handle)

        self._notify_dag_state_change(key, "stale")

    def invalidate_for_job(self, key: ArtifactKey):
        keys_to_invalidate = [
            ledger_key
            for ledger_key in self._ledger.keys()
            if extract_base_key(ledger_key) == key
        ]
        for ledger_key in keys_to_invalidate:
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                self.release_handle(entry.handle)
            self._remove_from_ledger(ledger_key)

        self._notify_dag_state_change(key, "stale")

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
        if key.group == "workpiece":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                handle = entry.handle
                if isinstance(handle, WorkPieceArtifactHandle):
                    return handle
            return None
        elif key.group == "step":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                handle = entry.handle
                if isinstance(handle, StepOpsArtifactHandle):
                    return handle
            return None
        elif key.group == "job":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                return entry.handle
            return None
        elif key.group == "view":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.handle is not None:
                handle = entry.handle
                if isinstance(handle, WorkPieceViewArtifactHandle):
                    return handle
            return None
        else:
            return self._step_render_handles.get(key.id)

    def _get_ledger_entry(
        self, key: Tuple[ArtifactKey, GenerationID]
    ) -> Optional[LedgerEntry]:
        """Returns the LedgerEntry for a key, or None if not found."""
        return self._ledger.get(key)

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

    def _get_dependents(self, key: ArtifactKey) -> List[ArtifactKey]:
        """
        Returns all parent keys that depend on this key.

        Dependencies are stored as {parent: [children]}, so this method
        finds all parent keys where the given key appears in their children
        list.
        """
        dependents: List[ArtifactKey] = []
        logger.debug(
            f"_get_dependents: looking for parents of {key}, "
            f"dependencies={self._dependencies}"
        )
        for parent_key, children in self._dependencies.items():
            if key in children:
                dependents.append(parent_key)
        logger.debug(f"_get_dependents: found {len(dependents)} parents")
        return dependents

    def _get_dependencies(
        self,
        key: ArtifactKey,
        generation_id: Optional[GenerationID] = None,
    ) -> List[Tuple[ArtifactKey, GenerationID]]:
        """Returns all child keys this key depends on."""
        deps = []
        if key in self._dependencies:
            deps.extend(self._dependencies[key])
        if generation_id is not None:
            return [make_composite_key(dep, generation_id) for dep in deps]
        return []

    def register_intent(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> None:
        """
        Creates a placeholder entry (key, generation_id) in the cache.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.

        Raises:
            AssertionError: If the entry already exists.
        """
        composite_key = make_composite_key(key, generation_id)
        if composite_key in self._ledger:
            raise AssertionError(
                f"Entry for {key} with generation_id={generation_id} "
                f"already exists"
            )
        self._ledger[composite_key] = LedgerEntry(
            generation_id=generation_id,
        )

    def mark_processing(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
        source_handle: Optional[BaseArtifactHandle] = None,
    ) -> None:
        """
        Notifies the DAG that processing has started.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.
            source_handle: Optional handle to store in metadata.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            entry = LedgerEntry(generation_id=generation_id)
            self._ledger[composite_key] = entry
        entry.generation_id = generation_id
        if source_handle is not None:
            entry.metadata["source_handle"] = source_handle
        self._notify_dag_state_change(key, "processing")

    def commit_artifact(
        self,
        key: ArtifactKey,
        handle: BaseArtifactHandle,
        generation_id: GenerationID,
    ) -> None:
        """
        Caches an artifact handle and notifies the DAG.

        This also retains the handle, creating a "Manager's Claim" on the
        shared memory. This ensures the data persists even after the
        GenerationContext releases its "Builder's Claim".

        Args:
            key: The ArtifactKey for the entry.
            handle: The handle to cache.
            generation_id: The generation ID for the entry.
        """
        logger.debug(f"commit: key={key}, generation_id={generation_id}")
        composite_key = make_composite_key(key, generation_id)
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            entry = LedgerEntry(generation_id=generation_id)
            self._ledger[composite_key] = entry

        logger.debug(f"commit: Committing entry {composite_key}")
        self._store.adopt(handle)
        self.retain_handle(handle)
        if entry.handle is not None:
            self._store.release(entry.handle)
        entry.handle = handle
        entry.generation_id = generation_id
        source_handle = entry.metadata.pop("source_handle", None)
        if source_handle is not None:
            self._store.release(source_handle)
        self._notify_dag_state_change(key, "valid")

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
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            logger.warning(
                f"mark_done called for {key} (gen_id={generation_id}) "
                f"but no ledger entry was found. Skipping."
            )
            return

        logger.debug(f"mark_done: Marking entry {composite_key} as done")
        entry.generation_id = generation_id
        if entry.handle is not None:
            self._store.release(entry.handle)
            entry.handle = None
        self._notify_dag_state_change(key, "valid")

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
                )
                self._notify_dag_state_change(key, "valid")
                return
            logger.debug(
                f"complete_generation: Creating new entry for {key} "
                f"(gen_id={generation_id})"
            )
            self._ledger[composite_key] = LedgerEntry(
                handle=handle,
                generation_id=generation_id,
            )
            self._notify_dag_state_change(key, "valid")
            return

        logger.debug(
            f"complete_generation: Marking entry {composite_key} as done"
        )
        entry.generation_id = generation_id
        if handle is not None:
            if entry.handle is not None:
                self._store.release(entry.handle)
            entry.handle = handle
        self._notify_dag_state_change(key, "valid")

    def fail_generation(
        self, key: ArtifactKey, error_msg: str, generation_id: GenerationID
    ) -> None:
        """
        Notifies the DAG that generation has failed.

        Args:
            key: The ArtifactKey for the entry.
            error_msg: The error message.
            generation_id: The generation ID for the entry.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            logger.warning(
                f"fail_generation called for {key} but no entry found"
            )
            return

        source_handle = entry.metadata.pop("source_handle", None)
        if source_handle is not None:
            self._store.release(source_handle)
        self._notify_dag_state_change(key, "error")

    def checkout_dependencies(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Dict[ArtifactKey, BaseArtifactHandle]:
        """
        Returns a dict mapping dependency keys to their handles.

        Looks up all children (dependencies) of key and verifies they
        all have handles.
        """
        deps = self._get_dependencies(key, generation_id)
        result = {}
        for dep in deps:
            entry = self._get_ledger_entry(dep)
            if entry is None:
                logger.error(
                    f"checkout_dependencies: Dependency {dep} not in ledger. "
                    f"Ledger keys: {list(self._ledger.keys())}"
                )
            assert entry is not None, f"Dependency {dep} not in ledger"
            assert entry.handle is not None, f"Dependency {dep} has no handle"
            base_key = extract_base_key(dep)
            if isinstance(base_key, ArtifactKey):
                result[base_key] = entry.handle
        return result

    def declare_generation(
        self,
        valid_keys: Set[ArtifactKey],
        generation_id: GenerationID,
    ) -> None:
        """
        Declares the expected artifacts for a specific generation.

        Creates placeholder entries in the ledger for any keys that do not yet
        exist for this generation ID.

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
                logger.debug(f"declare_generation: Creating entry at {key}")
                self._ledger[key] = LedgerEntry(
                    generation_id=generation_id,
                )

    def prune(
        self,
        active_data_gen_ids: Set[int],
        active_view_gen_ids: Set[int],
        processing_data_gen_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Removes ledger entries that do not belong to active generations.

        This method performs garbage collection on the ledger, removing
        entries that are no longer needed. It releases the "Manager's Claim"
        (the retain() call from commit_artifact) when removing entries.

        Args:
            active_data_gen_ids: Set of active data generation IDs.
            active_view_gen_ids: Set of active view generation IDs.
            processing_data_gen_ids: Set of data generation IDs that are
                currently being processed. View entries for these generations
                will not be pruned even if not in active_view_gen_ids.
        """
        if processing_data_gen_ids is None:
            processing_data_gen_ids = set()

        logger.debug(
            f"prune: active_data_gen_ids={active_data_gen_ids}, "
            f"active_view_gen_ids={active_view_gen_ids}, "
            f"processing_data_gen_ids={processing_data_gen_ids}, "
            f"ledger_size={len(self._ledger)}"
        )
        keys_to_remove = []
        for composite_key, entry in list(self._ledger.items()):
            base_key = extract_base_key(composite_key)
            generation_id = extract_generation_id(composite_key)

            is_view = base_key.group == "view"
            is_step = base_key.group == "step"
            if is_view:
                active_gen_ids = active_view_gen_ids
            else:
                active_gen_ids = active_data_gen_ids

            should_keep = generation_id in active_gen_ids
            if is_view and generation_id in processing_data_gen_ids:
                should_keep = True
            if is_step and generation_id in processing_data_gen_ids:
                should_keep = True

            if not should_keep:
                logger.debug(
                    f"prune: Scheduling removal of {composite_key} "
                    f"(gen_id={generation_id})"
                )
                keys_to_remove.append(composite_key)

        logger.debug(f"prune: Removing {len(keys_to_remove)} entries")
        for composite_key in keys_to_remove:
            entry = self._ledger.get(composite_key)
            if entry is not None and entry.handle is not None:
                base_key = extract_base_key(composite_key)
                if base_key.group != "view":
                    logger.debug(
                        f"prune: Releasing handle for {composite_key}"
                    )
                    self._store.release(entry.handle)
            del self._ledger[composite_key]
