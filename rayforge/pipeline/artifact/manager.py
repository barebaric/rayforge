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
from .key import ArtifactKey
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
    A centralized manager for artifact handles that understands the
    dependency graph between different artifact types.
    """

    def __init__(self, store: ArtifactStore):
        self._store = store
        self._step_render_handles: Dict[str, StepRenderArtifactHandle] = {}
        self._workpiece_view_handles: Dict[
            ArtifactKey, WorkPieceViewArtifactHandle
        ] = {}
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
            f"state={entry.state if entry else None}"
        )
        if entry is None or entry.state != ArtifactLifecycle.DONE:
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
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Optional[StepOpsArtifactHandle]:
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is None or entry.state != ArtifactLifecycle.DONE:
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
        if entry is None or entry.state != ArtifactLifecycle.DONE:
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
        if entry is None or entry.state != ArtifactLifecycle.DONE:
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

        if entry is None:
            return True

        if entry.state != ArtifactLifecycle.DONE:
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
        self,
        key: ArtifactKey,
        handle: StepOpsArtifactHandle,
        generation_id: GenerationID,
    ):
        """Stores a handle for a StepOpsArtifact."""
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is not None and entry.handle is not None:
            self.release_handle(entry.handle)
        # Don't set to DONE state - let commit handle that
        self._ledger[composite_key] = LedgerEntry(
            handle=handle,
            state=ArtifactLifecycle.PROCESSING,
            generation_id=generation_id,
            error=None,
        )

    def put_workpiece_view_handle(
        self,
        key: ArtifactKey,
        handle: WorkPieceViewArtifactHandle,
        generation_id: GenerationID,
    ):
        """Stores a handle for a WorkPieceViewArtifact."""
        composite_key = make_composite_key(key, generation_id)
        entry = self._ledger.get(composite_key)
        if entry is not None and entry.handle is not None:
            self.release_handle(entry.handle)
        self._ledger[composite_key] = LedgerEntry(
            handle=handle,
            state=ArtifactLifecycle.DONE,
            generation_id=generation_id,
            error=None,
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
            self.invalidate(ledger_key)

        # Invalidate the step ops artifact since it depends on workpieces
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
            self.invalidate(ledger_key)

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
            self.invalidate(ledger_key)

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
            self.invalidate(ledger_key)

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
            self.invalidate(ledger_key)

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
            self.invalidate(view_key)

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
            if entry and entry.state == ArtifactLifecycle.DONE:
                handle = entry.handle
                if isinstance(handle, WorkPieceArtifactHandle):
                    return handle
            return None
        elif key.group == "step":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.state == ArtifactLifecycle.DONE:
                handle = entry.handle
                if isinstance(handle, StepOpsArtifactHandle):
                    return handle
            return None
        elif key.group == "job":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.state == ArtifactLifecycle.DONE:
                return entry.handle
            return None
        elif key.group == "view":
            if generation_id is None:
                return None
            ledger_key = make_composite_key(key, generation_id)
            entry = self._ledger.get(ledger_key)
            if entry and entry.state == ArtifactLifecycle.DONE:
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

    def _register_dependency(
        self, child_key: ArtifactKey, parent_key: ArtifactKey
    ) -> None:
        """
        Registers that parent_key depends on child_key.

        Dependencies are stored as {parent: [children]}, so calling
        _register_dependency(step_key, wp_key) means step_key is a child
        (i.e., step_key appears in wp_key's children list).
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

    def invalidate(
        self,
        key: Union[ArtifactKey, Tuple[ArtifactKey, GenerationID]],
    ) -> None:
        """
        Sets the entry's state to STALE.

        Recursively invalidates all parent dependents found via the
        dependency graph. If key doesn't exist in ledger, does nothing.
        Accepts both base keys and composite keys.
        """
        base_key = extract_base_key(key)
        for composite_key, entry in list(self._ledger.items()):
            current_base = extract_base_key(composite_key)
            if current_base == base_key:
                # Check if generation_id matches if key is composite
                # Only invalidate entries with matching generation_id
                if isinstance(key, tuple) and len(key) == 2:
                    key_generation_id = key[1]
                    entry_generation_id = composite_key[1]
                    if entry_generation_id != key_generation_id:
                        continue
                # Don't invalidate PROCESSING entries - they're actively being
                # worked on
                if entry.state != ArtifactLifecycle.PROCESSING:
                    entry.state = ArtifactLifecycle.STALE
        for dependent in self._get_dependents(base_key):
            self.invalidate(dependent)

    def has_processing_work(self) -> bool:
        """
        Returns True if any artifact in the ledger is being generated,
        is stale, or is missing.
        """
        if not self._ledger:
            return False

        for entry in self._ledger.values():
            if entry.state in (
                ArtifactLifecycle.PROCESSING,
                ArtifactLifecycle.STALE,
                ArtifactLifecycle.INITIAL,
            ):
                return True
        return False

    def query_work_for_stage(
        self, stage_type: str, generation_id: GenerationID
    ) -> List[ArtifactKey]:
        """
        Returns a list of keys that are INITIAL and whose dependencies
        are all DONE.

        This is the core scheduling logic. Keys are filtered by
        stage_type (the group attribute of the ArtifactKey).
        """
        work_items = []
        for key, entry in self._ledger.items():
            if extract_generation_id(key) != generation_id:
                continue
            if entry.state != ArtifactLifecycle.INITIAL:
                continue
            base_key = extract_base_key(key)
            if base_key.group != stage_type:
                continue
            deps = self._get_dependencies(base_key, generation_id)
            all_ready = True
            for dep in deps:
                dep_entry = self._get_ledger_entry(dep)
                if dep_entry is None or dep_entry.state != (
                    ArtifactLifecycle.DONE
                ):
                    all_ready = False
                    break
            if all_ready:
                work_items.append(base_key)
        return work_items

    def register_intent(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> None:
        """
        Creates an entry (key, generation_id) with state INITIAL.

        Fails if the entry already exists.

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
            state=ArtifactLifecycle.INITIAL,
            generation_id=generation_id,
        )

    def mark_processing(
        self,
        key: ArtifactKey,
        generation_id: GenerationID,
        source_handle: Optional[BaseArtifactHandle] = None,
    ) -> None:
        """
        Transitions state from INITIAL or STALE to PROCESSING.

        Fails if not in INITIAL or STALE state.
        Stores the source_handle in metadata for later release.

        Args:
            key: The ArtifactKey for the entry.
            generation_id: The generation ID for the entry.
            source_handle: Optional handle to store in metadata.

        Raises:
            AssertionError: If entry not found or not in a valid state.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            raise AssertionError(
                f"Key {key} with generation_id={generation_id} "
                f"not found in ledger"
            )
        assert entry.state in (
            ArtifactLifecycle.INITIAL,
            ArtifactLifecycle.STALE,
        ), f"Cannot mark {entry.state} as PROCESSING, must be INITIAL or STALE"
        entry.state = ArtifactLifecycle.PROCESSING
        entry.generation_id = generation_id
        entry.error = None
        if source_handle is not None:
            entry.metadata["source_handle"] = source_handle

    def commit_artifact(
        self,
        key: ArtifactKey,
        handle: BaseArtifactHandle,
        generation_id: GenerationID,
    ) -> None:
        """
        Transitions state from PROCESSING or INITIAL to DONE.

        Fails if not in INITIAL or PROCESSING state.
        Adopts the new handle, releases the old handle if one existed,
        updates state to DONE, and clears error message. Releases the
        source_handle from metadata if present.

        Args:
            key: The ArtifactKey for the entry.
            handle: The handle to commit.
            generation_id: The generation ID for the entry.

        Raises:
            AssertionError: If entry not found or not in INITIAL or
                PROCESSING state.
        """
        logger.debug(
            f"commit: key={key}, generation_id={generation_id}, "
            f"ledger_keys={list(self._ledger.keys())}, "
            f"ledger_entries={[(k, v.state) for k, v in self._ledger.items()]}"
        )
        composite_key = make_composite_key(key, generation_id)
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            # It's possible for a task to complete so fast that the event
            # arrives before the stage has marked it as processing.
            # In this case, we create the entry.
            logger.warning(
                f"commit_artifact called for {key} (gen_id={generation_id}) "
                f"but no ledger entry was found. Creating one."
            )
            self.register_intent(key, generation_id)
            entry = self._get_ledger_entry(composite_key)
            assert entry is not None

        assert entry.state in (
            ArtifactLifecycle.INITIAL,
            ArtifactLifecycle.PROCESSING,
        ), (
            f"Cannot commit {entry.state} to DONE, "
            f"must be INITIAL or PROCESSING"
        )
        logger.debug(f"commit: Committing entry {composite_key}")
        self._store.adopt(handle)
        if entry.handle is not None:
            self._store.release(entry.handle)
        entry.handle = handle
        entry.state = ArtifactLifecycle.DONE
        entry.generation_id = generation_id
        entry.error = None
        source_handle = entry.metadata.pop("source_handle", None)
        if source_handle is not None:
            self._store.release(source_handle)

    def fail_generation(
        self, key: ArtifactKey, error_msg: str, generation_id: GenerationID
    ) -> None:
        """
        Marks a PROCESSING entry as ERROR.

        Args:
            key: The ArtifactKey for the entry.
            error_msg: The error message to store.
            generation_id: The generation ID for the entry.

        Raises:
            AssertionError: If the entry is not found or not in a valid state.
        """
        composite_key = make_composite_key(key, generation_id)
        entry = self._get_ledger_entry(composite_key)
        if entry is None:
            raise AssertionError(f"Key {key} not found in ledger")

        assert entry.state == ArtifactLifecycle.PROCESSING, (
            f"Cannot mark {entry.state} as ERROR, must be PROCESSING"
        )
        entry.state = ArtifactLifecycle.ERROR
        entry.error = error_msg
        source_handle = entry.metadata.pop("source_handle", None)
        if source_handle is not None:
            self._store.release(source_handle)

    def checkout_dependencies(
        self, key: ArtifactKey, generation_id: GenerationID
    ) -> Dict[ArtifactKey, BaseArtifactHandle]:
        """
        Returns a dict mapping dependency keys to their handles.

        Looks up all children (dependencies) of key and verifies they
        are all DONE.
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
            assert entry.state == ArtifactLifecycle.DONE, (
                f"Dependency {dep} is not DONE: {entry.state}"
            )
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

        Creates INITIAL entries in the ledger for any keys that do not yet
        exist for this generation ID. This method enforces immutable
        generations: it does not modify entries from previous generations
        or reset states.

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
                    state=ArtifactLifecycle.INITIAL,
                    generation_id=generation_id,
                )
            # If key exists, we do nothing. It might be PROCESSING or DONE
            # already.
