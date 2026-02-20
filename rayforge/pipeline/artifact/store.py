"""
Shared memory artifact storage for inter-process communication.

This module provides the ArtifactStore class which manages the lifecycle of
artifacts stored in shared memory blocks. The lifecycle is managed through
reference counting and supports two main ownership patterns:

1. Local Ownership: The creating process owns and releases the handle.
2. Inter-Process Handoff: A worker creates, transfers ownership to the
   main process via forget/adopt, and the main process releases.

Lifecycle State Diagram
-----------------------

                              put()
    ┌─────────┐         ┌──────────┐
    │  None   │────────>│ MANAGED  │
    └─────────┘         └──────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │ adopt()          │ retain()         │ release()
          ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐        ┌────────────┐
    │ ADOPTED  │       │ RETAINED │        │ RELEASED   │
    │ refcnt++ │       │ refcnt++ │        │ (closed    │
    └──────────┘       └──────────┘        │ + unlinked)│
          │                  │             └────────────┘
          │ forget()         │ forget()/release()
          ▼                  ▼
    ┌──────────┐       ┌──────────┐
    │ FORGOTTEN│       │ FORGOTTEN│
    │ (closed  │       │ or       │
    │  only)   │       │ RELEASED │
    └──────────┘       └──────────┘

Method Usage Guide
------------------

- release(handle): Close and unlink. Use when you own the SHM block.
- forget(handle): Close without unlinking. Use for inter-process handoff.
- retain(handle): Increment refcount. Use when multiple code paths need
  the same handle.
"""

from __future__ import annotations
import uuid
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Generator, Any, TYPE_CHECKING
from multiprocessing import shared_memory
import numpy as np
from ...shared.util.debug import safe_caller_stack
from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ..context import GenerationContext


logger = logging.getLogger(__name__)

MAX_TAG_LENGTH = 11


class ArtifactStore:
    """
    Manages the storage and retrieval of pipeline artifacts in shared memory
    to avoid costly inter-process communication.

    This class uses a multiprocessing.Manager to coordinate access to
    shared memory blocks across processes.
    """

    def __init__(self):
        """
        Initialize the ArtifactStore.
        """
        # This dictionary is the single source of truth for all shared memory
        # blocks for which this ArtifactStore instance has ownership. An open
        # handle is stored for each block this instance creates or adopts.
        self._managed_shms: Dict[str, shared_memory.SharedMemory] = {}
        # Track reference counts for shared memory blocks
        self._refcounts: Dict[str, int] = {}

    def __getstate__(self):
        """
        Exclude unpicklable SharedMemory handles when serializing.

        SharedMemory objects cannot be reliably transferred between
        processes via pickling after they've been unlinked. Workers
        should adopt their own handles when receiving artifact dicts.
        """
        return {"_refcounts": {}, "_managed_shms": {}}

    def __setstate__(self, state):
        """
        Initialize a fresh store when unpickling in a worker process.
        """
        self._refcounts = state.get("_refcounts", {})
        self._managed_shms = state.get("_managed_shms", {})

    def shutdown(self):
        for shm_name in list(self._managed_shms.keys()):
            self._release_by_name(shm_name)

    def adopt(self, handle: BaseArtifactHandle) -> None:
        """
        Takes ownership of a shared memory block created by another process.

        This method is called by the main process upon receiving an event that
        an artifact has been created in a worker. It opens its own handle to
        the shared memory block, ensuring it persists even if the creating
        worker process exits. This `ArtifactStore` instance now becomes
        responsible for the block's eventual release.

        Args:
            handle: The handle of the artifact whose shared memory block is
                    to be adopted.
        """
        shm_name = handle.shm_name
        if shm_name in self._managed_shms:
            # Increment refcount for already-managed block
            self._refcounts[shm_name] = self._refcounts.get(shm_name, 1) + 1
            logger.debug(
                f"Shared memory block {shm_name} refcount incremented to "
                f"{self._refcounts[shm_name]}"
            )
            return

        try:
            shm_obj = shared_memory.SharedMemory(name=shm_name)
            self._managed_shms[shm_name] = shm_obj
            self._refcounts[shm_name] = 1
            logger.debug(f"Adopted shared memory block: {shm_name}")
        except FileNotFoundError:
            logger.error(
                f"Failed to adopt shared memory block {shm_name}: "
                f"not found. It may have been released prematurely."
            )
        except Exception as e:
            logger.error(f"Error adopting shared memory block {shm_name}: {e}")

    @contextmanager
    def safe_adoption(
        self, handle_dict: Dict[str, Any]
    ) -> Generator[BaseArtifactHandle, None, None]:
        """
        Adopts a handle from a dictionary with automatic rollback on error.

        If the code executing within this context manager throws an exception,
        the adopted handle is immediately released (unlinked/destroyed).
        If the block completes successfully, the handle is kept and remains
        adopted.

        Args:
            handle_dict: The serialized handle dictionary.

        Yields:
            The adopted BaseArtifactHandle.
        """
        from .handle import create_handle_from_dict

        handle = create_handle_from_dict(handle_dict)
        self.adopt(handle)
        committed = False
        try:
            yield handle
            committed = True
        finally:
            if not committed:
                logger.debug(
                    f"Safe adoption rolled back (released) "
                    f"for {handle.shm_name}"
                )
                self.release(handle)

    def put(
        self,
        artifact: BaseArtifact,
        creator_tag: str = "unknown",
        generation_context: Optional["GenerationContext"] = None,
    ) -> BaseArtifactHandle:
        """
        Serializes an artifact into a new shared memory block and returns a
        handle.

        Args:
            artifact: The artifact to serialize.
            creator_tag: A tag identifying the creator for debugging.
                Must be alphanumeric and at most 11 characters.
            generation_context: Optional GenerationContext to track the
                created resource. When provided, the handle is registered
                with the context for lifecycle management.

        Returns:
            A handle to the serialized artifact.

        Raises:
            ValueError: If creator_tag exceeds MAX_TAG_LENGTH or contains
                invalid characters.
        """
        if len(creator_tag) > MAX_TAG_LENGTH:
            raise ValueError(
                f"creator_tag '{creator_tag}' exceeds maximum length of "
                f"{MAX_TAG_LENGTH} characters"
            )
        if not creator_tag or not all(
            c.isalnum() or c == "_" for c in creator_tag
        ):
            raise ValueError(
                f"creator_tag '{creator_tag}' must be non-empty and "
                f"contain only alphanumeric characters and underscores"
            )

        arrays = artifact.get_arrays_for_storage()
        total_bytes = sum(arr.nbytes for arr in arrays.values())

        shm_name = f"rf_{creator_tag}_{uuid.uuid4().hex[:16]}"
        try:
            # Prevent creating a zero-size block, which raises a ValueError.
            # A 1-byte block is a safe, minimal placeholder.
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=max(1, total_bytes)
            )
        except FileExistsError:
            # Handle rare UUID collision by retrying
            return self.put(
                artifact,
                creator_tag=creator_tag,
                generation_context=generation_context,
            )

        # Write data and collect metadata for the handle
        offset = 0
        array_metadata = {}
        for name, arr in arrays.items():
            # Create a view into the shared memory buffer at the correct offset
            dest_view = np.ndarray(
                arr.shape, dtype=arr.dtype, buffer=shm.buf, offset=offset
            )
            # Copy the data into the shared memory view
            dest_view[:] = arr[:]
            array_metadata[name] = {
                "dtype": str(arr.dtype),
                "shape": arr.shape,
                "offset": offset,
            }
            offset += arr.nbytes

        # The creating store is the owner of this block and must keep the
        # handle open to manage its lifecycle. This unified approach works
        # on all platforms and is required for the adoption model.
        self._managed_shms[shm_name] = shm
        self._refcounts[shm_name] = 1

        # Delegate handle creation to the artifact instance
        handle = artifact.create_handle(shm_name, array_metadata)

        if generation_context is not None:
            generation_context.add_resource(handle)

        return handle

    def get(self, handle: BaseArtifactHandle) -> BaseArtifact:
        """
        Reconstructs an artifact from a shared memory block using its handle.
        """
        shm = None
        close_after = False

        if handle.shm_name in self._managed_shms:
            managed_shm = self._managed_shms[handle.shm_name]
            assert managed_shm is not None
            try:
                buf = managed_shm.buf
                if buf is not None:
                    _ = buf[0]
                shm = managed_shm
                close_after = False
            except (FileNotFoundError, OSError):
                logger.debug(
                    f"Managed shared memory block {handle.shm_name} "
                    f"appears stale, reopening"
                )
                del self._managed_shms[handle.shm_name]
                self._refcounts.pop(handle.shm_name, None)

        if shm is None:
            try:
                shm = shared_memory.SharedMemory(name=handle.shm_name)
                close_after = True
            except FileNotFoundError:
                raise RuntimeError(
                    f"Shared memory block '{handle.shm_name}' not found."
                )

        try:
            arrays = {}
            for name, meta in handle.array_metadata.items():
                arr_view = np.ndarray(
                    meta["shape"],
                    dtype=np.dtype(meta["dtype"]),
                    buffer=shm.buf,
                    offset=meta["offset"],
                )
                arrays[name] = arr_view

            artifact_class = BaseArtifact.get_registered_class(
                handle.artifact_type_name
            )

            artifact = artifact_class.from_storage(handle, arrays)
        except (FileNotFoundError, OSError) as e:
            if close_after:
                shm.close()
            raise RuntimeError(
                f"Shared memory block '{handle.shm_name}' became "
                f"inaccessible: {e}"
            )

        if close_after:
            shm.close()

        return artifact

    def _release_by_name(self, shm_name: str) -> None:
        """
        Closes and unlinks a managed shared memory block by its name.
        Uses reference counting to ensure the block is not released while
        still in use.
        """
        caller_stack = safe_caller_stack()
        refcount = self._refcounts.get(shm_name, 0)
        if refcount > 1:
            # Decrement refcount and keep the block
            self._refcounts[shm_name] = refcount - 1
            if caller_stack:
                logger.debug(
                    f"Decremented refcount for {shm_name} to "
                    f"{self._refcounts[shm_name]} (caller: {caller_stack})"
                )
            return
        elif refcount == 1:
            # Refcount reached zero, release the block
            del self._refcounts[shm_name]
        else:
            # Refcount is 0 or not in refcounts
            # Check if block is still managed before warning
            if shm_name not in self._managed_shms:
                # Block was already released, this is fine
                logger.debug(
                    f"Shared memory block {shm_name} already released."
                )
                return
            # Block is managed but has no refcount, this is unusual
            msg = (
                f"Attempted to release block {shm_name}, which is "
                f"managed but has no refcount"
            )
            if caller_stack:
                msg += f" (caller: {caller_stack})"
            logger.warning(msg)
            return

        shm_obj = self._managed_shms.pop(shm_name, None)
        if not shm_obj:
            return

        try:
            shm_obj.close()
            msg = f"Unlinking shm: {shm_name}"
            if caller_stack:
                msg += f" (caller: {caller_stack})"
            logger.debug(msg)
            shm_obj.unlink()
            logger.debug(f"Released shared memory block: {shm_name}")
        except FileNotFoundError:
            # The block was already released externally, which is fine.
            logger.debug(f"SHM block {shm_name} was already unlinked.")
        except Exception as e:
            logger.warning(
                f"Error releasing shared memory block {shm_name}: {e}"
            )

    def release(self, handle: BaseArtifactHandle) -> None:
        """
        Closes and unlinks the shared memory block associated with a handle.

        This is the standard cleanup method when you own the SHM block and no
        other process needs it. It decrements the reference count and unlinks
        the memory when the count reaches zero.

        Use when:
            - You created the handle via put() and are done with it
            - You adopted a handle and no longer need it
            - Cleaning up cached artifacts in ArtifactManager

        Typical callers:
            - ArtifactManager (cache cleanup)
            - GenerationContext.shutdown()
            - ViewManager (cleanup old views)

        Args:
            handle: The handle of the artifact to release.
        """
        self._release_by_name(handle.shm_name)

    def close_handle(self, handle: BaseArtifactHandle) -> None:
        """
        Closes the shared memory block without unlinking it.

        Unlike release(), this method closes the file descriptor but leaves
        the shared memory block intact for other processes. The block will
        remain until explicitly unlinked or all handles across all processes
        are closed.

        Use when:
            - Closing a handle when the block is still needed elsewhere
            - Platform-specific cleanup where unlink timing matters

        Note:
            This method is rarely used. Prefer release() for normal cleanup
            or forget() for inter-process handoff.

        Args:
            handle: The handle of the artifact to close.
        """
        shm_name = handle.shm_name
        refcount = self._refcounts.get(shm_name, 0)
        if refcount > 1:
            # Decrement refcount and keep block
            self._refcounts[shm_name] = refcount - 1
            logger.debug(
                f"Decremented refcount for {shm_name} to "
                f"{self._refcounts[shm_name]}"
            )
            return

        # Refcount is 1 or less, close without unlinking
        shm_obj = self._managed_shms.get(shm_name, None)
        if not shm_obj:
            return

        try:
            logger.debug(
                f"Closing shared memory block without unlinking: {shm_name}"
            )
            shm_obj.close()
            # Keep in managed_shms for garbage collection
            # but don't unlink yet
        except Exception as e:
            logger.warning(
                f"Error closing shared memory block {shm_name}: {e}"
            )

    def retain(self, handle: BaseArtifactHandle) -> bool:
        """
        Increments the reference count for a shared memory block.

        Use when multiple code paths need the same handle and you want to
        prevent premature release. Every retain() must be matched with a
        release() or forget().

        Use when:
            - Multiple async callbacks will access the same artifact
            - Progressive rendering where chunks are reused across frames

        Typical callers:
            - ViewManager (progressive rendering chunks)

        Args:
            handle: The handle of the artifact whose shared memory block is
                    to be retained.

        Returns:
            True if the block was successfully retained, False otherwise.
        """
        shm_name = handle.shm_name
        if shm_name in self._managed_shms:
            self._refcounts[shm_name] = self._refcounts.get(shm_name, 1) + 1
            logger.debug(
                f"Retained {shm_name}, refcount now "
                f"{self._refcounts[shm_name]}"
            )
            return True
        return False

    def forget(self, handle: BaseArtifactHandle) -> None:
        """
        Closes the handle without unlinking the shared memory.

        This is used for inter-process handoff: a worker creates an artifact,
        sends it to the main process via IPC, and calls forget() after
        receiving acknowledgment. The main process has adopted the handle
        by then and will call release() when done.

        Use when:
            - Transferring ownership from worker to main process
            - After send_event_and_wait() returns successfully

        Typical callers:
            - workpiece_runner.py (after artifact_created acknowledged)
            - job_runner.py (after job_artifact_created acknowledged)
            - view_runner.py (after view_artifact_created acknowledged)
            - step_runner.py (after step artifacts acknowledged)

        On Windows, shared memory is destroyed when the last handle is closed.
        This method respects reference counting to ensure the block remains
        alive if other processes have adopted it.

        Args:
            handle: The handle of the artifact whose shared memory block is
                    to be forgotten.
        """
        shm_name = handle.shm_name
        refcount = self._refcounts.get(shm_name, 0)
        logger.debug(f"forget() called for {shm_name}, refcount={refcount}")

        shm_obj = self._managed_shms.get(shm_name)
        if not shm_obj:
            logger.warning(
                f"Attempted to forget block {shm_name}, which is not "
                f"managed or has already been released/forgotten."
            )
            return

        if refcount > 1:
            self._refcounts[shm_name] = refcount - 1
            logger.debug(
                f"Decremented refcount for {shm_name} to "
                f"{self._refcounts[shm_name]} in forget()"
            )
            # We do not pop it from _managed_shms here, because in
            # Windows this could lead to the garbage collector closing
            # it, which also destroys the underlying memory block
            return

        self._refcounts.pop(shm_name, None)
        self._managed_shms.pop(shm_name)

        try:
            logger.debug(f"Closing shared memory block: {shm_name}")
            shm_obj.close()
            logger.debug(f"Forgot shared memory block: {shm_name}")
        except Exception as e:
            logger.warning(
                f"Error forgetting shared memory block {shm_name}: {e}"
            )
