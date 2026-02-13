from __future__ import annotations
import uuid
import logging
from typing import Dict, Optional, TYPE_CHECKING
from multiprocessing import shared_memory
import numpy as np
from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ..context import GenerationContext


logger = logging.getLogger(__name__)


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
            generation_context: Optional GenerationContext to track the
                created resource. When provided, the handle is registered
                with the context for lifecycle management.

        Returns:
            A handle to the serialized artifact.
        """
        arrays = artifact.get_arrays_for_storage()
        total_bytes = sum(arr.nbytes for arr in arrays.values())

        # Create the shared memory block
        shm_name = f"rayforge_artifact_{creator_tag}_{uuid.uuid4()}"
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
        # If we manage this block, use our persistent handle to keep the
        # buffer valid.
        if handle.shm_name in self._managed_shms:
            shm = self._managed_shms[handle.shm_name]
            close_after = False
        else:
            # Otherwise, open a temporary handle. Warning: The buffer may
            # become invalid when this handle is closed!
            try:
                shm = shared_memory.SharedMemory(name=handle.shm_name)
                close_after = True
            except FileNotFoundError:
                raise RuntimeError(
                    f"Shared memory block '{handle.shm_name}' not found."
                )

        # Reconstruct views into the shared memory without copying data
        arrays = {}
        for name, meta in handle.array_metadata.items():
            arr_view = np.ndarray(
                meta["shape"],
                dtype=np.dtype(meta["dtype"]),
                buffer=shm.buf,
                offset=meta["offset"],
            )
            arrays[name] = arr_view

        # Look up the correct class from the central registry
        artifact_class = BaseArtifact.get_registered_class(
            handle.artifact_type_name
        )

        # Delegate reconstruction to the class
        artifact = artifact_class.from_storage(handle, arrays)

        if close_after:
            shm.close()

        return artifact

    def _release_by_name(self, shm_name: str) -> None:
        """
        Closes and unlinks a managed shared memory block by its name.
        Uses reference counting to ensure the block is not released while
        still in use.
        """
        refcount = self._refcounts.get(shm_name, 0)
        if refcount > 1:
            # Decrement refcount and keep the block
            self._refcounts[shm_name] = refcount - 1
            logger.debug(
                f"Decremented refcount for {shm_name} to "
                f"{self._refcounts[shm_name]}"
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
            logger.warning(
                f"Attempted to release block {shm_name}, which is "
                f"managed but has no refcount."
            )
            return

        shm_obj = self._managed_shms.pop(shm_name, None)
        if not shm_obj:
            return

        try:
            shm_obj.close()
            logger.info(
                f"[DIAGNOSTIC] About to unlink shared memory block: {shm_name}"
            )
            shm_obj.unlink()  # This actually frees the memory
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
        This must be called by the owner of the handle when it's no longer
        needed to prevent memory leaks.
        """
        self._release_by_name(handle.shm_name)

    def close_handle(self, handle: BaseArtifactHandle) -> None:
        """
        Closes the shared memory block associated with a handle without
        unlinking it. This is used when the block might still be
        in use by other processes (e.g., workers doing progressive rendering).
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
            logger.info(
                f"[DIAGNOSTIC] Closing shared memory block without unlinking: "
                f"{shm_name}"
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

        This is used to indicate that the block will be used by a subprocess
        and should not be released until the subprocess is done with it.

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
        Closes the handle to a shared memory block without destroying the
        underlying data.

        This is used when a worker process has transferred ownership of an
        artifact to another process (e.g., the main process). The worker
        closes its handle but does not unlink the shared memory, allowing
        the adopting process to continue accessing the data.

        Args:
            handle: The handle of the artifact whose shared memory block is
                    to be forgotten.
        """
        shm_name = handle.shm_name
        logger.info(
            f"[DIAGNOSTIC] forget() called for {shm_name}, refcount="
            f"{self._refcounts.get(shm_name, 0)}"
        )
        shm_obj = self._managed_shms.pop(shm_name, None)
        if not shm_obj:
            logger.warning(
                f"Attempted to forget block {shm_name}, which is not "
                f"managed or has already been released/forgotten."
            )
            return

        # Also remove from refcounts
        self._refcounts.pop(shm_name, None)

        try:
            logger.info(
                f"[DIAGNOSTIC] Closing shared memory block: {shm_name}"
            )
            shm_obj.close()
            logger.info(f"[DIAGNOSTIC] Forgot shared memory block: {shm_name}")
        except Exception as e:
            logger.warning(
                f"Error forgetting shared memory block {shm_name}: {e}"
            )
