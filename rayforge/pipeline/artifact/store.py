from __future__ import annotations
import uuid
import logging
from typing import Dict
from multiprocessing import shared_memory
import numpy as np
import threading
from .base import BaseArtifact
from .handle import BaseArtifactHandle


logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Manages the storage and retrieval of pipeline artifacts in shared memory
    to avoid costly inter-process communication.

    This class uses a reference counting system to manage the lifecycle of
    shared memory blocks across multiple processes and asynchronous consumers.
    """

    def __init__(self):
        """
        Initialize the ArtifactStore.
        """
        self._managed_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._ref_counts: Dict[str, int] = {}
        self._ref_counts_lock = threading.Lock()

    def shutdown(self):
        for shm_name in list(self._managed_shms.keys()):
            self._release_by_name(shm_name)

    def adopt(self, handle: BaseArtifactHandle) -> None:
        """
        Takes ownership of a shared memory block created by another process.
        The initial refcount is set to 1.

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
            logger.debug(f"Shared memory block {shm_name} is already managed.")
            return

        try:
            shm_obj = shared_memory.SharedMemory(name=shm_name)
            self._managed_shms[shm_name] = shm_obj
            with self._ref_counts_lock:
                if shm_name not in self._ref_counts:
                    self._ref_counts[shm_name] = 1
            logger.debug(f"Adopted shared memory block: {shm_name}")
        except FileNotFoundError:
            logger.error(
                f"Failed to adopt shared memory block {shm_name}: "
                f"not found. It may have been released prematurely."
            )
        except Exception as e:
            logger.error(f"Error adopting shared memory block {shm_name}: {e}")

    def put(
        self, artifact: BaseArtifact, creator_tag: str = "unknown"
    ) -> BaseArtifactHandle:
        """
        Serializes an artifact into a new shared memory block and returns a
        handle. The initial reference count is set to 1.
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
            return self.put(artifact, creator_tag=creator_tag)

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
        with self._ref_counts_lock:
            self._ref_counts[shm_name] = 1
        handle = artifact.create_handle(shm_name, array_metadata)
        return handle

    def get(self, handle: BaseArtifactHandle) -> BaseArtifact:
        """
        Reconstructs an artifact from a shared memory block using its handle.
        """
        shm = shared_memory.SharedMemory(name=handle.shm_name)

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

        shm.close()
        return artifact

    def _release_by_name(self, shm_name: str) -> None:
        """
        Internal method to close and unlink a managed shared memory block.
        This is now only called when the reference count reaches zero.
        """
        shm_obj = self._managed_shms.pop(shm_name, None)
        if not shm_obj:
            logger.warning(
                f"Attempted to release block {shm_name}, which is not "
                f"managed or has already been released."
            )
            return

        try:
            shm_obj.close()
            shm_obj.unlink()
            logger.debug(f"Released shared memory block: {shm_name}")
        except FileNotFoundError:
            # The block was already released externally, which is fine.
            logger.debug(f"SHM block {shm_name} was already unlinked.")
        except Exception as e:
            logger.warning(
                f"Error releasing shared memory block {shm_name}: {e}"
            )

    def acquire(self, handle: BaseArtifactHandle) -> None:
        """Increments the reference count for a shared memory block."""
        with self._ref_counts_lock:
            shm_name = handle.shm_name
            if shm_name in self._ref_counts:
                self._ref_counts[shm_name] += 1
                logger.debug(
                    f"Acquired {shm_name}. New ref count: "
                    f"{self._ref_counts[shm_name]}"
                )
            else:
                logger.warning(
                    f"Attempted to acquire un-tracked artifact {shm_name}"
                )

    def release(self, handle: BaseArtifactHandle) -> None:
        """
        Decrements the reference count for a block. If the count reaches
        zero, the block is closed and unlinked.
        """
        shm_name = handle.shm_name
        with self._ref_counts_lock:
            if shm_name not in self._ref_counts:
                logger.warning(
                    f"Attempted to release un-tracked artifact {shm_name}"
                )
                return

            self._ref_counts[shm_name] -= 1
            count = self._ref_counts[shm_name]
            logger.debug(f"Released {shm_name}. New ref count: {count}")

            if count <= 0:
                self._ref_counts.pop(shm_name, None)
                # Call the internal release method now that it's safe
                self._release_by_name(shm_name)
