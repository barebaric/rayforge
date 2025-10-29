from __future__ import annotations
import uuid
import logging
from typing import Dict
from multiprocessing import shared_memory
import numpy as np
from .base import BaseArtifact
from .handle import BaseArtifactHandle


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
            logger.debug(f"Shared memory block {shm_name} is already managed.")
            return

        try:
            shm_obj = shared_memory.SharedMemory(name=shm_name)
            self._managed_shms[shm_name] = shm_obj
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
        handle.
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

        # Delegate handle creation to the artifact instance
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
        Closes and unlinks a managed shared memory block by its name.
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
