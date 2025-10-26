from __future__ import annotations
import uuid
import sys
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
        Initialize the ArtifactStore with a multiprocessing Manager.
        """
        # On Windows, shared memory blocks are destroyed when all handles are
        # closed. To prevent a block from being destroyed immediately after
        # creation in `put()`, the creating process must keep a handle open.
        # This dictionary stores these handles. They are closed and removed
        # by `release()`. This is not needed on POSIX systems.
        self._managed_shms: Dict[str, shared_memory.SharedMemory] = {}

    def shutdown(self):
        for shm_name in list(self._managed_shms.keys()):
            self._release_by_name(shm_name)

    def put(self, artifact: BaseArtifact) -> BaseArtifactHandle:
        """
        Serializes an artifact into a new shared memory block and returns a
        handle.
        """
        arrays = artifact.get_arrays_for_storage()
        total_bytes = sum(arr.nbytes for arr in arrays.values())

        # Create the shared memory block
        shm_name = f"rayforge_artifact_{uuid.uuid4()}"
        try:
            # Prevent creating a zero-size block, which raises a ValueError.
            # A 1-byte block is a safe, minimal placeholder.
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=max(1, total_bytes)
            )
        except FileExistsError:
            # Handle rare UUID collision by retrying
            return self.put(artifact)

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

        # On POSIX, we can close our local handle; the block persists until
        # unlinked. On Windows, the block is destroyed when the last handle
        # is closed, so we must keep this handle open in the creating process.
        if sys.platform == "win32":
            self._managed_shms[shm_name] = shm
        else:
            shm.close()

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
        Closes and unlinks the shared memory block with the given name.
        This must be called by the owner of the block when it's no longer
        needed to prevent memory leaks.
        """
        if sys.platform == "win32":
            # If we are in the process that created the block, close the
            # handle we kept open to ensure the block's survival.
            if shm_name in self._managed_shms:
                shm_obj = self._managed_shms.pop(shm_name)
                shm_obj.close()

        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()  # This actually frees the memory
            logger.debug(f"Released shared memory block: {shm_name}")
        except FileNotFoundError:
            # The block was already released, which is fine.
            pass
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


artifact_store = ArtifactStore()
