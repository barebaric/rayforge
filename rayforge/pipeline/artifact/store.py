from __future__ import annotations
import uuid
import sys
import logging
from typing import Dict, Tuple
from multiprocessing import shared_memory
import numpy as np
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import Artifact
from .handle import ArtifactHandle
from .hybrid import HybridRasterArtifact
from .vector import VectorArtifact

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Manages the storage and retrieval of pipeline artifacts in shared memory
    to avoid costly inter-process communication.

    This class uses only class methods and is stateless, making it safe to
    use from multiple processes without needing an instance.
    """

    # On Windows, shared memory blocks are destroyed when all handles are
    # closed. To prevent a block from being destroyed immediately after
    # creation in `put()`, the creating process must keep a handle open.
    # This dictionary stores these handles. They are closed and removed
    # by `release()`. This is not needed on POSIX systems.
    if sys.platform == "win32":
        _managed_shms: Dict[str, shared_memory.SharedMemory] = {}

    @classmethod
    def put(cls, artifact: Artifact) -> ArtifactHandle:
        """
        Serializes an artifact into a new shared memory block and returns a
        handle.
        """
        arrays, total_bytes = cls._convert_artifact_to_arrays(artifact)

        # Create the shared memory block
        shm_name = f"rayforge_artifact_{uuid.uuid4()}"
        try:
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=total_bytes
            )
        except FileExistsError:
            # Handle rare UUID collision by retrying
            return cls.put(artifact)

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
            cls._managed_shms[shm_name] = shm
        else:
            shm.close()

        # Create the handle with all necessary metadata
        source_coord_system = artifact.source_coordinate_system
        handle = ArtifactHandle(
            shm_name=shm_name,
            artifact_type=artifact.type,
            is_scalable=artifact.is_scalable,
            source_coordinate_system_name=source_coord_system.name,
            source_dimensions=artifact.source_dimensions,
            generation_size=artifact.generation_size,
            array_metadata=array_metadata,
        )

        if isinstance(artifact, HybridRasterArtifact):
            handle.dimensions_mm = artifact.dimensions_mm
            handle.position_mm = artifact.position_mm

        return handle

    @classmethod
    def get(cls, handle: ArtifactHandle) -> Artifact:
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

        artifact = cls._reconstruct_artifact_from_arrays(handle, arrays)

        # This process is done with its connection to the SHM block.
        shm.close()

        return artifact

    @classmethod
    def release(cls, handle: ArtifactHandle) -> None:
        """
        Closes and unlinks the shared memory block associated with a handle.
        This must be called by the owner of the handle when it's no longer
        needed to prevent memory leaks.
        """
        if sys.platform == "win32":
            # If we are in the process that created the block, close the
            # handle we kept open to ensure the block's survival.
            if handle.shm_name in cls._managed_shms:
                shm_obj = cls._managed_shms.pop(handle.shm_name)
                shm_obj.close()

        try:
            shm = shared_memory.SharedMemory(name=handle.shm_name)
            shm.close()
            shm.unlink()  # This actually frees the memory
            logger.debug(f"Released shared memory block: {handle.shm_name}")
        except FileNotFoundError:
            # The block was already released, which is fine.
            pass
        except Exception as e:
            logger.warning(
                f"Error releasing shared memory block {handle.shm_name}: {e}"
            )

    @classmethod
    def _convert_artifact_to_arrays(
        cls, artifact: Artifact
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Converts an artifact's data into a dictionary of NumPy arrays and
        calculates total size. This is a pure function, making it easy to test.
        """
        arrays = artifact.ops.to_numpy_arrays()

        if isinstance(artifact, HybridRasterArtifact):
            arrays["power_texture_data"] = artifact.power_texture_data

        total_bytes = sum(arr.nbytes for arr in arrays.values())
        return arrays, total_bytes

    @classmethod
    def _reconstruct_artifact_from_arrays(
        cls, handle: ArtifactHandle, arrays: Dict[str, np.ndarray]
    ) -> Artifact:
        """
        Builds a full artifact object from a handle and a dictionary of
        NumPy array views. This is a pure function, making it easy to test.
        """
        # When reconstructing Ops, the from_numpy_arrays method implicitly
        # creates copies of the data by building new Python objects (tuples,
        # bytearrays, etc.), so it is already safe.
        ops = Ops.from_numpy_arrays(arrays)

        common_args = {
            "ops": ops,
            "is_scalable": handle.is_scalable,
            "source_coordinate_system": CoordinateSystem[
                handle.source_coordinate_system_name
            ],
            "source_dimensions": handle.source_dimensions,
            "generation_size": handle.generation_size,
        }

        if handle.artifact_type == "hybrid_raster":
            if handle.dimensions_mm is None or handle.position_mm is None:
                raise ValueError(
                    "HybridRasterArtifact handle is missing required "
                    "dimensions_mm or position_mm metadata."
                )

            # Explicitly copy the texture data from the shared memory view
            # into a new, owned NumPy array for the artifact.
            texture_copy = arrays["power_texture_data"].copy()

            return HybridRasterArtifact(
                power_texture_data=texture_copy,
                dimensions_mm=handle.dimensions_mm,
                position_mm=handle.position_mm,
                **common_args,
            )
        elif handle.artifact_type == "vector":
            return VectorArtifact(**common_args)
        else:
            raise TypeError(
                "Unknown artifact type for reconstruction: "
                f"{handle.artifact_type}"
            )
