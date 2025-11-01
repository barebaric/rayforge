from __future__ import annotations
from typing import cast, Any
import unittest
import json
import numpy as np
from multiprocessing import shared_memory
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.workpiece import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.artifact.job import JobArtifact, JobArtifactHandle
from rayforge.pipeline.artifact.base import (
    VertexData,
    TextureData,
    BaseArtifact,
)
from rayforge.pipeline.artifact.handle import BaseArtifactHandle


class EmptyArtifactHandle(BaseArtifactHandle):
    def __init__(self, ops: Ops, **kwargs: Any):
        super().__init__(**kwargs)
        self.ops = ops


class EmptyArtifact(BaseArtifact):
    """A minimal artifact for testing edge cases."""

    def __init__(self, ops: Ops):
        self.ops = ops
        self.artifact_type_name = self.__class__.__name__

    def get_arrays_for_storage(self) -> dict:
        return {}  # No arrays to store

    def create_handle(
        self, shm_name: str, array_metadata: dict
    ) -> EmptyArtifactHandle:
        return EmptyArtifactHandle(
            ops=self.ops,
            shm_name=shm_name,
            handle_class_name="EmptyArtifactHandle",
            artifact_type_name="EmptyArtifact",
            array_metadata=array_metadata,
        )

    @classmethod
    # Fixed (Style): Reformatted to fix line-too-long error.
    def from_storage(
        cls, handle: BaseArtifactHandle, arrays: dict
    ) -> EmptyArtifact:
        # Fixed: Reconstruct from the handle's ops.
        # We cast here to satisfy the type checker.
        typed_handle = cast(EmptyArtifactHandle, handle)
        return cls(ops=typed_handle.ops)


class TestArtifactStore(unittest.TestCase):
    """Test suite for the ArtifactStore shared memory manager."""

    def setUp(self):
        """Initializes a list to track created handles for cleanup."""
        self.store = get_context().artifact_store
        self.handles_to_release = []

    def tearDown(self):
        """
        Ensures all shared memory blocks created during tests are released.
        """
        for handle in self.handles_to_release:
            # We release without checking if it's already released,
            # as the store should handle this gracefully.
            try:
                self.store.release(handle)
            except Exception:
                # Suppress errors during teardown
                pass
        # Reset ref counts and shutdown to ensure a clean state
        self.store._ref_counts.clear()
        self.store.shutdown()

    def _create_sample_vertex_artifact(self) -> WorkPieceArtifact:
        """Helper to generate a consistent vertex artifact for tests."""
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(10, 0, 0)
        ops.arc_to(0, 10, i=-10, j=0, clockwise=False, z=0)

        vertex_data = VertexData(
            powered_vertices=np.array(
                [[0, 0, 0], [10, 0, 0]], dtype=np.float32
            ),
            powered_colors=np.array(
                [[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32
            ),
        )

        return WorkPieceArtifact(
            ops=ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(100, 100),
            generation_size=(50, 50),
            vertex_data=vertex_data,
        )

    def _create_sample_hybrid_artifact(self) -> WorkPieceArtifact:
        """Helper to generate a consistent hybrid artifact for tests."""
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(10, 0, 0, power_values=bytearray(range(256)))
        texture = np.arange(10000, dtype=np.uint8).reshape((100, 100))

        texture_data = TextureData(
            power_texture_data=texture,
            dimensions_mm=(50.0, 50.0),
            position_mm=(5.0, 10.0),
        )
        vertex_data = VertexData()
        return WorkPieceArtifact(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(200, 200),
            generation_size=(50, 50),
            vertex_data=vertex_data,
            texture_data=texture_data,
        )

    def _create_sample_final_job_artifact(self) -> JobArtifact:
        """Helper to generate a final job artifact for tests."""
        gcode_bytes = np.frombuffer(b"G1 X10 Y20", dtype=np.uint8)
        op_map = {"op_to_gcode": {0: [0, 1]}, "gcode_to_op": {0: 0, 1: 0}}
        op_map_bytes = np.frombuffer(
            json.dumps(op_map).encode("utf-8"), dtype=np.uint8
        )
        return JobArtifact(
            ops=Ops(),
            distance=15.0,
            gcode_bytes=gcode_bytes,
            op_map_bytes=op_map_bytes,
            vertex_data=VertexData(),  # Final jobs have vertex data
        )

    def test_put_get_release_vertex_artifact(self):
        """
        Tests the full put -> get -> release lifecycle with a
        vertex-based Artifact.
        """
        original_artifact = self._create_sample_vertex_artifact()

        # 1. Put the artifact into shared memory
        handle = self.store.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, WorkPieceArtifactHandle)

        # 2. Get the artifact back
        retrieved_artifact = self.store.get(handle)

        # 3. Verify the retrieved data
        assert isinstance(retrieved_artifact, WorkPieceArtifact)
        self.assertEqual(retrieved_artifact.artifact_type, "WorkPieceArtifact")
        self.assertIsNotNone(retrieved_artifact.vertex_data)
        self.assertIsNone(retrieved_artifact.texture_data)
        self.assertEqual(
            len(original_artifact.ops.commands),
            len(retrieved_artifact.ops.commands),
        )
        self.assertEqual(
            original_artifact.generation_size,
            retrieved_artifact.generation_size,
        )

        # 4. Release the memory
        self.store.release(handle)
        self.handles_to_release.remove(handle)

        # 5. Verify that the memory is no longer accessible
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_put_get_release_hybrid_artifact(self):
        """
        Tests the full put -> get -> release lifecycle with a
        Hybrid-like Artifact.
        """
        original_artifact = self._create_sample_hybrid_artifact()

        # 1. Put
        handle = self.store.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, WorkPieceArtifactHandle)

        # 2. Get
        retrieved_artifact = self.store.get(handle)

        # 3. Verify hybrid-specific attributes
        assert isinstance(retrieved_artifact, WorkPieceArtifact)
        self.assertEqual(retrieved_artifact.artifact_type, "WorkPieceArtifact")
        self.assertIsNotNone(retrieved_artifact.texture_data)
        self.assertIsNotNone(original_artifact.texture_data)
        assert retrieved_artifact.texture_data is not None
        assert original_artifact.texture_data is not None

        self.assertEqual(
            original_artifact.texture_data.dimensions_mm,
            retrieved_artifact.texture_data.dimensions_mm,
        )
        np.testing.assert_array_equal(
            original_artifact.texture_data.power_texture_data,
            retrieved_artifact.texture_data.power_texture_data,
        )
        self.assertEqual(
            len(original_artifact.ops.commands),
            len(retrieved_artifact.ops.commands),
        )

        # 4. Release
        self.store.release(handle)
        self.handles_to_release.remove(handle)

        # 5. Verify release
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_put_get_release_final_job_artifact(self):
        """
        Tests the full put -> get -> release lifecycle with a
        final_job Artifact.
        """
        original_artifact = self._create_sample_final_job_artifact()

        # 1. Put
        handle = self.store.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, JobArtifactHandle)

        # 2. Get
        retrieved_artifact = self.store.get(handle)

        # 3. Verify
        assert isinstance(retrieved_artifact, JobArtifact)
        self.assertEqual(retrieved_artifact.artifact_type, "JobArtifact")
        self.assertIsNotNone(retrieved_artifact.gcode_bytes)
        self.assertIsNotNone(retrieved_artifact.op_map_bytes)
        self.assertIsNotNone(retrieved_artifact.vertex_data)

        # Add assertions to satisfy the type checker
        assert retrieved_artifact.gcode_bytes is not None
        assert retrieved_artifact.op_map_bytes is not None

        # Decode and verify content
        gcode_str = retrieved_artifact.gcode_bytes.tobytes().decode("utf-8")
        op_map_str = retrieved_artifact.op_map_bytes.tobytes().decode("utf-8")
        raw_op_map = json.loads(op_map_str)

        # Reconstruct the map with integer keys, just like the app does.
        op_map = {
            "op_to_gcode": {
                int(k): v for k, v in raw_op_map["op_to_gcode"].items()
            },
            "gcode_to_op": {
                int(k): v for k, v in raw_op_map["gcode_to_op"].items()
            },
        }

        self.assertEqual(gcode_str, "G1 X10 Y20")
        self.assertDictEqual(
            op_map, {"op_to_gcode": {0: [0, 1]}, "gcode_to_op": {0: 0, 1: 0}}
        )

        # 4. Release
        self.store.release(handle)
        self.handles_to_release.remove(handle)

        # 5. Verify release
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_adopt_and_release(self):
        """
        Tests that `adopt` correctly takes ownership of a shared memory
        block, simulating a handover from a worker process.
        """
        original_artifact = self._create_sample_vertex_artifact()
        main_store = self.store

        # 1. Simulate a worker creating an artifact in its own store instance.
        worker_store = ArtifactStore()
        handle = worker_store.put(original_artifact)
        self.handles_to_release.append(handle)

        # The main store should not know about this block yet.
        self.assertNotIn(handle.shm_name, main_store._managed_shms)

        # 2. Simulate the main process receiving an event and adopting the
        # handle. Now, both the worker and main stores have open handles,
        # ensuring the block persists.
        main_store.adopt(handle)
        self.assertIn(handle.shm_name, main_store._managed_shms)

        # 3. Simulate the worker process transferring ownership by closing
        # its handle. The main process now holds the only handle.
        worker_store.shutdown()  # This will close and attempt to unlink

        # 4. Verify the block is still accessible by getting the artifact.
        try:
            retrieved = cast(WorkPieceArtifact, main_store.get(handle))
            self.assertEqual(
                original_artifact.generation_size, retrieved.generation_size
            )
        except FileNotFoundError:
            self.fail("Shared memory block was not found after adoption.")

        # 5. Release the memory via the main store's mechanism.
        main_store.release(handle)
        self.handles_to_release.remove(handle)

        # 6. Verify that the memory is now gone.
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

        # 7. Verify the block is no longer tracked by the main store.
        self.assertNotIn(handle.shm_name, main_store._managed_shms)

    def test_reference_counting_multiple_acquires(self):
        """
        Tests that acquire/release correctly manage the reference count
        and that the block is only unlinked on the final release.
        """
        artifact = self._create_sample_vertex_artifact()
        handle = self.store.put(artifact)
        self.handles_to_release.append(handle)

        self.assertEqual(self.store._ref_counts[handle.shm_name], 1)

        # Acquire two more references
        self.assertTrue(self.store.acquire(handle))
        self.assertTrue(self.store.acquire(handle))
        self.assertEqual(self.store._ref_counts[handle.shm_name], 3)

        # Release one reference, SHM should still exist
        self.store.release(handle)
        self.assertEqual(self.store._ref_counts[handle.shm_name], 2)
        # Verify SHM block is still present
        shm = shared_memory.SharedMemory(name=handle.shm_name)
        shm.close()

        # Release another reference
        self.store.release(handle)
        self.assertEqual(self.store._ref_counts[handle.shm_name], 1)

        # Final release should unlink the memory
        self.store.release(handle)
        self.assertNotIn(handle.shm_name, self.store._ref_counts)
        self.handles_to_release.remove(handle)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_scoped_reference_context_manager(self):
        """
        Tests the `scoped_reference` context manager for correct
        acquire/release behavior.
        """
        artifact = self._create_sample_vertex_artifact()
        handle = self.store.put(artifact)
        self.handles_to_release.append(handle)

        self.assertEqual(self.store._ref_counts[handle.shm_name], 1)

        with self.store.hold(handle) as acquired:
            self.assertTrue(acquired)
            self.assertEqual(self.store._ref_counts[handle.shm_name], 2)
            # Can get the artifact inside the scope
            retrieved = self.store.get(handle)
            # Fixed (Type Hinting): Add an isinstance check to inform the
            # type checker that this is a WorkPieceArtifact, resolving the
            # "reportAttributeAccessIssue".
            assert isinstance(retrieved, WorkPieceArtifact)
            self.assertIsNotNone(retrieved.vertex_data)

        # After exiting the context, ref count should be back to 1
        self.assertEqual(self.store._ref_counts[handle.shm_name], 1)

    def test_scoped_reference_on_released_handle(self):
        """
        Tests that `scoped_reference` correctly handles a handle that has
        already been released.
        """
        artifact = self._create_sample_vertex_artifact()
        handle = self.store.put(artifact)

        # Release the artifact immediately
        self.store.release(handle)

        # The context manager should yield False and not raise an error
        with self.store.hold(handle) as acquired:
            self.assertFalse(acquired)
            # The code inside this `if` block should not be executed
            if acquired:
                self.fail("Code inside 'if acquired' should not run.")

    def test_shutdown_cleans_up_all_memory(self):
        """
        Tests that the `shutdown` method correctly releases all managed
        shared memory blocks.
        """
        # Use a separate store to not interfere with the global context
        local_store = ArtifactStore()
        artifact1 = self._create_sample_vertex_artifact()
        artifact2 = self._create_sample_hybrid_artifact()

        handle1 = local_store.put(artifact1)
        handle2 = local_store.put(artifact2)

        # Verify blocks exist
        shm1 = shared_memory.SharedMemory(name=handle1.shm_name)
        shm1.close()
        shm2 = shared_memory.SharedMemory(name=handle2.shm_name)
        shm2.close()

        # Shutdown the store
        local_store.shutdown()

        # Verify both blocks are unlinked
        self.assertNotIn(handle1.shm_name, local_store._managed_shms)
        self.assertNotIn(handle2.shm_name, local_store._managed_shms)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle1.shm_name)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle2.shm_name)

    def test_put_empty_artifact(self):
        """
        Tests putting an artifact that has no numpy arrays. This should
        create a minimal (1-byte) shared memory block.
        """
        artifact = EmptyArtifact(ops=Ops())
        handle = self.store.put(artifact)
        self.handles_to_release.append(handle)

        # Check that a block was created and is managed
        self.assertIn(handle.shm_name, self.store._managed_shms)
        shm = self.store._managed_shms[handle.shm_name]
        self.assertGreaterEqual(shm.size, 1)

        # Check we can get it back
        retrieved = cast(EmptyArtifact, self.store.get(handle))
        self.assertIsInstance(retrieved, EmptyArtifact)
        # Verify ops were restored correctly
        self.assertEqual(retrieved.ops, artifact.ops)

        self.store.release(handle)
        self.handles_to_release.remove(handle)

    def test_get_on_released_handle_raises_error(self):
        """
        Tests that attempting to 'get' an artifact after it has been
        fully released raises a FileNotFoundError.
        """
        artifact = self._create_sample_vertex_artifact()
        handle = self.store.put(artifact)

        # Release it
        self.store.release(handle)

        # Attempting to get it now should fail
        with self.assertRaises(FileNotFoundError):
            self.store.get(handle)

    def test_double_release_is_safe(self):
        """
        Tests that releasing an already-released handle does not cause an
        error and is handled gracefully.
        """
        artifact = self._create_sample_vertex_artifact()
        handle = self.store.put(artifact)

        # First release (correct)
        self.store.release(handle)
        self.assertNotIn(handle.shm_name, self.store._ref_counts)

        # Second release (should be ignored without error)
        try:
            self.store.release(handle)
        except Exception as e:
            self.fail(f"Second release raised an unexpected exception: {e}")

    def test_adopt_already_managed_handle(self):
        """
        Tests that adopting a handle that the store already created and
        manages is a no-op and does not affect the ref count.
        """
        artifact = self._create_sample_vertex_artifact()
        handle = self.store.put(artifact)
        self.handles_to_release.append(handle)

        # Ref count should be 1 after putting
        self.assertEqual(self.store._ref_counts[handle.shm_name], 1)

        # Adopting it again should do nothing
        self.store.adopt(handle)
        self.assertEqual(self.store._ref_counts[handle.shm_name], 1)
        self.assertIn(handle.shm_name, self.store._managed_shms)


if __name__ == "__main__":
    unittest.main()
