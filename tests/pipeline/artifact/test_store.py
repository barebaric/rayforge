from typing import cast
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
from rayforge.pipeline.artifact.base import VertexData, TextureData


class TestArtifactStore(unittest.TestCase):
    """Test suite for the ArtifactStore shared memory manager."""

    def setUp(self):
        """Initializes a list to track created handles for cleanup."""
        self.handles_to_release = []

    def tearDown(self):
        """
        Ensures all shared memory blocks created during tests are released.
        """
        for handle in self.handles_to_release:
            get_context().artifact_store.release(handle)

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
        machine_code_bytes = np.frombuffer(b"G1 X10 Y20", dtype=np.uint8)
        op_map = {
            "op_to_machine_code": {0: [0, 1]},
            "machine_code_to_op": {0: 0, 1: 0},
        }
        op_map_bytes = np.frombuffer(
            json.dumps(op_map).encode("utf-8"), dtype=np.uint8
        )
        return JobArtifact(
            ops=Ops(),
            distance=15.0,
            machine_code_bytes=machine_code_bytes,
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
        handle = get_context().artifact_store.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, WorkPieceArtifactHandle)

        # 2. Get the artifact back
        retrieved_artifact = get_context().artifact_store.get(handle)

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
        get_context().artifact_store.release(handle)

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
        handle = get_context().artifact_store.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, WorkPieceArtifactHandle)

        # 2. Get
        retrieved_artifact = get_context().artifact_store.get(handle)

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
        get_context().artifact_store.release(handle)

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
        handle = get_context().artifact_store.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, JobArtifactHandle)

        # 2. Get
        retrieved_artifact = get_context().artifact_store.get(handle)

        # 3. Verify
        assert isinstance(retrieved_artifact, JobArtifact)
        self.assertEqual(retrieved_artifact.artifact_type, "JobArtifact")
        self.assertIsNotNone(retrieved_artifact.machine_code_bytes)
        self.assertIsNotNone(retrieved_artifact.op_map_bytes)
        self.assertIsNotNone(retrieved_artifact.vertex_data)

        # Add assertions to satisfy the type checker
        assert retrieved_artifact.machine_code_bytes is not None
        assert retrieved_artifact.op_map_bytes is not None

        # Decode and verify content
        gcode_str = retrieved_artifact.machine_code_bytes.tobytes().decode(
            "utf-8"
        )
        op_map_str = retrieved_artifact.op_map_bytes.tobytes().decode("utf-8")
        raw_op_map = json.loads(op_map_str)

        # Reconstruct the map with integer keys, just like the app does.
        op_map = {
            "op_to_machine_code": {
                int(k): v for k, v in raw_op_map["op_to_machine_code"].items()
            },
            "machine_code_to_op": {
                int(k): v for k, v in raw_op_map["machine_code_to_op"].items()
            },
        }

        self.assertEqual(gcode_str, "G1 X10 Y20")
        self.assertDictEqual(
            op_map,
            {
                "op_to_machine_code": {0: [0, 1]},
                "machine_code_to_op": {0: 0, 1: 0},
            },
        )

        # 4. Release
        get_context().artifact_store.release(handle)

        # 5. Verify release
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_adopt_and_release(self):
        """
        Tests that `adopt` correctly takes ownership of a shared memory
        block, simulating a handover from a worker process.
        """
        original_artifact = self._create_sample_vertex_artifact()
        main_store = get_context().artifact_store

        # 1. Simulate a worker creating an artifact in its own store instance.
        worker_store = ArtifactStore()
        handle = worker_store.put(original_artifact)

        # The main store should not know about this block yet.
        self.assertNotIn(handle.shm_name, main_store._managed_shms)

        # 2. Simulate the main process receiving an event and adopting the
        # handle. Now, both the worker and main stores have open handles,
        # ensuring the block persists.
        main_store.adopt(handle)
        self.assertIn(handle.shm_name, main_store._managed_shms)

        # 3. Simulate the worker process transferring ownership by closing
        # its handle. The main process now holds the only handle.
        worker_shm = worker_store._managed_shms.pop(handle.shm_name)
        worker_shm.close()

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

        # 6. Verify that the memory is now gone.
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

        # 7. Verify the block is no longer tracked by the main store.
        self.assertNotIn(handle.shm_name, main_store._managed_shms)


if __name__ == "__main__":
    unittest.main()
