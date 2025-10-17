import unittest
import json
import numpy as np
from multiprocessing import shared_memory
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact import ArtifactStore
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
            ArtifactStore.release(handle)

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
        op_map = {0: 0, 1: 1, 2: 2}
        op_map_bytes = np.frombuffer(
            json.dumps(op_map).encode("utf-8"), dtype=np.uint8
        )
        return JobArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
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
        handle = ArtifactStore.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, WorkPieceArtifactHandle)

        # 2. Get the artifact back
        retrieved_artifact = ArtifactStore.get(handle)

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
        ArtifactStore.release(handle)

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
        handle = ArtifactStore.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, WorkPieceArtifactHandle)

        # 2. Get
        retrieved_artifact = ArtifactStore.get(handle)

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
        ArtifactStore.release(handle)

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
        handle = ArtifactStore.put(original_artifact)
        self.handles_to_release.append(handle)
        self.assertIsInstance(handle, JobArtifactHandle)

        # 2. Get
        retrieved_artifact = ArtifactStore.get(handle)

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
        op_map = {int(k): v for k, v in json.loads(op_map_str).items()}

        self.assertEqual(gcode_str, "G1 X10 Y20")
        self.assertDictEqual(op_map, {0: 0, 1: 1, 2: 2})

        # 4. Release
        ArtifactStore.release(handle)

        # 5. Verify release
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)


if __name__ == "__main__":
    unittest.main()
