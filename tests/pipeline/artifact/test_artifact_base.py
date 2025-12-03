import unittest
import json
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.artifact import JobArtifact
from rayforge.pipeline.artifact import VertexData, TextureData
from rayforge.pipeline import CoordinateSystem


class TestArtifact(unittest.TestCase):
    """Test suite for the composable Artifact class."""

    def test_artifact_type_property(self):
        """Tests that specific artifact types are correctly identified."""
        # Test WorkPieceArtifact
        workpiece_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=(1, 1),
        )
        self.assertIsInstance(workpiece_artifact, WorkPieceArtifact)
        self.assertEqual(workpiece_artifact.artifact_type, "WorkPieceArtifact")

        # Test JobArtifact
        job_artifact = JobArtifact(
            ops=Ops(),
            distance=0.0,
            machine_code_bytes=np.array([72, 101, 108, 108, 111]),  # "Hello"
        )
        self.assertIsInstance(job_artifact, JobArtifact)
        self.assertEqual(job_artifact.artifact_type, "JobArtifact")

    def test_vector_serialization_round_trip(self):
        """Tests serialization for a vector-like artifact."""
        ops = Ops()
        ops.move_to(1, 2, 3)
        artifact = WorkPieceArtifact(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(100, 200),
            generation_size=(50, 100),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "WorkPieceArtifact")
        self.assertDictEqual(reconstructed.ops.to_dict(), ops.to_dict())
        self.assertFalse(reconstructed.is_scalable)
        self.assertEqual(
            reconstructed.source_coordinate_system,
            CoordinateSystem.PIXEL_SPACE,
        )
        self.assertEqual(reconstructed.source_dimensions, (100, 200))
        self.assertEqual(reconstructed.generation_size, (50, 100))
        self.assertIsNone(reconstructed.vertex_data)
        self.assertIsNone(reconstructed.texture_data)

    def test_vertex_serialization_round_trip(self):
        """Tests serialization for a vertex-like artifact."""
        vertex_data = VertexData(
            powered_vertices=np.array([[1, 2, 3]], dtype=np.float32),
            powered_colors=np.array([[0, 0, 0, 1]], dtype=np.float32),
            travel_vertices=np.array([[4, 5, 6]], dtype=np.float32),
        )
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            vertex_data=vertex_data,
            generation_size=(1, 1),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

        self.assertIsNotNone(reconstructed.vertex_data)
        self.assertIsNone(reconstructed.texture_data)

        assert reconstructed.vertex_data is not None
        np.testing.assert_array_equal(
            reconstructed.vertex_data.powered_vertices,
            vertex_data.powered_vertices,
        )
        np.testing.assert_array_equal(
            reconstructed.vertex_data.powered_colors,
            vertex_data.powered_colors,
        )
        np.testing.assert_array_equal(
            reconstructed.vertex_data.travel_vertices,
            vertex_data.travel_vertices,
        )

    def test_hybrid_serialization_round_trip(self):
        """Tests serialization for a hybrid raster artifact."""
        vertex_data = VertexData(
            powered_vertices=np.array([[1, 2, 3]], dtype=np.float32),
            powered_colors=np.array([[0, 0, 0, 1]], dtype=np.float32),
        )
        texture_data = TextureData(
            power_texture_data=np.array(
                [[0, 128], [128, 255]], dtype=np.uint8
            ),
            dimensions_mm=(10, 20),
            position_mm=(1, 2),
        )
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            vertex_data=vertex_data,
            texture_data=texture_data,
            generation_size=(1, 1),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

        self.assertIsNotNone(reconstructed.vertex_data)
        self.assertIsNotNone(reconstructed.texture_data)

        assert reconstructed.texture_data is not None
        np.testing.assert_array_equal(
            reconstructed.texture_data.power_texture_data,
            texture_data.power_texture_data,
        )
        self.assertEqual(reconstructed.texture_data.dimensions_mm, (10, 20))
        self.assertEqual(reconstructed.texture_data.position_mm, (1, 2))

    def test_final_job_serialization_round_trip(self):
        """Tests serialization for a final_job artifact."""
        machine_code_bytes = np.frombuffer(b"G1 X10", dtype=np.uint8)
        op_map_bytes = np.frombuffer(json.dumps({0: 0}).encode(), np.uint8)

        artifact = JobArtifact(
            ops=Ops(),
            distance=42.5,
            machine_code_bytes=machine_code_bytes,
            op_map_bytes=op_map_bytes,
        )

        reconstructed = JobArtifact.from_dict(artifact.to_dict())

        self.assertIsNotNone(reconstructed.machine_code_bytes)
        self.assertIsNotNone(reconstructed.op_map_bytes)
        self.assertEqual(reconstructed.distance, 42.5)

        # Add assertions to satisfy the type checker
        assert reconstructed.machine_code_bytes is not None
        assert artifact.machine_code_bytes is not None
        np.testing.assert_array_equal(
            reconstructed.machine_code_bytes, artifact.machine_code_bytes
        )

        assert reconstructed.op_map_bytes is not None
        assert artifact.op_map_bytes is not None
        np.testing.assert_array_equal(
            reconstructed.op_map_bytes, artifact.op_map_bytes
        )


if __name__ == "__main__":
    unittest.main()
