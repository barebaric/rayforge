import unittest
import json
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.artifact import JobArtifact
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

    def test_vertex_serialization_round_trip(self):
        """Tests serialization for a vertex-like artifact."""
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=(1, 1),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "WorkPieceArtifact")
        self.assertTrue(reconstructed.is_scalable)
        self.assertEqual(
            reconstructed.source_coordinate_system,
            CoordinateSystem.MILLIMETER_SPACE,
        )
        self.assertEqual(reconstructed.generation_size, (1, 1))

    def test_hybrid_serialization_round_trip(self):
        """Tests serialization for a hybrid raster artifact."""
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            generation_size=(1, 1),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "WorkPieceArtifact")
        self.assertFalse(reconstructed.is_scalable)
        self.assertEqual(
            reconstructed.source_coordinate_system,
            CoordinateSystem.PIXEL_SPACE,
        )
        self.assertEqual(reconstructed.generation_size, (1, 1))

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
