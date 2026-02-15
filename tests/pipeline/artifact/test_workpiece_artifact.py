import unittest
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.workpiece import WorkPieceArtifact
from rayforge.pipeline import CoordinateSystem


class TestWorkPieceArtifact(unittest.TestCase):
    """Test suite for the WorkPieceArtifact class."""

    def test_artifact_type_property(self):
        """Tests that the artifact type is correctly identified."""
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=(1, 1),
        )
        self.assertEqual(artifact.artifact_type, "WorkPieceArtifact")

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


if __name__ == "__main__":
    unittest.main()
