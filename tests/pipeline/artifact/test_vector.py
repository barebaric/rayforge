import unittest
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import VectorArtifact
from rayforge.pipeline import CoordinateSystem


class TestVectorArtifact(unittest.TestCase):
    """Test suite for the VectorArtifact class."""

    def test_serialization_round_trip(self):
        """Tests that a VectorArtifact can be converted to a dict and back."""
        ops = Ops()
        ops.move_to(10, 20, 0)
        ops.line_to(30, 40, 0)

        artifact = VectorArtifact(
            ops=ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(100.0, 150.0),
            generation_size=(50.0, 75.0),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = VectorArtifact.from_dict(artifact_dict)

        self.assertIsInstance(reconstructed, VectorArtifact)
        self.assertEqual(reconstructed.type, "vector")
        self.assertTrue(reconstructed.is_scalable)
        self.assertEqual(
            reconstructed.source_coordinate_system,
            CoordinateSystem.MILLIMETER_SPACE,
        )
        self.assertEqual(reconstructed.source_dimensions, (100.0, 150.0))
        self.assertEqual(reconstructed.generation_size, (50.0, 75.0))
        self.assertEqual(len(reconstructed.ops.commands), 2)
        self.assertEqual(reconstructed.ops.commands[1].end, (30.0, 40.0, 0.0))


if __name__ == "__main__":
    unittest.main()
