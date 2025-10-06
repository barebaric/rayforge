import unittest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import VertexArtifact
from rayforge.pipeline import CoordinateSystem


class TestVertexArtifact(unittest.TestCase):
    """Test suite for the VertexArtifact class."""

    def test_serialization_round_trip(self):
        """
        Tests that a VertexArtifact can be converted to dict and back.
        """
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(10, 5, 0)

        artifact = VertexArtifact(
            ops=ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            powered_vertices=np.array(
                [[0, 0, 0], [10, 5, 0]], dtype=np.float32
            ),
            powered_colors=np.array(
                [[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32
            ),
            travel_vertices=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
            zero_power_vertices=np.empty((0, 3), dtype=np.float32),
            source_dimensions=(100.0, 50.0),
            generation_size=(20.0, 10.0),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = VertexArtifact.from_dict(artifact_dict)

        self.assertIsInstance(reconstructed, VertexArtifact)
        self.assertEqual(reconstructed.type, "vertex")
        self.assertTrue(reconstructed.is_scalable)
        self.assertEqual(reconstructed.source_dimensions, (100.0, 50.0))
        self.assertEqual(reconstructed.generation_size, (20.0, 10.0))
        self.assertEqual(len(reconstructed.ops.commands), 2)

        np.testing.assert_array_equal(
            reconstructed.powered_vertices, artifact.powered_vertices
        )
        np.testing.assert_array_equal(
            reconstructed.powered_colors, artifact.powered_colors
        )
        np.testing.assert_array_equal(
            reconstructed.travel_vertices, artifact.travel_vertices
        )
        self.assertEqual(reconstructed.zero_power_vertices.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
