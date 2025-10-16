import unittest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import StepArtifact
from rayforge.pipeline.artifact import VertexData, TextureData
from rayforge.pipeline import CoordinateSystem


class TestStepArtifact(unittest.TestCase):
    """Test suite for the StepArtifact class."""

    def test_artifact_type_property(self):
        """Tests that the artifact type is correctly identified."""
        step_artifact = StepArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        self.assertEqual(step_artifact.artifact_type, "StepArtifact")

    def test_vertex_serialization_round_trip(self):
        """Tests serialization for an artifact with vertex data."""
        vertex_data = VertexData(
            powered_vertices=np.array([[1, 2, 3]], dtype=np.float32),
            powered_colors=np.array([[0, 0, 0, 1]], dtype=np.float32),
            travel_vertices=np.array([[4, 5, 6]], dtype=np.float32),
        )
        artifact = StepArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            vertex_data=vertex_data,
        )

        artifact_dict = artifact.to_dict()
        reconstructed = StepArtifact.from_dict(artifact_dict)

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
        """Tests serialization for a hybrid artifact with texture data."""
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
        artifact = StepArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            vertex_data=vertex_data,
            texture_data=texture_data,
        )

        artifact_dict = artifact.to_dict()
        reconstructed = StepArtifact.from_dict(artifact_dict)

        self.assertIsNotNone(reconstructed.vertex_data)
        self.assertIsNotNone(reconstructed.texture_data)

        assert reconstructed.texture_data is not None
        np.testing.assert_array_equal(
            reconstructed.texture_data.power_texture_data,
            texture_data.power_texture_data,
        )
        self.assertEqual(reconstructed.texture_data.dimensions_mm, (10, 20))
        self.assertEqual(reconstructed.texture_data.position_mm, (1, 2))


if __name__ == "__main__":
    unittest.main()
