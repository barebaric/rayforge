import unittest
import numpy as np

# Removed unused import: from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import (
    StepRenderArtifact,
    TextureData,
    TextureInstance,
    VertexData,
)
from rayforge.core.matrix import Matrix


class TestStepRenderArtifact(unittest.TestCase):
    """Test suite for the StepRenderArtifact class."""

    def test_artifact_type_property(self):
        """Tests that the artifact type is correctly identified."""
        artifact = StepRenderArtifact()
        self.assertEqual(artifact.artifact_type, "StepRenderArtifact")

    def test_vertex_serialization_round_trip(self):
        """Tests serialization for an artifact with only vertex data."""
        vertex_data = VertexData(
            powered_vertices=np.array([[1, 2, 3]], dtype=np.float32),
            travel_vertices=np.array([[4, 5, 6]], dtype=np.float32),
        )
        artifact = StepRenderArtifact(vertex_data=vertex_data)

        artifact_dict = artifact.to_dict()
        # Verify ops are not part of the serialization
        self.assertNotIn("ops", artifact_dict)

        reconstructed = StepRenderArtifact.from_dict(artifact_dict)

        self.assertIsNotNone(reconstructed.vertex_data)
        self.assertEqual(len(reconstructed.texture_instances), 0)

        assert reconstructed.vertex_data is not None
        np.testing.assert_array_equal(
            reconstructed.vertex_data.powered_vertices,
            vertex_data.powered_vertices,
        )
        np.testing.assert_array_equal(
            reconstructed.vertex_data.travel_vertices,
            vertex_data.travel_vertices,
        )
        # Ops attribute should no longer exist on StepRenderArtifact
        self.assertFalse(hasattr(reconstructed, "ops"))

    def test_hybrid_serialization_round_trip(self):
        """Tests serialization for an artifact with texture instances."""
        vertex_data = VertexData(
            powered_vertices=np.array([[7, 8, 9]], dtype=np.float32)
        )
        texture_data = TextureData(
            power_texture_data=np.array([[1, 2], [3, 4]], dtype=np.uint8),
            dimensions_mm=(10, 20),
            position_mm=(1, 2),
        )
        transform = Matrix.translation(100, 200).to_4x4_numpy()
        texture_instance = TextureInstance(
            texture_data=texture_data, world_transform=transform
        )

        artifact = StepRenderArtifact(
            vertex_data=vertex_data,
            texture_instances=[texture_instance],
        )

        artifact_dict = artifact.to_dict()
        reconstructed = StepRenderArtifact.from_dict(artifact_dict)

        self.assertIsNotNone(reconstructed.vertex_data)
        self.assertEqual(len(reconstructed.texture_instances), 1)

        # Check vertex data
        assert reconstructed.vertex_data is not None
        np.testing.assert_array_equal(
            reconstructed.vertex_data.powered_vertices,
            vertex_data.powered_vertices,
        )

        # Check texture instance data
        reconstructed_instance = reconstructed.texture_instances[0]
        np.testing.assert_array_equal(
            reconstructed_instance.texture_data.power_texture_data,
            texture_data.power_texture_data,
        )
        np.testing.assert_allclose(
            reconstructed_instance.world_transform, transform
        )


if __name__ == "__main__":
    unittest.main()
