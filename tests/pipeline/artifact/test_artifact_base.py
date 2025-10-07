import unittest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.base import Artifact, VertexData, TextureData
from rayforge.pipeline import CoordinateSystem


class TestArtifact(unittest.TestCase):
    """Test suite for the composable Artifact class."""

    def test_artifact_type_property(self):
        """Tests that the artifact_type property is correctly determined."""
        # Vector type (only ops)
        vector_artifact = Artifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        self.assertEqual(vector_artifact.artifact_type, "vector")

        # Vertex type (ops + vertex_data)
        vertex_artifact = Artifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            vertex_data=VertexData(),
        )
        self.assertEqual(vertex_artifact.artifact_type, "vertex")

        # Hybrid type (ops + vertex_data + texture_data)
        hybrid_artifact = Artifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            vertex_data=VertexData(),
            texture_data=TextureData(
                power_texture_data=np.empty((1, 1)),
                dimensions_mm=(1, 1),
                position_mm=(0, 0),
            ),
        )
        self.assertEqual(hybrid_artifact.artifact_type, "hybrid_raster")

    def test_vector_serialization_round_trip(self):
        """Tests serialization for a vector-like artifact."""
        ops = Ops()
        ops.move_to(1, 2, 3)
        artifact = Artifact(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(100, 200),
            generation_size=(50, 100),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = Artifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "vector")
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
        artifact = Artifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            vertex_data=vertex_data,
        )

        artifact_dict = artifact.to_dict()
        reconstructed = Artifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "vertex")
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
        artifact = Artifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            vertex_data=vertex_data,
            texture_data=texture_data,
        )

        artifact_dict = artifact.to_dict()
        reconstructed = Artifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "hybrid_raster")
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
