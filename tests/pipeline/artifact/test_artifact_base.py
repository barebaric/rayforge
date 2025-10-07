import unittest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import Artifact
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
            vertex_data={"powered_vertices": np.empty((0, 3))},
        )
        self.assertEqual(vertex_artifact.artifact_type, "vertex")

        # Hybrid type (ops + vertex_data + raster_data)
        hybrid_artifact = Artifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            vertex_data={"powered_vertices": np.empty((0, 3))},
            raster_data={"power_texture_data": np.empty((1, 1))},
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
        self.assertIsNone(reconstructed.raster_data)

    def test_vertex_serialization_round_trip(self):
        """Tests serialization for a vertex-like artifact."""
        vertex_data = {
            "powered_vertices": np.array([[1, 2, 3]], dtype=np.float32),
            "powered_colors": np.array([[0, 0, 0, 1]], dtype=np.float32),
            "travel_vertices": np.array([[4, 5, 6]], dtype=np.float32),
            "zero_power_vertices": np.empty((0, 3), dtype=np.float32),
        }
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
        self.assertIsNone(reconstructed.raster_data)

        assert reconstructed.vertex_data is not None
        for key, arr in vertex_data.items():
            self.assertIn(key, reconstructed.vertex_data)
            np.testing.assert_array_equal(reconstructed.vertex_data[key], arr)

    def test_hybrid_serialization_round_trip(self):
        """Tests serialization for a hybrid raster artifact."""
        vertex_data = {
            "powered_vertices": np.array([[1, 2, 3]], dtype=np.float32),
            "powered_colors": np.array([[0, 0, 0, 1]], dtype=np.float32),
            "travel_vertices": np.array([], dtype=np.float32).reshape(0, 3),
            "zero_power_vertices": np.array([], dtype=np.float32).reshape(
                0, 3
            ),
        }
        raster_data = {
            "power_texture_data": np.array(
                [[0, 128], [128, 255]], dtype=np.uint8
            ),
            "dimensions_mm": (10, 20),
            "position_mm": (1, 2),
        }
        artifact = Artifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            vertex_data=vertex_data,
            raster_data=raster_data,
        )

        artifact_dict = artifact.to_dict()
        reconstructed = Artifact.from_dict(artifact_dict)

        self.assertEqual(reconstructed.artifact_type, "hybrid_raster")
        self.assertIsNotNone(reconstructed.vertex_data)
        self.assertIsNotNone(reconstructed.raster_data)

        assert reconstructed.raster_data is not None
        self.assertIn("power_texture_data", reconstructed.raster_data)
        np.testing.assert_array_equal(
            reconstructed.raster_data["power_texture_data"],
            raster_data["power_texture_data"],
        )
        self.assertEqual(reconstructed.raster_data["dimensions_mm"], (10, 20))
        self.assertEqual(reconstructed.raster_data["position_mm"], (1, 2))


if __name__ == "__main__":
    unittest.main()
