import unittest
from rayforge.pipeline.artifact import ArtifactHandle


class TestArtifactHandle(unittest.TestCase):
    """Test suite for the ArtifactHandle data class."""

    def test_serialization_round_trip(self):
        """Tests that an ArtifactHandle can be converted to a dict and back."""
        handle = ArtifactHandle(
            shm_name="test_shm_123",
            artifact_type="hybrid_raster",
            is_scalable=False,
            source_coordinate_system_name="PIXEL_SPACE",
            source_dimensions=(1024, 768),
            generation_size=(100.0, 75.0),
            time_estimate=123.4,
            dimensions_mm=(100.0, 75.0),
            position_mm=(10.0, 20.0),
            array_metadata={
                "ops_types": {"dtype": "int32", "shape": (10,), "offset": 0},
                "power_texture": {
                    "dtype": "uint8",
                    "shape": (768, 1024),
                    "offset": 40,
                },
            },
        )

        handle_dict = handle.to_dict()
        reconstructed_handle = ArtifactHandle.from_dict(handle_dict)

        self.assertEqual(handle, reconstructed_handle)
        self.assertEqual(
            reconstructed_handle.array_metadata["power_texture"]["shape"],
            (768, 1024),
        )


if __name__ == "__main__":
    unittest.main()
