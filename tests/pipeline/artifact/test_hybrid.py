import unittest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import HybridRasterArtifact
from rayforge.pipeline import CoordinateSystem


class TestHybridRasterArtifact(unittest.TestCase):
    """Test suite for the HybridRasterArtifact class."""

    def test_serialization_round_trip(self):
        """
        Tests that a HybridRasterArtifact can be converted to dict and back.
        """
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(10, 0, 0, power_values=bytearray([128, 255]))
        texture = np.array([[0, 100], [200, 255]], dtype=np.uint8)

        artifact = HybridRasterArtifact(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            power_texture_data=texture,
            dimensions_mm=(50.0, 25.0),
            position_mm=(5.0, 10.0),
            source_dimensions=(100, 50),
            generation_size=(50.0, 25.0),
        )

        artifact_dict = artifact.to_dict()
        reconstructed = HybridRasterArtifact.from_dict(artifact_dict)

        self.assertIsInstance(reconstructed, HybridRasterArtifact)
        self.assertEqual(reconstructed.type, "hybrid_raster")
        self.assertFalse(reconstructed.is_scalable)
        self.assertEqual(reconstructed.dimensions_mm, (50.0, 25.0))
        self.assertEqual(reconstructed.position_mm, (5.0, 10.0))
        self.assertEqual(len(reconstructed.ops.commands), 2)
        np.testing.assert_array_equal(
            reconstructed.power_texture_data, texture
        )


if __name__ == "__main__":
    unittest.main()
