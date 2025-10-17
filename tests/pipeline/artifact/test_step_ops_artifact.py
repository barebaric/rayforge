import unittest
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import StepOpsArtifact
# Removed unused import: from rayforge.pipeline import CoordinateSystem


class TestStepOpsArtifact(unittest.TestCase):
    """Test suite for the StepOpsArtifact class."""

    def test_artifact_type_property(self):
        """Tests that the artifact type is correctly identified."""
        artifact = StepOpsArtifact(ops=Ops())
        self.assertEqual(artifact.artifact_type, "StepOpsArtifact")

    def test_serialization_round_trip(self):
        """Tests serialization for an artifact with ops and time."""
        ops = Ops()
        ops.move_to(10, 20)
        ops.set_power(0.5)
        ops.line_to(30, 40)

        artifact = StepOpsArtifact(
            ops=ops,
            time_estimate=42.5,
        )

        artifact_dict = artifact.to_dict()
        reconstructed = StepOpsArtifact.from_dict(artifact_dict)

        # Check properties
        self.assertEqual(reconstructed.artifact_type, "StepOpsArtifact")
        self.assertEqual(reconstructed.time_estimate, 42.5)

        # These attributes no longer exist on StepOpsArtifact
        self.assertFalse(hasattr(reconstructed, "is_scalable"))
        self.assertFalse(hasattr(reconstructed, "source_coordinate_system"))

        # Check Ops content
        self.assertDictEqual(reconstructed.ops.to_dict(), ops.to_dict())


if __name__ == "__main__":
    unittest.main()
