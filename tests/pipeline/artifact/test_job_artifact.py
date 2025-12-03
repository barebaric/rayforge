import unittest
import json
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.job import JobArtifact


class TestJobArtifact(unittest.TestCase):
    """Test suite for the JobArtifact class."""

    def test_artifact_type_property(self):
        """Tests that the artifact type is correctly identified."""
        job_artifact = JobArtifact(
            ops=Ops(),
            distance=0.0,
            machine_code_bytes=np.array([72, 101, 108, 108, 111]),  # "Hello"
        )
        self.assertEqual(job_artifact.artifact_type, "JobArtifact")

    def test_final_job_serialization_round_trip(self):
        """Tests serialization for a final_job artifact."""
        machine_code_bytes = np.frombuffer(b"G1 X10", dtype=np.uint8)
        op_map_bytes = np.frombuffer(json.dumps({0: 0}).encode(), np.uint8)

        artifact = JobArtifact(
            ops=Ops(),
            distance=42.5,
            machine_code_bytes=machine_code_bytes,
            op_map_bytes=op_map_bytes,
            time_estimate=123.45,
        )

        reconstructed = JobArtifact.from_dict(artifact.to_dict())

        self.assertIsNotNone(reconstructed.machine_code_bytes)
        self.assertIsNotNone(reconstructed.op_map_bytes)
        self.assertEqual(reconstructed.time_estimate, 123.45)
        self.assertEqual(reconstructed.distance, 42.5)

        # Add assertions to satisfy the type checker
        assert reconstructed.machine_code_bytes is not None
        assert artifact.machine_code_bytes is not None
        np.testing.assert_array_equal(
            reconstructed.machine_code_bytes, artifact.machine_code_bytes
        )

        assert reconstructed.op_map_bytes is not None
        assert artifact.op_map_bytes is not None
        np.testing.assert_array_equal(
            reconstructed.op_map_bytes, artifact.op_map_bytes
        )


if __name__ == "__main__":
    unittest.main()
