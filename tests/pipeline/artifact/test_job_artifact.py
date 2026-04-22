import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.job import JobArtifact
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


def test_artifact_type_property():
    """Tests that the artifact type is correctly identified."""
    job_artifact = JobArtifact(
        ops=Ops(),
        distance=0.0,
        generation_id=1,
    )
    assert job_artifact.artifact_type == "JobArtifact"


def test_final_job_serialization_round_trip():
    """Tests serialization for a final_job artifact."""
    encoded = EncodedOutput(text="G1 X10", op_map=MachineCodeOpMap())
    encoded_output_bytes = np.frombuffer(encoded.to_json().encode(), np.uint8)

    artifact = JobArtifact(
        ops=Ops(),
        distance=42.5,
        encoded_output_bytes=encoded_output_bytes,
        time_estimate=123.45,
        generation_id=1,
    )

    reconstructed = JobArtifact.from_dict(artifact.to_dict())

    assert reconstructed.encoded_output_bytes is not None
    assert reconstructed.time_estimate == 123.45
    assert reconstructed.distance == 42.5

    assert artifact.encoded_output_bytes is not None
    np.testing.assert_array_equal(
        reconstructed.encoded_output_bytes, artifact.encoded_output_bytes
    )
