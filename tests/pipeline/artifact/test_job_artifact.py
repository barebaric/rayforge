from raygeo.ops import Ops

from rayforge.pipeline.artifact.job import JobArtifact


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
    artifact = JobArtifact(
        ops=Ops(),
        distance=42.5,
        time_estimate=123.45,
        generation_id=1,
    )

    reconstructed = JobArtifact.from_dict(artifact.to_dict())

    assert reconstructed.time_estimate == 123.45
    assert reconstructed.distance == 42.5
    assert reconstructed.generation_id == 1
