from raygeo.ops import Ops

from rayforge.pipeline.artifact import JobArtifact, WorkPieceArtifact


def test_artifact_type_property():
    """Tests that specific artifact types are correctly identified."""
    workpiece_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        generation_size=(1, 1),
        generation_id=1,
    )
    assert isinstance(workpiece_artifact, WorkPieceArtifact)
    assert workpiece_artifact.artifact_type == "WorkPieceArtifact"

    job_artifact = JobArtifact(
        ops=Ops(),
        distance=0.0,
        generation_id=1,
    )
    assert isinstance(job_artifact, JobArtifact)
    assert job_artifact.artifact_type == "JobArtifact"


def test_final_job_serialization_round_trip():
    """Tests serialization for a final_job artifact."""
    artifact = JobArtifact(
        ops=Ops(),
        distance=42.5,
        generation_id=1,
    )

    reconstructed = JobArtifact.from_dict(artifact.to_dict())

    assert reconstructed.distance == 42.5
    assert reconstructed.generation_id == 1
