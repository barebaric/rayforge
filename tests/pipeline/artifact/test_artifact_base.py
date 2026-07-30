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


def test_vector_serialization_round_trip():
    """Tests serialization for a vector-like artifact."""
    ops = Ops()
    ops.move_to(1, 2, 3)
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_dimensions=(100, 200),
        generation_size=(50, 100),
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert reconstructed.ops.to_dict() == ops.to_dict()
    assert not reconstructed.is_scalable
    assert reconstructed.source_dimensions == (100, 200)
    assert reconstructed.generation_size == (50, 100)


def test_vertex_serialization_round_trip():
    """Tests serialization for a vertex-like artifact."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        generation_size=(1, 1),
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert reconstructed.is_scalable
    assert reconstructed.generation_size == (1, 1)


def test_hybrid_serialization_round_trip():
    """Tests serialization for a hybrid raster artifact."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        generation_size=(1, 1),
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert not reconstructed.is_scalable
    assert reconstructed.generation_size == (1, 1)


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
