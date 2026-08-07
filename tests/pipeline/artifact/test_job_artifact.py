from raygeo.ops import Ops

from rayforge.pipeline.artifact.job import JobArtifact


def _ops_with_line():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.set_power(1.0)
    ops.line_to(5.0, 0.0, 0.0)
    return ops


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


def test_preview_ops_prefers_mapped_ops():
    """preview_ops returns mapped_ops when it is set."""
    ops = _ops_with_line()
    mapped = _ops_with_line()
    mapped.set_power(0.9)

    artifact = JobArtifact(
        ops=ops,
        mapped_ops=mapped,
        distance=0.0,
        generation_id=1,
    )

    assert artifact.preview_ops is mapped


def test_preview_ops_falls_back_to_ops():
    """preview_ops falls back to ops when mapped_ops is None."""
    ops = _ops_with_line()

    artifact = JobArtifact(
        ops=ops,
        distance=0.0,
        generation_id=1,
    )

    assert artifact.preview_ops is ops
