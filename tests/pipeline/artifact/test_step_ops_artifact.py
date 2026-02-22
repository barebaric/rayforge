from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import StepOpsArtifact


def test_artifact_type_property():
    """Tests that the artifact type is correctly identified."""
    artifact = StepOpsArtifact(ops=Ops(), generation_id=1)
    assert artifact.artifact_type == "StepOpsArtifact"


def test_serialization_round_trip():
    """Tests serialization for an artifact with ops and time."""
    ops = Ops()
    ops.move_to(10, 20)
    ops.set_power(0.5)
    ops.line_to(30, 40)

    artifact = StepOpsArtifact(
        ops=ops,
        time_estimate=42.5,
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = StepOpsArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "StepOpsArtifact"
    assert reconstructed.time_estimate == 42.5

    assert not hasattr(reconstructed, "is_scalable")
    assert not hasattr(reconstructed, "source_coordinate_system")

    assert reconstructed.ops.to_dict() == ops.to_dict()
