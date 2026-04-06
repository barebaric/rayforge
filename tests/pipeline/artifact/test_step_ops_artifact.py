from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import StepOpsArtifact


def test_artifact_type_property():
    """Tests that the artifact type is correctly identified."""
    artifact = StepOpsArtifact(ops=Ops(), generation_id=1)
    assert artifact.artifact_type == "StepOpsArtifact"


def test_ops_preserved():
    """Tests that ops are preserved through artifact creation."""
    ops = Ops()
    ops.move_to(10, 20)
    ops.set_power(0.5)
    ops.line_to(30, 40)

    artifact = StepOpsArtifact(ops=ops, generation_id=1)
    assert artifact.ops.to_dict() == ops.to_dict()
    assert artifact.generation_id == 1
    assert not hasattr(artifact, "is_scalable")
    assert not hasattr(artifact, "source_coordinate_system")
