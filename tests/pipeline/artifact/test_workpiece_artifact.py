from raygeo.ops import Ops

from rayforge.pipeline.artifact.workpiece import WorkPieceArtifact


def test_artifact_type_property():
    """Tests that the artifact type is correctly identified."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        generation_size=(1, 1),
        generation_id=1,
    )
    assert artifact.artifact_type == "WorkPieceArtifact"
