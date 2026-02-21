from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.workpiece import WorkPieceArtifact
from rayforge.pipeline import CoordinateSystem


def test_artifact_type_property():
    """Tests that the artifact type is correctly identified."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(1, 1),
        generation_id=1,
    )
    assert artifact.artifact_type == "WorkPieceArtifact"


def test_vector_serialization_round_trip():
    """Tests serialization for a vector-like artifact."""
    ops = Ops()
    ops.move_to(1, 2, 3)
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(100, 200),
        generation_size=(50, 100),
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert reconstructed.ops.to_dict() == ops.to_dict()
    assert not reconstructed.is_scalable
    assert (
        reconstructed.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
    )
    assert reconstructed.source_dimensions == (100, 200)
    assert reconstructed.generation_size == (50, 100)


def test_vertex_serialization_round_trip():
    """Tests serialization for a vertex-like artifact."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(1, 1),
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert reconstructed.is_scalable
    assert (
        reconstructed.source_coordinate_system
        == CoordinateSystem.MILLIMETER_SPACE
    )
    assert reconstructed.generation_size == (1, 1)


def test_hybrid_serialization_round_trip():
    """Tests serialization for a hybrid raster artifact."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(1, 1),
        generation_id=1,
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert not reconstructed.is_scalable
    assert (
        reconstructed.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
    )
    assert reconstructed.generation_size == (1, 1)
