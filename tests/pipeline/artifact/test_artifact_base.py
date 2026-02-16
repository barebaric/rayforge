import json
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.artifact import JobArtifact
from rayforge.pipeline import CoordinateSystem


def test_artifact_type_property():
    """Tests that specific artifact types are correctly identified."""
    workpiece_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(1, 1),
    )
    assert isinstance(workpiece_artifact, WorkPieceArtifact)
    assert workpiece_artifact.artifact_type == "WorkPieceArtifact"

    job_artifact = JobArtifact(
        ops=Ops(),
        distance=0.0,
        machine_code_bytes=np.array([72, 101, 108, 108, 111]),
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
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(100, 200),
        generation_size=(50, 100),
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
    )

    artifact_dict = artifact.to_dict()
    reconstructed = WorkPieceArtifact.from_dict(artifact_dict)

    assert reconstructed.artifact_type == "WorkPieceArtifact"
    assert not reconstructed.is_scalable
    assert (
        reconstructed.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
    )
    assert reconstructed.generation_size == (1, 1)


def test_final_job_serialization_round_trip():
    """Tests serialization for a final_job artifact."""
    machine_code_bytes = np.frombuffer(b"G1 X10", dtype=np.uint8)
    op_map_bytes = np.frombuffer(json.dumps({0: 0}).encode(), np.uint8)

    artifact = JobArtifact(
        ops=Ops(),
        distance=42.5,
        machine_code_bytes=machine_code_bytes,
        op_map_bytes=op_map_bytes,
    )

    reconstructed = JobArtifact.from_dict(artifact.to_dict())

    assert reconstructed.machine_code_bytes is not None
    assert reconstructed.op_map_bytes is not None
    assert reconstructed.distance == 42.5

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
