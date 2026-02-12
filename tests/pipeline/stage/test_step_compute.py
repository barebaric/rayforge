import pytest
import numpy as np

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand
from rayforge.core.matrix import Matrix
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    StepRenderArtifact,
    StepOpsArtifact,
)
from rayforge.pipeline.artifact.base import TextureData, VertexData
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.transformer import Optimize, Smooth
from rayforge.pipeline.stage.step_compute import (
    compute_step_artifacts,
    _apply_artifact_scaling,
    _create_workpiece_placement_matrix,
    _calculate_texture_dimensions,
    _create_texture_data,
    _create_texture_instance,
    _process_artifact,
    _apply_transformers_to_ops,
    _encode_vertex_data,
)


@pytest.fixture
def setup_two_artifacts():
    """Creates two workpiece artifacts for testing."""
    doc = Doc()
    layer = doc.active_layer

    wp1 = WorkPiece(name="wp1")
    wp1.set_size(20, 10)
    wp1.pos = 50, 60
    wp1.angle = 90
    layer.add_workpiece(wp1)

    base_ops1 = Ops()
    base_ops1.move_to(0, 0)
    base_ops1.line_to(100, 0)
    artifact1 = WorkPieceArtifact(
        ops=base_ops1,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(20, 10),
    )

    wp2 = WorkPiece(name="wp2")
    wp2.set_size(30, 15)
    wp2.pos = 100, 120
    wp2.angle = 0
    layer.add_workpiece(wp2)

    base_ops2 = Ops()
    base_ops2.move_to(0, 0)
    base_ops2.line_to(50, 0)
    artifact2 = WorkPieceArtifact(
        ops=base_ops2,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(50, 50),
        generation_size=(30, 15),
    )

    artifacts = [
        (artifact1, wp1.get_world_transform(), wp1),
        (artifact2, wp2.get_world_transform(), wp2),
    ]

    return artifacts


def test_compute_step_artifacts_merges_ops(setup_two_artifacts):
    """Test that compute_step_artifacts merges ops from multiple artifacts."""
    artifacts = setup_two_artifacts

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[],
    )

    assert isinstance(render_artifact, StepRenderArtifact)
    assert isinstance(ops_artifact, StepOpsArtifact)

    line_commands = [
        c for c in ops_artifact.ops if isinstance(c, LineToCommand)
    ]
    assert len(line_commands) == 2


def test_compute_step_artifacts_applies_transforms(setup_two_artifacts):
    """Test that compute_step_artifacts applies world transforms."""
    artifacts = setup_two_artifacts

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[],
    )

    line_commands = [
        c for c in ops_artifact.ops if isinstance(c, LineToCommand)
    ]

    first_end = line_commands[0].end
    expected_first_end = (65.0, 75.0, 0.0)
    assert first_end == pytest.approx(expected_first_end)

    second_end = line_commands[1].end
    expected_second_end = (130.0, 120.0, 0.0)
    assert second_end == pytest.approx(expected_second_end)


def test_compute_step_artifacts_with_progress_callback(
    setup_two_artifacts, mock_progress_context
):
    """Test that compute_step_artifacts calls progress callback."""
    artifacts = setup_two_artifacts

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[],
        context=mock_progress_context,
    )

    assert len(mock_progress_context.progress_calls) > 0
    assert mock_progress_context.progress_calls[-1][0] == 1.0


def test_compute_step_artifacts_with_transformers(setup_two_artifacts):
    """Test that compute_step_artifacts applies transformers."""
    artifacts = setup_two_artifacts
    transformer = Optimize()

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[transformer],
    )

    assert isinstance(ops_artifact, StepOpsArtifact)
    assert not ops_artifact.ops.is_empty()


def test_compute_step_artifacts_empty_list():
    """Test compute_step_artifacts with empty artifacts list."""
    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=[],
        transformers=[],
    )

    assert isinstance(render_artifact, StepRenderArtifact)
    assert isinstance(ops_artifact, StepOpsArtifact)
    assert ops_artifact.ops.is_empty()


def test_compute_step_artifacts_with_message_callback(
    setup_two_artifacts, mock_progress_context
):
    """Test that compute_step_artifacts calls message callback."""
    artifacts = setup_two_artifacts

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[Optimize()],
        context=mock_progress_context,
    )

    assert len(mock_progress_context.message_calls) > 0


def test_compute_step_artifacts_raster_artifact():
    """Test compute_step_artifacts with non-scalable artifact."""
    doc = Doc()
    layer = doc.active_layer

    wp = WorkPiece(name="raster_wp")
    wp.set_size(20, 10)
    wp.pos = 50, 60
    layer.add_workpiece(wp)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)
    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(20, 10),
    )

    artifacts = [(artifact, wp.get_world_transform(), wp)]

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[],
    )

    assert isinstance(render_artifact, StepRenderArtifact)
    assert isinstance(ops_artifact, StepOpsArtifact)
    assert not ops_artifact.ops.is_empty()


def test_compute_step_artifacts_multiple_transformers(
    setup_two_artifacts,
):
    """Test compute_step_artifacts with multiple transformers."""
    artifacts = setup_two_artifacts
    transformers = [Optimize(), Smooth()]

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=transformers,
    )

    assert isinstance(render_artifact, StepRenderArtifact)
    assert isinstance(ops_artifact, StepOpsArtifact)
    assert not ops_artifact.ops.is_empty()


def test_compute_step_artifacts_without_source_dimensions(
    setup_two_artifacts,
):
    """Test compute_step_artifacts with artifact without source dims."""
    artifacts = setup_two_artifacts

    artifact1, matrix1, wp1 = artifacts[0]
    artifact1.source_dimensions = None

    new_artifacts = [
        (artifact1, matrix1, wp1),
        artifacts[1],
    ]

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=new_artifacts,
        transformers=[],
    )

    assert isinstance(render_artifact, StepRenderArtifact)
    assert isinstance(ops_artifact, StepOpsArtifact)


def test_apply_artifact_scaling_scalable_with_source_dimensions():
    """Test _apply_artifact_scaling for scalable artifact with source dims."""
    wp = WorkPiece(name="test_wp")
    wp.set_size(20, 10)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(20, 10),
    )

    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)

    _apply_artifact_scaling(ops, artifact, wp)

    line_commands = [c for c in ops if isinstance(c, LineToCommand)]
    assert len(line_commands) == 1
    assert line_commands[0].end == pytest.approx((20.0, 0.0, 0.0))


def test_apply_artifact_scaling_non_scalable():
    """Test _apply_artifact_scaling for non-scalable artifact."""
    wp = WorkPiece(name="test_wp")
    wp.set_size(20, 10)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(20, 10),
    )

    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)

    _apply_artifact_scaling(ops, artifact, wp)

    line_commands = [c for c in ops if isinstance(c, LineToCommand)]
    assert len(line_commands) == 1
    assert line_commands[0].end == pytest.approx((100.0, 0.0, 0.0))


def test_apply_artifact_scaling_without_source_dimensions():
    """Test _apply_artifact_scaling without source dimensions."""
    wp = WorkPiece(name="test_wp")
    wp.set_size(20, 10)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=None,
        generation_size=(20, 10),
    )

    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)

    _apply_artifact_scaling(ops, artifact, wp)

    line_commands = [c for c in ops if isinstance(c, LineToCommand)]
    assert len(line_commands) == 1
    assert line_commands[0].end == pytest.approx((100.0, 0.0, 0.0))


def test_create_workpiece_placement_matrix():
    """Test _create_workpiece_placement_matrix."""
    matrix = _create_workpiece_placement_matrix(
        tx=10.0, ty=20.0, angle=45.0, sy=1.0, skew=0.0
    )

    tx, ty, angle, sx, sy, skew = matrix.decompose()
    assert tx == pytest.approx(10.0)
    assert ty == pytest.approx(20.0)
    assert angle == pytest.approx(45.0)
    assert sx == pytest.approx(1.0)
    assert sy == pytest.approx(1.0)
    assert skew == pytest.approx(0.0)


def test_create_workpiece_placement_matrix_with_flip():
    """Test _create_workpiece_placement_matrix with Y flip."""
    matrix = _create_workpiece_placement_matrix(
        tx=10.0, ty=20.0, angle=0.0, sy=-1.0, skew=0.0
    )

    tx, ty, angle, sx, sy, skew = matrix.decompose()
    assert sy == pytest.approx(-1.0)


def test_calculate_texture_dimensions_with_source_dimensions():
    """Test _calculate_texture_dimensions with source dimensions."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(500, 250),
        generation_size=(10.0, 5.0),
    )

    width_px, height_px, px_per_mm_x, px_per_mm_y = (
        _calculate_texture_dimensions(artifact)
    )

    assert width_px == 500
    assert height_px == 250
    assert px_per_mm_x == pytest.approx(50.0)
    assert px_per_mm_y == pytest.approx(50.0)


def test_calculate_texture_dimensions_without_source_dimensions():
    """Test _calculate_texture_dimensions without source dimensions."""
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=None,
        generation_size=(10.0, 5.0),
    )

    width_px, height_px, px_per_mm_x, px_per_mm_y = (
        _calculate_texture_dimensions(artifact)
    )

    assert width_px == 500
    assert height_px == 250
    assert px_per_mm_x == pytest.approx(50.0)
    assert px_per_mm_y == pytest.approx(50.0)


def test_create_texture_data_non_scalable():
    """Test _create_texture_data for non-scalable artifact."""
    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(500, 250),
        generation_size=(10.0, 5.0),
    )

    texture_data = _create_texture_data(artifact)

    assert texture_data is not None
    assert texture_data.dimensions_mm == pytest.approx((10.0, 5.0))
    assert texture_data.position_mm == pytest.approx((0.0, 0.0))


def test_create_texture_data_scalable():
    """Test _create_texture_data returns None for scalable artifact."""
    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(10.0, 5.0),
    )

    texture_data = _create_texture_data(artifact)

    assert texture_data is None


def test_create_texture_instance():
    """Test _create_texture_instance."""
    texture_data = TextureData(
        power_texture_data=np.array([1, 2, 3], dtype=np.uint8),
        dimensions_mm=(10.0, 5.0),
        position_mm=(1.0, 2.0),
    )

    placement_matrix = Matrix.translation(10.0, 20.0)

    instance = _create_texture_instance(texture_data, placement_matrix)

    assert instance.texture_data is texture_data
    assert instance.world_transform is not None


def test_process_artifact_scalable():
    """Test _process_artifact for scalable artifact."""
    wp = WorkPiece(name="test_wp")
    wp.set_size(20, 10)
    wp.pos = 50, 60
    wp.angle = 90

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(20, 10),
    )

    world_matrix = wp.get_world_transform()

    ops, texture_instance = _process_artifact(artifact, world_matrix, wp)

    assert not ops.is_empty()
    assert texture_instance is None


def test_process_artifact_non_scalable():
    """Test _process_artifact for non-scalable artifact."""
    wp = WorkPiece(name="test_wp")
    wp.set_size(20, 10)
    wp.pos = 50, 60
    wp.angle = 0

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)

    artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(500, 250),
        generation_size=(10.0, 5.0),
    )

    world_matrix = wp.get_world_transform()

    ops, texture_instance = _process_artifact(artifact, world_matrix, wp)

    assert not ops.is_empty()
    assert texture_instance is not None


def test_apply_transformers_to_ops_empty_list():
    """Test _apply_transformers_to_ops with empty transformer list."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)

    wp = WorkPiece(name="test")
    wp.set_size(10, 10)
    _apply_transformers_to_ops(ops, [], wp)

    assert not ops.is_empty()


def test_apply_transformers_to_ops_with_transformer():
    """Test _apply_transformers_to_ops with transformer."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)
    ops.line_to(100, 50)

    transformer = Optimize()

    wp = WorkPiece(name="test")
    wp.set_size(10, 10)
    _apply_transformers_to_ops(ops, [transformer], wp)

    assert not ops.is_empty()


def test_apply_transformers_to_ops_with_progress_context(
    mock_progress_context,
):
    """Test _apply_transformers_to_ops with progress context."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)

    transformer = Optimize()

    wp = WorkPiece(name="test")
    wp.set_size(10, 10)
    _apply_transformers_to_ops(ops, [transformer], wp, mock_progress_context)

    assert len(mock_progress_context.message_calls) > 0


def test_encode_vertex_data():
    """Test _encode_vertex_data."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(100, 0)

    vertex_data = _encode_vertex_data(ops)

    assert isinstance(vertex_data, VertexData)
    assert vertex_data is not None


def test_encode_vertex_data_empty_ops():
    """Test _encode_vertex_data with empty ops."""
    ops = Ops()

    vertex_data = _encode_vertex_data(ops)

    assert isinstance(vertex_data, VertexData)
