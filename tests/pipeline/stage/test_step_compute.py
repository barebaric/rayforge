import pytest

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    StepRenderArtifact,
    StepOpsArtifact,
)
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.stage.step_compute import compute_step_artifacts


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
    from rayforge.pipeline.transformer import Optimize

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

    from rayforge.pipeline.transformer import Optimize

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=[Optimize()],
        context=mock_progress_context,
    )

    assert len(mock_progress_context.message_calls) > 0


def test_compute_step_artifacts_raster_artifact():
    """Test compute_step_artifacts with non-scalable artifact."""
    from rayforge.pipeline.coord import CoordinateSystem

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
    from rayforge.pipeline.transformer import Optimize, Smooth

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
