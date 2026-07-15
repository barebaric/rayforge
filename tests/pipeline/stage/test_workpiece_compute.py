from pathlib import Path
from unittest.mock import MagicMock

import pytest
from raygeo.geo import Geometry, Matrix
from raygeo.ops import Ops
from raygeo.ops.types import CommandType

from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.step import Step


from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.stage.workpiece_compute import (
    MAX_VECTOR_TRACE_PIXELS,
    _apply_transformers,
    _calculate_vector_render_size,
    _merge_artifact_ops,
    _validate_workpiece_size,
)
from rayforge.pipeline.transformer.base import OpsTransformer
from rayforge.shared.tasker.progress import set_progress


@pytest.fixture
def base_workpiece():
    """Creates a WorkPiece with basic vector data."""
    geo = Geometry()
    geo.move_to(0, 0, 0)
    geo.line_to(1, 0, 0)
    geo.line_to(1, 1, 0)
    geo.line_to(0, 1, 0)
    geo.close_path()

    source = SourceAsset(
        source_file=Path("test.dxf"), original_data=b"", renderer=MagicMock()
    )

    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=geo,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )

    wp = WorkPiece(name="test_wp", source_segment=segment)
    wp.set_size(25, 25)
    return wp


def test_set_progress_with_callback(mock_progress_context):
    """Test set_progress calls callback when provided."""
    set_progress(mock_progress_context, 0.5, "Test message")

    assert len(mock_progress_context.progress_calls) == 1
    assert mock_progress_context.progress_calls[0][0] == 0.5
    assert len(mock_progress_context.message_calls) == 1
    assert mock_progress_context.message_calls[0] == "Test message"


def test_set_progress_without_callback():
    """Test set_progress does not raise when callback is None."""
    set_progress(None, 0.5, "Test message")


def test_create_initial_ops():
    """Test Step.create_initial_ops creates configured Ops."""
    step = Step(typelabel="test")
    step.power = 0.8
    step.cut_speed = 15
    step.travel_speed = 30
    step.air_assist = True

    ops = step.create_initial_ops()

    assert isinstance(ops, Ops)
    assert not ops.is_empty()


def test_create_initial_ops_with_frequency_and_pulse_width():
    """Test Step.create_initial_ops injects frequency/pulse_width cmds."""
    step = Step(typelabel="test")
    step.power = 0.8
    step.cut_speed = 15
    step.travel_speed = 30
    step.air_assist = True
    step.frequency = 1000
    step.pulse_width = 50

    ops = step.create_initial_ops()

    freq_idxs = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SET_FREQUENCY
    ]
    pw_idxs = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SET_PULSE_WIDTH
    ]
    assert len(freq_idxs) == 1
    assert ops.frequency(freq_idxs[0]) == 1000
    assert len(pw_idxs) == 1
    assert ops.pulse_width(pw_idxs[0]) == 50


def test_create_initial_ops_zero_frequency_no_command():
    """Test Step.create_initial_ops skips frequency when value is 0."""
    step = Step(typelabel="test")
    step.power = 0.8
    step.cut_speed = 15
    step.travel_speed = 30
    step.air_assist = True
    step.frequency = 0

    ops = step.create_initial_ops()

    freq_idxs = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SET_FREQUENCY
    ]
    assert len(freq_idxs) == 0


def test_create_initial_ops_missing_frequency_no_error():
    """Test Step.create_initial_ops handles empty frequency."""
    step = Step(typelabel="test")
    step.power = 0.8
    step.cut_speed = 15
    step.travel_speed = 30
    step.air_assist = True

    ops = step.create_initial_ops()

    freq_idxs = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SET_FREQUENCY
    ]
    assert len(freq_idxs) == 0


def test_validate_workpiece_size_valid():
    """Test _validate_workpiece_size returns True for valid size."""
    assert _validate_workpiece_size((10.0, 20.0)) is True
    assert _validate_workpiece_size((0.1, 0.1)) is True


def test_validate_workpiece_size_none():
    """Test _validate_workpiece_size returns False for None."""
    assert _validate_workpiece_size(None) is False


def test_validate_workpiece_size_zero():
    """Test _validate_workpiece_size returns False for zero size."""
    assert _validate_workpiece_size((0.0, 10.0)) is False
    assert _validate_workpiece_size((10.0, 0.0)) is False


def test_validate_workpiece_size_negative():
    """Test _validate_workpiece_size returns False for negative size."""
    assert _validate_workpiece_size((-10.0, 20.0)) is False
    assert _validate_workpiece_size((10.0, -20.0)) is False


def test_calculate_vector_render_size_no_scaling():
    """Test _calculate_vector_render_size without scaling."""
    size_mm = (100.0, 100.0)
    px_per_mm_x = 10.0
    px_per_mm_y = 10.0

    width_px, height_px = _calculate_vector_render_size(
        size_mm, px_per_mm_x, px_per_mm_y
    )

    assert width_px == 1000
    assert height_px == 1000


def test_calculate_vector_render_size_with_scaling():
    """Test _calculate_vector_render_size applies scaling when needed."""
    size_mm = (10000.0, 10000.0)
    px_per_mm_x = 10.0
    px_per_mm_y = 10.0

    width_px, height_px = _calculate_vector_render_size(
        size_mm, px_per_mm_x, px_per_mm_y
    )

    num_pixels = width_px * height_px
    assert num_pixels <= MAX_VECTOR_TRACE_PIXELS


def test_calculate_vector_render_size_boundary():
    """Test _calculate_vector_render_size at max pixel boundary."""
    size_mm = (4096.0, 4096.0)
    px_per_mm_x = 10.0
    px_per_mm_y = 10.0

    width_px, height_px = _calculate_vector_render_size(
        size_mm, px_per_mm_x, px_per_mm_y
    )

    num_pixels = width_px * height_px
    assert num_pixels <= MAX_VECTOR_TRACE_PIXELS


def test_merge_artifact_ops_first_chunk():
    """Test _merge_artifact_ops with first chunk (no final artifact)."""
    chunk_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(10.0, 10.0),
        generation_id=1,
    )
    initial_ops = Ops()
    initial_ops.set_power(1.0)

    result = _merge_artifact_ops(None, chunk_artifact, initial_ops)

    assert result is not None
    assert result == chunk_artifact


def test_merge_artifact_ops_subsequent_chunks():
    """Test _merge_artifact_ops with subsequent chunks."""
    final_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(10.0, 10.0),
        generation_id=1,
    )
    chunk_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(10.0, 10.0),
        generation_id=1,
    )
    initial_ops = Ops()

    result = _merge_artifact_ops(final_artifact, chunk_artifact, initial_ops)

    assert result is final_artifact


def test_apply_transformers_empty_list(base_workpiece, mock_progress_context):
    """Test _apply_transformers with empty transformer list."""
    ops = Ops()
    transformers = []

    _apply_transformers(
        ops,
        transformers,
        base_workpiece,
        0.5,
        0.5,
        mock_progress_context,
    )

    assert len(mock_progress_context.progress_calls) == 0


def test_apply_transformers_disabled(
    base_workpiece, mock_progress_context, get_transformer
):
    """Test _apply_transformers with disabled transformer."""
    Optimize = get_transformer("Optimize")
    ops = Ops()
    transformer = Optimize()
    transformer.enabled = False
    transformers: list[OpsTransformer] = [transformer]

    _apply_transformers(
        ops,
        transformers,
        base_workpiece,
        0.5,
        0.5,
        mock_progress_context,
    )

    assert len(mock_progress_context.progress_calls) == 0
