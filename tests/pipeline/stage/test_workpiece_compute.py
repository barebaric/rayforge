import cairo
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.ops import Ops
from rayforge.machine.models.machine import Laser
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.producer import ContourProducer, Rasterizer
from rayforge.pipeline.transformer import Optimize
from rayforge.pipeline.transformer.base import OpsTransformer
from rayforge.pipeline.stage.workpiece_compute import (
    _create_initial_ops,
    _validate_workpiece_size,
    _calculate_vector_render_size,
    _execute_vector,
    _execute_raster,
    _apply_transformers,
    _merge_artifact_ops,
    compute_workpiece_artifact_vector,
    compute_workpiece_artifact_raster,
    compute_workpiece_artifact,
    MAX_VECTOR_TRACE_PIXELS,
)
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


def test_compute_workpiece_artifact_returns_valid_artifact(
    base_workpiece,
):
    """Test that compute_workpiece_artifact returns a valid artifact."""
    opsproducer = ContourProducer()
    laser = Laser()
    transformers = []
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }
    pixels_per_mm = (10.0, 10.0)
    generation_size = (25.0, 25.0)

    result = compute_workpiece_artifact(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        transformers=transformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        generation_id=1,
    )

    assert result is not None
    assert isinstance(result, WorkPieceArtifact)
    assert not result.ops.is_empty()
    assert result.generation_size == generation_size


def test_compute_workpiece_artifact_with_progress_callback(
    base_workpiece, mock_progress_context
):
    """Test that compute_workpiece_artifact calls progress callback."""
    opsproducer = ContourProducer()
    laser = Laser()
    transformers = []
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }
    pixels_per_mm = (10.0, 10.0)
    generation_size = (25.0, 25.0)

    result = compute_workpiece_artifact(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        transformers=transformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        generation_id=1,
        context=mock_progress_context,
    )

    assert result is not None
    assert len(mock_progress_context.progress_calls) > 0
    assert mock_progress_context.progress_calls[-1][0] == 1.0


def test_compute_workpiece_artifact_with_empty_workpiece():
    """Test compute_workpiece_artifact returns None for empty workpiece."""
    empty_source = SourceAsset(
        source_file=Path("empty"), original_data=b"", renderer=MagicMock()
    )
    empty_segment = SourceAssetSegment(
        source_asset_uid=empty_source.uid,
        pristine_geometry=Geometry(),
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )
    empty_workpiece = WorkPiece(name="empty_wp", source_segment=empty_segment)
    empty_workpiece.set_size(10, 10)

    opsproducer = ContourProducer()
    laser = Laser()
    transformers = []
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }
    pixels_per_mm = (10.0, 10.0)
    generation_size = (10.0, 10.0)

    result = compute_workpiece_artifact(
        workpiece=empty_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        transformers=transformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        generation_id=1,
    )

    assert result is None


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
    """Test _create_initial_ops creates configured Ops."""
    settings = {
        "power": 0.8,
        "cut_speed": 15,
        "travel_speed": 30,
        "air_assist": True,
    }

    ops = _create_initial_ops(settings)

    assert isinstance(ops, Ops)
    assert not ops.is_empty()


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


def test_execute_vector_with_boundaries(base_workpiece):
    """Test _execute_vector with boundaries (vector path)."""
    base_workpiece._edited_boundaries = base_workpiece.boundaries

    opsproducer = ContourProducer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    results = list(
        _execute_vector(
            base_workpiece,
            opsproducer,
            laser,
            settings,
            1,
            base_workpiece.size,
        )
    )

    assert len(results) == 1
    artifact, progress = results[0]
    assert progress == 1.0


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


def test_apply_transformers_disabled(base_workpiece, mock_progress_context):
    """Test _apply_transformers with disabled transformer."""
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


def test_compute_workpiece_artifact_vector(base_workpiece):
    """Test compute_workpiece_artifact_vector returns valid artifact."""
    opsproducer = ContourProducer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    result = compute_workpiece_artifact_vector(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        settings=settings,
        generation_id=1,
        generation_size=base_workpiece.size,
    )

    assert result is not None
    assert isinstance(result, WorkPieceArtifact)


def test_compute_workpiece_artifact_vector_with_progress(
    base_workpiece, mock_progress_context
):
    """Test compute_workpiece_artifact_vector with progress callback."""
    opsproducer = ContourProducer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    result = compute_workpiece_artifact_vector(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        settings=settings,
        generation_id=1,
        generation_size=base_workpiece.size,
        context=mock_progress_context,
    )

    assert result is not None
    assert len(mock_progress_context.progress_calls) > 0


def test_compute_workpiece_artifact_vector_with_boundaries(
    base_workpiece,
):
    """Test compute_workpiece_artifact_vector with boundaries."""
    base_workpiece._edited_boundaries = base_workpiece.boundaries

    opsproducer = ContourProducer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    result = compute_workpiece_artifact_vector(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        settings=settings,
        generation_id=1,
        generation_size=base_workpiece.size,
    )

    assert result is not None


def test_execute_raster_invalid_size(base_workpiece):
    """Test _execute_raster returns nothing for invalid size."""
    base_workpiece.set_size(0, 0)

    opsproducer = ContourProducer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    results = list(
        _execute_raster(
            base_workpiece,
            opsproducer,
            laser,
            settings,
            1,
            base_workpiece.size,
        )
    )

    assert len(results) == 0


def test_compute_workpiece_artifact_with_air_assist(
    base_workpiece,
):
    """Test compute_workpiece_artifact with air assist enabled."""
    opsproducer = ContourProducer()
    laser = Laser()
    transformers = []
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": True,
    }
    pixels_per_mm = (10.0, 10.0)
    generation_size = (25.0, 25.0)

    result = compute_workpiece_artifact(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        transformers=transformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        generation_id=1,
    )

    assert result is not None
    assert isinstance(result, WorkPieceArtifact)


def test_compute_workpiece_artifact_raster(base_workpiece):
    """Test compute_workpiece_artifact_raster returns valid artifact."""
    opsproducer = Rasterizer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 250, 250)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()

    with patch.object(
        base_workpiece, "render_chunk", return_value=[(surface, (0, 0))]
    ):
        result = compute_workpiece_artifact_raster(
            workpiece=base_workpiece,
            opsproducer=opsproducer,
            laser=laser,
            settings=settings,
            generation_id=1,
            generation_size=base_workpiece.size,
        )

    assert result is not None
    assert isinstance(result, WorkPieceArtifact)


def test_compute_workpiece_artifact_raster_with_progress(
    base_workpiece, mock_progress_context
):
    """Test compute_workpiece_artifact_raster with progress callback."""
    opsproducer = Rasterizer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 250, 250)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()

    with patch.object(
        base_workpiece, "render_chunk", return_value=[(surface, (0, 0))]
    ):
        result = compute_workpiece_artifact_raster(
            workpiece=base_workpiece,
            opsproducer=opsproducer,
            laser=laser,
            settings=settings,
            generation_id=1,
            generation_size=base_workpiece.size,
            context=mock_progress_context,
        )

    assert result is not None
    assert len(mock_progress_context.progress_calls) > 0


def test_compute_workpiece_artifact_raster_empty_workpiece():
    """Test compute_workpiece_artifact_raster with empty workpiece."""
    empty_source = SourceAsset(
        source_file=Path("empty"), original_data=b"", renderer=MagicMock()
    )
    empty_segment = SourceAssetSegment(
        source_asset_uid=empty_source.uid,
        pristine_geometry=Geometry(),
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )
    empty_workpiece = WorkPiece(name="empty_wp", source_segment=empty_segment)
    empty_workpiece.set_size(10, 10)

    opsproducer = Rasterizer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    result = compute_workpiece_artifact_raster(
        workpiece=empty_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        settings=settings,
        generation_id=1,
        generation_size=empty_workpiece.size,
    )

    assert result is None


def test_compute_workpiece_artifact_with_transformers(
    base_workpiece,
):
    """Test compute_workpiece_artifact applies transformers."""
    opsproducer = ContourProducer()
    laser = Laser()
    transformers: list[OpsTransformer] = [Optimize()]
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }
    pixels_per_mm = (10.0, 10.0)
    generation_size = (25.0, 25.0)

    result = compute_workpiece_artifact(
        workpiece=base_workpiece,
        opsproducer=opsproducer,
        laser=laser,
        transformers=transformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        generation_id=1,
    )

    assert result is not None
    assert isinstance(result, WorkPieceArtifact)


def test_chunk_artifact_has_generation_size(base_workpiece):
    """Test that chunk artifacts carry the full generation_size."""
    opsproducer = Rasterizer()
    laser = Laser()
    settings = {
        "pixels_per_mm": (10.0, 10.0),
        "power": 1.0,
        "cut_speed": 10,
        "travel_speed": 20,
        "air_assist": False,
    }

    chunks_received = []

    def on_chunk_callback(chunk_artifact):
        chunks_received.append(chunk_artifact)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 250, 250)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()

    with patch.object(
        base_workpiece, "render_chunk", return_value=[(surface, (0, 0))]
    ):
        compute_workpiece_artifact_raster(
            workpiece=base_workpiece,
            opsproducer=opsproducer,
            laser=laser,
            settings=settings,
            generation_id=1,
            generation_size=base_workpiece.size,
            on_chunk=on_chunk_callback,
        )

    assert len(chunks_received) > 0
    for chunk in chunks_received:
        assert chunk.generation_size == base_workpiece.size
