import cairo
import pytest
from rayforge.pipeline.producer.rasterize import (
    rasterize_at_angle,
    Rasterizer,
)
from rayforge.core.ops import (
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    MoveToCommand,
    LineToCommand,
)
from unittest.mock import MagicMock
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def white_surface():
    """Creates a 10x10 white surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)  # White
    ctx.paint()
    return surface


@pytest.fixture
def black_surface():
    """Creates a 10x10 black surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black
    ctx.paint()
    return surface


@pytest.fixture
def checkerboard_surface():
    """Creates a 10x10 checkerboard surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                ctx.set_source_rgb(0, 0, 0)
            else:
                ctx.set_source_rgb(1, 1, 1)
            ctx.rectangle(i, j, 1, 1)
            ctx.fill()
    return surface


@pytest.fixture
def mock_workpiece() -> WorkPiece:
    """Returns a mock workpiece with a default size."""
    wp = WorkPiece(name="test_wp")
    wp.uid = "wp_test_123"
    wp.set_size(1.0, 1.0)  # Corresponds to 10x10 surface at 10ppm
    return wp


def test_rasterize_horizontally_white(white_surface):
    """
    Tests horizontal rasterization on a white surface should produce no ops.
    """
    ops = rasterize_at_angle(
        white_surface,
        ymax=1.0,
        pixels_per_mm=(10, 10),
        raster_size_mm=0.1,
        direction_degrees=0,
    )
    assert len(ops.commands) == 0


def test_rasterize_vertically_white(white_surface):
    """
    Tests vertical rasterization on a white surface should produce no ops.
    """
    ops = rasterize_at_angle(
        white_surface,
        ymax=1.0,
        pixels_per_mm=(10, 10),
        raster_size_mm=0.1,
        direction_degrees=90,
    )
    assert len(ops.commands) == 0


def test_rasterize_horizontally_black(black_surface):
    """Tests horizontal rasterization on a black surface."""
    ops = rasterize_at_angle(
        black_surface,
        ymax=1.0,
        pixels_per_mm=(10, 10),
        raster_size_mm=0.1,
        direction_degrees=0,
    )
    assert len(ops.commands) > 0
    # More specific assertions can be added here based on expected output


def test_rasterize_vertically_black(black_surface):
    """Tests vertical rasterization on a black surface."""
    ops = rasterize_at_angle(
        black_surface,
        ymax=1.0,
        pixels_per_mm=(10, 10),
        raster_size_mm=0.1,
        direction_degrees=90,
    )
    assert len(ops.commands) > 0
    # More specific assertions can be added here based on expected output


def test_rasterizer_serialization():
    """Tests serialization and deserialization of the Rasterizer producer."""
    original_producer = Rasterizer(cross_hatch=True, invert=True)
    data = original_producer.to_dict()
    recreated_producer = OpsProducer.from_dict(data)

    assert data["type"] == "Rasterizer"
    assert data["params"]["cross_hatch"] is True
    assert data["params"]["invert"] is True
    assert isinstance(recreated_producer, Rasterizer)
    assert recreated_producer.cross_hatch is True
    assert recreated_producer.invert is True


def test_rasterizer_run_requires_workpiece(white_surface):
    """
    Tests that the run method raises an error if no workpiece is provided.
    """
    laser = MagicMock()
    rasterizer = Rasterizer()
    with pytest.raises(ValueError, match="requires a workpiece context"):
        rasterizer.run(laser, white_surface, (10, 10), settings={})


def test_run_with_empty_surface_returns_empty_ops():
    """Test that a zero-dimension surface produces no errors and empty Ops."""
    empty_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    laser = MagicMock()
    rasterizer = Rasterizer()
    mock_workpiece = WorkPiece(name="wp_123")
    mock_workpiece.uid = "wp_123"
    mock_workpiece.set_size(0, 0)
    artifact = rasterizer.run(
        laser, empty_surface, (10, 10), workpiece=mock_workpiece, settings={}
    )
    assert len(artifact.ops.commands) == 2  # Start, End
    assert isinstance(artifact.ops.commands[0], OpsSectionStartCommand)
    assert isinstance(artifact.ops.commands[1], OpsSectionEndCommand)


def test_rasterizer_run_wraps_ops_in_section_markers(
    white_surface, mock_workpiece
):
    """
    Even with an empty result from a white surface, the output should be
    correctly wrapped.
    """
    laser = MagicMock()
    rasterizer = Rasterizer()
    artifact = rasterizer.run(
        laser, white_surface, (10, 10), workpiece=mock_workpiece, settings={}
    )
    assert len(artifact.ops.commands) == 2  # Start, End
    start_cmd, end_cmd = artifact.ops.commands[0], artifact.ops.commands[1]
    assert isinstance(start_cmd, OpsSectionStartCommand)
    assert start_cmd.section_type == SectionType.RASTER_FILL
    assert start_cmd.workpiece_uid == "wp_test_123"
    assert isinstance(end_cmd, OpsSectionEndCommand)
    assert end_cmd.section_type == SectionType.RASTER_FILL


def test_rasterizer_cross_hatch(black_surface, mock_workpiece):
    """Tests the Rasterizer class with cross-hatch enabled."""
    laser = MagicMock()
    laser.spot_size_mm = (0.1, 0.1)
    rasterizer = Rasterizer(cross_hatch=True)
    artifact = rasterizer.run(
        laser,
        black_surface,
        pixels_per_mm=(10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    horizontal_lines = 0
    vertical_lines = 0
    last_pos = None

    for cmd in artifact.ops.commands:
        if isinstance(cmd, MoveToCommand):
            last_pos = cmd.end
        elif isinstance(cmd, LineToCommand):
            if last_pos is not None:
                if last_pos[0] == cmd.end[0]:  # x is same, so vertical
                    vertical_lines += 1
                elif last_pos[1] == cmd.end[1]:  # y is same, so horizontal
                    horizontal_lines += 1
            last_pos = cmd.end

    assert horizontal_lines == 10
    assert vertical_lines == 10


def test_rasterizer_invert_white_surface(white_surface, mock_workpiece):
    """Tests that invert=True on a white surface produces raster ops."""
    laser = MagicMock()
    laser.spot_size_mm = (0.1, 0.1)
    rasterizer = Rasterizer(invert=True)
    artifact = rasterizer.run(
        laser, white_surface, (10, 10), workpiece=mock_workpiece, settings={}
    )

    horizontal_lines = 0
    for cmd in artifact.ops.commands:
        if isinstance(cmd, LineToCommand):
            horizontal_lines += 1

    assert horizontal_lines == 10


def test_rasterizer_invert_black_surface(black_surface, mock_workpiece):
    """Tests that invert=True on a black surface produces no ops."""
    laser = MagicMock()
    laser.spot_size_mm = (0.1, 0.1)
    rasterizer = Rasterizer(invert=True)
    artifact = rasterizer.run(
        laser, black_surface, (10, 10), workpiece=mock_workpiece, settings={}
    )

    horizontal_lines = 0
    for cmd in artifact.ops.commands:
        if isinstance(cmd, LineToCommand):
            horizontal_lines += 1

    assert horizontal_lines == 0


def test_rasterizer_invert_respects_alpha():
    """Tests that invert=True still respects alpha transparency."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.set_source_rgba(1, 1, 1, 0)
    ctx.rectangle(2, 2, 6, 6)
    ctx.fill()

    laser = MagicMock()
    laser.spot_size_mm = (0.1, 0.1)
    rasterizer = Rasterizer(invert=True)
    mock_workpiece = WorkPiece(name="test_wp")
    mock_workpiece.uid = "wp_test_123"
    mock_workpiece.set_size(1.0, 1.0)

    artifact = rasterizer.run(
        laser, surface, (10, 10), workpiece=mock_workpiece, settings={}
    )

    horizontal_lines = 0
    for cmd in artifact.ops.commands:
        if isinstance(cmd, LineToCommand):
            horizontal_lines += 1

    assert horizontal_lines > 0
