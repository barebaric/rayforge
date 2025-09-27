import pytest
import cairo
from rayforge.core.ops import (
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    LineToCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.producer.depth import DepthEngraver, DepthMode
from rayforge.machine.models.machine import Laser
from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def producer() -> DepthEngraver:
    """Returns a default-initialized DepthEngraver instance."""
    return DepthEngraver()


@pytest.fixture
def laser() -> Laser:
    """Returns a default laser model."""
    return Laser()


@pytest.fixture
def white_surface() -> cairo.ImageSurface:
    """Returns a 10x10 pixel pure white Cairo surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)  # White
    ctx.paint()
    return surface


@pytest.fixture
def mock_workpiece() -> WorkPiece:
    """Returns a mock workpiece with a default size."""
    wp = WorkPiece(name="mock_wp")
    wp.uid = "wp_123"
    wp.set_size(10.0, 10.0)  # 10mm x 10mm
    return wp


def test_initialization_defaults(producer: DepthEngraver):
    """Verify the producer initializes with expected default values."""
    assert producer.depth_mode == DepthMode.POWER_MODULATION
    assert producer.scan_angle == 0.0
    assert producer.line_interval == 0.1
    assert producer.bidirectional is True
    assert producer.speed == 3000.0
    assert producer.min_power == 0.0
    assert producer.max_power == 100.0
    assert producer.num_depth_levels == 5


def test_is_vector_producer_is_false(producer: DepthEngraver):
    """The producer is for rastering, so it should not be scalable."""
    assert producer.is_vector_producer() is False


def test_serialization_and_deserialization():
    """
    Tests that the producer can be serialized to a dict and recreated,
    including correct enum handling.
    """
    # Arrange
    original_producer = DepthEngraver(
        scan_angle=45.0,
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=8,
        z_step_down=0.2,
    )

    # Act
    data = original_producer.to_dict()
    # Use the base class factory, which is the correct pattern
    recreated_producer = OpsProducer.from_dict(data)

    # Assert
    assert data["type"] == "DepthEngraver"
    params = data["params"]
    assert params["scan_angle"] == 45.0
    assert params["depth_mode"] == "MULTI_PASS"  # Stored as string name
    assert params["num_depth_levels"] == 8

    # Assert that we got the right type back
    assert isinstance(recreated_producer, DepthEngraver)
    assert recreated_producer.scan_angle == 45.0
    # Should be converted back to enum member
    assert recreated_producer.depth_mode == DepthMode.MULTI_PASS
    assert recreated_producer.num_depth_levels == 8
    assert recreated_producer.z_step_down == 0.2


def test_deserialization_with_invalid_enum_falls_back():
    """
    Tests that deserializing with an unknown enum value falls back to the
    default instead of crashing.
    """
    # Arrange
    data = {
        "type": "DepthEngraver",
        "params": {"depth_mode": "INVALID_MODE"},
    }

    # Act
    producer = OpsProducer.from_dict(data)

    # Assert
    assert isinstance(producer, DepthEngraver)
    assert producer.depth_mode == DepthMode.POWER_MODULATION


def test_run_requires_workpiece(
    producer: DepthEngraver,
    laser: Laser,
    white_surface: cairo.ImageSurface,
):
    """
    Verify that the run method raises an error if no workpiece is provided.
    """
    with pytest.raises(ValueError, match="requires a workpiece context"):
        producer.run(laser, white_surface, (10, 10))


def test_run_wraps_ops_in_section_markers(
    producer: DepthEngraver,
    laser: Laser,
    white_surface: cairo.ImageSurface,
    mock_workpiece: WorkPiece,
):
    """
    Even with an empty result from a white surface, the output should be
    correctly wrapped in start and end section commands.
    """
    # Arrange
    producer.min_power = (
        0  # Set min_power to 0 to trigger the "skip white line" optimization
    )

    # Act
    artifact = producer.run(
        laser, white_surface, (1.0, 1.0), workpiece=mock_workpiece
    )

    # Assert
    assert len(artifact.ops.commands) == 2
    start_cmd, end_cmd = artifact.ops.commands
    assert isinstance(start_cmd, OpsSectionStartCommand)
    assert start_cmd.section_type == SectionType.RASTER_FILL
    assert start_cmd.workpiece_uid == "wp_123"
    assert isinstance(end_cmd, OpsSectionEndCommand)
    assert end_cmd.section_type == SectionType.RASTER_FILL


def test_run_with_empty_surface_returns_empty_ops(
    producer: DepthEngraver, laser: Laser, mock_workpiece: WorkPiece
):
    """
    Test that a zero-dimension surface produces no errors and empty Ops.
    """
    empty_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    artifact = producer.run(
        laser, empty_surface, (1.0, 1.0), workpiece=mock_workpiece
    )
    # Should only contain the start/end markers
    assert len(artifact.ops.commands) == 2
    assert isinstance(artifact.ops.commands[0], OpsSectionStartCommand)
    assert isinstance(artifact.ops.commands[1], OpsSectionEndCommand)


def test_power_modulation_logic(laser: Laser, mock_workpiece: WorkPiece):
    """Tests the power modulation logic with solid color blocks."""
    # Arrange: Create a 10px wide surface with solid blocks.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)  # White
    ctx.rectangle(9, 0, 1, 1)
    ctx.fill()
    mock_workpiece.set_size(1.0, 0.1)  # 1mm wide, 0.1mm tall

    producer = DepthEngraver(
        depth_mode=DepthMode.POWER_MODULATION,
        min_power=10,
        max_power=90,
        line_interval=0.1,  # Corresponds to 1 pixel row
        overscan=0,
    )

    # Act
    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    # Assert
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) == 1
    power_vals = scan_cmds[0].power_values
    # Solid Black (start) should be exactly max_power
    assert power_vals[0] == 90
    # Solid White (end) should be exactly min_power
    assert power_vals[-1] == 10


def test_multi_pass_logic(laser: Laser, mock_workpiece: WorkPiece):
    """Tests the multi-pass logic with stepped gray values."""
    # Arrange: 3 bars: black, gray (127), white
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black -> pass_map value 4
    ctx.rectangle(0, 0, 1, 10)
    ctx.fill()
    ctx.set_source_rgb(0.5, 0.5, 0.5)  # Gray (127.5) -> pass_map value 2
    ctx.rectangle(1, 0, 1, 10)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)  # White -> pass_map value 0
    ctx.rectangle(2, 0, 1, 10)
    ctx.fill()
    mock_workpiece.set_size(0.3, 1.0)  # 0.3mm wide, 1mm tall

    producer = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=4,
        z_step_down=0.1,
        overscan=0,
        line_interval=0.1,  # One line per pixel row
    )

    # Act
    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    # Assert: Group commands by their Z-coordinate
    lines_by_z = {}
    for cmd in artifact.ops.commands:
        if isinstance(cmd, LineToCommand):
            z = round(cmd.end[2], 2)
            lines_by_z.setdefault(z, 0)
            lines_by_z[z] += 1

    # The rasterizer combines adjacent active pixels into single lines.
    # Pass 1 (z=0.0): black(4) and gray(2) are active and adjacent.
    # A single line per row is created. 10 lines total.
    # Pass 2 (z=-0.1): black(4) and gray(2) are still active. 10 lines.
    # Pass 3 (z=-0.2): Only black(4) is active. 10 lines.
    # Pass 4 (z=-0.3): Only black(4) is active. 10 lines.
    assert lines_by_z.get(0.0, 0) == 10
    assert lines_by_z.get(-0.1, 0) == 10
    assert lines_by_z.get(-0.2, 0) == 10
    assert lines_by_z.get(-0.3, 0) == 10
    # Check that there are no deeper passes
    assert -0.4 not in lines_by_z
