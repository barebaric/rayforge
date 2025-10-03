import pytest
import cairo
import numpy as np
from rayforge.core.ops import (
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    LineToCommand,
    ScanLinePowerCommand,
    MoveToCommand,
)
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.producer.depth import DepthEngraver, DepthMode
from rayforge.machine.models.laser import Laser
from rayforge.core.workpiece import WorkPiece
from rayforge.core.matrix import Matrix


@pytest.fixture
def producer() -> DepthEngraver:
    """Returns a default-initialized DepthEngraver instance."""
    return DepthEngraver()


@pytest.fixture
def laser() -> Laser:
    """Returns a default laser model."""
    laser_instance = Laser()
    laser_instance.max_power = 1000
    return laser_instance


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
    wp.matrix.scale(10.0, 10.0)  # 10mm x 10mm
    return wp


def test_initialization_defaults(producer: DepthEngraver):
    """Verify the producer initializes with expected default values."""
    assert producer.depth_mode == DepthMode.POWER_MODULATION
    assert producer.scan_angle == 0.0
    assert producer.line_interval == 0.1
    assert producer.bidirectional is True
    assert producer.speed == 3000.0
    assert producer.min_power == 0.0
    assert producer.max_power == 1.0
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
    producer.min_power = 0  # Skip white lines
    settings = {"power": 1000}

    # Act
    artifact = producer.run(
        laser,
        white_surface,
        (1.0, 1.0),
        workpiece=mock_workpiece,
        settings=settings,
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
    # Should only contain the start/SetLaser/end markers
    assert len(artifact.ops.commands) == 2
    assert isinstance(artifact.ops.commands[0], OpsSectionStartCommand)
    assert isinstance(artifact.ops.commands[1], OpsSectionEndCommand)


def test_power_modulation_with_gray_and_master_power(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that modulation correctly scales with the step's master power
    and handles intermediate gray values.
    """
    # Arrange: 3px surface: Black, 50% Gray (128), White
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(128 / 255, 128 / 255, 128 / 255)  # 50% Gray
    ctx.rectangle(1, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)  # White
    ctx.rectangle(2, 0, 1, 1)
    ctx.fill()

    mock_workpiece.matrix = Matrix()  # Reset matrix from fixture default
    mock_workpiece.matrix.scale(0.3, 0.1)  # 0.3mm wide, 0.1mm tall

    producer = DepthEngraver(min_power=0.1, max_power=0.9, line_interval=0.1)

    # Simulate step setting of 50% power (500 out of 1000)
    settings = {"power": 500}

    # Act
    artifact = producer.run(
        laser,
        surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings=settings,
    )

    # Assert
    scan_cmd = next(
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    )
    power_vals = scan_cmd.power_values

    # Expected values calculation:
    # The producer generates values in its configured min/max power range.
    # final_byte = (min_frac + gray_factor * (max_frac-min_frac)) * 255
    # Black (gray_factor=1.0): (0.1 + 1.0 * 0.8) * 255 = 0.9 * 255 = 229.5
    # Gray (gray_factor~0.5): (0.1 + (1-128/255)*0.8) * 255 = 0.498 * 255 = 127
    # White (gray_factor=0.0): (0.1 + 0.0 * 0.8) * 255 = 0.1 * 255 = 25.5
    assert len(power_vals) == 3
    assert power_vals[0] == pytest.approx(229, 1)
    assert power_vals[1] == pytest.approx(127, 1)
    assert power_vals[2] == pytest.approx(25, 1)


def test_multi_pass_logic_line_widths(laser: Laser, mock_workpiece: WorkPiece):
    """
    Tests that multi-pass creates lines of the correct width for each pass
    based on the depth map.
    """
    # Arrange: 2 bars: black, gray (127.5), on a 10px high surface
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black -> pass_map value 4
    ctx.rectangle(0, 0, 1, 10)
    ctx.fill()
    ctx.set_source_rgb(
        127 / 255, 127 / 255, 127 / 255
    )  # Gray -> pass_map value 3
    ctx.rectangle(1, 0, 1, 10)
    ctx.fill()

    mock_workpiece.matrix = Matrix()  # Reset matrix from fixture default
    mock_workpiece.matrix.scale(0.2, 1.0)  # 0.2mm wide, 1mm tall
    px_per_mm = 10

    producer = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=4,
        z_step_down=0.1,
        line_interval=0.1,  # One line per pixel row
    )

    # Act
    artifact = producer.run(
        laser, surface, (px_per_mm, px_per_mm), workpiece=mock_workpiece
    )

    # Assert: Group commands by their Z-coordinate and check line widths
    lines_by_z = {}
    all_commands = artifact.ops.commands
    for i, cmd in enumerate(all_commands):
        if isinstance(cmd, LineToCommand) and i > 0:
            prev_cmd = all_commands[i - 1]
            if isinstance(prev_cmd, MoveToCommand):
                z = round(cmd.end[2], 2)
                lines_by_z.setdefault(z, []).append((prev_cmd.end, cmd.end))

    # Pass 1, 2, 3 (z=0.0, -0.1, -0.2): Black and Gray are active. Line should
    # span 2 pixels. Expected width: 0.1mm
    wide_width = 0.1
    for z in [0.0, -0.1, -0.2]:
        assert len(lines_by_z[z]) == 10
        for start, end in lines_by_z[z]:
            width = abs(end[0] - start[0])
            assert np.isclose(width, wide_width)

    # Pass 4 (z=-0.3): Only Black is active. Line should span 1 pixel.
    # Expected width: 0.0mm
    narrow_width = 0.0
    z = -0.3
    assert len(lines_by_z[z]) == 10
    for start, end in lines_by_z[z]:
        width = abs(end[0] - start[0])
        assert np.isclose(width, narrow_width)

    # Check that there are no deeper passes
    assert -0.4 not in lines_by_z
