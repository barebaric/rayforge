import pytest
import cairo
import numpy as np
from rayforge.core.ops import (
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    LineToCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline.producer.base import (
    OpsProducer,
    HybridRasterArtifact,
    CoordinateSystem,
)
from rayforge.pipeline.producer.depth import DepthEngraver, DepthMode
from rayforge.machine.models.laser import Laser
from rayforge.core.workpiece import WorkPiece


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
    wp.set_size(10.0, 10.0)
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
    """The producer is for rastering, so it should use the chunked path."""
    assert producer.is_vector_producer() is False


def test_serialization_and_deserialization():
    """
    Tests that the producer can be serialized to a dict and recreated,
    including correct enum handling.
    """
    original = DepthEngraver(
        scan_angle=45.0,
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=8,
        z_step_down=0.2,
    )
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)

    assert isinstance(recreated, DepthEngraver)
    assert recreated.scan_angle == 45.0
    assert recreated.depth_mode == DepthMode.MULTI_PASS
    assert recreated.num_depth_levels == 8
    assert recreated.z_step_down == 0.2


def test_deserialization_with_invalid_enum_falls_back():
    """Tests that an unknown enum value falls back to the default."""
    data = {
        "type": "DepthEngraver",
        "params": {"depth_mode": "INVALID_MODE"},
    }
    producer = OpsProducer.from_dict(data)
    assert isinstance(producer, DepthEngraver)
    assert producer.depth_mode == DepthMode.POWER_MODULATION


def test_run_requires_workpiece(
    producer: DepthEngraver, laser: Laser, white_surface: cairo.ImageSurface
):
    """Verify run() raises an error if no workpiece is provided."""
    with pytest.raises(ValueError, match="requires a workpiece context"):
        producer.run(laser, white_surface, (10, 10))


def test_run_returns_hybrid_raster_artifact_with_correct_metadata(
    producer: DepthEngraver,
    laser: Laser,
    white_surface: cairo.ImageSurface,
    mock_workpiece: WorkPiece,
):
    """
    Test that run() returns a HybridRasterArtifact with valid structure
    and metadata.
    """
    artifact = producer.run(
        laser, white_surface, (1.0, 1.0), workpiece=mock_workpiece
    )

    assert isinstance(artifact, HybridRasterArtifact)
    assert artifact.is_scalable is False
    assert artifact.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
    assert artifact.source_dimensions == (10, 10)
    assert artifact.generation_size == (10.0, 10.0)
    assert artifact.dimensions_mm == (10.0, 10.0)
    assert artifact.position_mm == (0.0, 0.0)
    assert artifact.ops is not None
    assert artifact.power_texture_data is not None
    assert artifact.power_texture_data.shape == (10, 10)


def test_run_wraps_ops_in_section_markers(
    producer: DepthEngraver,
    laser: Laser,
    white_surface: cairo.ImageSurface,
    mock_workpiece: WorkPiece,
):
    """
    Even with an empty result, output should be wrapped in section commands.
    """
    artifact = producer.run(
        laser, white_surface, (1.0, 1.0), workpiece=mock_workpiece
    )

    # Assert
    cmds = list(artifact.ops)
    assert len(cmds) == 2
    start_cmd, end_cmd = cmds
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
    cmds = list(artifact.ops)
    assert len(cmds) == 2
    assert isinstance(cmds[0], OpsSectionStartCommand)
    assert isinstance(cmds[1], OpsSectionEndCommand)


def test_power_modulation_generates_correct_ops_and_texture(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that power modulation correctly generates both the ScanLineCommands
    and the power texture data from a grayscale image.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(128 / 255, 128 / 255, 128 / 255)
    ctx.rectangle(1, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(2, 0, 1, 1)
    ctx.fill()

    mock_workpiece.set_size(0.3, 0.1)  # 0.3mm wide, 0.1mm tall
    producer = DepthEngraver(min_power=0.1, max_power=0.9, line_interval=0.1)

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    # Add type check to satisfy Pylance
    assert isinstance(artifact, HybridRasterArtifact)

    # Expected values calculation:
    # final_byte = (min_frac + gray_factor * (max_frac-min_frac)) * 255
    # Black (gray_factor=1.0): (0.1 + 1.0 * 0.8) * 255 = 0.9 * 255 = 229.5
    # Gray (gray_factor~0.5): (0.1 + (1-128/255)*0.8) * 255 = 0.498 * 255 = 127
    # White (gray_factor=0.0): (0.1 + 0.0 * 0.8) * 255 = 0.1 * 255 = 25.5
    expected_texture_row = [229, 127, 26]

    # Assert Ops data
    scan_cmd = next(
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    )
    power_vals = scan_cmd.power_values
    assert len(power_vals) == 3
    assert power_vals[0] == pytest.approx(expected_texture_row[0], 1)
    assert power_vals[1] == pytest.approx(expected_texture_row[1], 1)
    assert power_vals[2] == pytest.approx(expected_texture_row[2], 1)

    # Assert Texture data
    assert artifact.power_texture_data.shape == (1, 3)
    texture_row = artifact.power_texture_data[0]
    assert texture_row[0] == pytest.approx(229, 1)
    assert texture_row[1] == pytest.approx(127, 1)
    assert texture_row[2] == pytest.approx(26, 1)


def test_multi_pass_generates_correct_ops_and_texture(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that multi-pass generates correct Z-stepped Ops AND the correct
    source power texture data.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 1, 10)
    ctx.fill()
    ctx.set_source_rgb(127 / 255, 127 / 255, 127 / 255)
    ctx.rectangle(1, 0, 1, 10)
    ctx.fill()

    mock_workpiece.set_size(0.2, 1.0)  # 0.2mm wide, 1mm tall
    px_per_mm = 10
    producer = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=4,
        z_step_down=0.1,
        line_interval=0.1,
    )

    artifact = producer.run(
        laser, surface, (px_per_mm, px_per_mm), workpiece=mock_workpiece
    )

    # Add type check to satisfy Pylance
    assert isinstance(artifact, HybridRasterArtifact)

    # -- Assert Ops Data (Z-stepping) --
    lines_by_z = {}
    for cmd in artifact.ops.commands:
        if isinstance(cmd, LineToCommand):
            z = round(cmd.end[2], 2)
            lines_by_z.setdefault(z, 0)
            lines_by_z[z] += 1

    assert 0.0 in lines_by_z and lines_by_z[0.0] > 0
    assert -0.1 in lines_by_z and lines_by_z[-0.1] > 0
    assert -0.2 in lines_by_z and lines_by_z[-0.2] > 0
    assert -0.3 in lines_by_z and lines_by_z[-0.3] > 0
    assert -0.4 not in lines_by_z

    # -- Assert Texture Data (Source depth map) --
    # The texture should represent the source image, but mapped to power
    # based on the number of passes.
    # Black (gray=0) -> 4 passes -> power = 4/4 = 1.0 -> 255
    # Gray (gray=127) -> ceil((1-127/255)*4) = ceil(2.007) = 3 passes
    #   -> power = 3/4 = 0.75 -> 191.25
    assert artifact.power_texture_data.shape == (10, 2)
    black_col = artifact.power_texture_data[:, 0]
    gray_col = artifact.power_texture_data[:, 1]
    assert np.all(np.isclose(black_col, 255))
    assert np.all(np.isclose(gray_col, 191, atol=1))
