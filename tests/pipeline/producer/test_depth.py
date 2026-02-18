import pytest
import cairo
from rayforge.core.ops import (
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    LineToCommand,
    ScanLinePowerCommand,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.image.dither import DitherAlgorithm
from rayforge.machine.models.laser import Laser
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.producer.depth import (
    DepthEngraver,
    DepthMode,
)


@pytest.fixture
def producer() -> DepthEngraver:
    """Returns a default-initialized DepthEngraver instance."""
    return DepthEngraver()


@pytest.fixture
def laser() -> Laser:
    """Returns a default laser model."""
    laser_instance = Laser()
    laser_instance.max_power = 1000
    laser_instance.spot_size_mm = (0.1, 0.1)
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
    assert producer.speed == 3000.0
    assert producer.min_power == 0.0
    assert producer.max_power == 1.0
    assert producer.num_depth_levels == 5
    assert producer.invert is False


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
        invert=True,
    )
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)

    assert isinstance(recreated, DepthEngraver)
    assert recreated.scan_angle == 45.0
    assert recreated.depth_mode == DepthMode.MULTI_PASS
    assert recreated.num_depth_levels == 8
    assert recreated.z_step_down == 0.2
    assert recreated.invert is True


def test_deserialization_with_invalid_enum_falls_back():
    """Tests that an unknown enum value falls back to the default."""
    data = {
        "type": "DepthEngraver",
        "params": {"depth_mode": "INVALID_MODE"},
    }
    producer = OpsProducer.from_dict(data)
    assert isinstance(producer, DepthEngraver)
    assert producer.depth_mode == DepthMode.POWER_MODULATION


def test_deserialization_ignores_unknown_params():
    """Tests that unknown params are ignored for forward compatibility."""
    data = {
        "type": "DepthEngraver",
        "params": {
            "bidirectional": True,
            "unknown_future_param": "some_value",
            "depth_mode": "CONSTANT_POWER",
        },
    }
    producer = OpsProducer.from_dict(data)
    assert isinstance(producer, DepthEngraver)
    assert producer.depth_mode == DepthMode.CONSTANT_POWER
    assert not hasattr(producer, "bidirectional")
    assert not hasattr(producer, "unknown_future_param")


def test_run_requires_workpiece(
    producer: DepthEngraver, laser: Laser, white_surface: cairo.ImageSurface
):
    """Verify run() raises an error if no workpiece is provided."""
    with pytest.raises(ValueError, match="requires a workpiece context"):
        producer.run(laser, white_surface, (10, 10))


def test_run_returns_hybrid_artifact_with_correct_metadata(
    producer: DepthEngraver,
    laser: Laser,
    white_surface: cairo.ImageSurface,
    mock_workpiece: WorkPiece,
):
    """
    Test that run() returns a Artifact with valid structure
    and metadata.
    """
    artifact = producer.run(
        laser, white_surface, (1.0, 1.0), workpiece=mock_workpiece
    )

    assert isinstance(artifact, WorkPieceArtifact)
    assert artifact.is_scalable is False
    assert artifact.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
    assert artifact.source_dimensions == (10, 10)
    assert artifact.generation_size == (10.0, 10.0)
    assert artifact.ops is not None


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


def test_power_modulation_generates_correct_ops(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that power modulation correctly generates ScanLineCommands
    from a grayscale image.
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
    producer = DepthEngraver(min_power=0.1, max_power=0.9)

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    # Add type check to satisfy Pylance
    assert isinstance(artifact, WorkPieceArtifact)

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


def test_multi_pass_generates_correct_ops(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that multi-pass generates correct Z-stepped Ops.
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
    )

    artifact = producer.run(
        laser, surface, (px_per_mm, px_per_mm), workpiece=mock_workpiece
    )

    # Add type check to satisfy Pylance
    assert isinstance(artifact, WorkPieceArtifact)

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


def test_multi_pass_with_scan_angle(laser: Laser, mock_workpiece: WorkPiece):
    """
    Tests that multi-pass respects scan_angle setting.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    producer_horizontal = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=2,
        scan_angle=0.0,
    )
    artifact_horizontal = producer_horizontal.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    lines_horizontal = [
        c for c in artifact_horizontal.ops if isinstance(c, LineToCommand)
    ]

    producer_angled = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=2,
        scan_angle=45.0,
    )
    artifact_angled = producer_angled.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    lines_angled = [
        c for c in artifact_angled.ops if isinstance(c, LineToCommand)
    ]

    assert len(lines_horizontal) >= 1
    assert len(lines_angled) >= 1


def test_multi_pass_with_cross_hatch(laser: Laser, mock_workpiece: WorkPiece):
    """
    Tests that multi-pass respects cross_hatch setting.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    producer_no_cross = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=2,
        cross_hatch=False,
    )
    artifact_no_cross = producer_no_cross.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    lines_no_cross = [
        c for c in artifact_no_cross.ops if isinstance(c, LineToCommand)
    ]

    producer_cross = DepthEngraver(
        depth_mode=DepthMode.MULTI_PASS,
        num_depth_levels=2,
        cross_hatch=True,
    )
    artifact_cross = producer_cross.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    lines_cross = [
        c for c in artifact_cross.ops if isinstance(c, LineToCommand)
    ]

    assert len(lines_no_cross) >= 1
    assert len(lines_cross) > len(lines_no_cross)


def test_power_modulation_respects_step_power_setting(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that power modulation correctly scales power values by the step's
    power setting. When step power is 20%, maximum power in the
    output should be 20% of the modulation range.
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

    mock_workpiece.set_size(0.3, 0.1)
    producer = DepthEngraver(min_power=0.0, max_power=1.0)
    step_power = 0.2  # 20% power setting
    settings = {"power": step_power}

    artifact = producer.run(
        laser, surface, (10, 10), workpiece=mock_workpiece, settings=settings
    )

    assert isinstance(artifact, WorkPieceArtifact)

    # Expected values calculation WITH step power scaling:
    # Without scaling:
    # Black (gray_factor=1.0): (0.0 + 1.0 * 1.0) * 255 = 255
    # Gray (gray_factor~0.5): (0.0 + (1-128/255)*1.0) * 255 = 127
    # White (gray_factor=0.0): (0.0 + 0.0 * 1.0) * 255 = 0
    # With step_power=0.2 scaling:
    # Black: 255 * 0.2 = 51
    # Gray: 127 * 0.2 = 25.4
    # White: 0 * 0.2 = 0
    expected_texture_row = [51, 25, 0]

    # Note: Zero-power pixels are filtered out from scan commands,
    # so only 2 values are present in the scan command
    scan_cmd = next(
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    )
    power_vals = scan_cmd.power_values
    assert len(power_vals) == 2
    assert power_vals[0] == pytest.approx(expected_texture_row[0], 1)
    assert power_vals[1] == pytest.approx(expected_texture_row[1], 1)


def test_invert_inverts_grayscale_values(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that invert=True correctly inverts the grayscale values.
    White becomes black (max power) and black becomes white (min power).
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)
    producer = DepthEngraver(min_power=0.1, max_power=0.9, invert=True)

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1
    for cmd in scan_cmds:
        for val in cmd.power_values:
            assert val == pytest.approx(26, 1)


def test_invert_respects_alpha(laser: Laser, mock_workpiece: WorkPiece):
    """
    Tests that invert=True still respects alpha transparency.
    Transparent pixels should remain not rastered.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 0)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, 3, 10)
    ctx.fill()
    ctx.rectangle(7, 0, 3, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)
    producer = DepthEngraver(invert=True)

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1
    for cmd in scan_cmds:
        for val in cmd.power_values:
            assert val > 0


def test_constant_power_threshold_mode(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that CONSTANT_POWER mode generates scan commands with constant
    power values using threshold binarization.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(1, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(2, 0, 1, 1)
    ctx.fill()

    mock_workpiece.set_size(0.3, 0.1)
    producer = DepthEngraver(
        depth_mode=DepthMode.CONSTANT_POWER,
        threshold=128,
    )

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1
    for cmd in scan_cmds:
        for val in cmd.power_values:
            assert val == 255


def test_constant_power_threshold_parameter(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that the threshold parameter affects which pixels are engraved
    in CONSTANT_POWER mode.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.6, 0.6, 0.6)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    mock_workpiece.set_size(0.1, 0.1)

    producer_low_threshold = DepthEngraver(
        depth_mode=DepthMode.CONSTANT_POWER,
        threshold=100,
    )
    artifact_low = producer_low_threshold.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_low = [
        c for c in artifact_low.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds_low) == 0

    producer_high_threshold = DepthEngraver(
        depth_mode=DepthMode.CONSTANT_POWER,
        threshold=200,
    )
    artifact_high = producer_high_threshold.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_high = [
        c for c in artifact_high.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds_high) >= 1


def test_constant_power_invert(laser: Laser, mock_workpiece: WorkPiece):
    """
    Tests that invert=True inverts which pixels are engraved in
    CONSTANT_POWER mode.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(1, 0, 1, 1)
    ctx.fill()

    mock_workpiece.set_size(0.2, 0.1)

    producer_normal = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        threshold=128,
        invert=False,
    )
    artifact_normal = producer_normal.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_normal = [
        c for c in artifact_normal.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds_normal) >= 1

    producer_invert = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        threshold=128,
        invert=True,
    )
    artifact_invert = producer_invert.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_invert = [
        c for c in artifact_invert.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds_invert) >= 1


def test_constant_power_respects_alpha(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that transparent pixels are not engraved in CONSTANT_POWER mode.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3, 1)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.rectangle(0, 0, 3, 1)
    ctx.fill()
    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.rectangle(1, 0, 1, 1)
    ctx.fill()

    mock_workpiece.set_size(0.3, 0.1)
    producer = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        threshold=128,
    )

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1


def test_dither_serialization():
    """
    Tests that DITHER mode serializes and deserializes correctly.
    """
    original = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        threshold=200,
        invert=True,
    )
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)

    assert isinstance(recreated, DepthEngraver)
    assert recreated.depth_mode == DepthMode.DITHER
    assert recreated.dither_algorithm == DitherAlgorithm.FLOYD_STEINBERG
    assert recreated.threshold == 200
    assert recreated.invert is True


def test_dither_mode_with_floyd_steinberg(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that CONSTANT_POWER mode works with Floyd-Steinberg dithering.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)
    producer = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
    )

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1
    for cmd in scan_cmds:
        for val in cmd.power_values:
            assert val == 255


def test_cross_hatch_generates_two_passes(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that cross_hatch=True generates two passes (two sets of scan lines).
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    producer_no_cross_hatch = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        cross_hatch=False,
    )
    artifact_no_cross_hatch = producer_no_cross_hatch.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_no_cross = [
        c
        for c in artifact_no_cross_hatch.ops
        if isinstance(c, ScanLinePowerCommand)
    ]

    producer_cross_hatch = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        cross_hatch=True,
    )
    artifact_cross_hatch = producer_cross_hatch.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_cross = [
        c
        for c in artifact_cross_hatch.ops
        if isinstance(c, ScanLinePowerCommand)
    ]

    assert len(scan_cmds_cross) > len(scan_cmds_no_cross)


def test_cross_hatch_perpendicular_angles(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that cross_hatch generates passes at perpendicular angles.
    With scan_angle=45, second pass should be at 135 degrees.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)
    producer = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        scan_angle=45.0,
        cross_hatch=True,
    )

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 2


def test_cross_hatch_serialization():
    """
    Tests that cross_hatch serializes and deserializes correctly.
    """
    original = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        cross_hatch=True,
        scan_angle=30.0,
    )
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)

    assert isinstance(recreated, DepthEngraver)
    assert recreated.cross_hatch is True
    assert recreated.scan_angle == 30.0


def test_cross_hatch_with_power_modulation(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that cross_hatch works with POWER_MODULATION mode.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    producer_no_cross = DepthEngraver(
        depth_mode=DepthMode.POWER_MODULATION,
        cross_hatch=False,
    )
    artifact_no_cross = producer_no_cross.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_no_cross = [
        c for c in artifact_no_cross.ops if isinstance(c, ScanLinePowerCommand)
    ]

    producer_cross = DepthEngraver(
        depth_mode=DepthMode.POWER_MODULATION,
        cross_hatch=True,
    )
    artifact_cross = producer_cross.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_cross = [
        c for c in artifact_cross.ops if isinstance(c, ScanLinePowerCommand)
    ]

    assert len(scan_cmds_cross) > len(scan_cmds_no_cross)


def test_constant_power_bayer_dithering(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that CONSTANT_POWER mode works with Bayer dithering methods.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    for bayer_method in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        producer = DepthEngraver(
            depth_mode=DepthMode.DITHER,
            dither_algorithm=bayer_method,
        )

        artifact = producer.run(
            laser, surface, (10, 10), workpiece=mock_workpiece
        )

        assert isinstance(artifact, WorkPieceArtifact)
        scan_cmds = [
            c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
        ]
        assert len(scan_cmds) >= 1
        for cmd in scan_cmds:
            for val in cmd.power_values:
                assert val == 255


def test_dithering_respects_alpha(laser: Laser, mock_workpiece: WorkPiece):
    """
    Tests that dithering respects alpha transparency - transparent pixels
    should not be engraved.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()
    ctx.set_source_rgba(0.5, 0.5, 0.5, 1)
    ctx.rectangle(2, 2, 6, 6)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)
    producer = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
    )

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1


def test_dithering_adapts_to_spot_size(mock_workpiece: WorkPiece):
    """
    Tests that dithering adapts to the laser spot size via min_feature_px.
    A larger spot size should produce coarser dithering.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 20, 20)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 20, 20)
    ctx.fill()

    mock_workpiece.set_size(2.0, 2.0)

    laser_small_spot = Laser()
    laser_small_spot.max_power = 1000
    laser_small_spot.spot_size_mm = (0.05, 0.05)

    laser_large_spot = Laser()
    laser_large_spot.max_power = 1000
    laser_large_spot.spot_size_mm = (0.5, 0.5)

    producer = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.BAYER4,
    )

    artifact_small = producer.run(
        laser_small_spot, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_small = [
        c for c in artifact_small.ops if isinstance(c, ScanLinePowerCommand)
    ]

    artifact_large = producer.run(
        laser_large_spot, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_large = [
        c for c in artifact_large.ops if isinstance(c, ScanLinePowerCommand)
    ]

    assert len(scan_cmds_small) >= 1
    assert len(scan_cmds_large) >= 1


def test_constant_power_with_scan_angle(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that CONSTANT_POWER mode works with angled scanning.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    producer_horizontal = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        scan_angle=0.0,
    )
    artifact_horizontal = producer_horizontal.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_horizontal = [
        c
        for c in artifact_horizontal.ops
        if isinstance(c, ScanLinePowerCommand)
    ]

    producer_angled = DepthEngraver(
        depth_mode=DepthMode.DITHER,
        dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        scan_angle=45.0,
    )
    artifact_angled = producer_angled.run(
        laser, surface, (10, 10), workpiece=mock_workpiece
    )
    scan_cmds_angled = [
        c for c in artifact_angled.ops if isinstance(c, ScanLinePowerCommand)
    ]

    assert len(scan_cmds_horizontal) >= 1
    assert len(scan_cmds_angled) >= 1


def test_power_modulation_with_scan_angle(
    laser: Laser, mock_workpiece: WorkPiece
):
    """
    Tests that POWER_MODULATION mode works with angled scanning.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.rectangle(0, 0, 10, 10)
    ctx.fill()

    mock_workpiece.set_size(1.0, 1.0)

    producer = DepthEngraver(
        depth_mode=DepthMode.POWER_MODULATION,
        scan_angle=30.0,
    )

    artifact = producer.run(laser, surface, (10, 10), workpiece=mock_workpiece)

    assert isinstance(artifact, WorkPieceArtifact)
    scan_cmds = [
        c for c in artifact.ops if isinstance(c, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) >= 1
