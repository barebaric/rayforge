import pytest
import math

from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    ScanLinePowerCommand,
    SetPowerCommand,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    EnableAirAssistCommand,
)
from rayforge.pipeline.transformer.overscan import (
    OverscanTransformer,
    ExecutionPhase,
)


@pytest.fixture
def transformer() -> OverscanTransformer:
    """Provides a default OverscanTransformer instance."""
    return OverscanTransformer(enabled=True, distance_mm=5.0)


def test_initialization_and_properties():
    """Tests the constructor and property setters."""
    t = OverscanTransformer(enabled=True, distance_mm=2.5)
    assert t.enabled is True
    assert t.distance_mm == 2.5
    t.distance_mm = -10.0
    assert t.distance_mm == 0.0
    t.distance_mm = 7.0
    assert t.distance_mm == 7.0


def test_serialization_and_deserialization():
    """
    Tests that the transformer can be serialized to a dict and recreated.
    """
    original = OverscanTransformer(enabled=False, distance_mm=3.14)
    data = original.to_dict()
    recreated = OverscanTransformer.from_dict(data)
    assert data["name"] == "OverscanTransformer"
    assert data["enabled"] is False
    assert data["distance_mm"] == 3.14
    assert isinstance(recreated, OverscanTransformer)
    assert recreated.enabled is False
    assert recreated.distance_mm == 3.14


def test_no_op_when_disabled(transformer: OverscanTransformer):
    """Verify the run method does nothing if the transformer is disabled."""
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    # Keep a reference to the original list object
    original_commands_list = ops.commands

    transformer.enabled = False
    transformer.run(ops)

    # Assert that the list object itself was not replaced.
    assert ops.commands is original_commands_list


def test_no_op_with_zero_distance(transformer: OverscanTransformer):
    """Verify the run method does nothing if the distance is zero."""
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    # Keep a reference to the original list object
    original_commands_list = ops.commands

    transformer.distance_mm = 0.0
    transformer.run(ops)

    # Assert that the list object itself was not replaced.
    assert ops.commands is original_commands_list


def test_execution_phase_is_correct(transformer: OverscanTransformer):
    """Overscan must run before optimization."""
    assert transformer.execution_phase == ExecutionPhase.POST_PROCESSING


def test_run_with_constant_power_lines_from_rasterizer(
    transformer: OverscanTransformer,
):
    """
    Tests overscan on a simple constant-power line, typical of output
    from the Rasterizer producer.
    """
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 20, 5)  # Horizontal line, length 20mm, at z=5
    ops.line_to(30, 20, 5)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    transformer.run(ops)

    cmds = ops.commands
    # Expected: Start, [Move, SP(0), Line, SP(orig), Line, SP(0), Line], End
    assert len(cmds) == 9
    assert isinstance(cmds[1], MoveToCommand)
    assert cmds[1].end == pytest.approx((5.0, 20.0, 5.0))  # 10 - 5
    assert isinstance(cmds[5], LineToCommand)
    assert cmds[5].end == pytest.approx((30.0, 20.0, 5.0))  # Original end
    assert isinstance(cmds[7], LineToCommand)
    assert cmds[7].end == pytest.approx((35.0, 20.0, 5.0))  # 30 + 5


def test_preserves_state_for_constant_power_lines(
    transformer: OverscanTransformer,
):
    """
    Verify the overscan transformation for LineToCommands is precise,
    checking for correct power state management and geometry without
    relying on preload_state. This test includes an intermediate power
    change to validate handling of more complex sequences.
    """
    # Arrange: A sequence with two raster lines. The second line has a
    # SetPower command between its MoveTo and LineTo.
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(EnableAirAssistCommand())
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    # Line 1: Standard
    ops.move_to(10, 20, 0)
    ops.line_to(20, 20, 0)
    # Line 2: With intermediate state change
    ops.move_to(30, 20, 0)
    ops.add(SetPowerCommand(0.4))
    ops.line_to(40, 20, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))

    # Act
    transformer.run(ops)

    # Assert: Manually verify the exact output command sequence.
    cmds = ops.commands

    # --- Verification for Line 1 ---
    # Expected sequence: Move, SP(0), Line, SP(0.8), Line, SP(0), Line
    line_1_cmds = cmds[3:10]
    assert isinstance(line_1_cmds[0], MoveToCommand)
    assert line_1_cmds[0].end == pytest.approx((5.0, 20.0, 0.0))
    assert (
        isinstance(line_1_cmds[1], SetPowerCommand)
        and line_1_cmds[1].power == 0
    )
    assert isinstance(line_1_cmds[2], LineToCommand)
    assert line_1_cmds[2].end == pytest.approx((10.0, 20.0, 0.0))
    assert (
        isinstance(line_1_cmds[3], SetPowerCommand)
        and line_1_cmds[3].power == 0.8
    )
    assert isinstance(line_1_cmds[4], LineToCommand)
    assert line_1_cmds[4].end == pytest.approx((20.0, 20.0, 0.0))
    assert (
        isinstance(line_1_cmds[5], SetPowerCommand)
        and line_1_cmds[5].power == 0
    )
    assert isinstance(line_1_cmds[6], LineToCommand)
    assert line_1_cmds[6].end == pytest.approx((25.0, 20.0, 0.0))

    # --- Verification for Line 2 ---
    # The intermediate SetPower(0.4) must be preserved inside the
    # overscan wrap.
    # Expected sequence: Move, SP(0), Line, SP(0.4), Line, SP(0), Line
    line_2_cmds = cmds[10:17]
    assert isinstance(line_2_cmds[0], MoveToCommand)
    assert line_2_cmds[0].end == pytest.approx((25.0, 20.0, 0.0))
    assert (
        isinstance(line_2_cmds[1], SetPowerCommand)
        and line_2_cmds[1].power == 0
    )
    assert isinstance(line_2_cmds[2], LineToCommand)
    assert line_2_cmds[2].end == pytest.approx((30.0, 20.0, 0.0))
    # This is the critical check: the original intermediate SetPower
    # is preserved.
    # Note: Because the original buffer is extended, the power command is at
    # index 3, and the original LineTo is at index 4.
    assert (
        isinstance(line_2_cmds[3], SetPowerCommand)
        and line_2_cmds[3].power == 0.4
    )
    assert isinstance(line_2_cmds[4], LineToCommand)
    assert line_2_cmds[4].end == pytest.approx((40.0, 20.0, 0.0))
    assert (
        isinstance(line_2_cmds[5], SetPowerCommand)
        and line_2_cmds[5].power == 0
    )
    assert isinstance(line_2_cmds[6], LineToCommand)
    assert line_2_cmds[6].end == pytest.approx((45.0, 20.0, 0.0))

    # --- Final structure check ---
    # Total commands:
    # 2 (header) + 1 (start) + 7 (line 1) + 7 (line 2) + 1 (end) = 18
    assert len(cmds) == 18
    assert isinstance(cmds[0], SetPowerCommand) and cmds[0].power == 0.8
    assert isinstance(cmds[1], EnableAirAssistCommand)
    assert isinstance(cmds[2], OpsSectionStartCommand)
    assert isinstance(cmds[17], OpsSectionEndCommand)


def test_run_with_variable_power_scanlines_from_depth(
    transformer: OverscanTransformer,
):
    """
    Tests overscan on a variable-power scanline, typical of output from
    the DepthEngraver producer.
    """
    power_vals = bytearray(range(1, 41))
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 20, 0)
    ops.add(ScanLinePowerCommand(end=(30, 20, 0), power_values=power_vals))
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    transformer.run(ops)

    cmds = ops.commands
    assert len(cmds) == 4  # Start, Move, ScanLine, End
    move_cmd = cmds[1]
    scan_cmd = cmds[2]

    assert isinstance(move_cmd, MoveToCommand)
    assert move_cmd.end == pytest.approx((5.0, 20.0, 0.0))

    assert isinstance(scan_cmd, ScanLinePowerCommand)
    assert scan_cmd.end == pytest.approx((35.0, 20.0, 0.0))

    num_pad_pixels = 10  # 5mm distance * (40px / 20mm) = 10px
    pad_bytes = bytearray([0] * num_pad_pixels)
    expected_power = pad_bytes + power_vals + pad_bytes
    assert scan_cmd.power_values == expected_power


def test_preserves_state_for_scanline_commands(
    transformer: OverscanTransformer,
):
    """
    Verify the overscan transformation for ScanLinePowerCommands is precise
    and does not rely on preload_state. Checks for correct geometry extension
    and power value padding, while preserving preceding state commands.
    """
    # Arrange: A master power setting followed by a raster section with a
    # single ScanLine. This simulates a DepthEngraver output.
    ops = Ops()
    ops.add(SetPowerCommand(0.5))  # Master power setting
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 20, 0)
    ops.add(
        ScanLinePowerCommand(
            end=(20, 20, 0), power_values=bytearray([100, 200])
        )
    )
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))

    # The transformer should have a 5mm distance from the fixture
    assert transformer.distance_mm == 5.0

    # Act
    transformer.run(ops)

    # Assert: Manually verify the exact command sequence and their properties
    # without using preload_state, which could mask bugs.
    # Expected output structure:
    # [0] SetPower(0.5) - Preserved from before the section
    # [1] OpsSectionStart - Preserved
    # [2] MoveTo(5, 20, 0) - New overscan start point
    # [3] ScanLinePowerCommand - Modified with new geometry and padded power
    # [4] OpsSectionEnd - Preserved

    cmds = ops.commands
    assert len(cmds) == 5

    # 1. Check preserved master power command
    master_power_cmd = cmds[0]
    assert isinstance(master_power_cmd, SetPowerCommand)
    assert master_power_cmd.power == 0.5

    # 2. Check preserved section start
    assert isinstance(cmds[1], OpsSectionStartCommand)

    # 3. Check new overscan MoveTo command
    move_cmd = cmds[2]
    assert isinstance(move_cmd, MoveToCommand)
    assert move_cmd.end == pytest.approx(
        (5.0, 20.0, 0.0)
    )  # Original start (10) - 5mm

    # 4. Check modified ScanLinePowerCommand
    scan_cmd = cmds[3]
    assert isinstance(scan_cmd, ScanLinePowerCommand)
    assert scan_cmd.end == pytest.approx(
        (25.0, 20.0, 0.0)
    )  # Original end (20) + 5mm

    # Calculate expected padding.
    # Line length = 10mm. Power values length = 2.
    # Pixels per mm = 2 / 10 = 0.2
    # Pad pixels = round(5.0mm * 0.2px/mm) = round(1.0) = 1
    num_pad_pixels = 1
    pad_bytes = bytearray([0] * num_pad_pixels)
    expected_power_values = pad_bytes + bytearray([100, 200]) + pad_bytes
    assert scan_cmd.power_values == expected_power_values

    # 5. Check preserved section end
    assert isinstance(cmds[4], OpsSectionEndCommand)


def test_does_not_modify_commands_outside_raster_section(
    transformer: OverscanTransformer,
):
    """
    Ensures that only commands inside a RASTER_FILL section are modified.
    """
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(5, 5, 0)
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    original_vector_cmds = ops.commands[:2]

    transformer.run(ops)

    assert ops.commands[0] is original_vector_cmds[0]
    assert ops.commands[1] is original_vector_cmds[1]
    assert len(ops.commands) > 5


def test_handles_multiple_bidirectional_lines(
    transformer: OverscanTransformer,
):
    """
    Tests overscan on a typical bidirectional raster pattern.
    """
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 20, 0)
    ops.line_to(30, 20, 0)
    ops.move_to(30, 22, 0)
    ops.line_to(10, 22, 0)
    ops.move_to(5, 30, 0)
    ops.line_to(15, 40, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    dist = transformer.distance_mm

    transformer.run(ops)

    # Each line is rewritten from 2 moving commands to 4 + state changes
    # so we can't just count moving commands easily.
    # We check the final endpoints instead.
    all_cmds = ops.commands
    move_cmds = [c for c in all_cmds if isinstance(c, MoveToCommand)]
    line_cmds = [c for c in all_cmds if isinstance(c, LineToCommand)]

    # Expected moves: to start of overscan for each of the 3 lines
    assert len(move_cmds) == 3
    # Expected lines: 3 lead-in + 3 content + 3 lead-out = 9
    assert len(line_cmds) == 9

    # Check endpoints of the rewritten lines
    # Line 1
    assert move_cmds[0].end == pytest.approx((10 - dist, 20, 0))
    assert line_cmds[2].end == pytest.approx((30 + dist, 20, 0))
    # Line 2
    assert move_cmds[1].end == pytest.approx((30 + dist, 22, 0))
    assert line_cmds[5].end == pytest.approx((10 - dist, 22, 0))
    # Line 3 (diagonal)
    norm_v = 1 / math.sqrt(2)
    offset_x = offset_y = dist * norm_v
    assert move_cmds[2].end == pytest.approx((5 - offset_x, 30 - offset_y, 0))
    assert line_cmds[8].end == pytest.approx((15 + offset_x, 40 + offset_y, 0))


def test_handles_zero_length_line(transformer: OverscanTransformer):
    """
    Tests that a raster "line" that is just a point is not modified.
    """
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    original_cmds = ops.commands[:]

    transformer.run(ops)

    assert ops.commands == original_cmds
