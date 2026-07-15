import math

import pytest
from post_processors.transformers import OverscanTransformer
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode
from raygeo.ops.types import CommandType, RasterMode, SectionType

from rayforge.pipeline.transformer.base import ExecutionPhase


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
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 10, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    original_len = ops.len()

    transformer.enabled = False
    transformer.run(ops)

    assert ops.len() == original_len


def test_no_op_with_native_overscan(transformer: OverscanTransformer):
    """Verify the transformer is skipped when the driver does overscan."""
    ops = Ops()
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 10, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    original_len = ops.len()

    transformer.run(ops, settings={"driver_native_overscan": True})

    assert ops.len() == original_len


def test_no_op_with_zero_distance(transformer: OverscanTransformer):
    """Verify the run method does nothing if the distance is zero."""
    ops = Ops()
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 10, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    original_len = ops.len()

    transformer.distance_mm = 0.0
    transformer.run(ops)

    assert ops.len() == original_len


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
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 20, 5)  # Horizontal line, length 20mm, at z=5
    ops.line_to(30, 20, 5)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    transformer.run(ops)

    # Expected: Start, [Move, SP(0), Line, SP(orig), Line, SP(0), Line], End
    assert ops.len() == 9
    assert ops.command_type(1) == CommandType.MOVE_TO
    assert ops.endpoint(1) == pytest.approx((5.0, 20.0, 5.0))  # 10 - 5
    assert ops.command_type(5) == CommandType.LINE_TO
    assert ops.endpoint(5) == pytest.approx((30.0, 20.0, 5.0))  # Original end
    assert ops.command_type(7) == CommandType.LINE_TO
    assert ops.endpoint(7) == pytest.approx((35.0, 20.0, 5.0))  # 30 + 5


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
    ops.set_power(0.8)
    ops.set_air_assist(AirAssistMode.ON)
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    # Line 1: Standard
    ops.move_to(10, 20, 0)
    ops.line_to(20, 20, 0)
    # Line 2: With intermediate state change
    ops.move_to(30, 20, 0)
    ops.set_power(0.4)
    ops.line_to(40, 20, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )

    # Act
    transformer.run(ops)

    # --- Verification for Line 1 ---
    # Expected sequence: Move, SP(0), Line, SP(0.8), Line, SP(0), Line
    assert ops.command_type(3) == CommandType.MOVE_TO
    assert ops.endpoint(3) == pytest.approx((5.0, 20.0, 0.0))
    assert ops.command_type(4) == CommandType.SET_POWER and ops.power(4) == 0
    assert ops.command_type(5) == CommandType.LINE_TO
    assert ops.endpoint(5) == pytest.approx((10.0, 20.0, 0.0))
    assert ops.command_type(6) == CommandType.SET_POWER and ops.power(6) == 0.8
    assert ops.command_type(7) == CommandType.LINE_TO
    assert ops.endpoint(7) == pytest.approx((20.0, 20.0, 0.0))
    assert ops.command_type(8) == CommandType.SET_POWER and ops.power(8) == 0
    assert ops.command_type(9) == CommandType.LINE_TO
    assert ops.endpoint(9) == pytest.approx((25.0, 20.0, 0.0))

    # --- Verification for Line 2 ---
    # The intermediate SetPower(0.4) must be preserved inside the
    # overscan wrap.
    # Expected sequence: Move, SP(0), Line, SP(0.4), Line, SP(0), Line
    assert ops.command_type(10) == CommandType.MOVE_TO
    assert ops.endpoint(10) == pytest.approx((25.0, 20.0, 0.0))
    assert ops.command_type(11) == CommandType.SET_POWER and ops.power(11) == 0
    assert ops.command_type(12) == CommandType.LINE_TO
    assert ops.endpoint(12) == pytest.approx((30.0, 20.0, 0.0))
    # This is the critical check: the original intermediate SetPower
    # is preserved.
    # Note: Because the original buffer is extended, the power command is at
    # index 3, and the original LineTo is at index 4.
    assert (
        ops.command_type(13) == CommandType.SET_POWER and ops.power(13) == 0.4
    )
    assert ops.command_type(14) == CommandType.LINE_TO
    assert ops.endpoint(14) == pytest.approx((40.0, 20.0, 0.0))
    assert ops.command_type(15) == CommandType.SET_POWER and ops.power(15) == 0
    assert ops.command_type(16) == CommandType.LINE_TO
    assert ops.endpoint(16) == pytest.approx((45.0, 20.0, 0.0))

    # --- Final structure check ---
    # Total commands:
    # 2 (header) + 1 (start) + 7 (line 1) + 7 (line 2) + 1 (end) = 18
    assert ops.len() == 18
    assert ops.command_type(0) == CommandType.SET_POWER and ops.power(0) == 0.8
    assert ops.command_type(1) == CommandType.SET_AIR_ASSIST
    assert ops.command_type(2) == CommandType.OPS_SECTION_START
    assert ops.command_type(17) == CommandType.OPS_SECTION_END


def test_run_with_variable_power_scanlines_from_depth(
    transformer: OverscanTransformer,
):
    """
    Tests overscan on a variable-power scanline, typical of output from
    the Rasterizer producer in POWER_MODULATION mode.
    """
    power_vals = bytearray(range(1, 41))
    ops = Ops()
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 20, 0)
    ops.scan_to(30, 20, 0, power_values=power_vals)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    transformer.run(ops)

    assert ops.len() == 4  # Start, Move, ScanLine, End

    assert ops.command_type(1) == CommandType.MOVE_TO
    assert ops.endpoint(1) == pytest.approx((5.0, 20.0, 0.0))

    assert ops.command_type(2) == CommandType.SCAN_LINE
    assert ops.endpoint(2) == pytest.approx((35.0, 20.0, 0.0))

    num_pad_pixels = 10  # 5mm distance * (40px / 20mm) = 10px
    pad_bytes = bytearray([0] * num_pad_pixels)
    expected_power = pad_bytes + power_vals + pad_bytes
    assert ops.scanline_data(2) == expected_power


def test_preserves_state_for_scanline_commands(
    transformer: OverscanTransformer,
):
    """
    Verify the overscan transformation for ScanLinePowerCommands is precise
    and does not rely on preload_state. Checks for correct geometry extension
    and power value padding, while preserving preceding state commands.
    """
    # Arrange: A master power setting followed by a raster section with a
    # single ScanLine. This simulates a Rasterizer output.
    ops = Ops()
    ops.set_power(0.5)  # Master power setting
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 20, 0)
    ops.scan_to(20, 20, 0, power_values=bytearray([100, 200]))
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )

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

    assert ops.len() == 5

    # 1. Check preserved master power command
    assert ops.command_type(0) == CommandType.SET_POWER
    assert ops.power(0) == 0.5

    # 2. Check preserved section start
    assert ops.command_type(1) == CommandType.OPS_SECTION_START

    # 3. Check new overscan MoveTo command
    assert ops.command_type(2) == CommandType.MOVE_TO
    assert ops.endpoint(2) == pytest.approx(
        (5.0, 20.0, 0.0)
    )  # Original start (10) - 5mm

    # 4. Check modified ScanLinePowerCommand
    assert ops.command_type(3) == CommandType.SCAN_LINE
    assert ops.endpoint(3) == pytest.approx(
        (25.0, 20.0, 0.0)
    )  # Original end (20) + 5mm

    # Calculate expected padding.
    # Line length = 10mm. Power values length = 2.
    # Pixels per mm = 2 / 10 = 0.2
    # Pad pixels = round(5.0mm * 0.2px/mm) = round(1.0) = 1
    num_pad_pixels = 1
    pad_bytes = bytearray([0] * num_pad_pixels)
    expected_power_values = pad_bytes + bytearray([100, 200]) + pad_bytes
    assert ops.scanline_data(3) == expected_power_values

    # 5. Check preserved section end
    assert ops.command_type(4) == CommandType.OPS_SECTION_END


def test_does_not_modify_commands_outside_raster_section(
    transformer: OverscanTransformer,
):
    """
    Ensures that only commands inside a RASTER_FILL section are modified.
    """
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(5, 5, 0)
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    original_ep0 = ops.endpoint(0)
    original_ep1 = ops.endpoint(1)

    transformer.run(ops)

    assert ops.endpoint(0) == original_ep0
    assert ops.endpoint(1) == original_ep1
    assert ops.len() > 5


def test_handles_multiple_bidirectional_lines(
    transformer: OverscanTransformer,
):
    """
    Tests overscan on a typical bidirectional raster pattern.
    """
    ops = Ops()
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 20, 0)
    ops.line_to(30, 20, 0)
    ops.move_to(30, 22, 0)
    ops.line_to(10, 22, 0)
    ops.move_to(5, 30, 0)
    ops.line_to(15, 40, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    dist = transformer.distance_mm

    transformer.run(ops)

    # Each line is rewritten from 2 moving commands to 4 + state changes
    # so we can't just count moving commands easily.
    # We check the final endpoints instead.
    move_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.MOVE_TO
    ]
    line_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.LINE_TO
    ]

    # Expected moves: to start of overscan for each of the 3 lines
    assert len(move_indices) == 3
    # Expected lines: 3 lead-in + 3 content + 3 lead-out = 9
    assert len(line_indices) == 9

    # Check endpoints of the rewritten lines
    # Line 1
    assert ops.endpoint(move_indices[0]) == pytest.approx((10 - dist, 20, 0))
    assert ops.endpoint(line_indices[2]) == pytest.approx((30 + dist, 20, 0))
    # Line 2
    assert ops.endpoint(move_indices[1]) == pytest.approx((30 + dist, 22, 0))
    assert ops.endpoint(line_indices[5]) == pytest.approx((10 - dist, 22, 0))
    # Line 3 (diagonal)
    norm_v = 1 / math.sqrt(2)
    offset_x = offset_y = dist * norm_v
    assert ops.endpoint(move_indices[2]) == pytest.approx(
        (5 - offset_x, 30 - offset_y, 0)
    )
    assert ops.endpoint(line_indices[8]) == pytest.approx(
        (15 + offset_x, 40 + offset_y, 0)
    )


def test_handles_zero_length_line(transformer: OverscanTransformer):
    """
    Tests that a raster "line" that is just a point is not modified.
    """
    ops = Ops()
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp_123",
        raster_mode=RasterMode.CONSTANT_POWER,
    )
    ops.move_to(10, 10, 0)
    ops.line_to(10, 10, 0)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER,
    )
    original_len = ops.len()

    transformer.run(ops)

    assert ops.len() == original_len
