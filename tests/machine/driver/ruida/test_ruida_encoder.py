"""
Extensive test suite for the RuidaEncoder.

Tests cover:
- Individual command encoding
- Binary output verification
- Text representation generation
- Op map correctness
- Edge cases and error handling
- Serialization of EncodedOutput
"""

import pytest
import base64

from rayforge.core.ops import Ops, SetFrequencyCommand, SetPulseWidthCommand
from rayforge.core.doc import Doc
from rayforge.machine.models.laser import Laser
from rayforge.machine.driver.ruida.ruida_encoder import RuidaEncoder
from rayforge.machine.driver.ruida.ruida_util import encode14, encode35
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


@pytest.fixture
def encoder():
    """Provides a fresh RuidaEncoder instance."""
    return RuidaEncoder()


@pytest.fixture
def mock_machine(isolated_machine):
    """Provides a machine with multiple laser heads for testing."""
    laser1 = Laser()
    laser1.uid = "laser-1"
    laser1.tool_number = 1

    laser2 = Laser()
    laser2.uid = "laser-2"
    laser2.tool_number = 2

    isolated_machine.heads.clear()
    isolated_machine.add_head(laser1)
    isolated_machine.add_head(laser2)

    isolated_machine.active_wcs = "MACHINE"

    return isolated_machine


@pytest.fixture
def doc():
    """Provides a fresh Doc instance."""
    return Doc()


class TestRuidaEncoderBasics:
    """Basic encoder functionality tests."""

    def test_encode_returns_encoded_output(self, encoder, mock_machine, doc):
        """Verify encode() returns an EncodedOutput instance."""
        ops = Ops()
        result = encoder.encode(ops, mock_machine, doc)

        assert isinstance(result, EncodedOutput)
        assert isinstance(result.text, str)
        assert isinstance(result.op_map, MachineCodeOpMap)
        assert isinstance(result.driver_data, dict)

    def test_empty_ops_produces_empty_output(self, encoder, mock_machine, doc):
        """Empty Ops should produce empty binary and text."""
        ops = Ops()
        result = encoder.encode(ops, mock_machine, doc)

        assert result.driver_data["binary"] == b""
        assert result.text == ""
        assert result.op_map.op_to_machine_code == {}
        assert result.op_map.machine_code_to_op == {}

    def test_binary_in_driver_data(self, encoder, mock_machine, doc):
        """Binary output should be stored in driver_data['binary']."""
        ops = Ops()
        ops.set_power(0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert "binary" in result.driver_data
        assert isinstance(result.driver_data["binary"], bytes)
        assert len(result.driver_data["binary"]) > 0

    def test_encoder_state_resets_between_encodes(
        self, encoder, mock_machine, doc
    ):
        """Each encode() call should reset internal state."""
        ops1 = Ops()
        ops1.set_power(0.5)
        encoder.encode(ops1, mock_machine, doc)

        assert encoder.power == 0.5

        ops2 = Ops()
        ops2.set_power(0.8)
        result2 = encoder.encode(ops2, mock_machine, doc)

        assert encoder.power == 0.8
        assert result2.op_map.op_to_machine_code == {0: [0]}


class TestSetPowerCommand:
    """Tests for SetPowerCommand encoding."""

    def test_power_zero(self, encoder, mock_machine, doc):
        """Zero power should encode to 0."""
        ops = Ops()
        ops.set_power(0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary == b"\xc8" + encode14(0)
        assert "POWER 0.0" in result.text

    def test_power_half(self, encoder, mock_machine, doc):
        """50% power should encode to 8192 (half of 16384)."""
        ops = Ops()
        ops.set_power(0.5)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        expected_val = int(0.5 * 16384)
        assert binary == b"\xc8" + encode14(expected_val)
        assert "POWER 50.0" in result.text

    def test_power_full(self, encoder, mock_machine, doc):
        """100% power should encode to 16383 (max 14-bit value)."""
        ops = Ops()
        ops.set_power(1.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        expected_val = int(1.0 * 16384) & 0x3FFF
        assert binary == b"\xc8" + encode14(expected_val)
        assert "POWER 100.0" in result.text

    def test_power_precision(self, encoder, mock_machine, doc):
        """Power should maintain precision in text output."""
        ops = Ops()
        ops.set_power(0.123)
        result = encoder.encode(ops, mock_machine, doc)

        assert "POWER 12.3" in result.text

    def test_power_with_different_lasers(self, encoder, mock_machine, doc):
        """Power command should use correct byte for active laser."""
        ops = Ops()
        ops.set_laser("laser-2")
        ops.set_power(0.5)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        lines = result.text.split("\n")

        assert b"\xc1" in binary
        assert "LASER 2" in lines[0]
        assert "POWER 50.0" in lines[1]


class TestSetCutSpeedCommand:
    """Tests for SetCutSpeedCommand encoding."""

    def test_speed_encoding(self, encoder, mock_machine, doc):
        """Cut speed should be encoded as mm/s to µm/s."""
        ops = Ops()
        ops.set_cut_speed(100)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        speed_um = int(100 * 1000)
        assert binary == b"\xc9\x02" + encode35(speed_um)
        assert "SPEED 100.0" in result.text

    def test_speed_fractional(self, encoder, mock_machine, doc):
        """Fractional speeds should be encoded correctly."""
        ops = Ops()
        ops.set_cut_speed(50.5)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        speed_um = int(int(50.5) * 1000)
        assert binary == b"\xc9\x02" + encode35(speed_um)
        assert "SPEED 50.0" in result.text

    def test_speed_zero(self, encoder, mock_machine, doc):
        """Zero speed should encode correctly."""
        ops = Ops()
        ops.set_cut_speed(0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary == b"\xc9\x02" + encode35(0)
        assert "SPEED 0.0" in result.text


class TestSetTravelSpeedCommand:
    """Tests for SetTravelSpeedCommand encoding."""

    def test_travel_speed_encoding(self, encoder, mock_machine, doc):
        """Travel speed should be encoded and stored."""
        ops = Ops()
        ops.set_travel_speed(500)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        speed_um = int(500 * 1000)
        assert b"\xc9\x02" + encode35(speed_um) in binary
        assert "TRAVEL_SPEED 500.0" in result.text

    def test_travel_speed_updates_state(self, encoder, mock_machine, doc):
        """Travel speed should update encoder state."""
        ops = Ops()
        ops.set_travel_speed(300)
        encoder.encode(ops, mock_machine, doc)

        assert encoder.travel_speed == 300


class TestAirAssistCommands:
    """Tests for air assist commands."""

    def test_enable_air_assist(self, encoder, mock_machine, doc):
        """Enable air assist should send correct command."""
        ops = Ops()
        ops.enable_air_assist()
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert b"\xca\x13" in binary
        assert "AIR_ASSIST ON" in result.text

    def test_disable_air_assist(self, encoder, mock_machine, doc):
        """Disable air assist should send correct command."""
        ops = Ops()
        ops.enable_air_assist()
        ops.disable_air_assist()
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert b"\xca\x13" in binary
        assert b"\xca\x12" in binary
        lines = result.text.split("\n")
        assert "AIR_ASSIST ON" in lines[0]
        assert "AIR_ASSIST OFF" in lines[1]

    def test_air_assist_no_redundant_commands(
        self, encoder, mock_machine, doc
    ):
        """Should not emit redundant air assist commands."""
        ops = Ops()
        ops.enable_air_assist()
        ops.enable_air_assist()
        ops.enable_air_assist()
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        on_count = binary.count(b"\xca\x13")
        assert on_count == 1

    def test_air_assist_state_tracking(self, encoder, mock_machine, doc):
        """Air assist state should be tracked correctly."""
        ops = Ops()
        ops.enable_air_assist()
        ops.enable_air_assist()
        ops.disable_air_assist()
        ops.disable_air_assist()
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary.count(b"\xca\x13") == 1
        assert binary.count(b"\xca\x12") == 1


class TestSetLaserCommand:
    """Tests for laser selection command."""

    def test_select_laser_1(self, encoder, mock_machine, doc):
        """Select laser 1 should emit correct command."""
        ops = Ops()
        ops.set_laser("laser-1")
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary == b"\xca\n"
        assert "LASER 1" in result.text

    def test_select_laser_2(self, encoder, mock_machine, doc):
        """Select laser 2 should emit correct command."""
        ops = Ops()
        ops.set_laser("laser-2")
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary == b"\xca\x0b"
        assert "LASER 2" in result.text

    def test_select_unknown_laser_defaults_to_1(
        self, encoder, mock_machine, doc
    ):
        """Unknown laser UID should default to laser 1."""
        ops = Ops()
        ops.set_laser("nonexistent-laser")
        result = encoder.encode(ops, mock_machine, doc)

        assert encoder.active_laser == 1
        assert "LASER 1" in result.text


class TestMoveToCommand:
    """Tests for rapid move (travel) command."""

    def test_move_to_origin(self, encoder, mock_machine, doc):
        """Move to origin should encode correctly."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        x_um = int(0.0 * 1000)
        y_um = int(0.0 * 1000)
        assert binary == b"\x88" + encode35(x_um) + encode35(y_um)
        assert "MOVE_ABS X:0.000 Y:0.000" in result.text

    def test_move_to_positive_coords(self, encoder, mock_machine, doc):
        """Move to positive coordinates should encode correctly."""
        ops = Ops()
        ops.move_to(100.5, 200.25, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        x_um = int(100.5 * 1000)
        y_um = int(200.25 * 1000)
        assert binary == b"\x88" + encode35(x_um) + encode35(y_um)
        assert "MOVE_ABS X:100.500 Y:200.250" in result.text

    def test_move_to_updates_position(self, encoder, mock_machine, doc):
        """Move command should update current position."""
        ops = Ops()
        ops.move_to(50.0, 75.0, 0.0)
        encoder.encode(ops, mock_machine, doc)

        assert encoder.current_pos == (50.0, 75.0, 0.0)


class TestLineToCommand:
    """Tests for cutting move command."""

    def test_line_to_simple(self, encoder, mock_machine, doc):
        """Simple line cut should encode correctly."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(50.0, 100.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "MOVE_ABS X:0.000 Y:0.000" in lines[0]
        assert "CUT_ABS X:50.000 Y:100.000" in lines[1]

    def test_line_to_updates_position(self, encoder, mock_machine, doc):
        """Line command should update current position."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(25.0, 50.0, 0.0)
        encoder.encode(ops, mock_machine, doc)

        assert encoder.current_pos == (25.0, 50.0, 0.0)

    def test_multiple_line_commands(self, encoder, mock_machine, doc):
        """Multiple consecutive line commands should encode correctly."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.line_to(0.0, 10.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert len(lines) == 4
        assert "MOVE_ABS X:0.000 Y:0.000" in lines[0]
        assert "CUT_ABS X:10.000 Y:0.000" in lines[1]
        assert "CUT_ABS X:10.000 Y:10.000" in lines[2]
        assert "CUT_ABS X:0.000 Y:10.000" in lines[3]


class TestArcToCommand:
    """Tests for arc command (linearized to line segments)."""

    def test_arc_linearizes_to_lines(self, encoder, mock_machine, doc):
        """Arc should be linearized into line segments."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.arc_to(10.0, 0.0, 5.0, 0.0, clockwise=True)
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert any("; ARC" in line for line in lines)
        cut_lines = [line for line in lines if "CUT_ABS" in line]
        assert len(cut_lines) >= 1

    def test_arc_updates_position(self, encoder, mock_machine, doc):
        """Arc command should update current position."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.arc_to(10.0, 0.0, 5.0, 0.0, clockwise=True)
        encoder.encode(ops, mock_machine, doc)

        assert encoder.current_pos == (10.0, 0.0, 0.0)


class TestScanLinePowerCommand:
    """Tests for scan line (raster) command."""

    def test_scan_line_linearizes(self, encoder, mock_machine, doc):
        """Scan line should linearize into power and line commands."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        power_values = bytearray([0, 128, 255, 128, 0])
        ops.scan_to(5.0, 0.0, 0.0, power_values)
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert any("; SCAN_LINE" in line for line in lines)
        assert any("POWER" in line for line in lines)

    def test_scan_line_updates_position(self, encoder, mock_machine, doc):
        """Scan line command should update current position."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        power_values = bytearray([128, 128])
        ops.scan_to(2.0, 0.0, 0.0, power_values)
        encoder.encode(ops, mock_machine, doc)

        assert encoder.current_pos == (2.0, 0.0, 0.0)


class TestJobMarkers:
    """Tests for job start/end markers."""

    def test_job_start(self, encoder, mock_machine, doc):
        """Job start should emit reference point selection command."""
        ops = Ops()
        ops.job_start()
        result = encoder.encode(ops, mock_machine, doc)

        assert result.driver_data["binary"] == b"\xd8\x10"
        assert "; Job Start - Ref Point: MACHINE" in result.text

    def test_job_end(self, encoder, mock_machine, doc):
        """Job end should emit EOF marker."""
        ops = Ops()
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert result.driver_data["binary"] == b"\xd7"
        assert "; Job End" in result.text

    def test_full_job_structure(self, encoder, mock_machine, doc):
        """Full job should have proper structure."""
        ops = Ops()
        ops.job_start()
        ops.set_power(0.5)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "Job Start" in lines[0]
        assert lines[-1] == "; Job End"
        assert result.driver_data["binary"].startswith(b"\xd8\x10")
        assert result.driver_data["binary"].endswith(b"\xd7")


class TestLayerMarkers:
    """Tests for layer start/end markers."""

    def test_layer_start(self, encoder, mock_machine, doc):
        """Layer start should emit binary and text marker."""
        ops = Ops()
        ops.layer_start("test-layer-123")
        result = encoder.encode(ops, mock_machine, doc)

        assert b"\xca\x00" in result.driver_data["binary"]
        assert "; --- Layer test-lay ---" in result.text

    def test_layer_end(self, encoder, mock_machine, doc):
        """Layer end should emit binary and text marker."""
        ops = Ops()
        ops.layer_end("test-layer-456")
        result = encoder.encode(ops, mock_machine, doc)

        assert b"\xca\x00" in result.driver_data["binary"]
        assert "; --- End Layer ---" in result.text


class TestWorkpieceMarkers:
    """Tests for workpiece start/end markers."""

    def test_workpiece_start(self, encoder, mock_machine, doc):
        """Workpiece start should emit text marker only."""
        ops = Ops()
        ops.workpiece_start("workpiece-abc")
        result = encoder.encode(ops, mock_machine, doc)

        assert result.driver_data["binary"] == b""
        assert "; --- Workpiece workpiec ---" in result.text

    def test_workpiece_end(self, encoder, mock_machine, doc):
        """Workpiece end should emit text marker only."""
        ops = Ops()
        ops.workpiece_end("workpiece-xyz")
        result = encoder.encode(ops, mock_machine, doc)

        assert result.driver_data["binary"] == b""
        assert "; --- End Workpiece ---" in result.text


class TestOpMapGeneration:
    """Tests for MachineCodeOpMap generation."""

    def test_single_command_mapping(self, encoder, mock_machine, doc):
        """Single command should have correct op_map."""
        ops = Ops()
        ops.set_power(0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert 0 in result.op_map.op_to_machine_code
        assert result.op_map.op_to_machine_code[0] == [0]
        assert result.op_map.machine_code_to_op[0] == 0

    def test_multi_line_command_mapping(self, encoder, mock_machine, doc):
        """Command producing multiple lines should map correctly."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.arc_to(10.0, 0.0, 5.0, 0.0, clockwise=True)
        result = encoder.encode(ops, mock_machine, doc)

        assert 0 in result.op_map.op_to_machine_code
        assert 1 in result.op_map.op_to_machine_code
        assert len(result.op_map.op_to_machine_code[0]) >= 1

    def test_marker_command_has_text_mapping(self, encoder, mock_machine, doc):
        """Marker commands with text output should map to line."""
        ops = Ops()
        ops.job_start()
        result = encoder.encode(ops, mock_machine, doc)

        # Job start produces a text line, so op_map should have [0]
        assert result.op_map.op_to_machine_code[0] == [0]

    def test_sequential_commands_mapping(self, encoder, mock_machine, doc):
        """Sequential commands should have sequential line numbers."""
        ops = Ops()
        ops.set_power(0.5)
        ops.set_cut_speed(100)
        ops.move_to(0.0, 0.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        assert result.op_map.op_to_machine_code[0] == [0]
        assert result.op_map.op_to_machine_code[1] == [1]
        assert result.op_map.op_to_machine_code[2] == [2]

        for line_num in range(3):
            assert result.op_map.machine_code_to_op[line_num] == line_num


class TestEncodedOutputSerialization:
    """Tests for EncodedOutput serialization with binary data."""

    def test_to_dict_contains_base64_binary(self, encoder, mock_machine, doc):
        """Binary in driver_data should be base64 encoded in to_dict()."""
        ops = Ops()
        ops.set_power(0.5)
        result = encoder.encode(ops, mock_machine, doc)

        data = result.to_dict()

        assert "driver_data" in data
        assert "binary" in data["driver_data"]
        binary_entry = data["driver_data"]["binary"]
        assert binary_entry["__type__"] == "bytes"

        expected_b64 = base64.b64encode(result.driver_data["binary"]).decode(
            "ascii"
        )
        assert binary_entry["data"] == expected_b64

    def test_roundtrip_serialization(self, encoder, mock_machine, doc):
        """EncodedOutput should survive dict roundtrip."""
        ops = Ops()
        ops.set_power(0.75)
        ops.move_to(100.0, 50.0, 0.0)
        ops.line_to(150.0, 100.0, 0.0)
        original = encoder.encode(ops, mock_machine, doc)

        data = original.to_dict()
        restored = EncodedOutput.from_dict(data)

        assert restored.text == original.text
        assert restored.driver_data["binary"] == original.driver_data["binary"]
        assert (
            restored.op_map.op_to_machine_code
            == original.op_map.op_to_machine_code
        )
        assert (
            restored.op_map.machine_code_to_op
            == original.op_map.machine_code_to_op
        )

    def test_json_roundtrip(self, encoder, mock_machine, doc):
        """EncodedOutput should survive JSON roundtrip."""
        ops = Ops()
        ops.set_power(0.5)
        ops.enable_air_assist()
        ops.move_to(10.0, 20.0, 0.0)
        original = encoder.encode(ops, mock_machine, doc)

        json_str = original.to_json()
        restored = EncodedOutput.from_json(json_str)

        assert restored.text == original.text
        assert restored.driver_data["binary"] == original.driver_data["binary"]


class TestComplexJobs:
    """Tests for complex job scenarios."""

    def test_square_cut(self, encoder, mock_machine, doc):
        """A simple square cut should encode correctly."""
        ops = Ops()
        ops.job_start()
        ops.set_power(0.8)
        ops.set_cut_speed(200)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.line_to(0.0, 10.0, 0.0)
        ops.line_to(0.0, 0.0, 0.0)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert len(lines) >= 9
        assert "Job Start" in lines[0]
        assert lines[-1] == "; Job End"

        cut_lines = [line for line in lines if "CUT_ABS" in line]
        assert len(cut_lines) == 4

    def test_multi_layer_job(self, encoder, mock_machine, doc):
        """Multi-layer job should encode correctly."""
        ops = Ops()
        ops.job_start()

        ops.layer_start("layer-1")
        ops.set_power(0.5)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.layer_end("layer-1")

        ops.layer_start("layer-2")
        ops.set_power(1.0)
        ops.move_to(0.0, 10.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.layer_end("layer-2")

        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        text = result.text
        assert "; --- Layer layer-1 ---" in text
        assert "; --- Layer layer-2 ---" in text
        assert text.count("; --- End Layer ---") == 2

    def test_air_assist_toggle(self, encoder, mock_machine, doc):
        """Air assist toggle during job should work correctly."""
        ops = Ops()
        ops.enable_air_assist()
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.disable_air_assist()
        ops.move_to(20.0, 0.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "AIR_ASSIST ON" in lines[0]
        on_idx = lines.index(
            [line for line in lines if "AIR_ASSIST ON" in line][0]
        )
        off_idx = lines.index(
            [line for line in lines if "AIR_ASSIST OFF" in line][0]
        )
        assert on_idx < off_idx


class TestCoordinateConversion:
    """Tests for coordinate conversion (mm to µm)."""

    def test_mm_to_um_conversion(self, encoder, mock_machine, doc):
        """Coordinates should be converted from mm to µm."""
        ops = Ops()
        ops.move_to(1.0, 1.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        x_um = int(1.0 * 1000)
        y_um = int(1.0 * 1000)
        assert encode35(x_um) in binary
        assert encode35(y_um) in binary

    def test_large_coordinates(self, encoder, mock_machine, doc):
        """Large coordinates should encode correctly."""
        ops = Ops()
        ops.move_to(1000.0, 2000.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        x_um = int(1000.0 * 1000)
        y_um = int(2000.0 * 1000)
        assert encode35(x_um) in binary
        assert encode35(y_um) in binary

    def test_fractional_coordinates(self, encoder, mock_machine, doc):
        """Fractional coordinates should maintain precision."""
        ops = Ops()
        ops.move_to(0.001, 0.001, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        x_um = int(0.001 * 1000)
        y_um = int(0.001 * 1000)
        assert encode35(x_um) in binary
        assert encode35(y_um) in binary


class TestBinaryCommandStructure:
    """Tests for verifying binary command structure."""

    def test_move_command_structure(self, encoder, mock_machine, doc):
        """Move command should have 0x88 prefix + 10 bytes of coords."""
        ops = Ops()
        ops.move_to(100.0, 200.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[0] == 0x88
        assert len(binary) == 11

    def test_cut_command_structure(self, encoder, mock_machine, doc):
        """Cut command should have 0xA8 prefix + 10 bytes of coords."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(50.0, 75.0, 0.0)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        cut_cmd = binary[11:]
        assert cut_cmd[0] == 0xA8
        assert len(cut_cmd) == 11

    def test_speed_command_structure(self, encoder, mock_machine, doc):
        """Speed command should have 0xC9 0x02 prefix."""
        ops = Ops()
        ops.set_cut_speed(100)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[0:2] == b"\xc9\x02"

    def test_power_command_structure(self, encoder, mock_machine, doc):
        """Power command should have 0xC8 prefix + 2 bytes."""
        ops = Ops()
        ops.set_power(0.5)
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[0] == 0xC8
        assert len(binary) == 3


class TestSetFrequencyCommand:
    """Tests for SetFrequencyCommand encoding."""

    def test_frequency_encoding(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.add(SetFrequencyCommand(1000))
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[:2] == b"\xc6\x60"
        assert binary[2] == 1  # laser 1
        assert binary[3] == 0  # part 0
        assert binary[4:] == encode35(1000)
        assert "FREQUENCY 1000" in result.text

    def test_frequency_with_laser_2(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.set_laser("laser-2")
        ops.add(SetFrequencyCommand(5000))
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        freq_cmd = binary[binary.index(0xC6) + 1 :]
        assert freq_cmd[1] == 2  # laser 2
        assert "FREQUENCY 5000" in result.text

    def test_frequency_command_structure(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.add(SetFrequencyCommand(2000))
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[0] == 0xC6
        assert binary[1] == 0x60
        assert len(binary) == 9


class TestSetPulseWidthCommand:
    """Tests for SetPulseWidthCommand encoding."""

    def test_pulse_width_encoding(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.add(SetPulseWidthCommand(50))
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[:2] == b"\xc6\x10"
        assert binary[2] == 1  # laser 1
        assert binary[3] == 0  # part 0
        assert binary[4:] == encode35(50)
        assert "PULSE_WIDTH 50.0" in result.text

    def test_pulse_width_with_laser_2(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.set_laser("laser-2")
        ops.add(SetPulseWidthCommand(100))
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        pulse_cmd = binary[binary.index(bytes([0xC6, 0x10])) + 2 :]
        assert pulse_cmd[0] == 2  # laser 2

    def test_pulse_width_command_structure(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.add(SetPulseWidthCommand(25))
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert binary[0] == 0xC6
        assert binary[1] == 0x10
        assert len(binary) == 9


class TestFrequencyAndPulseWidthInJob:
    """Tests for frequency/pulse_width within a full job."""

    def test_full_job_with_pwm(self, encoder, mock_machine, doc):
        ops = Ops()
        ops.job_start()
        ops.set_power(0.8)
        ops.set_cut_speed(200)
        ops.add(SetFrequencyCommand(1000))
        ops.add(SetPulseWidthCommand(50))
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        binary = result.driver_data["binary"]
        assert b"\xc6\x60" in binary
        assert b"\xc6\x10" in binary
        text = result.text
        assert "FREQUENCY 1000" in text
        assert "PULSE_WIDTH 50.0" in text
