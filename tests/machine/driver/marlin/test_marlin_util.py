import pytest
from rayforge.machine.driver.grbl.grbl_util import strip_gcode_comments
from rayforge.machine.driver.marlin.marlin_util import (
    extract_marlin_device_name,
    gcode_to_p_number,
    is_boot_message,
    is_error_response,
    is_ok_response,
    parse_error_message,
    parse_marlin_version,
    parse_m114_position,
    parse_m115_firmware_info,
    parse_m211_endstops,
    parse_m503_settings,
)


class TestParseM114Position:
    """Tests for parse_m114_position function."""

    def test_valid_m114_response(self):
        """Test parsing a valid M114 response line."""
        lines = ["X:10.5 Y:20.3 Z:-1.0 E:0.0 Count X:1050 Y:2030"]
        result = parse_m114_position(lines)
        assert result == (10.5, 20.3, -1.0)

    def test_valid_m114_response_integer_values(self):
        """Test parsing M114 response with integer coordinates."""
        lines = ["X:10 Y:20 Z:30"]
        result = parse_m114_position(lines)
        assert result == (10.0, 20.0, 30.0)

    def test_no_matching_lines_returns_none(self):
        """Test that no matching lines returns None."""
        lines = ["ok", "some other response"]
        result = parse_m114_position(lines)
        assert result is None

    def test_empty_list_returns_none(self):
        """Test that empty list returns None."""
        result = parse_m114_position([])
        assert result is None

    def test_multiple_lines_finds_first_match(self):
        """Test that the first matching line is returned."""
        lines = [
            "ok",
            "X:1.0 Y:2.0 Z:3.0",
            "X:10.0 Y:20.0 Z:30.0",
        ]
        result = parse_m114_position(lines)
        assert result == (1.0, 2.0, 3.0)

    def test_negative_coordinates(self):
        """Test parsing negative coordinate values."""
        lines = ["X:-10.5 Y:-20.3 Z:-1.0"]
        result = parse_m114_position(lines)
        assert result == (-10.5, -20.3, -1.0)


class TestParseMarlinVersion:
    """Tests for parse_marlin_version function."""

    def test_valid_version_string(self):
        """Test parsing 'Marlin 2.1.2.7'."""
        result = parse_marlin_version("Marlin 2.1.2.7")
        assert result == "2.1.2.7"

    def test_non_matching_line_returns_none(self):
        """Test that non-matching line returns None."""
        result = parse_marlin_version("ok")
        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = parse_marlin_version("")
        assert result is None

    def test_version_with_extra_text(self):
        """Test version embedded in longer line."""
        result = parse_marlin_version("Marlin 1.1.0 on some board")
        assert result == "1.1.0"


class TestIsOkResponse:
    """Tests for is_ok_response function."""

    def test_plain_ok(self):
        """Test plain 'ok' response."""
        assert is_ok_response("ok") is True

    def test_ok_with_temperature(self):
        """Test 'ok T:19.5 /200.0 B:60.0 /60.0'."""
        assert is_ok_response("ok T:19.5 /200.0 B:60.0 /60.0") is True

    def test_ok_with_p_and_b(self):
        """Test 'ok P:15 B:3'."""
        assert is_ok_response("ok P:15 B:3") is True

    def test_ok_with_newline(self):
        """Test 'ok\\n' is True."""
        assert is_ok_response("ok\n") is True

    def test_okeh_returns_false(self):
        """Test 'okeh' returns False."""
        assert is_ok_response("okeh") is False

    def test_not_ok_returns_false(self):
        """Test 'not ok' returns False."""
        assert is_ok_response("not ok") is False

    def test_empty_string_returns_false(self):
        """Test empty string returns False."""
        assert is_ok_response("") is False

    def test_ok_with_leading_space(self):
        """Test ' ok' returns True."""
        assert is_ok_response(" ok") is True


class TestIsErrorResponse:
    """Tests for is_error_response function."""

    def test_error_with_message(self):
        """Test 'Error:Unknown command' returns True."""
        assert is_error_response("Error:Unknown command") is True

    def test_error_case_sensitive(self):
        """Test 'error:foo' returns False (lowercase)."""
        assert is_error_response("error:foo") is False

    def test_ok_returns_false(self):
        """Test 'ok' returns False."""
        assert is_error_response("ok") is False

    def test_empty_string_returns_false(self):
        """Test empty string returns False."""
        assert is_error_response("") is False

    def test_error_with_leading_space(self):
        """Test ' Error:foo' returns True."""
        assert is_error_response(" Error:foo") is True


class TestParseErrorMessage:
    """Tests for parse_error_message function."""

    def test_error_with_quoted_command(self):
        """Test 'Error:Unknown command: \"?\"'."""
        result = parse_error_message('Error:Unknown command: "?"')
        assert result == 'Unknown command: "?"'

    def test_no_match_returns_empty_string(self):
        """Test non-matching line returns empty string."""
        result = parse_error_message("ok")
        assert result == ""

    def test_error_message_stripped(self):
        """Test that message is whitespace-stripped."""
        result = parse_error_message("Error:  spaced message  ")
        assert result == "spaced message"


class TestIsBootMessage:
    """Tests for is_boot_message function."""

    def test_start(self):
        """Test 'start' returns True."""
        assert is_boot_message("start") is True

    def test_marlin_version(self):
        """Test 'Marlin 2.1.2.7' returns True."""
        assert is_boot_message("Marlin 2.1.2.7") is True

    def test_echo_stored_settings(self):
        """Test 'echo:V88 stored settings retrieved'."""
        assert is_boot_message("echo:V88 stored settings retrieved") is True

    def test_external_reset(self):
        """Test 'External Reset' returns True."""
        assert is_boot_message("External Reset") is True

    def test_external_reset_with_leading_space(self):
        """Test ' External Reset' returns True after strip()."""
        assert is_boot_message(" External Reset") is True

    def test_g28_returns_false(self):
        """Test 'G28' returns False."""
        assert is_boot_message("G28") is False

    def test_ok_returns_false(self):
        """Test 'ok' returns False."""
        assert is_boot_message("ok") is False

    def test_empty_string_returns_false(self):
        """Test empty string returns False."""
        assert is_boot_message("") is False


class TestGcodeToPNumber:
    """Tests for gcode_to_p_number function."""

    def test_g54_to_1(self):
        """Test 'G54' returns 1."""
        assert gcode_to_p_number("G54") == 1

    def test_g55_to_2(self):
        """Test 'G55' returns 2."""
        assert gcode_to_p_number("G55") == 2

    def test_g59_to_6(self):
        """Test 'G59' returns 6."""
        assert gcode_to_p_number("G59") == 6

    def test_g60_returns_none(self):
        """Test 'G60' returns None (out of range)."""
        assert gcode_to_p_number("G60") is None

    def test_m999_returns_none(self):
        """Test 'M999' returns None (not a G-code)."""
        assert gcode_to_p_number("M999") is None

    def test_g53_returns_none(self):
        """Test 'G53' returns None (P=0, out of range)."""
        assert gcode_to_p_number("G53") is None

    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        assert gcode_to_p_number("") is None

    def test_invalid_returns_none(self):
        """Test 'invalid' returns None."""
        assert gcode_to_p_number("invalid") is None


class TestStripGcodeComments:
    @pytest.mark.parametrize(
        "input_line, expected",
        [
            ("G0 X10 ; rapid move", "G0 X10"),
            ("G0 X10;rapid move", "G0 X10"),
            ("; just a comment", ""),
            ("G0 X10", "G0 X10"),
            ("G0 X10 (rapid move)", "G0 X10"),
            ("G0 X10 (rapid) Y20 (another)", "G0 X10  Y20"),
            ("(whole line comment)", ""),
            ("G0 X10 (inline) ; trailing", "G0 X10"),
            ("  G0 X10  ", "G0 X10"),
            ("", ""),
        ],
    )
    def test_strip_comments(self, input_line, expected):
        assert strip_gcode_comments(input_line) == expected


class TestParseM115FirmwareInfo:
    def test_full_m115_response(self):
        lines = [
            "FIRMWARE_NAME:Marlin 2.1.2.7 "
            "(Marlin 3D Printer Firmware) "
            "SOURCE_CODE_URL:https://github.com/MarlinFirmware"
            "/Marlin PROTOCOL_VERSION:1.0 "
            "MACHINE_TYPE:Custom Laser "
            "EXTRUDER_COUNT:1 UUID:abcdef",
        ]
        result = parse_m115_firmware_info(lines)
        assert result["firmware_name"] == "Marlin"
        assert result["machine_type"] == "Custom Laser"

    def test_firmware_name_only(self):
        lines = ["FIRMWARE_NAME:Marlin 2.1.2.7 PROTOCOL_VERSION:1.0"]
        result = parse_m115_firmware_info(lines)
        assert result["firmware_name"] == "Marlin"
        assert "machine_type" not in result

    def test_empty_lines(self):
        assert parse_m115_firmware_info([]) == {}

    def test_no_match(self):
        assert parse_m115_firmware_info(["ok"]) == {}


class TestParseM211Endstops:
    def test_m211_output(self):
        lines = ["echo: M211 S1 X200.00 Y200.00 Z200.00"]
        result = parse_m211_endstops(lines)
        assert result == (200.0, 200.0)

    def test_m211_report_style(self):
        lines = ["X: 300.00 Y: 300.00 Z: 200.00 S: 1 (Enabled)"]
        result = parse_m211_endstops(lines)
        assert result == (300.0, 300.0)

    def test_no_match(self):
        assert parse_m211_endstops(["ok"]) is None

    def test_empty(self):
        assert parse_m211_endstops([]) is None


class TestParseM503Settings:
    def test_full_m503(self):
        lines = [
            "echo:  G21    ; Units in mm",
            "echo:; Steps per unit:",
            "echo: M92 X80.00 Y80.00 Z400.00 E93.00",
            "echo:; Maximum feedrates (units/s):",
            "echo: M203 X300.00 Y300.00 Z5.00 E25.00",
            "echo:; Maximum Acceleration (units/s2):",
            "echo: M201 X3000.00 Y3000.00 Z100.00 E5000.00",
            "echo:; Acceleration: S=acceleration",
            "echo: M204 S3000.00 T3000.00",
        ]
        result = parse_m503_settings(lines)
        assert result["max_feedrate_x"] == 300.0
        assert result["max_feedrate_y"] == 300.0
        assert result["acceleration"] == 3000.0

    def test_different_feedrates(self):
        lines = [
            "echo: M203 X500.00 Y300.00 Z5.00 E25.00",
            "echo: M204 S500.00 T500.00",
        ]
        result = parse_m503_settings(lines)
        assert result["max_feedrate_x"] == 500.0
        assert result["max_feedrate_y"] == 300.0
        assert result["acceleration"] == 500.0

    def test_no_match(self):
        assert parse_m503_settings(["ok", "echo: M92 X80.00"]) == {}

    def test_empty(self):
        assert parse_m503_settings([]) == {}


class TestExtractMarlinDeviceName:
    def test_machine_type_from_m115(self):
        lines = [
            "FIRMWARE_NAME:Marlin 2.1.2.7 "
            "MACHINE_TYPE:CustomCNC EXTRUDER_COUNT:1"
        ]
        assert extract_marlin_device_name(lines) == "CustomCNC"

    def test_generic_machine_type_falls_back(self):
        lines = [
            "FIRMWARE_NAME:Marlin 2.1.2.7 "
            "MACHINE_TYPE:3D Printer EXTRUDER_COUNT:1"
        ]
        assert extract_marlin_device_name(lines) == "Marlin"

    def test_no_m115_uses_boot_lines(self):
        result = extract_marlin_device_name([], boot_lines=["Marlin 2.1.2.7"])
        assert result == "Marlin 2.1.2.7"

    def test_no_info_returns_unknown(self):
        assert extract_marlin_device_name([]) == ("Unknown Marlin Device")
