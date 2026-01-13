from rayforge.machine.driver.grbl_util import (
    _split_status_line,
    _parse_status_part,
    _parse_position_attribute,
    _parse_feed_rate,
    _parse_buffer_state,
    _recalculate_positions,
    parse_state,
    error_code_to_device_error,
    parse_grbl_parser_state,
    gcode_to_p_number,
    _parse_pos_triplet,
)
from rayforge.machine.driver.driver import DeviceStatus, DeviceState


class TestSplitStatusLine:
    """Tests for _split_status_line function."""

    def test_basic_status_line(self):
        """Test splitting a basic status line."""
        status_part, attribs = _split_status_line("<Idle|MPos:10,20,30>")
        assert status_part == "Idle"
        assert attribs == ["MPos:10,20,30"]

    def test_status_line_with_multiple_attributes(self):
        """Test splitting status line with multiple attributes."""
        status_part, attribs = _split_status_line(
            "<Run|MPos:10,20,30|WPos:0,0,0|FS:1000,0>"
        )
        assert status_part == "Run"
        assert attribs == ["MPos:10,20,30", "WPos:0,0,0", "FS:1000,0"]

    def test_status_line_with_alarm(self):
        """Test splitting status line with alarm status."""
        status_part, attribs = _split_status_line("<Alarm:1|MPos:0,0,0>")
        assert status_part == "Alarm:1"
        assert attribs == ["MPos:0,0,0"]

    def test_status_line_only_status(self):
        """Test splitting status line with only status."""
        status_part, attribs = _split_status_line("<Idle>")
        assert status_part == "Idle"
        assert attribs == []

    def test_status_line_with_empty_parts(self):
        """Test splitting status line with empty parts."""
        status_part, attribs = _split_status_line("<Idle||MPos:10,20,30>")
        assert status_part == "Idle"
        assert attribs == ["MPos:10,20,30"]


class TestParseStatusPart:
    """Tests for _parse_status_part function."""

    def test_parse_idle_status(self):
        """Test parsing idle status."""
        status, error_code = _parse_status_part("Idle")
        assert status == DeviceStatus.IDLE
        assert error_code is None

    def test_parse_run_status(self):
        """Test parsing run status."""
        status, error_code = _parse_status_part("Run")
        assert status == DeviceStatus.RUN
        assert error_code is None

    def test_parse_alarm_with_error_code(self):
        """Test parsing alarm with error code."""
        status, error_code = _parse_status_part("Alarm:1")
        assert status == DeviceStatus.ALARM
        assert error_code == "1"

    def test_parse_unknown_status(self):
        """Test parsing unknown status."""
        status, error_code = _parse_status_part("UnknownStatus")
        assert status == DeviceStatus.UNKNOWN
        assert error_code is None

    def test_parse_hold_status(self):
        """Test parsing hold status."""
        status, error_code = _parse_status_part("Hold:0")
        assert status == DeviceStatus.HOLD
        assert error_code == "0"

    def test_parse_jog_status(self):
        """Test parsing jog status."""
        status, error_code = _parse_status_part("Jog")
        assert status == DeviceStatus.JOG
        assert error_code is None


class TestParsePositionAttribute:
    """Tests for _parse_position_attribute function."""

    def test_parse_mpos_attribute(self):
        """Test parsing MPos attribute."""
        result = _parse_position_attribute("MPos:10.5,20.3,-1.0", "MPos")
        assert result == (10.5, 20.3, -1.0)

    def test_parse_wpos_attribute(self):
        """Test parsing WPos attribute."""
        result = _parse_position_attribute("WPos:0.0,0.0,0.0", "WPos")
        assert result == (0.0, 0.0, 0.0)

    def test_parse_wco_attribute(self):
        """Test parsing WCO attribute."""
        result = _parse_position_attribute("WCO:-10.0,-20.0,-30.0", "WCO")
        assert result == (-10.0, -20.0, -30.0)

    def test_parse_wrong_type_returns_none(self):
        """Test that parsing wrong type returns None."""
        result = _parse_position_attribute("MPos:10,20,30", "WPos")
        assert result is None

    def test_parse_invalid_format_returns_none(self):
        """Test that parsing invalid format returns None."""
        result = _parse_position_attribute("MPos:invalid", "MPos")
        assert result is None


class TestParseFeedRate:
    """Tests for _parse_feed_rate function."""

    def test_parse_valid_feed_rate(self):
        """Test parsing valid feed rate."""
        result = _parse_feed_rate("FS:1000,0")
        assert result == 1000

    def test_parse_feed_rate_with_zero(self):
        """Test parsing feed rate with zero."""
        result = _parse_feed_rate("FS:0,0")
        assert result == 0

    def test_parse_feed_rate_large_value(self):
        """Test parsing large feed rate."""
        result = _parse_feed_rate("FS:10000,5000")
        assert result == 10000

    def test_parse_non_fs_attribute_returns_none(self):
        """Test that non-FS attribute returns None."""
        result = _parse_feed_rate("MPos:10,20,30")
        assert result is None

    def test_parse_invalid_fs_format_returns_none(self):
        """Test that invalid FS format returns None."""
        result = _parse_feed_rate("FS:invalid")
        assert result is None


class TestParseBufferState:
    """Tests for _parse_buffer_state function."""

    def test_parse_valid_buffer_state(self):
        """Test parsing valid buffer state."""
        result = _parse_buffer_state("Bf:62,0")
        assert result == (62, 0)

    def test_parse_buffer_state_with_zero(self):
        """Test parsing buffer state with zero."""
        result = _parse_buffer_state("Bf:0,0")
        assert result == (0, 0)

    def test_parse_buffer_state_large_values(self):
        """Test parsing buffer state with large values."""
        result = _parse_buffer_state("Bf:128,64")
        assert result == (128, 64)

    def test_non_bf_attribute_returns_none(self):
        """Test that non-Bf attribute returns None."""
        result = _parse_buffer_state("MPos:10,20,30")
        assert result is None

    def test_invalid_bf_format_returns_none(self):
        """Test that invalid Bf format returns None."""
        result = _parse_buffer_state("Bf:invalid")
        assert result is None


class TestRecalculatePositions:
    """Tests for _recalculate_positions function."""

    def test_recalculate_wpos_from_mpos_and_wco(self):
        """Test recalculating WPos from MPos and WCO."""
        mpos = (100.0, 200.0, 50.0)
        wpos = (None, None, None)
        wco = (10.0, 20.0, 5.0)
        result_mpos, result_wpos, result_wco = _recalculate_positions(
            mpos, wpos, wco, mpos_found=True, wpos_found=False, wco_found=True
        )
        assert result_mpos == mpos
        assert result_wpos == (90.0, 180.0, 45.0)
        assert result_wco == wco

    def test_recalculate_mpos_from_wpos_and_wco(self):
        """Test recalculating MPos from WPos and WCO."""
        mpos = (None, None, None)
        wpos = (90.0, 180.0, 45.0)
        wco = (10.0, 20.0, 5.0)
        result_mpos, result_wpos, result_wco = _recalculate_positions(
            mpos, wpos, wco, mpos_found=False, wpos_found=True, wco_found=True
        )
        assert result_mpos == (100.0, 200.0, 50.0)
        assert result_wpos == wpos
        assert result_wco == wco

    def test_infer_wco_from_mpos_and_wpos(self):
        """Test inferring WCO from MPos and WPos."""
        mpos = (100.0, 200.0, 50.0)
        wpos = (90.0, 180.0, 45.0)
        wco = (0.0, 0.0, 0.0)  # Default/stale WCO
        result_mpos, result_wpos, result_wco = _recalculate_positions(
            mpos, wpos, wco, mpos_found=True, wpos_found=True, wco_found=False
        )
        assert result_mpos == mpos
        assert result_wpos == wpos
        assert result_wco == (10.0, 20.0, 5.0)

    def test_no_recalculation_when_mpos_not_found(self):
        """Test no recalculation when MPos not found."""
        mpos = (None, None, None)
        wpos = (90.0, 180.0, 45.0)
        wco = (10.0, 20.0, 5.0)
        result_mpos, result_wpos, result_wco = _recalculate_positions(
            mpos, wpos, wco, mpos_found=False, wpos_found=False, wco_found=True
        )
        assert result_mpos == mpos
        assert result_wpos == wpos
        assert result_wco == wco

    def test_no_recalculation_when_wco_incomplete(self):
        """Test no recalculation when WCO is incomplete."""
        mpos = (100.0, 200.0, 50.0)
        wpos = (None, None, None)
        wco = (10.0, None, 5.0)
        result_mpos, result_wpos, result_wco = _recalculate_positions(
            mpos, wpos, wco, mpos_found=True, wpos_found=False, wco_found=True
        )
        assert result_mpos == mpos
        assert result_wpos == wpos
        assert result_wco == wco

    def test_no_recalculation_when_mpos_incomplete(self):
        """Test no recalculation when MPos is incomplete."""
        mpos = (100.0, None, 50.0)
        wpos = (None, None, None)
        wco = (10.0, 20.0, 5.0)
        result_mpos, result_wpos, result_wco = _recalculate_positions(
            mpos, wpos, wco, mpos_found=True, wpos_found=False, wco_found=True
        )
        assert result_mpos == mpos
        assert result_wpos == wpos
        assert result_wco == wco


class TestParseState:
    """Tests for parse_state function."""

    def test_parse_basic_idle_state(self):
        """Test parsing basic idle state."""
        default = DeviceState()
        result = parse_state("<Idle|MPos:10,20,30>", default)
        assert result.status == DeviceStatus.IDLE
        assert result.machine_pos == (10.0, 20.0, 30.0)

    def test_parse_state_with_wpos_and_wco(self):
        """Test parsing state with WPos and WCO."""
        default = DeviceState()
        result = parse_state(
            "<Idle|MPos:100,200,50|WPos:90,180,45|WCO:10,20,5>",
            default,
        )
        assert result.status == DeviceStatus.IDLE
        assert result.machine_pos == (100.0, 200.0, 50.0)
        assert result.work_pos == (90.0, 180.0, 45.0)
        assert result.wco == (10.0, 20.0, 5.0)

    def test_parse_state_infers_wco(self):
        """
        Test that WCO is inferred if missing but MPos and WPos are present.
        """
        default = DeviceState()
        result = parse_state("<Idle|MPos:100,200,50|WPos:90,180,45>", default)
        assert result.machine_pos == (100.0, 200.0, 50.0)
        assert result.work_pos == (90.0, 180.0, 45.0)
        assert result.wco == (10.0, 20.0, 5.0)

    def test_parse_state_with_feed_rate(self):
        """Test parsing state with feed rate."""
        default = DeviceState()
        result = parse_state("<Run|MPos:10,20,30|FS:1000,500>", default)
        assert result.status == DeviceStatus.RUN
        assert result.feed_rate == 1000

    def test_parse_state_with_alarm_and_error_code(self):
        """Test parsing state with alarm and error code."""
        default = DeviceState()
        result = parse_state("<Alarm:1|MPos:0,0,0>", default)
        assert result.status == DeviceStatus.ALARM
        assert result.error is not None
        assert result.error.code == 1

    def test_parse_state_recalculates_wpos(self):
        """Test that WPos is recalculated from MPos and WCO."""
        default = DeviceState()
        result = parse_state("<Idle|MPos:100,200,50|WCO:10,20,5>", default)
        assert result.machine_pos == (100.0, 200.0, 50.0)
        assert result.work_pos == (90.0, 180.0, 45.0)
        assert result.wco == (10.0, 20.0, 5.0)

    def test_parse_state_recalculates_mpos(self):
        """Test that MPos is recalculated from WPos and WCO."""
        default = DeviceState()
        result = parse_state("<Idle|WPos:90,180,45|WCO:10,20,5>", default)
        assert result.work_pos == (90.0, 180.0, 45.0)
        assert result.machine_pos == (100.0, 200.0, 50.0)
        assert result.wco == (10.0, 20.0, 5.0)

    def test_parse_state_preserves_default_values(self):
        """Test that default values are preserved when not in input."""
        default = DeviceState(
            status=DeviceStatus.UNKNOWN,
            machine_pos=(1.0, 2.0, 3.0),
            work_pos=(4.0, 5.0, 6.0),
            wco=(7.0, 8.0, 9.0),
            feed_rate=100,
        )
        result = parse_state("<Idle>", default)
        assert result.status == DeviceStatus.IDLE
        assert result.machine_pos == (1.0, 2.0, 3.0)
        assert result.work_pos == (4.0, 5.0, 6.0)
        assert result.wco == (7.0, 8.0, 9.0)
        assert result.feed_rate == 100

    def test_parse_state_with_invalid_format(self):
        """Test parsing state with invalid format."""
        default = DeviceState()
        result = parse_state("invalid", default)
        assert result.status == DeviceStatus.UNKNOWN

    def test_parse_state_with_logger(self):
        """Test parsing state with logger callback."""
        default = DeviceState()
        log_messages = []

        def logger(**kwargs):
            log_messages.append(kwargs.get("message", ""))

        result = parse_state("<Idle|MPos:10,20,30>", default, logger)
        assert result.status == DeviceStatus.IDLE
        assert any("Parsed status: IDLE" in msg for msg in log_messages)

    def test_parse_state_with_unknown_error_code(self):
        """Test parsing state with unknown error code."""
        default = DeviceState()
        result = parse_state("<Alarm:999|MPos:0,0,0>", default)
        assert result.status == DeviceStatus.ALARM
        assert result.error is not None
        assert result.error.code == -1

    def test_parse_state_with_buffer_state(self):
        """Test parsing state with buffer state."""
        default = DeviceState()
        result = parse_state(
            "<Run|MPos:9.212,4.287,0.000|WPos:9.212,4.287,0.000|"
            "Bf:62,0|FS:313,10>",
            default,
        )
        assert result.status == DeviceStatus.RUN
        assert result.buffer_available == 62
        assert result.buffer_rx_available == 0


class TestParsePosTriplet:
    """Tests for _parse_pos_triplet function."""

    def test_parse_valid_position(self):
        """Test parsing valid position triplet."""
        result = _parse_pos_triplet("MPos:10.5,20.3,-1.0")
        assert result == (10.5, 20.3, -1.0)

    def test_parse_position_with_negative_values(self):
        """Test parsing position with negative values."""
        result = _parse_pos_triplet("WPos:-10.5,-20.3,-1.0")
        assert result == (-10.5, -20.3, -1.0)

    def test_parse_position_with_zero(self):
        """Test parsing position with zero values."""
        result = _parse_pos_triplet("WCO:0.0,0.0,0.0")
        assert result == (0.0, 0.0, 0.0)

    def test_parse_invalid_position_returns_none(self):
        """Test that invalid position returns None."""
        result = _parse_pos_triplet("Invalid")
        assert result is None

    def test_parse_incomplete_position_returns_none(self):
        """Test that incomplete position returns None."""
        result = _parse_pos_triplet("MPos:10,20")
        assert result is None


class TestErrorCodeToDeviceError:
    """Tests for error_code_to_device_error function."""

    def test_parse_known_error_code(self):
        """Test parsing known error code."""
        result = error_code_to_device_error("1")
        assert result.code == 1
        assert "Expected Command Letter" in result.title

    def test_parse_unknown_error_code(self):
        """Test parsing unknown error code."""
        result = error_code_to_device_error("999")
        assert result.code == -1
        assert "Unknown Error" in result.title

    def test_parse_invalid_error_code_string(self):
        """Test parsing invalid error code string."""
        result = error_code_to_device_error("invalid")
        assert result.code == -1


class TestParseGrblParserState:
    """Tests for parse_grbl_parser_state function."""

    def test_parse_g54_state(self):
        """Test parsing G54 state."""
        state_line = "[G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]"
        result = parse_grbl_parser_state([state_line])
        assert result == "G54"

    def test_parse_g59_state(self):
        """Test parsing G59 state."""
        state_line = "[G59 G17 G21 G90 G94 M5 M9 T0 F0 S0]"
        result = parse_grbl_parser_state([state_line])
        assert result == "G59"

    def test_parse_no_wcs_found(self):
        """Test when no WCS is found."""
        result = parse_grbl_parser_state(["[G17 G21 G90 G94 M5 M9 T0 F0 S0]"])
        assert result is None

    def test_parse_multiple_lines(self):
        """Test parsing multiple lines."""
        lines = [
            "[G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]",
            "ok",
        ]
        result = parse_grbl_parser_state(lines)
        assert result == "G54"

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        result = parse_grbl_parser_state([])
        assert result is None


class TestGcodeToPNumber:
    """Tests for gcode_to_p_number function."""

    def test_g54_to_p1(self):
        """Test converting G54 to P1."""
        result = gcode_to_p_number("G54")
        assert result == 1

    def test_g55_to_p2(self):
        """Test converting G55 to P2."""
        result = gcode_to_p_number("G55")
        assert result == 2

    def test_g59_to_p6(self):
        """Test converting G59 to P6."""
        result = gcode_to_p_number("G59")
        assert result == 6

    def test_invalid_format_returns_none(self):
        """Test that invalid format returns None."""
        result = gcode_to_p_number("invalid")
        assert result is None

    def test_out_of_range_returns_none(self):
        """Test that out of range returns None."""
        result = gcode_to_p_number("G50")
        assert result is None

    def test_g53_returns_none(self):
        """Test that G53 returns None (not G54-G59)."""
        result = gcode_to_p_number("G53")
        assert result is None
