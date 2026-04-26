"""
Tests for the Ruida simulator.
"""

from rayforge.machine.driver.ruida.ruida_simulator import RuidaSimulator
from rayforge.machine.driver.ruida.ruida_util import (
    build_swizzle_lut,
    decode14,
    decode35,
    decodeu14,
    decodeu35,
    encode14,
    encode35,
    swizzle_byte,
    unswizzle_byte,
)


class TestEncodingFunctions:
    """Test byte encoding/decoding functions."""

    def test_swizzle_unswizzle_roundtrip(self):
        for val in range(256):
            swizzled = swizzle_byte(val, 0x88)
            unswizzled = unswizzle_byte(swizzled, 0x88)
            assert unswizzled == val

    def test_swizzle_different_magic(self):
        val = 0x42
        swizzled_88 = swizzle_byte(val, 0x88)
        swizzled_11 = swizzle_byte(val, 0x11)
        assert swizzled_88 != swizzled_11

    def test_build_swizzle_lut(self):
        swizzle, unswizzle = build_swizzle_lut(0x88)
        assert len(swizzle) == 256
        assert len(unswizzle) == 256
        for i in range(256):
            assert unswizzle[swizzle[i]] == i

    def test_encode14_zero(self):
        assert encode14(0) == b"\x00\x00"

    def test_encode14_max(self):
        assert encode14(0x3FFF) == b"\x7f\x7f"

    def test_encode32_zero(self):
        assert encode35(0) == b"\x00\x00\x00\x00\x00"

    def test_encode32_max(self):
        assert encode35(0xFFFFFFFF) == b"\x0f\x7f\x7f\x7f\x7f"

    def test_decode14_positive(self):
        assert decode14(b"\x00\x00") == 0
        assert decode14(b"\x00\x7f") == 127

    def test_decode14_negative(self):
        result = decode14(b"\x40\x00")
        assert result == -8192

    def test_decodeu14(self):
        assert decodeu14(b"\x00\x00") == 0
        assert decodeu14(b"\x7f\x7f") == 0x3FFF

    def test_decode32_positive(self):
        assert decode35(b"\x00\x00\x00\x00\x00") == 0

    def test_decode32_negative(self):
        result = decode35(b"\x7c\x00\x00\x00\x00")
        assert result == -1073741824

    def test_decodeu35(self):
        assert decodeu35(b"\x00\x00\x00\x00\x00") == 0
        assert decodeu35(b"\x0f\x7f\x7f\x7f\x7f") == 0xFFFFFFFF


class TestSimulatorBasics:
    """Test basic simulator functionality."""

    def test_default_values(self):
        sim = RuidaSimulator()
        assert sim.bed_x == RuidaSimulator.DEFAULT_BED_X
        assert sim.bed_y == RuidaSimulator.DEFAULT_BED_Y


class TestCommandHandling:
    """Test command parsing and handling."""

    def test_handle_ack(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xcc")
        assert response == b""
        assert length == 1

    def test_handle_eof(self):
        sim = RuidaSimulator()
        sim.program_mode = True
        response, length = sim._process_single_command(b"\xd7")
        assert response == b""
        assert length == 1
        assert sim.program_mode is False


class TestPacketHandling:
    """Test packet handling directly (no UDP)."""

    def test_send_card_id_request(self):
        sim = RuidaSimulator()
        cmd = b"\xda\x00\x05\x7e"
        response = sim.process_commands(cmd)
        assert response[0] == 0xDA
        assert response[1] == 0x01

    def test_jog_keepalive(self):
        sim = RuidaSimulator()
        response = sim.handle_jog_packet(b"\xcc")
        assert response == b"\xcc"


class TestD8Commands:
    """Test D8 realtime commands."""

    def test_start_process(self):
        sim = RuidaSimulator()
        sim.program_mode = False
        response, length = sim._process_single_command(b"\xd8\x00")
        assert response == b""
        assert length == 2
        assert sim.program_mode is True

    def test_stop_process(self):
        sim = RuidaSimulator()
        sim.program_mode = True
        response, length = sim._process_single_command(b"\xd8\x01")
        assert response == b""
        assert length == 2

    def test_pause_process(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xd8\x02")
        assert response == b""
        assert length == 2

    def test_restore_process(self):
        sim = RuidaSimulator()
        sim.program_mode = True
        response, length = sim._process_single_command(b"\xd8\x03")
        assert response == b""
        assert length == 2

    def test_home_xy(self):
        sim = RuidaSimulator()
        sim.x = 1000
        sim.y = 2000
        response, length = sim._process_single_command(b"\xd8\x2a")
        assert response == b""
        assert length == 2
        assert sim.x == 0
        assert sim.y == 0

    def test_home_z(self):
        sim = RuidaSimulator()
        sim.z = 500
        response, length = sim._process_single_command(b"\xd8\x2c")
        assert response == b""
        assert length == 2
        assert sim.z == 0

    def test_home_u(self):
        sim = RuidaSimulator()
        sim.u = 300
        response, length = sim._process_single_command(b"\xd8\x2d")
        assert response == b""
        assert length == 2
        assert sim.u == 0

    def test_ref_point_mode_0(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xd8\x12")
        assert response == b""
        assert length == 2

    def test_ref_point_mode_1(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xd8\x11")
        assert response == b""
        assert length == 2

    def test_ref_point_mode_2(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xd8\x10")
        assert response == b""
        assert length == 2


class TestD9Commands:
    """Test D9 rapid move commands."""

    def test_rapid_move_x(self):
        sim = RuidaSimulator()
        initial_x = sim.x
        coord_bytes = encode35(50000)
        cmd = b"\xd9\x00\x00" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 8
        assert sim.x == initial_x + 50000

    def test_rapid_move_extended_x(self):
        sim = RuidaSimulator()
        initial_x = sim.x
        coord_bytes = encode35(75000)
        cmd = b"\xd9\x50\x00" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 8
        assert sim.x == initial_x + 75000


class TestMoveCommands:
    """Test move and cut commands."""

    def test_move_abs(self):
        sim = RuidaSimulator()
        x_bytes = encode35(100000)
        y_bytes = encode35(200000)
        cmd = b"\x88" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 11
        assert sim.x == 100000
        assert sim.y == 200000

    def test_move_rel(self):
        sim = RuidaSimulator()
        sim.x = 1000
        sim.y = 2000
        dx_bytes = encode14(100)
        dy_bytes = encode14(200)
        cmd = b"\x89" + dx_bytes + dy_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 5
        assert sim.x == 1100
        assert sim.y == 2200

    def test_cut_abs(self):
        sim = RuidaSimulator()
        x_bytes = encode35(50000)
        y_bytes = encode35(60000)
        cmd = b"\xa8" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 11
        assert sim.x == 50000
        assert sim.y == 60000

    def test_cut_rel(self):
        sim = RuidaSimulator()
        sim.x = 5000
        sim.y = 6000
        dx_bytes = encode14(500)
        dy_bytes = encode14(600)
        cmd = b"\xa9" + dx_bytes + dy_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 5
        assert sim.x == 5500
        assert sim.y == 6600


class TestPowerCommands:
    """Test power setting commands."""

    def test_immediate_power(self):
        sim = RuidaSimulator()
        power_bytes = encode14(16384)
        cmd = b"\xc7" + power_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_end_power(self):
        sim = RuidaSimulator()
        power_bytes = encode14(8192)
        cmd = b"\xc8" + power_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_min_power(self):
        sim = RuidaSimulator()
        power_bytes = encode14(3277)
        cmd = b"\xc6\x01" + power_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 4

    def test_max_power(self):
        sim = RuidaSimulator()
        power_bytes = encode14(16384)
        cmd = b"\xc6\x02" + power_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 4


class TestSpeedCommands:
    """Test speed setting commands."""

    def test_speed_laser_1(self):
        sim = RuidaSimulator()
        speed_bytes = encode35(20000)
        cmd = b"\xc9\x02" + speed_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7

    def test_axis_speed(self):
        sim = RuidaSimulator()
        speed_bytes = encode35(50000)
        cmd = b"\xc9\x03" + speed_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7

    def test_part_speed(self):
        sim = RuidaSimulator()
        speed_bytes = encode35(30000)
        cmd = b"\xc9\x04\x00" + speed_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 8

    def test_force_eng_speed(self):
        sim = RuidaSimulator()
        speed_bytes = encode35(15000)
        cmd = b"\xc9\x05" + speed_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7

    def test_axis_move_speed(self):
        sim = RuidaSimulator()
        speed_bytes = encode35(25000)
        cmd = b"\xc9\x06" + speed_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7


class TestE7Commands:
    """Test E7 file layout commands."""

    def test_filename(self):
        sim = RuidaSimulator()
        cmd = b"\xe7\x01test\x00"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
        assert sim.filename == "test"

    def test_block_end(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xe7\x00")
        assert response == b""
        assert length == 2

    def test_process_top_left(self):
        sim = RuidaSimulator()
        x_bytes = encode35(0)
        y_bytes = encode35(0)
        cmd = b"\xe7\x03" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 12

    def test_process_bottom_right(self):
        sim = RuidaSimulator()
        x_bytes = encode35(100000)
        y_bytes = encode35(100000)
        cmd = b"\xe7\x07" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 12

    def test_document_min_point(self):
        sim = RuidaSimulator()
        x_bytes = encode35(0)
        y_bytes = encode35(0)
        cmd = b"\xe7\x50" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 12

    def test_document_max_point(self):
        sim = RuidaSimulator()
        x_bytes = encode35(320000)
        y_bytes = encode35(220000)
        cmd = b"\xe7\x51" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 12


class TestF1F2Commands:
    """Test F1 and F2 element commands."""

    def test_f1_element_max_index(self):
        sim = RuidaSimulator()
        cmd = b"\xf1\x00\x0a"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_f1_element_name_max_index(self):
        sim = RuidaSimulator()
        cmd = b"\xf1\x01\x05"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_f1_enable_block_cutting(self):
        sim = RuidaSimulator()
        cmd = b"\xf1\x02\x01"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_f2_element_index(self):
        sim = RuidaSimulator()
        cmd = b"\xf2\x00\x00"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_f2_element_name(self):
        sim = RuidaSimulator()
        cmd = b"\xf2\x02TestName\x00"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 11

    def test_f2_element_array_min_point(self):
        sim = RuidaSimulator()
        x_bytes = encode35(1000)
        y_bytes = encode35(2000)
        cmd = b"\xf2\x03" + x_bytes + y_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 12


class TestE8Commands:
    """Test E8 file interaction commands."""

    def test_delete_document(self):
        sim = RuidaSimulator()
        cmd = b"\xe8\x00\x00\x00"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xE8
        assert response[1] == 0x00

    def test_document_name(self):
        sim = RuidaSimulator()
        cmd = b"\xe8\x01\x00\x01"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xE8
        assert response[1] == 0x01

    def test_file_transfer(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xe8\x02")
        assert response == b""
        assert length == 2


class TestChecksumAccumulation:
    """Test file checksum accumulation."""

    def test_checksum_accumulates_on_cut(self):
        sim = RuidaSimulator()
        sim.file_checksum_accumulator = 0
        x_bytes = encode35(10000)
        y_bytes = encode35(20000)
        cmd = b"\xa8" + x_bytes + y_bytes
        sim._process_single_command(cmd)
        assert sim.file_checksum_accumulator > 0

    def test_checksum_accumulates_on_move(self):
        sim = RuidaSimulator()
        sim.file_checksum_accumulator = 0
        x_bytes = encode35(5000)
        y_bytes = encode35(10000)
        cmd = b"\x88" + x_bytes + y_bytes
        sim._process_single_command(cmd)
        assert sim.file_checksum_accumulator > 0

    def test_checksum_not_accumulated_on_memory(self):
        sim = RuidaSimulator()
        initial = sim.file_checksum_accumulator
        cmd = b"\xda\x00\x05\x7e"
        sim._process_single_command(cmd)
        assert sim.file_checksum_accumulator == initial


class TestDACommands:
    """Test DA memory commands."""

    def test_get_card_id(self):
        sim = RuidaSimulator()
        cmd = b"\xda\x00\x05\x7e"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xDA
        assert response[1] == 0x01

    def test_get_bed_x(self):
        sim = RuidaSimulator()
        cmd = b"\xda\x00\x00\x26"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xDA
        assert response[1] == 0x01

    def test_get_bed_y(self):
        sim = RuidaSimulator()
        cmd = b"\xda\x00\x00\x36"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xDA
        assert response[1] == 0x01

    def test_read_run_info(self):
        sim = RuidaSimulator()
        cmd = b"\xda\x05\x00\x00"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xDA
        assert response[1] == 0x05


class TestCACommands:
    """Test CA layer/mode commands."""

    def test_end_layer(self):
        sim = RuidaSimulator()
        cmd = b"\xca\x01\x00"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_work_mode_1(self):
        sim = RuidaSimulator()
        cmd = b"\xca\x01\x01"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3

    def test_layer_color(self):
        sim = RuidaSimulator()
        color_bytes = encode35(0x0000FF)
        cmd = b"\xca\x05" + color_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7

    def test_part_layer_number(self):
        sim = RuidaSimulator()
        cmd = b"\xca\x02\x00"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 3


class TestMemoryPersistence:
    """Test DA 0x01 memory write persistence."""

    def test_write_then_read_value(self):
        sim = RuidaSimulator()
        write_val = encode35(12345) + encode35(0)
        cmd = b"\xda\x01\x00\x30" + write_val
        sim._process_single_command(cmd)
        read_cmd = b"\xda\x00\x00\x30"
        response, _ = sim._process_single_command(read_cmd)
        assert response[0] == 0xDA
        assert response[1] == 0x01
        decoded = decode35(response[4:9])
        assert decoded == 12345


class TestRefPointMode:
    """Test D8 reference point mode tracking."""

    def test_ref_point_mode_0_tracked(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x12")
        assert sim._ref_point_mode == 0

    def test_ref_point_mode_1_tracked(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x11")
        assert sim._ref_point_mode == 1

    def test_ref_point_mode_2_tracked(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x10")
        assert sim._ref_point_mode == 2


class TestDA04Response:
    """Test DA 0x04 OEM/CardIO response (required for some models)."""

    def test_da04_returns_response(self):
        sim = RuidaSimulator()
        cmd = b"\xda\x04\x00\x00"
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xDA
        assert response[1] == 0x04
        assert len(response) == 12


class TestKeepAlive:
    """Test CE keep alive command."""

    def test_keepalive_returns_ack(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xce")
        assert response == b"\xcc"
        assert length == 1


class TestAxisMoveCommands:
    """Test 0x80 and 0xA0 axis move commands."""

    def test_axis_x_move(self):
        sim = RuidaSimulator()
        coord_bytes = encode35(50000)
        cmd = b"\x80\x00" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
        assert sim.x == 50000

    def test_axis_z_move_0x08(self):
        sim = RuidaSimulator()
        coord_bytes = encode35(3000)
        cmd = b"\x80\x08" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
        assert sim.z == 3000

    def test_axis_y_move(self):
        sim = RuidaSimulator()
        coord_bytes = encode35(60000)
        cmd = b"\xa0\x00" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
        assert sim.y == 60000

    def test_axis_u_move(self):
        sim = RuidaSimulator()
        coord_bytes = encode35(2000)
        cmd = b"\xa0\x08" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
        assert sim.u == 2000


class TestE5Commands:
    """Test E5 document commands."""

    def test_e5_00_document_page_number(self):
        sim = RuidaSimulator()
        filenum_bytes = encode14(5)
        cmd = b"\xe5\x00" + filenum_bytes
        response, length = sim._process_single_command(cmd)
        assert response != b""
        assert length == 4
        assert response[0] == 0xE5
        assert response[1] == 0x00
        assert len(response) == 12

    def test_e5_02_document_data_end(self):
        sim = RuidaSimulator()
        response, length = sim._process_single_command(b"\xe5\x02")
        assert response == b""
        assert length == 2

    def test_e5_05_set_file_sum(self):
        sim = RuidaSimulator()
        checksum_bytes = encode35(12345)
        cmd = b"\xe5\x05" + checksum_bytes
        sim.file_checksum_accumulator = 12345
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
        assert sim.file_checksum == 12345
        assert sim.file_checksum_accumulator == 0


class TestD9RapidFeedAxis:
    """Test D9 0x0F rapid feed axis move command."""

    def test_rapid_feed_axis_move(self):
        sim = RuidaSimulator()
        coord_bytes = encode35(10000)
        cmd = b"\xd9\x0f\x00" + coord_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 8

    def test_rapid_feed_axis_move_insufficient_data(self):
        sim = RuidaSimulator()
        cmd = b"\xd9\x0f\x00"
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 1


class TestD8JogTracking:
    """Test D8 KeyDown/KeyUp jog state tracking."""

    def test_keydown_x_left(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x20")
        assert sim._jog_active["x"] < 0

    def test_keydown_x_right(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x21")
        assert sim._jog_active["x"] > 0

    def test_keydown_y_top(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x22")
        assert sim._jog_active["y"] > 0

    def test_keydown_y_bottom(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x23")
        assert sim._jog_active["y"] < 0

    def test_keydown_z_up(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x24")
        assert sim._jog_active["z"] > 0

    def test_keydown_z_down(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x25")
        assert sim._jog_active["z"] < 0

    def test_keydown_u_forward(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x26")
        assert sim._jog_active["u"] > 0

    def test_keydown_u_backward(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x27")
        assert sim._jog_active["u"] < 0

    def test_keyup_x_stops_jog(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x20")
        assert sim._jog_active["x"] != 0
        sim._process_single_command(b"\xd8\x30")
        assert sim._jog_active["x"] == 0

    def test_keyup_y_stops_jog(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x22")
        assert sim._jog_active["y"] != 0
        sim._process_single_command(b"\xd8\x32")
        assert sim._jog_active["y"] == 0

    def test_keyup_z_stops_jog(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x24")
        assert sim._jog_active["z"] != 0
        sim._process_single_command(b"\xd8\x34")
        assert sim._jog_active["z"] == 0

    def test_keyup_u_stops_jog(self):
        sim = RuidaSimulator()
        sim._process_single_command(b"\xd8\x26")
        assert sim._jog_active["u"] != 0
        sim._process_single_command(b"\xd8\x36")
        assert sim._jog_active["u"] == 0

    def test_jog_uses_jog_speed(self):
        sim = RuidaSimulator()
        sim._jog_speed = 5000
        sim._process_single_command(b"\xd8\x21")
        assert sim._jog_active["x"] == 5000

    def test_jog_negative_direction(self):
        sim = RuidaSimulator()
        sim._jog_speed = 8000
        sim._process_single_command(b"\xd8\x20")
        assert sim._jog_active["x"] == -8000


class TestE7BYTest:
    """Test E7 0x46 BY Test command."""

    def test_by_test(self):
        sim = RuidaSimulator()
        val_bytes = encode35(0x11227766)
        cmd = b"\xe7\x46" + val_bytes
        response, length = sim._process_single_command(cmd)
        assert response == b""
        assert length == 7
