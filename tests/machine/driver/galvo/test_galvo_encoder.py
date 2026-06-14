"""
Tests for GalvoEncoder.

Verifies conversion of Ops commands to EzCad2 binary protocol.
"""

from unittest.mock import MagicMock

from raygeo.ops import Ops

from rayforge.machine.driver.galvo.galvo_encoder import GalvoEncoder
from rayforge.machine.driver.galvo.galvo_consts import GALVO_CENTER


class TestGalvoEncoder:
    def _make_machine(self):
        machine = MagicMock()
        machine.active_wcs = "MACHINE"
        return machine

    def _make_doc(self):
        doc = MagicMock()
        return doc

    def test_encode_empty_ops(self):
        """Test encoding empty ops produces empty output."""
        encoder = GalvoEncoder()
        ops = Ops()
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert result.text == ""
        assert result.driver_data.get("binary", b"") == b""

    def test_encode_move_to(self):
        """Test encoding a move (jump) command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.move_to(10.0, 20.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "JUMP" in result.text

    def test_encode_line_to(self):
        """Test encoding a line (mark) command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.move_to(0.0, 0.0)
        ops.line_to(10.0, 15.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "MARK" in result.text

    def test_encode_job_start_end(self):
        """Test encoding job start/end commands."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.job_start()
        ops.job_end()
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "JOB START" in result.text
        assert "JOB END" in result.text

    def test_encode_set_power(self):
        """Test encoding set power command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.set_power(0.5)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "POWER" in result.text
        assert "50.0" in result.text

    def test_encode_set_cut_speed(self):
        """Test encoding set cut speed command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.set_cut_speed(100.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "MARK_SPEED" in result.text

    def test_encode_set_travel_speed(self):
        """Test encoding set travel speed command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.set_travel_speed(2000.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "JUMP_SPEED" in result.text

    def test_encode_set_frequency(self):
        """Test encoding set frequency command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.set_frequency(30000)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "FREQUENCY" in result.text

    def test_encode_set_pulse_width(self):
        """Test encoding set pulse width command."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.set_pulse_width(10.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "PULSE_WIDTH" in result.text

    def test_encode_binary_data_present(self):
        """Test that binary data is produced."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.job_start()
        ops.move_to(0.0, 0.0)
        ops.line_to(10.0, 15.0)
        ops.job_end()
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert len(result.driver_data.get("binary", b"")) > 0

    def test_encode_binary_is_valid_packets(self):
        """Test that binary data consists of valid 12-byte packets."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.move_to(0.0, 0.0)
        ops.line_to(10.0, 15.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)
        binary = result.driver_data.get("binary", b"")

        assert len(binary) % 12 == 0

    def test_encode_op_map(self):
        """Test that op_map is correctly built."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.move_to(0.0, 0.0)
        ops.line_to(10.0, 15.0)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert len(result.op_map.op_to_machine_code) > 0
        assert len(result.op_map.machine_code_to_op) > 0

    def test_co2_source_uses_power_ratio(self):
        """Test CO2 source uses power ratio command."""
        encoder = GalvoEncoder(source="co2")
        ops = Ops()
        ops.set_power(0.5)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "POWER" in result.text

    def test_uv_source_uses_power_ratio(self):
        """Test UV source uses power ratio command."""
        encoder = GalvoEncoder(source="uv")
        ops = Ops()
        ops.set_power(0.5)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "POWER" in result.text

    def test_encode_layer_start_end(self):
        """Test encoding layer start/end commands."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.layer_start("layer-1")
        ops.move_to(0.0, 0.0)
        ops.layer_end("layer-1")
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "Layer" in result.text

    def test_encode_arc(self):
        """Test encoding arc command linearizes to lines."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.move_to(0.0, 0.0)
        ops.arc_to(10.0, 10.0, -5.0, 5.0, clockwise=True)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "ARC" in result.text or "MARK" in result.text

    def test_encode_air_assist(self):
        """Test encoding air assist commands."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.enable_air_assist(True)
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        assert "AIR_ASSIST" in result.text

    def test_encode_multiple_commands(self):
        """Test encoding a sequence of multiple commands."""
        encoder = GalvoEncoder()
        ops = Ops()
        ops.job_start()
        ops.set_power(0.5)
        ops.set_cut_speed(100.0)
        ops.set_frequency(30000)
        ops.move_to(0.0, 0.0)
        ops.line_to(10.0, 20.0)
        ops.line_to(30.0, 40.0)
        ops.move_to(50.0, 50.0)
        ops.job_end()
        machine = self._make_machine()
        doc = self._make_doc()

        result = encoder.encode(ops, machine, doc)

        lines = result.text.split("\n")
        assert len(lines) > 5
        assert any("JOB START" in ln for ln in lines)
        assert any("POWER" in ln for ln in lines)
        assert any("MARK_SPEED" in ln for ln in lines)
        assert any("JUMP" in ln for ln in lines)
        assert any("MARK" in ln for ln in lines)
        assert any("JOB END" in ln for ln in lines)

    def test_mm_to_galvo_center(self):
        """Test that center position maps to 0x8000."""
        encoder = GalvoEncoder()
        result = encoder._mm_to_galvo(0.0)
        assert result == GALVO_CENTER

    def test_mm_to_galvo_positive(self):
        """Test positive coordinate conversion."""
        encoder = GalvoEncoder()
        result = encoder._mm_to_galvo(100.0)
        assert result == 0xC000

    def test_mm_to_galvo_negative(self):
        """Test negative coordinate conversion."""
        encoder = GalvoEncoder()
        result = encoder._mm_to_galvo(-100.0)
        assert result == 0x4000

    def test_mm_to_galvo_clamp(self):
        """Test clamping of out-of-range coordinates."""
        encoder = GalvoEncoder()
        assert encoder._mm_to_galvo(-200.0) == 0
        assert encoder._mm_to_galvo(200.0) == 0xFFFF

    def test_power_to_int(self):
        """Test power conversion from normalized to percent."""
        encoder = GalvoEncoder()
        assert encoder._power_to_int(0.5) == 50
        assert encoder._power_to_int(1.0) == 100
        assert encoder._power_to_int(0.0) == 0

    def test_speed_to_int(self):
        """Test speed conversion from mm/s to galvo units."""
        encoder = GalvoEncoder(galvos_per_mm=500)
        expected = int(100.0 * 500 / 1000.0)
        assert encoder._speed_to_int(100.0) == expected

    def test_frequency_to_period(self):
        """Test frequency to q-switch period conversion."""
        encoder = GalvoEncoder()
        result = encoder._frequency_to_period(30.0, base=20000.0)
        expected = int(round(20000.0 / 30.0)) & 0xFFFF
        assert result == expected

    def test_encoder_default_values(self):
        """Test encoder default values."""
        encoder = GalvoEncoder()
        assert encoder._source == "fiber"
        assert encoder._galvos_per_mm == 500
        assert encoder._default_mark_speed == 100.0
        assert encoder._default_jump_speed == 2000.0
