import math
from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.state import CoolantMode

from rayforge.machine.models.dialect.grbl import GRBL_DIALECT
from rayforge.machine.models.dialect.grbl_raster import GRBL_RASTER_DIALECT
from rayforge.machine.models.dialect.marlin import MARLIN_DIALECT
from rayforge.pipeline.encoder.context import GcodeContext, JobInfo
from rayforge.pipeline.encoder.gcode import GcodeEncoder


def _make_encoder(dialect) -> GcodeEncoder:
    encoder = GcodeEncoder(dialect)
    encoder._coord_format = "{:.3f}"
    encoder._feed_format = "{:.3f}"
    encoder._power_format = "{:.3f}"
    return encoder


def _make_context() -> GcodeContext:
    machine = MagicMock()
    machine.get_default_head.return_value = MagicMock(
        uid="head0", max_power=100.0, tool_number=1
    )
    machine.get_head_by_uid.return_value = MagicMock(
        uid="head0", max_power=100.0, tool_number=1
    )
    machine.max_travel_speed = 5000.0
    machine.active_wcs = "G54"
    machine.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.hookmacros = {}
    doc = MagicMock()
    doc.active_layer = MagicMock(
        rotary_enabled=False,
        rotary_diameter=25.0,
        rotary_module_uid=None,
    )
    doc.find_descendant_by_uid.return_value = None
    extents = (0.0, 0.0, 100.0, 100.0)
    return GcodeContext(machine=machine, doc=doc, job=JobInfo(extents=extents))


def _dispatch_all(gcode, ops, encoder, context):
    for i in range(ops.len()):
        encoder._handle_command(gcode, ops, i, context)


class TestHandleBezierTo:
    def test_bezier_with_g5_template(self):
        encoder = _make_encoder(MARLIN_DIALECT)
        encoder.current_pos = {Axis.X: 0.0, Axis.Y: 0.0, Axis.Z: 0.0}
        encoder.power = 1.0
        encoder.cut_speed = 1000.0
        context = _make_context()

        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.bezier_to(
            (3.0, 0.0, 0.0),
            (7.0, 10.0, 0.0),
            (10.0, 10.0, 0.0),
        )

        gcode = []
        ea = ops.extra_axes(1)
        encoder._handle_bezier_to(context, gcode, ops, 1, ea)

        assert len(gcode) >= 1
        line = gcode[-1]
        assert line.startswith("G5")
        assert "X10" in line
        assert "Y10" in line
        assert "I3" in line
        assert "J0" in line
        assert "P-3" in line
        assert "Q0" in line

    def test_bezier_with_empty_template_falls_back_to_lines(self):
        encoder = _make_encoder(GRBL_DIALECT)
        encoder.current_pos = {Axis.X: 0.0, Axis.Y: 0.0, Axis.Z: 0.0}
        encoder.power = 1.0
        encoder.cut_speed = 1000.0
        context = _make_context()

        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.bezier_to(
            (3.0, 0.0, 0.0),
            (7.0, 10.0, 0.0),
            (10.0, 10.0, 0.0),
        )

        gcode = []
        ea = ops.extra_axes(1)
        encoder._handle_bezier_to(context, gcode, ops, 1, ea)

        assert len(gcode) > 1
        motion_lines = [
            line for line in gcode if line.strip().startswith("G1")
        ]
        assert len(motion_lines) > 1

    def test_bezier_offset_computation(self):
        encoder = _make_encoder(MARLIN_DIALECT)
        encoder.current_pos = {Axis.X: 5.0, Axis.Y: 5.0, Axis.Z: 0.0}
        encoder.power = 1.0
        encoder.cut_speed = 1000.0
        context = _make_context()

        ops = Ops()
        ops.move_to(5.0, 5.0, 0.0)
        ops.bezier_to(
            (8.0, 5.0, 0.0),
            (12.0, 15.0, 0.0),
            (15.0, 15.0, 0.0),
        )

        gcode = []
        ea = ops.extra_axes(1)
        encoder._handle_bezier_to(context, gcode, ops, 1, ea)

        line = gcode[-1]
        assert "I3" in line
        assert "J0" in line
        assert "P-3" in line
        assert "Q0" in line

    def test_dispatch_in_handle_command(self):
        encoder = _make_encoder(MARLIN_DIALECT)
        encoder.current_pos = {Axis.X: 0.0, Axis.Y: 0.0, Axis.Z: 0.0}
        encoder.power = 1.0
        encoder.cut_speed = 1000.0
        context = _make_context()

        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.bezier_to(
            (3.0, 0.0, 0.0),
            (7.0, 10.0, 0.0),
            (10.0, 10.0, 0.0),
        )

        gcode = []
        encoder._handle_command(gcode, ops, 1, context)

        assert any("G5" in line for line in gcode)
        assert math.isclose(encoder.current_pos[Axis.X], 10.0)
        assert math.isclose(encoder.current_pos[Axis.Y], 10.0)

    def test_bezier_in_ops_encoding(self):
        ops = Ops()
        ops.job_start()
        ops.set_power(1.0)
        ops.set_feed_rate(1000)
        ops.move_to(0.0, 0.0, 0.0)
        ops.bezier_to(
            (3.0, 0.0, 0.0),
            (7.0, 10.0, 0.0),
            (10.0, 10.0, 0.0),
        )
        ops.job_end()

        encoder = _make_encoder(MARLIN_DIALECT)
        context = _make_context()
        gcode = []

        _dispatch_all(gcode, ops, encoder, context)

        joined = "\n".join(gcode)
        assert "G5" in joined


class TestG0G1FeedrateSharing:
    """
    Regression test for issue #210.

    In GRBL, the F word is modal across G0 and G1. When G0 emits an F
    parameter, subsequent G1 commands must re-emit their own F value,
    even when modal_feedrate is enabled.
    """

    def _encode_ops(self, ops, dialect):
        encoder = _make_encoder(dialect)
        context = _make_context()
        gcode = []
        _dispatch_all(gcode, ops, encoder, context)
        return gcode

    def test_g1_re_emits_feedrate_after_g0_with_modal_feedrate(self):
        ops = Ops()
        ops.job_start()
        ops.set_power(1.0)
        ops.set_feed_rate(3000)
        ops.set_rapid_rate(1000)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.move_to(20.0, 0.0, 0.0)
        ops.line_to(30.0, 0.0, 0.0)
        ops.job_end()

        gcode = self._encode_ops(ops, GRBL_RASTER_DIALECT)

        g1_lines = [line for line in gcode if line.startswith("G1")]
        assert len(g1_lines) == 3
        assert "F3000" in g1_lines[0]
        assert "F3000" in g1_lines[2]

    def test_g1_omits_feedrate_when_no_g0_intervenes(self):
        ops = Ops()
        ops.job_start()
        ops.set_power(1.0)
        ops.set_feed_rate(3000)
        ops.set_rapid_rate(1000)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.line_to(10.0, 10.0, 0.0)
        ops.line_to(0.0, 10.0, 0.0)
        ops.job_end()

        gcode = self._encode_ops(ops, GRBL_RASTER_DIALECT)

        g1_lines = [line for line in gcode if line.startswith("G1")]
        assert len(g1_lines) == 3
        assert "F3000" in g1_lines[0]
        assert "F" not in g1_lines[1]
        assert "F" not in g1_lines[2]

    def test_non_modal_dialect_always_emits_feedrate(self):
        ops = Ops()
        ops.job_start()
        ops.set_power(1.0)
        ops.set_feed_rate(3000)
        ops.set_rapid_rate(1000)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.move_to(20.0, 0.0, 0.0)
        ops.line_to(30.0, 0.0, 0.0)
        ops.job_end()

        gcode = self._encode_ops(ops, GRBL_DIALECT)

        g1_lines = [line for line in gcode if line.startswith("G1")]
        assert len(g1_lines) == 2
        assert "F3000" in g1_lines[0]
        assert "F3000" in g1_lines[1]


def test_frequency_command_no_gcode_output():
    encoder = _make_encoder(GRBL_DIALECT)
    context = _make_context()

    ops = Ops()
    ops.set_frequency(1000)

    gcode = []
    encoder._handle_command(gcode, ops, 0, context)

    assert len(gcode) == 0
    assert encoder.frequency == 1000


def test_pulse_width_command_no_gcode_output():
    encoder = _make_encoder(GRBL_DIALECT)
    context = _make_context()

    ops = Ops()
    ops.set_pulse_width(50)

    gcode = []
    encoder._handle_command(gcode, ops, 0, context)

    assert len(gcode) == 0
    assert encoder.pulse_width == 50


def test_frequency_and_pulse_width_sets_state():
    encoder = _make_encoder(GRBL_DIALECT)
    context = _make_context()

    ops = Ops()
    ops.job_start()
    ops.set_power(1.0)
    ops.set_feed_rate(1000)
    ops.set_frequency(2000)
    ops.set_pulse_width(100)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.job_end()

    gcode = []
    _dispatch_all(gcode, ops, encoder, context)

    assert encoder.frequency == 2000
    assert encoder.pulse_width == 100


def test_encode_resets_frequency_and_pulse_width():
    encoder = _make_encoder(GRBL_DIALECT)
    encoder.frequency = 5000
    encoder.pulse_width = 200

    machine = MagicMock()
    machine.dialect = GRBL_DIALECT
    machine.gcode_precision = 3
    machine.max_travel_speed = 5000.0
    machine.active_wcs = "G54"
    machine.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.hookmacros = {}
    machine.heads = []
    doc = MagicMock()
    doc.find_descendant_by_uid.return_value = None

    ops = Ops()
    ops.job_start()
    ops.job_end()

    encoder.encode(ops, machine, doc)

    assert encoder.frequency is None
    assert encoder.pulse_width is None


class TestSpindle:
    def test_spindle_on_emits_m3_with_rpm(self):
        encoder = _make_encoder(GRBL_DIALECT)
        context = _make_context()

        ops = Ops()
        ops.set_spindle_rpm(12000)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == ["M3 S12000"]
        assert encoder.spindle_rpm == 12000

    def test_spindle_off_when_rpm_zero_after_on(self):
        encoder = _make_encoder(GRBL_DIALECT)
        encoder.spindle_rpm = 12000
        context = _make_context()

        ops = Ops()
        ops.set_spindle_rpm(0)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == ["M5"]
        assert encoder.spindle_rpm == 0

    def test_spindle_off_when_rpm_zero_and_already_off(self):
        encoder = _make_encoder(GRBL_DIALECT)
        encoder.spindle_rpm = 0
        context = _make_context()

        ops = Ops()
        ops.set_spindle_rpm(0)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == []

    def test_spindle_change_reissues_off_then_on(self):
        encoder = _make_encoder(GRBL_DIALECT)
        encoder.spindle_rpm = 8000
        context = _make_context()

        ops = Ops()
        ops.set_spindle_rpm(12000)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == ["M5", "M3 S12000"]
        assert encoder.spindle_rpm == 12000


class TestCoolant:
    def test_coolant_flood_emits_m8(self):
        encoder = _make_encoder(GRBL_DIALECT)
        context = _make_context()

        ops = Ops()
        ops.set_coolant(CoolantMode.FLOOD)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == ["M8"]
        assert encoder.coolant_mode == CoolantMode.FLOOD

    def test_coolant_mist_emits_m7(self):
        encoder = _make_encoder(GRBL_DIALECT)
        context = _make_context()

        ops = Ops()
        ops.set_coolant(CoolantMode.MIST)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == ["M7"]
        assert encoder.coolant_mode == CoolantMode.MIST

    def test_coolant_off_emits_m9(self):
        encoder = _make_encoder(GRBL_DIALECT)
        encoder.coolant_mode = CoolantMode.FLOOD
        context = _make_context()

        ops = Ops()
        ops.set_coolant(CoolantMode.OFF)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == ["M9"]
        assert encoder.coolant_mode == CoolantMode.OFF

    def test_coolant_noop_when_same_mode(self):
        encoder = _make_encoder(GRBL_DIALECT)
        encoder.coolant_mode = CoolantMode.FLOOD
        context = _make_context()

        ops = Ops()
        ops.set_coolant(CoolantMode.FLOOD)

        gcode = []
        encoder._handle_command(gcode, ops, 0, context)

        assert gcode == []


def test_job_end_shuts_off_spindle_and_coolant():
    encoder = _make_encoder(GRBL_DIALECT)
    encoder.spindle_rpm = 12000
    encoder.coolant_mode = CoolantMode.FLOOD
    context = _make_context()

    ops = Ops()
    ops.job_start()
    ops.job_end()

    gcode = []
    _dispatch_all(gcode, ops, encoder, context)

    assert "M5" in gcode
    assert "M9" in gcode
    assert encoder.spindle_rpm == 0
    assert encoder.coolant_mode == CoolantMode.OFF


def test_encode_resets_spindle_and_coolant():
    encoder = _make_encoder(GRBL_DIALECT)
    encoder.spindle_rpm = 12000
    encoder.coolant_mode = CoolantMode.FLOOD

    machine = MagicMock()
    machine.dialect = GRBL_DIALECT
    machine.gcode_precision = 3
    machine.max_travel_speed = 5000.0
    machine.active_wcs = "G54"
    machine.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.hookmacros = {}
    machine.heads = []
    doc = MagicMock()
    doc.find_descendant_by_uid.return_value = None

    ops = Ops()
    ops.job_start()
    ops.job_end()

    encoder.encode(ops, machine, doc)

    assert encoder.spindle_rpm == 0
    assert encoder.coolant_mode is None


def test_dialect_template_fields_include_spindle_coolant():
    varsets = GRBL_DIALECT.get_editor_varsets()
    keys = list(varsets["templates"].keys())
    assert "spindle_on_cw" in keys
    assert "spindle_on_ccw" in keys
    assert "spindle_off" in keys
    assert "coolant_flood" in keys
    assert "coolant_mist" in keys
    assert "coolant_off" in keys
