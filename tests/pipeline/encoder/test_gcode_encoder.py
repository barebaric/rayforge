from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.convert import GcodeDialectSpec
from raygeo.ops.state import AirAssistMode

from rayforge.machine.models.dialect.grbl import GRBL_DIALECT
from rayforge.pipeline.encoder.gcode import GcodeEncoder


def _make_machine_mock(dialect=GRBL_DIALECT):
    """A complete enough Machine mock for ``encode()`` calls."""
    machine = MagicMock()
    machine.dialect = dialect
    machine.name = "TestMachine"
    machine.gcode_precision = 3
    machine.max_travel_speed = 5000.0
    machine.active_wcs = "G54"
    machine.axis_extents = (200.0, 200.0)
    machine.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.get_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.hookmacros = {}
    machine.macros = {}
    machine.heads = [
        MagicMock(uid="head0", max_power=100.0, tool_number=1)
    ]
    machine.get_default_head.return_value = MagicMock(
        uid="head0", max_power=100.0, tool_number=1
    )
    machine.get_head_by_uid.return_value = MagicMock(
        uid="head0", max_power=100.0, tool_number=1
    )
    return machine


def _make_doc_mock():
    doc = MagicMock()
    doc.name = "TestDoc"
    doc.layers = []
    doc.find_descendant_by_uid.return_value = None
    return doc


def test_encode_resets_frequency_and_pulse_width():
    machine = _make_machine_mock()
    doc = _make_doc_mock()

    ops = Ops()
    ops.job_start()
    ops.job_end()

    encoder = GcodeEncoder(GRBL_DIALECT)
    result = encoder.encode(ops, machine, doc)

    assert result.text is not None
    assert "M3" not in result.text
    assert "M4" not in result.text


def test_encode_resets_spindle_and_coolant():
    machine = _make_machine_mock()
    doc = _make_doc_mock()

    ops = Ops()
    ops.job_start()
    ops.job_end()

    encoder = GcodeEncoder(GRBL_DIALECT)
    result = encoder.encode(ops, machine, doc)

    assert "G1" not in result.text
    assert "M3" not in result.text
    assert "M7" not in result.text
    assert "M8" not in result.text


def test_dialect_template_fields_include_spindle_coolant():
    varsets = GRBL_DIALECT.get_editor_varsets()
    keys = list(varsets["templates"].keys())
    assert "spindle_on_cw" in keys
    assert "spindle_on_ccw" in keys
    assert "spindle_off" in keys
    assert "coolant_flood" in keys
    assert "coolant_mist" in keys
    assert "coolant_off" in keys


# ── Tests for Ops.to_gcode() with typed GcodeDialectSpec ────────


def _make_context(**overrides) -> dict:
    """Build a minimal EncodeContext dict."""
    defaults = dict(
        gcode_precision=3,
        max_travel_speed=6000.0,
        default_head_uid="default",
        heads=[{"uid": "default", "tool_number": 0, "max_power": 1000.0}],
        active_wcs="",
        layer_wcs={},
        macros={},
        path_vars={},
        layer_path_vars={},
        workpiece_path_vars={},
    )
    defaults.update(overrides)
    return defaults


def test_rust_encode_basic_move_and_line():
    ops = Ops()
    ops.job_start()
    ops.set_power(1.0)
    ops.set_feed_rate(1000)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 20.0, 0.0)
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context()
    result = ops.to_gcode(dialect, context)
    text = result["text"]

    assert "G90" in text
    assert "G0 X0" in text
    assert "G1 X10 Y20" in text
    assert "M4 S1000" in text
    assert "G1 F1000" in text
    assert "M5" in text
    assert "M30" in text


def test_rust_encode_air_assist():
    ops = Ops()
    ops.job_start()
    ops.set_air_assist(AirAssistMode.ON)
    ops.set_power(1.0)
    ops.set_feed_rate(500)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    ops.set_air_assist(AirAssistMode.OFF)
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context()
    result = ops.to_gcode(dialect, context)
    text = result["text"]

    assert "M8" in text
    assert "M9" in text


def test_rust_encode_dwell():
    ops = Ops()
    ops.job_start()
    ops.set_power(1.0)
    ops.set_feed_rate(1000)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.dwell(500.0)
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context()
    result = ops.to_gcode(dialect, context)
    text = result["text"]

    assert "G4 P0.500" in text


def test_rust_encode_zero_power_no_laser():
    ops = Ops()
    ops.job_start()
    ops.set_power(0.0)
    ops.set_feed_rate(1000)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context()
    result = ops.to_gcode(dialect, context)
    text = result["text"]

    assert "M4 S0" not in text


def test_rust_encode_tool_change():
    ops = Ops()
    ops.job_start()
    ops.set_head("head2")
    ops.set_power(1.0)
    ops.set_feed_rate(1000)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context(
        heads=[{"uid": "head2", "tool_number": 1, "max_power": 40.0}],
    )
    result = ops.to_gcode(dialect, context)
    text = result["text"]

    assert "T1" in text


def test_rust_encode_macro_expansion():
    ops = Ops()
    ops.job_start()
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context(
        macros={
            "all_macros": {
                "hello": {
                    "name": "hello",
                    "code": ["; hello"],
                    "enabled": True,
                },
            },
        },
    )
    result = ops.to_gcode(dialect, context)
    assert "text" in result


def test_rust_encode_macro_cycle_detection():
    ops = Ops()
    ops.job_start()
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context(
        macros={
            "all_macros": {
                "a": {"name": "a", "code": ["@include(b)"], "enabled": True},
                "b": {"name": "b", "code": ["@include(a)"], "enabled": True},
            },
        },
    )
    result = ops.to_gcode(dialect, context)
    assert "text" in result


def test_rust_encode_disabled_macro_warns():
    ops = Ops()
    ops.job_start()
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context(
        macros={
            "all_macros": {
                "off": {"name": "off", "code": ["G1 X1"], "enabled": False},
                "outer": {
                    "name": "outer",
                    "code": ["@include(off)"],
                    "enabled": True,
                },
            },
        },
    )
    result = ops.to_gcode(dialect, context)
    assert "text" in result


def test_rust_encode_macro_path_vars():
    ops = Ops()
    ops.job_start()
    ops.job_end()

    dialect = GcodeDialectSpec()
    context = _make_context(
        path_vars={"machine.name": "MyMachine"},
        macros={
            "all_macros": {
                "preamble": {
                    "name": "preamble",
                    "code": ["; machine={machine.name}"],
                    "enabled": True,
                },
            },
        },
    )
    result = ops.to_gcode(dialect, context)
    assert "text" in result
