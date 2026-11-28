from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.state import PowerMode
from raygeo.template import CounterSettings

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.dialect.grbl import GRBL_DIALECT
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.encode_inputs import (
    build_encode_context,
    encode_gcode,
)
from rayforge.shared.units.system import UnitSystem


def _make_machine_mock(dialect=GRBL_DIALECT):
    """A complete enough Machine mock for ``encode_gcode()`` calls."""
    machine = MagicMock()
    machine.dialect = dialect
    machine.name = "TestMachine"
    machine.gcode_precision = 3
    machine.max_travel_speed = 5000.0
    machine.unit_system = UnitSystem.METRIC
    machine.active_wcs = "G54"
    machine.axis_extents = (200.0, 200.0)
    machine.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.get_wcs_offset.return_value = (0.0, 0.0, 0.0)
    machine.get_job_power_cap.return_value = None
    machine.hookmacros = {}
    machine.macros = {}
    head = LaserHead()
    head.uid = "head0"
    head.tool_number = 1
    machine.heads = [head]
    machine.get_default_head.return_value = head
    machine.get_head_by_uid.return_value = head
    machine.has_z_axis = True
    return machine


def _make_doc_mock():
    doc = MagicMock()
    doc.name = "TestDoc"
    doc.setups = []
    doc.find_descendant_by_uid.return_value = None
    return doc


def test_dialect_template_fields_include_spindle_coolant():
    varsets = GRBL_DIALECT.get_editor_varsets()
    keys = list(varsets["templates"].keys())
    assert "spindle_on_cw" in keys
    assert "spindle_on_ccw" in keys
    assert "spindle_off" in keys
    assert "coolant_flood" in keys
    assert "coolant_mist" in keys
    assert "coolant_off" in keys


def test_encodes_frame_opstream():
    """The frame's synthetic job/layer/workpiece structure encodes.

    The frame layer uid does not exist in the doc; the encoder must
    still emit the trace with constant power (M3) and the frame speed.
    """
    machine = _make_machine_mock()
    doc = _make_doc_mock()

    ops = Ops()
    ops.job_start()
    ops.layer_start("rayforge-frame")
    ops.workpiece_start("rayforge-frame-workpiece")
    ops.set_head("head0")
    ops.set_power_mode(PowerMode.CONSTANT)
    ops.set_power(0.1)
    ops.set_feed_rate(3000)
    ops.move_to(10.0, 20.0, 0.0)
    ops.line_to(10.0, 40.0, 0.0)
    ops.line_to(30.0, 40.0, 0.0)
    ops.line_to(30.0, 20.0, 0.0)
    ops.line_to(10.0, 20.0, 0.0)
    ops.workpiece_end("rayforge-frame-workpiece")
    ops.layer_end("rayforge-frame")
    ops.job_end()

    encoded = encode_gcode(ops, machine, doc)

    assert "M3" in encoded.text
    assert "F3000" in encoded.text
    assert "G1 X30" in encoded.text


def test_build_encode_context_includes_counter():
    machine = _make_machine_mock()
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=7, step=2)
    wp = WorkPiece("part")
    wp.variable_offset = 2
    doc.active_setup.add_child(wp)

    ops = Ops()
    context = build_encode_context(ops, machine, doc)

    assert context["path_vars"]["counter"] == "7"
    assert context["path_vars"]["job.counter"] == "7"
    assert context["workpiece_path_vars"][wp.uid]["workpiece.counter"] == "11"
