from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.state import PowerMode
from raygeo.template import CounterSettings

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workflow import Workflow
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.dialect.grbl import GRBL_DIALECT
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.macro import Macro, MacroTrigger
from rayforge.pipeline.encode_inputs import (
    build_encode_context,
    encode_gcode,
)
from rayforge.shared.units.system import UnitSystem


def _macro_table(machine, doc) -> dict:
    """The encoder's macro table, via the public context builder."""
    return build_encode_context(Ops(), machine, doc)["macros"]


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
    doc.layers = []
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
    doc.active_layer.add_child(wp)

    ops = Ops()
    context = build_encode_context(ops, machine, doc)

    assert context["path_vars"]["counter"] == "7"
    assert context["path_vars"]["job.counter"] == "7"
    assert context["workpiece_path_vars"][wp.uid]["workpiece.counter"] == "11"


def _make_layer(name: str, workflow_count: int = 1) -> Layer:
    layer = Layer(name)
    layer.add_child(WorkPiece(f"{name} part"))
    while len(layer.workflows) < workflow_count:
        layer.add_child(
            Workflow(f"{name} Workflow {len(layer.workflows) + 1}")
        )
    for workflow in layer.workflows:
        workflow.add_step(Step("Do it"))
    return layer


def _make_doc(*layers: Layer) -> Doc:
    doc = Doc()
    doc.set_children([])
    for layer in layers:
        doc.add_layer(layer)
    return doc


def _make_hooks() -> dict:
    return {
        MacroTrigger.LAYER_START: Macro(name="ls", code=["; LAYER_START"]),
        MacroTrigger.LAYER_END: Macro(name="le", code=["; LAYER_END"]),
        MacroTrigger.WORKFLOW_START: Macro(
            name="ws", code=["; WORKFLOW_START"]
        ),
        MacroTrigger.WORKFLOW_END: Macro(name="we", code=["; WORKFLOW_END"]),
    }


def test_macro_table_fires_layer_hooks_once_per_layer():
    machine = _make_machine_mock()
    machine.hookmacros = _make_hooks()
    first = _make_layer("First", workflow_count=2)
    second = _make_layer("Second")
    doc = _make_doc(first, second)
    wf1, wf2 = first.workflows
    (wf3,) = second.workflows

    table = _macro_table(machine, doc)
    starts = table["layer_start_by_uid"]
    ends = table["layer_end_by_uid"]

    assert starts[wf1.uid]["code"] == ["; LAYER_START", "; WORKFLOW_START"]
    assert starts[wf2.uid]["code"] == ["; WORKFLOW_START"]
    assert starts[wf3.uid]["code"] == ["; LAYER_START", "; WORKFLOW_START"]

    assert ends[wf1.uid]["code"] == ["; WORKFLOW_END"]
    assert ends[wf2.uid]["code"] == ["; WORKFLOW_END", "; LAYER_END"]
    assert ends[wf3.uid]["code"] == ["; WORKFLOW_END", "; LAYER_END"]

    # Unknown spans (framing) keep the layer-only fallback.
    assert table["layer_start"]["code"] == ["; LAYER_START"]
    assert table["layer_end"]["code"] == ["; LAYER_END"]


def test_macro_table_skips_workflows_without_runnable_steps():
    machine = _make_machine_mock()
    machine.hookmacros = _make_hooks()
    layer = _make_layer("Layer", workflow_count=3)
    wf1, wf2, wf3 = layer.workflows
    wf1.set_children([])
    wf3.enabled = False
    wf2.steps[0].workpiece_uids = ["missing"]
    doc = _make_doc(layer)

    table = _macro_table(machine, doc)
    starts = table["layer_start_by_uid"]
    ends = table["layer_end_by_uid"]

    for workflow in layer.workflows:
        assert workflow.uid not in starts
        assert workflow.uid not in ends


def test_macro_table_suppresses_hooks_for_workflows_with_no_hooks():
    machine = _make_machine_mock()
    layer = _make_layer("Layer")
    doc = _make_doc(layer)
    workflow = layer.workflows[0]

    table = _macro_table(machine, doc)

    assert table["layer_start"] is None
    assert table["layer_end"] is None
    assert table["layer_start_by_uid"] == {workflow.uid: None}
    assert table["layer_end_by_uid"] == {workflow.uid: None}


def test_encode_applies_hooks_at_layer_and_workflow_boundaries():
    machine = _make_machine_mock()
    machine.hookmacros = _make_hooks()
    layer = _make_layer("Layer", workflow_count=2)
    doc = _make_doc(layer)
    first, second = layer.workflows

    ops = Ops()
    ops.job_start()
    ops.set_head("head0")
    ops.set_power_mode(PowerMode.CONSTANT)
    ops.set_power(0.5)
    ops.set_feed_rate(1000)
    ops.layer_start(first.uid)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.layer_end(first.uid)
    ops.layer_start(second.uid)
    ops.move_to(0.0, 10.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.layer_end(second.uid)
    ops.job_end()

    text = encode_gcode(ops, machine, doc).text
    hooks = [
        line
        for line in text.splitlines()
        if line
        in {
            "; LAYER_START",
            "; LAYER_END",
            "; WORKFLOW_START",
            "; WORKFLOW_END",
        }
    ]

    assert hooks == [
        "; LAYER_START",
        "; WORKFLOW_START",
        "; WORKFLOW_END",
        "; WORKFLOW_START",
        "; WORKFLOW_END",
        "; LAYER_END",
    ]
