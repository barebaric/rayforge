"""
Tests for geometry-less (Command) steps in the intent builder.

Covers the wiring added for the Command step (issue #449): a step
with ``needs_workpieces == False`` gets a single compute node that
runs once at its workflow position, with or without workpieces in
the layer, and its custom lines land in the job ops at exactly that
position.
"""

from raygeo.pipeline.execute import execute_stages

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
    command_key,
    job_key,
    job_machinexform_key,
    step_key,
)


class _GeometryStep(Step):
    """A regular per-workpiece step (mirrors test_intent_builder)."""

    def __init__(self, name: str = "geometry"):
        super().__init__(typelabel="geometry", name=name)


class _CommandLikeStep(Step):
    """Minimal geometry-less step for builder-level tests."""

    needs_workpieces = False

    def __init__(self, name: str = "command", lines: list[str] | None = None):
        super().__init__(typelabel="command", name=name)
        self.lines = lines or ["M101"]

    def get_cache_params(self):
        params = super().get_cache_params()
        params["command_text"] = "\n".join(self.lines)
        return params

    def build_command_payload(self, machine):
        from raygeo.cnc.execution.specs import ComputePayload
        from raygeo.ops.assembly import Assembler
        from raygeo.ops.assembly.custom import CustomSpec
        from raygeo.ops.part import Part

        return Part(size_mm=(0.0, 0.0)), ComputePayload(
            assembler=Assembler(CustomSpec(lines=self.lines))
        )


def _make_doc(*items) -> Doc:
    doc = Doc()
    layer = doc.active_layer
    wf = layer.workflow
    assert wf is not None
    for item in items:
        if isinstance(item, Step):
            wf.add_child(item)
        else:
            layer.add_child(item)
    return doc


def _run_nodes(nodes):
    completed = []
    execute_stages(nodes, completed.append, None)
    return {c.key: c for c in completed}


# ----------------------------------------------------------------------
# Node keys
# ----------------------------------------------------------------------


def test_command_step_without_workpieces_emits_job(isolated_machine):
    step = _CommandLikeStep()
    doc = _make_doc(step)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert command_key(step.uid) in keys
    assert step_key(step.uid) in keys
    assert job_key() in keys
    assert job_machinexform_key() in keys
    # No workpieces: exactly one compute node for the command step.
    assert len([k for k in keys if k.startswith("command:")]) == 1
    assert not any(k.startswith("workpiece:") for k in keys)


def test_command_step_with_workpieces_runs_once(isolated_machine):
    step = _CommandLikeStep()
    doc = _make_doc(step, WorkPiece(name="wp1"), WorkPiece(name="wp2"))

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    # One command compute node regardless of workpiece count.
    assert len([k for k in keys if k.startswith("command:")]) == 1


def test_doc_with_only_workpiece_steps_and_no_workpieces_is_empty(
    isolated_machine,
):
    doc = _make_doc(_GeometryStep())
    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    assert nodes == []


# ----------------------------------------------------------------------
# Job ops ordering
# ----------------------------------------------------------------------


def test_command_lines_keep_workflow_position(isolated_machine):
    """The command runs between the two neighboring steps."""
    before = _CommandLikeStep(name="before", lines=["; BEFORE"])
    cmd = _CommandLikeStep(name="cmd", lines=["M101"])
    after = _CommandLikeStep(name="after", lines=["; AFTER"])
    doc = _make_doc(before, cmd, after)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    completed = _run_nodes(nodes)

    ops = completed[job_key()].output.ops
    texts = [
        ops.custom_text(i)
        for i in range(ops.len())
        if ops.command_type(i).name == "CUSTOM"
    ]
    # LayerStart/JobStart markers wrap the content; the three steps'
    # lines must appear in workflow order.
    assert texts == ["; BEFORE", "M101", "; AFTER"]


def test_command_only_doc_produces_custom_ops(isolated_machine):
    step = _CommandLikeStep(lines=["M101", "M102"])
    doc = _make_doc(step)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    completed = _run_nodes(nodes)
    ops = completed[job_key()].output.ops

    texts = [
        ops.custom_text(i)
        for i in range(ops.len())
        if ops.command_type(i).name == "CUSTOM"
    ]
    assert texts == ["M101", "M102"]


def test_command_text_change_invalidates_compute_token(isolated_machine):
    step = _CommandLikeStep(lines=["M101"])
    doc = _make_doc(step)
    builder = IntentBuilder(machine=isolated_machine)

    n1 = builder.build(doc)
    step.lines = ["M102"]
    n2 = builder.build(doc)

    token1 = {n.key: n.version_token for n in n1}
    token2 = {n.key: n.version_token for n in n2}
    assert token1[command_key(step.uid)] != token2[command_key(step.uid)]
    assert token1[step_key(step.uid)] != token2[step_key(step.uid)]


def test_command_lines_reach_gcode_output(isolated_machine):
    """End to end: the custom lines survive the full pipeline and
    appear in the encoded G-code text."""
    from rayforge.pipeline.intent_builder import job_encode_key

    step = _CommandLikeStep(lines=["M101", "; {machine.name}"])
    doc = _make_doc(step)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    completed = _run_nodes(nodes)

    encoded = completed[job_encode_key()]
    assert encoded.output is not None
    text = encoded.output.text
    assert "M101" in text.splitlines()
