"""
Tests for :mod:`rayforge.pipeline.intent_builder`.

These tests build a small :class:`~rayforge.core.doc.Doc` with a
single step (and two workpieces) and verify the keys, version tokens,
and the position-sensitive folding rule.
"""

import math
from unittest.mock import MagicMock

import pytest
from raygeo.cnc.execution.intent import create_intent_from_nodes
from raygeo.cnc.execution.specs import (
    AggregateGroup,
    AggregateInput,
    AggregateSpec,
    ComputePayload,
    EncodeSpec,
    MachineParams,
    MachineTransformSpec,
    Marker,
    RotaryMappingSpec,
)
from raygeo.geo import Geometry, Matrix
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.axis import Axis
from raygeo.ops.convert import Encoder, GcodeSpec
from raygeo.ops.part import Part
from raygeo.pipeline.execute import execute_stages
from raygeo.pipeline.request import NodeRequest
from raygeo.pipeline.stage import StageSpec

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.dialect.grbl import GRBL_DIALECT
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.models.rotary_module import RotaryMode, RotaryModule
from rayforge.pipeline.encoder.rust_helpers import dialect_to_spec
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
    job_encode_key,
    job_key,
    job_machinexform_key,
    step_key,
    workpiece_key,
)


class _TestStep(Step):
    """
    Concrete ``Step`` subclass for tests.  ``Step`` is normally
    subclassed by addons (ContourStep, FrameStep, ...) but it is
    fully instantiable on its own and exposes every attribute the
    IntentBuilder consumes.  The position-sensitive flag is
    overridable per-instance via ``_position_sensitive``.
    """

    def __init__(self, name: str = "test", position_sensitive: bool = False):
        super().__init__(typelabel="test", name=name)
        self._position_sensitive = position_sensitive

    def is_position_sensitive(self) -> bool:
        return self._position_sensitive


def _make_doc(step: _TestStep, *workpieces: WorkPiece) -> Doc:
    doc = Doc()
    layer = doc.active_layer
    wf = layer.workflow
    assert wf is not None
    wf.add_child(step)
    for wp in workpieces:
        layer.add_child(wp)
    return doc


# ----------------------------------------------------------------------
# Key stability
# ----------------------------------------------------------------------


def test_builds_one_node_per_workpiece_plus_step_and_job():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    wp2 = WorkPiece(name="wp2")
    doc = _make_doc(step, wp1, wp2)

    nodes = IntentBuilder().build(doc)
    keys = [n.key for n in nodes]
    assert workpiece_key(wp1.uid, step.uid) in keys
    assert workpiece_key(wp2.uid, step.uid) in keys
    assert step_key(step.uid) in keys
    assert job_key() in keys
    assert len(keys) == 4


def test_keys_are_stable_across_rebuilds():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    n1 = IntentBuilder().build(doc)
    n2 = IntentBuilder().build(doc)
    assert [n.key for n in n1] == [n.key for n in n2]


# ----------------------------------------------------------------------
# Token stability
# ----------------------------------------------------------------------


def test_version_tokens_stable_across_rebuilds():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    n1 = IntentBuilder().build(doc)
    n2 = IntentBuilder().build(doc)
    tokens_1 = {n.key: n.version_token for n in n1}
    tokens_2 = {n.key: n.version_token for n in n2}
    assert tokens_1 == tokens_2
    for t in tokens_1.values():
        assert isinstance(t, int)
        assert t > 0


def test_compute_token_changes_on_geometry_revision_bump():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    wp1.updated.send(wp1)  # bumps geometry_revision
    after = IntentBuilder().build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


def test_compute_token_changes_on_step_param_change():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    step.cut_speed = 999
    after = IntentBuilder().build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


def test_compute_token_changes_on_workpiece_transformer_change():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    step.per_workpiece_transformers_dicts.append(
        {"name": "Optimize", "enabled": True}
    )
    after = IntentBuilder().build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# POSITION_SENSITIVE folding
# ----------------------------------------------------------------------


def test_transform_revision_omitted_when_not_position_sensitive():
    """
    A pure move must NOT change the workpiece compute token when the
    step is not position-sensitive.
    """
    step = _TestStep(name="s1", position_sensitive=False)
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    wp1.transform_changed.send(wp1, old_matrix=wp1.matrix)
    assert wp1.geometry_revision == 0
    assert wp1.transform_revision == 1
    after = IntentBuilder().build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t == after_t


def test_transform_revision_folded_when_position_sensitive():
    """
    When the step is position-sensitive a move must change the workpiece
    compute token.
    """
    step = _TestStep(name="s1", position_sensitive=True)
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    wp1.transform_changed.send(wp1, old_matrix=wp1.matrix)
    assert wp1.transform_revision == 1
    after = IntentBuilder().build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# Aggregate tokens
# ----------------------------------------------------------------------


def test_step_aggregate_token_changes_on_step_param_change():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    step.cut_speed = 2222
    after = IntentBuilder().build(doc)

    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_step_aggregate_token_changes_on_upstream_token_change():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    wp1.updated.send(wp1)  # bump upstream compute token
    after = IntentBuilder().build(doc)

    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_step_aggregate_token_changes_on_per_step_transformer_change():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    step.per_step_transformers_dicts.append(
        {"name": "Smooth", "enabled": True}
    )
    after = IntentBuilder().build(doc)

    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_job_token_changes_on_step_param_change():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder().build(doc)
    step.cut_speed = 7777
    after = IntentBuilder().build(doc)

    jk = job_key()
    before_t = next(n.version_token for n in before if n.key == jk)
    after_t = next(n.version_token for n in after if n.key == jk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# Visibility / structural filtering
# ----------------------------------------------------------------------


def test_hidden_steps_excluded():
    step = _TestStep(name="s1")
    step.visible = False
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    keys = [n.key for n in nodes]

    assert workpiece_key(wp1.uid, step.uid) not in keys
    assert step_key(step.uid) not in keys
    assert job_key() in keys


def test_layers_without_workpieces_skipped_for_compute():
    step = _TestStep(name="s1")
    doc = Doc()
    layer = doc.active_layer
    workflow = layer.workflow
    assert workflow is not None
    workflow.add_child(step)
    # No workpieces added.

    nodes = IntentBuilder().build(doc)
    keys = [n.key for n in nodes]
    assert step_key(step.uid) not in keys
    assert workpiece_key("any", step.uid) not in keys
    assert job_key() in keys


def test_second_layer_without_skips_affect_existing():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    layer2 = Layer(name="empty")
    doc.add_child(layer2)
    layer2.add_child(WorkPiece(name="wp2"))

    nodes = IntentBuilder().build(doc)
    keys = [n.key for n in nodes]
    # Only one step aggregate node and one job node.
    assert keys.count(step_key(step.uid)) == 1
    assert keys.count(job_key()) == 1


# ----------------------------------------------------------------------
# Generation ID and stage payload
# ----------------------------------------------------------------------


def test_generation_id_propagated_to_nodes():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(generation_id=42).build(doc)
    assert all(n.generation_id == 42 for n in nodes)


def test_stage_payload_is_valid_raygeo_stagespec():
    """The IntentBuilder's stage payloads must be real raygeo StageSpec
    instances so ``create_intent_from_nodes`` accepts them."""
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)

    # All stages must be StageSpec.Compute or StageSpec.Aggregate.
    for n in nodes:
        assert isinstance(n.stage, (StageSpec.Compute, StageSpec.Aggregate))

    # create_intent_from_nodes must succeed (validates the StageSpec).
    intent = create_intent_from_nodes(nodes)
    assert intent is not None
    assert intent.step_count == 2  # one compute + one step aggregate


def test_stage_wpspec_for_workpiece_node_is_compute():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    wp_node = next(
        n for n in nodes if n.key == workpiece_key(wp1.uid, step.uid)
    )
    assert isinstance(wp_node.stage, StageSpec.Compute)


def test_stage_step_node_is_aggregate():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    st_node = next(n for n in nodes if n.key == step_key(step.uid))
    assert isinstance(st_node.stage, StageSpec.Aggregate)


def test_stage_job_node_is_aggregate():
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    job_node = next(n for n in nodes if n.key == job_key())
    assert isinstance(job_node.stage, StageSpec.Aggregate)


# ----------------------------------------------------------------------
# Contour compute spec wiring (B2)
# ----------------------------------------------------------------------


def test_contour_workpiece_node_uses_contour_spec(
    contour_step_class, test_machine_and_config
):
    """Contour workpiece compute nodes must carry a ContourSpec assembler
    populated from the step's resolved assembler kwargs and machine
    defaults."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    wp_node = next(n for n in nodes if n.key == wpk)
    assert isinstance(wp_node.stage, StageSpec.Compute)
    payload = wp_node.stage.params
    assert isinstance(payload.assembler, Assembler)
    assert isinstance(payload.assembler.spec, ContourSpec)


def test_contour_spec_reflects_step_params(
    contour_step_class, test_machine_and_config
):
    """Changing the step's ``cut_side`` is reflected in the ContourSpec
    and in the compute version token."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    step.cut_side = "outside"
    step.path_offset_mm = 0.5
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    wp_node = next(n for n in nodes if n.key == wpk)
    spec = wp_node.stage.params.assembler.spec
    assert spec.cut_side == "outside"
    assert spec.path_offset_mm == 0.5

    # The compute token must change when the contour params change.
    step.cut_side = "inside"
    after = IntentBuilder(machine=machine).build(doc)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert after_t != wp_node.version_token


def test_compute_token_changes_on_machine_arc_tolerance(
    contour_step_class, test_machine_and_config
):
    """A machine-level default (arc_tolerance) folds into the compute
    token via the resolved assembler kwargs, so a machine swap
    invalidates contour workpiece caches."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    before = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)

    machine.arc_tolerance = 0.25
    after = IntentBuilder(machine=machine).build(doc)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


def test_contour_compute_node_executes_through_raygeo(
    contour_step_class, test_machine_and_config
):
    """The contour compute stage built by IntentBuilder must run end-to-end
    through ``execute_stages`` and produce a non-empty Ops output."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")

    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()

    wp = WorkPiece(name="rect")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    sk = step_key(step.uid)

    completed = []

    def on_completed(node):
        completed.append(node)

    execute_stages(nodes, on_completed)

    wp_result = next(c for c in completed if c.key == wpk)
    assert wp_result.error is None, wp_result.error
    assert wp_result.output is not None
    assert wp_result.output.ops.len() > 0

    # The step aggregate must also run and concatenate the workpiece
    # ops with placement + workpiece start/end markers.
    step_result = next(c for c in completed if c.key == sk)
    assert step_result.error is None, step_result.error
    assert step_result.output is not None
    assert step_result.output.ops.len() >= wp_result.output.ops.len()


def test_step_aggregate_carries_workpiece_placement(
    contour_step_class, test_machine_and_config
):
    """The step aggregate's AggregateInput must carry the workpiece's
    world placement (translation + rotation, scale normalised) and the
    workpiece's physical size as target_dimensions."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")

    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()

    wp = WorkPiece(name="rect")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    wp.pos = 10.0, 20.0
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    sk = step_key(step.uid)
    step_node = next(n for n in nodes if n.key == sk)
    assert isinstance(step_node.stage, StageSpec.Aggregate)
    group = step_node.stage.spec.groups[0]
    inp = group.inputs[0]
    assert inp.source_key == workpiece_key(wp.uid, step.uid)
    assert inp.target_dimensions == (50.0, 30.0)

    # Placement matrix should carry the translation (10, 20) but not
    # the absolute scale (scale normalised to 1).
    tx = inp.placement_matrix[0][3]
    ty = inp.placement_matrix[1][3]
    assert (tx, ty) == (10.0, 20.0)
    sx = inp.placement_matrix[0][0]
    sy = inp.placement_matrix[1][1]
    assert abs(sx - 1.0) < 1e-9
    assert abs(abs(sy) - 1.0) < 1e-9


def test_step_aggregate_machine_params_from_machine(
    contour_step_class, test_machine_and_config
):
    """The step aggregate's MachineParams is populated from the
    resolved machine so the aggregate's time estimate is correct."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    sk = step_key(step.uid)
    step_node = next(n for n in nodes if n.key == sk)
    mp = step_node.stage.spec.machine
    assert mp.default_feed_rate == float(machine.max_cut_speed)
    assert mp.default_rapid_rate == float(machine.max_travel_speed)
    assert mp.acceleration == float(machine.acceleration)


def test_job_aggregate_has_layer_and_job_markers(
    contour_step_class, test_machine_and_config
):
    """The job aggregate wraps layers with LayerStart/End and the whole
    job with JobStart/End."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    job_node = next(n for n in nodes if n.key == job_key())
    assert isinstance(job_node.stage, StageSpec.Aggregate)
    spec = job_node.stage.spec
    assert len(spec.wrap_start) == 1
    assert len(spec.wrap_end) == 1
    # One group per layer (single layer here).
    assert len(spec.groups) == 1
    group = spec.groups[0]
    assert len(group.start_markers) == 1
    assert len(group.end_markers) == 1


def test_job_encode_node_emits_encode_spec(
    contour_step_class, test_machine_and_config
):
    """The IntentBuilder appends a job encode node carrying an
    EncodeSpec (Compute stage wrapping an encoder) after the job
    aggregate."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    ek = job_encode_key()
    encode_nodes = [n for n in nodes if n.key == ek]
    assert len(encode_nodes) == 1
    assert isinstance(encode_nodes[0].stage, EncodeSpec)
    assert encode_nodes[0].stage.source_key == job_machinexform_key()


def test_job_encode_token_changes_on_machine_swap(
    contour_step_class, test_machine_and_config
):
    """A machine-level change (gcode_precision) invalidates the encode
    token."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    before = IntentBuilder(machine=machine).build(doc)
    ek = job_encode_key()
    before_t = next(n.version_token for n in before if n.key == ek)

    machine.gcode_precision = 5
    after = IntentBuilder(machine=machine).build(doc)
    after_t = next(n.version_token for n in after if n.key == ek)
    assert before_t != after_t


def test_contour_job_encodes_through_raygeo(
    contour_step_class, test_machine_and_config
):
    """A contour-only document runs end-to-end through execute_stages:
    workpiece compute → step aggregate → job aggregate → encode,
    producing G-code machine output."""
    machine, context = test_machine_and_config
    machine.hydrate()

    step = contour_step_class.create(context, name="cut")

    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()

    wp = WorkPiece(name="rect")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    ek = job_encode_key()

    completed = []

    def on_completed(node):
        completed.append(node)

    execute_stages(nodes, on_completed)

    enc_result = next(c for c in completed if c.key == ek)
    assert enc_result.error is None, enc_result.error
    assert enc_result.output is not None
    # The MachineCode variant carries non-empty G-code text.
    assert enc_result.output.text is not None
    assert len(enc_result.output.text) > 0


def test_machine_transform_node_present(
    contour_step_class, test_machine_and_config
):
    """The IntentBuilder emits a MachineTransformSpec node between the
    job aggregate and the encoder."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    keys = [n.key for n in nodes]
    job_idx = keys.index(job_key())
    mx_idx = keys.index(job_machinexform_key())
    enc_idx = keys.index(job_encode_key())
    assert job_idx < mx_idx < enc_idx


def test_machine_transform_linearizes_curves(
    contour_step_class, test_machine_and_config
):
    """The machine-transform stage linearizes Bezier curves when the
    machine does not support curves."""
    machine, context = test_machine_and_config
    machine.set_supports_curves(False)
    step = contour_step_class.create(context, name="cut")

    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.bezier_to(0.0, 10.0, 10.0, 0.0, 10.0, 10.0)
    geo.close_path()

    wp = WorkPiece(name="curve")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    ek = job_encode_key()
    completed = []
    execute_stages(nodes, lambda n: completed.append(n))

    enc_result = next(c for c in completed if c.key == ek)
    assert enc_result.error is None, enc_result.error
    assert enc_result.output is not None
    gcode = enc_result.output.text
    assert "G5 " not in gcode
    cut_lines = [ln for ln in gcode.split("\n") if ln.startswith("G1")]
    assert len(cut_lines) > 0


def test_machine_transform_true_4th_axis_rotary():
    """TRUE_4TH_AXIS rotary mapping converts world-space Y to A-axis
    degrees.  A 10 mm Y movement at 25 mm diameter produces
    45.837 degrees."""
    part = Part.from_polygons(
        [(0, 0), (10, 0), (10, 10), (0, 10)], size_mm=(10, 10)
    )
    completed = _build_rotary_pipeline(
        part, 25.0, axis="A", mode="true_4th_axis"
    )

    enc = completed.get("enc")
    assert enc is not None
    assert enc.error is None, enc.error
    text = enc.output.text

    degrees = (10.0 / (25.0 * math.pi)) * 360.0
    formatted = f"{degrees:.3f}".rstrip("0").rstrip(".")
    assert f"A{formatted}" in text


def test_machine_transform_rotary_extra_axes(
    contour_step_class, test_machine_and_config
):
    """The machine transform stage stores rotary degree values in
    extra_axes for TRUE_4TH_AXIS mode."""
    machine, context = test_machine_and_config
    machine.hydrate()

    rm = RotaryModule()
    rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
    rm.set_axis(Axis.A)
    machine.add_rotary_module(rm)

    step = contour_step_class.create(context, name="cut")
    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()
    wp = WorkPiece(name="wp")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    layer = doc.active_layer
    layer.set_rotary_enabled(True)
    layer.set_rotary_diameter(25.0)
    layer.set_rotary_module_uid(rm.uid)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    completed = {}
    execute_stages(nodes, lambda n: completed.__setitem__(n.key, n))

    mx_node = completed.get(job_machinexform_key())
    assert mx_node is not None
    assert mx_node.error is None, mx_node.error
    assert mx_node.output is not None
    ops = mx_node.output.ops

    # At least one moving command should have A-axis extra_axes.
    found_a = False
    for i in range(ops.len()):
        ea = ops.extra_axes(i)
        if ea is not None:
            for axis in ea:
                if axis.name == "A":
                    found_a = True
                    assert abs(ea[axis]) > 0.0
    assert found_a, "expected A-axis extra_axes in machine transform output"


def _build_rotary_pipeline(
    part, diameter, axis="A", mode="true_4th_axis", mu_per_rotation=0.0
):
    """Helper: create a 4-node pipeline (compute → aggregate → mxform
    → encode) that feeds known geometry through the machine-transform
    stage with a rotary module and encodes the result."""
    cs = ContourSpec(
        cut_side="centerline",
        cut_order="inside_outside",
        remove_inner=False,
        kerf_mm=0.0,
        path_offset_mm=0.0,
        overcut=0.0,
    )
    assembler = Assembler(cs)
    payload = ComputePayload(assembler=assembler)

    compute_node = NodeRequest(
        key="src",
        generation_id=1,
        stage=StageSpec.Compute(part=part, params=payload),
        version_token=0,
    )

    agg_spec = AggregateSpec(
        wrap_start=[Marker.JobStart(_tag=True)],
        groups=[
            AggregateGroup(
                start_markers=[Marker.LayerStart(uid="test", _tag=True)],
                inputs=[
                    AggregateInput(
                        source_key="src",
                        placement_matrix=[
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    )
                ],
                end_markers=[Marker.LayerEnd(uid="test", _tag=True)],
            )
        ],
        wrap_end=[Marker.JobEnd(_tag=True)],
        machine=MachineParams(),
    )
    agg_node = NodeRequest(
        key="agg",
        generation_id=1,
        stage=StageSpec.Aggregate(spec=agg_spec),
        version_token=0,
    )

    rotary = RotaryMappingSpec(
        layer_uid="test",
        diameter=diameter,
        gear_ratio=1.0,
        reverse=False,
        axis_position_3d=[0, 0, 0],
        cylinder_dir=[1, 0, 0],
        rotary_axis=axis,
        replaced_axis=None if mode == "true_4th_axis" else axis,
        mu_per_rotation=mu_per_rotation,
    )

    mt_spec = MachineTransformSpec(
        source_key="agg",
        linearize_curves=False,
        world_to_machine=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        default_wcs_offset=[0, 0, 0],
        layer_wcs_offsets=[],
        reverse_z=False,
        rotary_mappings=[rotary],
    )
    mt_node = NodeRequest(
        key="mx",
        generation_id=1,
        stage=mt_spec,
        version_token=0,
    )

    mm = MagicMock()
    mm.gcode_precision = 3
    dialect_spec = dialect_to_spec(GRBL_DIALECT, mm)
    context_json = "{}"
    gcode_spec = GcodeSpec(dialect=dialect_spec, context_json=context_json)
    encoder = Encoder(gcode_spec)

    enc_node = NodeRequest(
        key="enc",
        generation_id=1,
        stage=EncodeSpec(source_key="mx", encoder=encoder),
        version_token=0,
    )

    completed = {}
    execute_stages(
        [compute_node, agg_node, mt_node, enc_node],
        lambda n: completed.__setitem__(n.key, n),
    )
    return completed


def test_machine_transform_rotary_replaces_y_for_axis_replacement(
    contour_step_class, test_machine_and_config
):
    """For AXIS_REPLACEMENT, the machine transform stage clears
    extra_axes and writes scaled-mu values into the endpoint Y."""
    machine, context = test_machine_and_config
    machine.hydrate()

    rm = RotaryModule()
    rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
    rm.set_mm_per_rotation(100.0)
    rm.set_axis(Axis.A)
    machine.add_rotary_module(rm)

    step = contour_step_class.create(context, name="cut")
    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()
    wp = WorkPiece(name="wp")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    layer = doc.active_layer
    layer.set_rotary_enabled(True)
    layer.set_rotary_diameter(25.0)
    layer.set_rotary_module_uid(rm.uid)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    completed = {}
    execute_stages(nodes, lambda n: completed.__setitem__(n.key, n))

    mx_node = completed.get(job_machinexform_key())
    assert mx_node is not None
    assert mx_node.error is None, mx_node.error
    assert mx_node.output is not None
    ops = mx_node.output.ops

    # For AXIS_REPLACEMENT, extra_axes should be cleared after
    # the downstream pass.  Each moving command's Y position is
    # the scaled-mu value.
    for i in range(ops.len()):
        ea = ops.extra_axes(i)
        if ea is not None:
            for axis in ea:
                assert axis.name not in ("A", "Y"), (
                    f"unexpected extra_axes after AXIS_REPLACEMENT: "
                    f"{axis.name}={ea[axis]}"
                )


def test_machine_transform_axis_replacement():
    """AXIS_REPLACEMENT converts world-space Y to scaled-mu.  A 10 mm
    Y movement at 25 mm diameter with 100 mu/rotation produces
    Y=12.732 in the G-code output."""
    part = Part.from_polygons(
        [(0, 0), (10, 0), (10, 10), (0, 10)], size_mm=(10, 10)
    )
    completed = _build_rotary_pipeline(
        part,
        25.0,
        axis="Y",
        mode="axis_replacement",
        mu_per_rotation=100.0,
    )

    enc = completed.get("enc")
    assert enc is not None
    assert enc.error is None, enc.error
    text = enc.output.text

    degrees = (10.0 / (25.0 * math.pi)) * 360.0
    scaled = degrees * 100.0 / 360.0
    formatted = f"{scaled:.3f}".rstrip("0").rstrip(".")
    assert f"Y{formatted}" in text


# ----------------------------------------------------------------------
# Post-process transformer wiring (regression: raster ops too small,
# overscan missing, post-processors had no effect on Raster)
# ----------------------------------------------------------------------


def test_contour_workpiece_node_carries_per_workpiece_transformers(
    contour_step_class, test_machine_and_config
):
    """Per-workpiece transformer dicts must be resolved into typed
    Rust specs and attached to the workpiece compute node's
    ComputePayload so that the Rust compute stage applies them after
    assembly."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    wp.set_size(10.0, 10.0)
    doc = _make_doc(step, wp)

    # The default contour step ships with several per-workpiece
    # transformers (Optimize, LeadInOut, Tabs, ...). At least one
    # must be wired.
    assert len(step.per_workpiece_transformers_dicts) > 0

    nodes = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    wp_node = next(n for n in nodes if n.key == wpk)
    assert isinstance(wp_node.stage, StageSpec.Compute)
    payload = wp_node.stage.params
    assert len(payload.transformers) > 0


def test_step_aggregate_carries_per_step_transformers(
    contour_step_class, test_machine_and_config
):
    """Per-step transformer dicts must be resolved into typed Rust
    specs and attached to the step aggregate's AggregateSpec so that
    the Rust aggregate stage applies them after concatenation."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    wp.set_size(10.0, 10.0)
    doc = _make_doc(step, wp)

    # The default contour step ships with per-step transformers
    # (Optimize, MultiPass).
    assert len(step.per_step_transformers_dicts) > 0

    nodes = IntentBuilder(machine=machine).build(doc)
    sk = step_key(step.uid)
    step_node = next(n for n in nodes if n.key == sk)
    assert isinstance(step_node.stage, StageSpec.Aggregate)
    spec = step_node.stage.spec
    assert len(spec.transformers) > 0


def test_disabled_per_workpiece_transformer_not_wired(
    contour_step_class, test_machine_and_config
):
    """A transformer dict with ``enabled=False`` must be skipped."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    # Disable every per-workpiece transformer.
    for t in step.per_workpiece_transformers_dicts:
        t["enabled"] = False
    wp = WorkPiece(name="wp")
    wp.set_size(10.0, 10.0)
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    wp_node = next(n for n in nodes if n.key == wpk)
    payload = wp_node.stage.params
    assert payload.transformers == []


# ----------------------------------------------------------------------
# Workpiece-move cache invalidation (regression: 3D canvas stale)
# ----------------------------------------------------------------------


def test_step_aggregate_token_changes_on_workpiece_move(
    contour_step_class, test_machine_and_config
):
    """A pure position change (no geometry_revision bump) must
    invalidate the step aggregate cache because the aggregate applies
    the workpiece placement matrix to the (possibly cached) workpiece
    compute output. If the token did not change, the cached aggregate
    would be reused with the old placement baked in and the 3D canvas
    would display a stale position."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    before = IntentBuilder(machine=machine).build(doc)
    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)

    # Pure move — bumps transform_revision but not geometry_revision.
    wp.pos = 100.0, 100.0
    assert wp.geometry_revision == 0
    after = IntentBuilder(machine=machine).build(doc)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_job_token_changes_on_workpiece_move(
    contour_step_class, test_machine_and_config
):
    """The job aggregate token folds in the step aggregate tokens, so
    a workpiece move must propagate through to the job/encode cache
    too. Without this, the encoded G-code shown in the 3D canvas
    would be the pre-move output."""
    machine, context = test_machine_and_config
    step = contour_step_class.create(context, name="cut")
    wp = WorkPiece(name="wp")
    wp.set_size(50.0, 30.0)
    doc = _make_doc(step, wp)

    before = IntentBuilder(machine=machine).build(doc)
    jk = job_key()
    before_t = next(n.version_token for n in before if n.key == jk)

    wp.pos = 25.0, 25.0
    after = IntentBuilder(machine=machine).build(doc)
    after_t = next(n.version_token for n in after if n.key == jk)
    assert before_t != after_t


def test_raster_compute_token_changes_on_workpiece_move(
    engrave_step_class, test_machine_and_config
):
    """The raster assembler bakes ``workpiece.bbox`` into its output
    via ``offset_x_mm`` / ``offset_y_mm``, so a position change must
    invalidate the workpiece compute cache. Otherwise the cached
    workpiece-local-but-offset ops would be re-displaced by the
    aggregate's new placement matrix and land at the wrong world
    position."""
    machine, context = test_machine_and_config
    step = engrave_step_class.create(context, name="engrave")
    wp = WorkPiece(name="wp")
    wp.set_size(20.0, 20.0)
    doc = _make_doc(step, wp)

    before = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)

    wp.pos = 50.0, 50.0
    after = IntentBuilder(machine=machine).build(doc)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


def test_raster_step_is_position_sensitive(engrave_step_class):
    """EngraveStep.assemble's RasterSpec uses ``workpiece.bbox`` for
    its offsets, so the step must declare itself position-sensitive
    so the IntentBuilder folds ``transform_revision`` into the
    compute token."""
    step = engrave_step_class(name="engrave")
    assert step.is_position_sensitive() is True


# ----------------------------------------------------------------------
# Stock resolution
# ----------------------------------------------------------------------


def _make_doc_with_stock(
    step: _TestStep,
    wp: WorkPiece,
    stock_visible: bool = True,
) -> Doc:
    """Build a Doc containing *step*, *wp*, and a single visible
    StockItem backed by a rectangular StockAsset."""
    doc = _make_doc(step, wp)
    asset = StockAsset(name="sheet", geometry=None)

    # Give the asset a 100×80 mm rectangle.
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 0)
    geo.line_to(100, 80)
    geo.line_to(0, 80)
    geo.close_path()
    asset.geometry = geo
    doc.add_asset(asset)
    item = StockItem(stock_asset_uid=asset.uid, name="sheet")
    item.visible = stock_visible
    doc.add_child(item)
    return doc


def test_stock_items_resolved_into_geometries():
    """Visible StockItems must produce world-rect geometries in the
    resolved stock list (not just the machine workarea fallback)."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc_with_stock(step, wp)

    builder = IntentBuilder()
    builder.build(doc)
    geos = builder._resolve_stock_geometries()

    assert geos is not None
    assert len(geos) == 1
    assert not geos[0].is_empty()


def test_hidden_stock_skipped():
    """Hidden StockItems must not appear in the stock geometries."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc_with_stock(step, wp, stock_visible=False)

    builder = IntentBuilder()
    builder.build(doc)
    geos = builder._resolve_stock_geometries()

    assert geos is not None
    assert len(geos) == 0


def test_no_stock_falls_back_to_workarea(lite_context):
    """When no StockItems exist, the machine workarea rectangle must be
    used as the stock geometry fallback."""
    machine = Machine(lite_context)
    machine.set_axis_extents(300, 200)

    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    builder = IntentBuilder(machine=machine)
    builder.build(doc)
    geos = builder._resolve_stock_geometries()

    assert geos is not None
    assert len(geos) == 1
    assert not geos[0].is_empty()


def test_stock_move_invalidates_compute_token():
    """Moving a StockItem must change the compute token for a
    position-sensitive step (e.g. one with CropTransformer)."""
    step = _TestStep(name="s1", position_sensitive=True)
    wp = WorkPiece(name="wp")
    doc = _make_doc_with_stock(step, wp)

    before = IntentBuilder().build(doc)

    # Move the stock item by translating its matrix.
    stock_item = doc.stock_items[0]
    stock_item.matrix = stock_item.matrix @ Matrix.translation(50, 50)

    after = IntentBuilder().build(doc)

    wpk = workpiece_key(wp.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# Machine transform E2E tests
# ----------------------------------------------------------------------


def _run_full_pipeline(machine, context, contour_step_class):
    """Build a doc with a 10×10 square workpiece, run the full
    IntentBuilder pipeline, and return the G-code text."""
    step = contour_step_class.create(context, name="cut")
    step.set_cut_speed(3000)
    step.set_power(0.5)

    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()

    wp = WorkPiece(name="square")
    wp._edited_boundaries = geo
    wp.set_size(10.0, 10.0)
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    completed = []
    execute_stages(nodes, lambda n: completed.append(n))

    ek = job_encode_key()
    enc = next(c for c in completed if c.key == ek)
    assert enc.error is None, enc.error
    assert enc.output is not None
    return enc.output.text


def _extract_cut_coords(gcode):
    """Return a list of (x, y) tuples from G0/G1 motion lines in the
    G-code, tracking the current position across lines.

    The first entry is the rapid-move starting position (G0), followed
    by each cut move (G1).
    """
    coords = []
    x = y = 0.0
    for line in gcode.split("\n"):
        line = line.split(";")[0].strip()
        if not (line.startswith("G0") or line.startswith("G1")):
            continue
        # Skip the final return-to-origin G0.
        if "Return to origin" in line:
            continue
        for token in line.split():
            if token.startswith("X"):
                x = float(token[1:])
            elif token.startswith("Y"):
                y = float(token[1:])
        coords.append((x, y))
    return coords


@pytest.mark.parametrize(
    "origin,expected_x,expected_y",
    [
        # 10×10 square workpiece at world (0,0).  With kerf offset the
        # actual cut starts at approximately (-4.5, -4.5) in world space.
        # Machine extents 200×150.
        (Origin.BOTTOM_LEFT, -4.5, -4.5),
        (Origin.TOP_LEFT, -4.5, 154.5),
        (Origin.TOP_RIGHT, 204.5, 154.5),
        (Origin.BOTTOM_RIGHT, 204.5, -4.5),
    ],
)
def test_machine_transform_origin_in_gcode(
    contour_step_class, test_machine_and_config, origin, expected_x, expected_y
):
    """The machine-transform stage flips coordinates according to the
    machine origin corner.  A square at world (0,0) should appear at
    the opposite corner for non-bottom-left origins."""
    machine, context = test_machine_and_config
    machine.set_origin(origin)

    gcode = _run_full_pipeline(machine, context, contour_step_class)
    coords = _extract_cut_coords(gcode)
    assert len(coords) >= 4

    # coords[0] is the G0 rapid to the first corner of the square.
    assert coords[0][0] == pytest.approx(expected_x, abs=0.1)
    assert coords[0][1] == pytest.approx(expected_y, abs=0.1)


def test_machine_transform_wcs_offset_in_gcode(
    contour_step_class, test_machine_and_config
):
    """The machine-transform stage subtracts the active WCS offset from
    the G-code coordinates."""
    machine, context = test_machine_and_config
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.set_active_wcs("G54")
    machine.update_wcs_offset("G54", (50.0, 30.0, 0.0))

    gcode = _run_full_pipeline(machine, context, contour_step_class)
    coords = _extract_cut_coords(gcode)
    assert len(coords) >= 4

    # G0 rapid to first corner: world (-4.5, -4.5) minus WCS (50, 30)
    # = (-54.5, -34.5).
    assert coords[0][0] == pytest.approx(-54.5, abs=0.1)
    assert coords[0][1] == pytest.approx(-34.5, abs=0.1)

    # First G1 cut: world (95.5, -4.5) minus WCS (50, 30) = (45.5, -34.5).
    assert coords[1][0] == pytest.approx(45.5, abs=0.1)
    assert coords[1][1] == pytest.approx(-34.5, abs=0.1)
