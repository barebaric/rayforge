"""
Tests for :mod:`rayforge.pipeline.intent_builder`.

These tests build a small :class:`~rayforge.core.doc.Doc` with a
single step (and two workpieces) and verify the keys, version tokens,
and the position-sensitive folding rule.
"""

import math
from unittest.mock import MagicMock

import numpy as np
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
from raygeo.ops.material.spec import CylinderStock, MaterialFoldSpec
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
from rayforge.machine.models.machine_panel import PanelOrientation
from rayforge.machine.models.rotary_module import RotaryMode, RotaryModule
from rayforge.pipeline.encoder.rust_helpers import dialect_to_spec
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
    UnsupportedRotaryPanelOrientationError,
    job_encode_key,
    job_key,
    job_machinexform_key,
    parse_stock_key,
    step_key,
    stock_key,
    workpiece_key,
)


class _TestStep(Step):
    """
    Concrete ``Step`` subclass for tests.  ``Step`` is normally
    subclassed by addons (ContourStep, FrameStep, ...); the generic
    base exposes every attribute the IntentBuilder consumes.  The
    position-sensitive flag is overridable per-instance via
    ``_position_sensitive``.
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


def test_builds_one_node_per_workpiece_plus_step_and_job(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    wp2 = WorkPiece(name="wp2")
    doc = _make_doc(step, wp1, wp2)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert workpiece_key(wp1.uid, step.uid) in keys
    assert workpiece_key(wp2.uid, step.uid) in keys
    assert step_key(step.uid) in keys
    assert job_key() in keys
    assert job_machinexform_key() in keys
    assert job_encode_key() in keys
    assert len(keys) == 6


def test_step_with_generated_workpiece_only_applies_to_that_workpiece(
    isolated_machine,
):
    """A step that generated a workpiece (e.g. material test) must not
    be applied to the other workpieces in its layer."""
    step = _TestStep(name="s1")
    step.generated_workpiece_uid = "wp-gen"
    wp_gen = WorkPiece(name="wp-gen")
    wp_gen.uid = "wp-gen"
    wp_other = WorkPiece(name="wp-other")
    doc = _make_doc(step, wp_gen, wp_other)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert workpiece_key("wp-gen", step.uid) in keys
    assert workpiece_key("wp-other", step.uid) not in keys


def test_step_with_missing_generated_workpiece_emits_no_nodes(
    isolated_machine,
):
    """When a step's generated workpiece is not in the layer, the step
    emits no workpiece or step aggregate nodes."""
    step = _TestStep(name="s1")
    step.generated_workpiece_uid = "wp-gone"
    doc = _make_doc(step, WorkPiece(name="wp-other"))

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert step_key(step.uid) not in keys
    assert job_key() not in keys


def test_keys_are_stable_across_rebuilds(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    n1 = IntentBuilder(machine=isolated_machine).build(doc)
    n2 = IntentBuilder(machine=isolated_machine).build(doc)
    assert [n.key for n in n1] == [n.key for n in n2]


def test_rotary_layer_rejects_rotated_panel(isolated_machine):
    """Rotary uses its own cylindrical coordinate space, not the
    rotated flat-bed presentation."""
    doc = _make_doc(_TestStep(name="s1"), WorkPiece(name="wp"))
    doc.active_layer.set_rotary_enabled(True)
    isolated_machine.set_panel_orientation(PanelOrientation.ROTATED_RIGHT)

    with pytest.raises(
        UnsupportedRotaryPanelOrientationError,
        match="Rotary layers require the Native panel orientation",
    ):
        IntentBuilder(machine=isolated_machine).build(doc)


# ----------------------------------------------------------------------
# Token stability
# ----------------------------------------------------------------------


def test_version_tokens_stable_across_rebuilds(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    n1 = IntentBuilder(machine=isolated_machine).build(doc)
    n2 = IntentBuilder(machine=isolated_machine).build(doc)
    tokens_1 = {n.key: n.version_token for n in n1}
    tokens_2 = {n.key: n.version_token for n in n2}
    assert tokens_1 == tokens_2
    for t in tokens_1.values():
        assert isinstance(t, int)
        assert t > 0


def test_compute_token_changes_on_geometry_revision_bump(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    wp1.updated.send(wp1)  # bumps geometry_revision
    after = IntentBuilder(machine=isolated_machine).build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


def test_compute_token_changes_on_step_param_change(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    step.cut_speed = 999
    after = IntentBuilder(machine=isolated_machine).build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


def test_compute_token_changes_on_workpiece_transformer_change(
    isolated_machine,
):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    step.per_workpiece_transformers_dicts.append(
        {"name": "Optimize", "enabled": True}
    )
    after = IntentBuilder(machine=isolated_machine).build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# POSITION_SENSITIVE folding
# ----------------------------------------------------------------------


def test_transform_revision_omitted_when_not_position_sensitive(
    isolated_machine,
):
    """
    A pure move must NOT change the workpiece compute token when the
    step is not position-sensitive.
    """
    step = _TestStep(name="s1", position_sensitive=False)
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    wp1.transform_changed.send(wp1, old_matrix=wp1.matrix)
    assert wp1.geometry_revision == 0
    assert wp1.transform_revision == 1
    after = IntentBuilder(machine=isolated_machine).build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t == after_t


def test_transform_revision_folded_when_position_sensitive(isolated_machine):
    """
    When the step is position-sensitive a move must change the workpiece
    compute token.
    """
    step = _TestStep(name="s1", position_sensitive=True)
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    wp1.transform_changed.send(wp1, old_matrix=wp1.matrix)
    assert wp1.transform_revision == 1
    after = IntentBuilder(machine=isolated_machine).build(doc)

    wpk = workpiece_key(wp1.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# Aggregate tokens
# ----------------------------------------------------------------------


def test_step_aggregate_token_changes_on_step_param_change(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    step.cut_speed = 2222
    after = IntentBuilder(machine=isolated_machine).build(doc)

    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_step_aggregate_token_changes_on_upstream_token_change(
    isolated_machine,
):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    wp1.updated.send(wp1)  # bump upstream compute token
    after = IntentBuilder(machine=isolated_machine).build(doc)

    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_step_aggregate_token_changes_on_per_step_transformer_change(
    isolated_machine,
):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    step.per_step_transformers_dicts.append(
        {"name": "Smooth", "enabled": True}
    )
    after = IntentBuilder(machine=isolated_machine).build(doc)

    sk = step_key(step.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_job_token_changes_on_step_param_change(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    step.cut_speed = 7777
    after = IntentBuilder(machine=isolated_machine).build(doc)

    jk = job_key()
    before_t = next(n.version_token for n in before if n.key == jk)
    after_t = next(n.version_token for n in after if n.key == jk)
    assert before_t != after_t


# ----------------------------------------------------------------------
# Visibility / structural filtering
# ----------------------------------------------------------------------


def test_hidden_steps_excluded(isolated_machine):
    step = _TestStep(name="s1")
    step.visible = False
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]

    assert workpiece_key(wp1.uid, step.uid) not in keys
    assert step_key(step.uid) not in keys
    assert job_key() not in keys


def test_layers_without_workpieces_skipped_for_compute(isolated_machine):
    step = _TestStep(name="s1")
    doc = Doc()
    layer = doc.active_layer
    workflow = layer.workflow
    assert workflow is not None
    workflow.add_child(step)
    # No workpieces added.

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert step_key(step.uid) not in keys
    assert workpiece_key("any", step.uid) not in keys
    assert job_key() not in keys


def test_second_layer_without_skips_affect_existing(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    layer2 = Layer(name="empty")
    doc.add_child(layer2)
    layer2.add_child(WorkPiece(name="wp2"))

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    # Only one step aggregate node and one job node.
    assert keys.count(step_key(step.uid)) == 1
    assert keys.count(job_key()) == 1


# ----------------------------------------------------------------------
# Generation ID and stage payload
# ----------------------------------------------------------------------


def test_generation_id_propagated_to_nodes(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(machine=isolated_machine, generation_id=42).build(
        doc
    )
    assert all(n.generation_id == 42 for n in nodes)


def test_stage_payload_is_valid_raygeo_stagespec(isolated_machine):
    """The IntentBuilder's stage payloads must be real raygeo StageSpec
    instances so ``create_intent_from_nodes`` accepts them."""
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)

    # All stages must be valid raygeo stage payloads.
    valid_stages = (
        StageSpec.Compute,
        StageSpec.Aggregate,
        MachineTransformSpec,
        EncodeSpec,
    )
    for n in nodes:
        assert isinstance(n.stage, valid_stages)

    # create_intent_from_nodes must succeed (validates the StageSpec).
    intent = create_intent_from_nodes(nodes)
    assert intent is not None
    assert intent.step_count == 4  # machine-aware compute + aggregate stages


def test_stage_wpspec_for_workpiece_node_is_compute(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    wp_node = next(
        n for n in nodes if n.key == workpiece_key(wp1.uid, step.uid)
    )
    assert isinstance(wp_node.stage, StageSpec.Compute)


def test_stage_step_node_is_aggregate(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    st_node = next(n for n in nodes if n.key == step_key(step.uid))
    assert isinstance(st_node.stage, StageSpec.Aggregate)


def test_stage_job_node_is_aggregate(isolated_machine):
    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
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
    step.offset_mm = 0.5
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)

    nodes = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    wp_node = next(n for n in nodes if n.key == wpk)
    spec = wp_node.stage.params.assembler.spec
    assert spec.cut_side == "outside"
    assert spec.offset_mm == 0.5

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


def test_machine_token_ignores_unused_wcs_offset(isolated_machine):
    step = _TestStep(name="s1")
    doc = _make_doc(step, WorkPiece(name="wp"))
    before = IntentBuilder(machine=isolated_machine).build(doc)

    isolated_machine.update_wcs_offset("G55", (10.0, 20.0, 0.0))
    after = IntentBuilder(machine=isolated_machine).build(doc)

    before_t = next(
        n.version_token for n in before if n.key == job_machinexform_key()
    )
    after_t = next(
        n.version_token for n in after if n.key == job_machinexform_key()
    )
    assert after_t == before_t


def test_machine_token_tracks_effective_layer_wcs_offset(isolated_machine):
    step = _TestStep(name="s1")
    doc = _make_doc(step, WorkPiece(name="wp"))
    doc.active_layer.set_wcs("G55")
    before = IntentBuilder(machine=isolated_machine).build(doc)

    isolated_machine.update_wcs_offset("G55", (10.0, 20.0, 0.0))
    after = IntentBuilder(machine=isolated_machine).build(doc)

    before_t = next(
        n.version_token for n in before if n.key == job_machinexform_key()
    )
    after_t = next(
        n.version_token for n in after if n.key == job_machinexform_key()
    )
    assert after_t != before_t


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


def test_machine_transform_ignores_panel_rotation(isolated_machine):
    """The output stage is native: the panel rotation never reaches the
    encoder, because the document lives in WORLD space."""
    isolated_machine.set_axis_extents(400.0, 800.0)
    isolated_machine.set_panel_orientation(PanelOrientation.ROTATED_RIGHT)
    doc = _make_doc(_TestStep())

    spec = IntentBuilder(
        machine=isolated_machine
    )._build_machine_transform_stage(doc)

    assert spec.world_to_machine == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_panel_rotation_keeps_machine_transform_token(
    contour_step_class, test_machine_and_config
):
    """Changing the panel orientation does not invalidate cached
    machine-space output."""
    machine, context = test_machine_and_config
    doc = _make_doc(
        contour_step_class.create(context, name="cut"),
        WorkPiece(name="wp"),
    )
    before = IntentBuilder(machine=machine).build(doc)

    machine.set_panel_orientation(PanelOrientation.ROTATED_LEFT)
    after = IntentBuilder(machine=machine).build(doc)

    before_token = next(
        node.version_token
        for node in before
        if node.key == job_machinexform_key()
    )
    after_token = next(
        node.version_token
        for node in after
        if node.key == job_machinexform_key()
    )
    assert before_token == after_token


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
    part, diameter, axis="A", mode="true_4th_axis", mm_per_rotation=0.0
):
    """Helper: create a 4-node pipeline (compute → aggregate → mxform
    → encode) that feeds known geometry through the machine-transform
    stage with a rotary module and encodes the result."""
    cs = ContourSpec(
        cut_side="centerline",
        cut_order="inside_outside",
        remove_inner=False,
        offset_mm=0.0,
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
        mm_per_rotation=mm_per_rotation,
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
        mm_per_rotation=100.0,
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


@pytest.mark.parametrize(
    "attr,value",
    [
        ("invert", True),
        ("auto_levels", False),
        ("black_point", 80),
        ("white_point", 200),
        ("threshold", 200),
        ("dither_algorithm", "BAYER4"),
    ],
)
def test_raster_compute_token_changes_on_image_preprocess_params(
    engrave_step_class,
    test_machine_and_config,
    attr,
    value,
):
    """Changing any raster image-preprocessing setting must invalidate
    the workpiece compute cache.

    The levels (black/white points), inversion, threshold and dither
    are baked into the preprocessed image the assembler consumes, so a
    token that omits them would replay the exact same ops regardless of
    the brightness settings.
    """
    machine, context = test_machine_and_config
    step = engrave_step_class.create(context, name="engrave")
    wp = WorkPiece(name="wp")
    wp.set_size(20.0, 20.0)
    doc = _make_doc(step, wp)

    before = IntentBuilder(machine=machine).build(doc)
    wpk = workpiece_key(wp.uid, step.uid)
    before_t = next(n.version_token for n in before if n.key == wpk)

    if attr == "dither_algorithm":
        step.set_dither_algorithm(value)
    else:
        setattr(step, attr, value)
    after = IntentBuilder(machine=machine).build(doc)
    after_t = next(n.version_token for n in after if n.key == wpk)
    assert before_t != after_t


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


def test_stock_items_resolved_into_geometries(isolated_machine):
    """Visible StockItems must produce world-rect geometries in the
    resolved stock list (not just the machine workarea fallback)."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc_with_stock(step, wp)

    builder = IntentBuilder(machine=isolated_machine)
    builder.build(doc)
    geos = builder._resolve_stock_geometries()

    assert geos is not None
    assert len(geos) == 1
    assert not geos[0].is_empty()


def test_hidden_stock_skipped(isolated_machine):
    """Hidden StockItems must not appear in the stock geometries.

    The hidden stock contributes no geometry, so only the machine
    workarea fallback remains (1 geometry instead of 2)."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc_with_stock(step, wp, stock_visible=False)

    builder = IntentBuilder(machine=isolated_machine)
    builder.build(doc)
    geos = builder._resolve_stock_geometries()

    assert geos is not None
    assert len(geos) == 1


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


def test_stock_move_invalidates_compute_token(isolated_machine):
    """Moving a StockItem must change the compute token for a
    position-sensitive step (e.g. one with CropTransformer)."""
    step = _TestStep(name="s1", position_sensitive=True)
    wp = WorkPiece(name="wp")
    doc = _make_doc_with_stock(step, wp)

    before = IntentBuilder(machine=isolated_machine).build(doc)

    # Move the stock item by translating its matrix.
    stock_item = doc.stock_items[0]
    stock_item.matrix = stock_item.matrix @ Matrix.translation(50, 50)

    after = IntentBuilder(machine=isolated_machine).build(doc)

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
        if not (line.startswith(("G0", "G1"))):
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


@pytest.mark.parametrize("origin", list(Origin))
@pytest.mark.parametrize("reverse_x", [False, True])
@pytest.mark.parametrize("reverse_y", [False, True])
def test_machine_transform_wcs_with_axis_reversal(
    contour_step_class,
    test_machine_and_config,
    origin,
    reverse_x,
    reverse_y,
):
    """WCS offset must be subtracted in machine space (after w2m), not
    in world space (before w2m).

    A 10x10 square at world (0,0) is zeroed at world (10,20) via WCS.
    The G-code for the first corner must equal the machine-space
    distance from the WCS origin to that corner — regardless of origin
    corner or axis reversal settings.
    """
    machine, context = test_machine_and_config
    machine.set_axis_extents(200, 150)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(reverse_x)
    machine.set_reverse_y_axis(reverse_y)
    machine.set_active_wcs("G54")

    # WCS zeroed at world (10, 20): stores the machine coordinate.
    space = machine.get_coordinate_space()
    wcs_machine = space.world_point_to_machine(10.0, 20.0)
    machine.update_wcs_offset("G54", (wcs_machine[0], wcs_machine[1], 0.0))

    gcode = _run_full_pipeline(machine, context, contour_step_class)
    coords = _extract_cut_coords(gcode)
    assert len(coords) >= 4

    # Independently compute expected G-code for the first corner
    # (world point after kerf offset ≈ (-4.5, -4.5)).
    w2m = space.get_world_to_machine_matrix()
    cmd = space.get_command_offset(
        wcs_offset=(wcs_machine[0], wcs_machine[1], 0.0),
        wcs_is_workarea_origin=False,
    )
    # The kerf offset shifts the first point to approximately (-4.5, -4.5)
    first_world = np.array([-4.5, -4.5, 0.0, 1.0])
    first_machine = w2m @ first_world
    expected_x = first_machine[0] - cmd[0]
    expected_y = first_machine[1] - cmd[1]

    assert coords[0][0] == pytest.approx(expected_x, abs=0.5)
    assert coords[0][1] == pytest.approx(expected_y, abs=0.5)


@pytest.mark.parametrize(
    "orientation",
    [
        PanelOrientation.ROTATED_RIGHT,
        PanelOrientation.ROTATED_LEFT,
    ],
)
def test_panel_rotation_does_not_change_gcode(
    contour_step_class,
    test_machine_and_config,
    orientation,
):
    """The panel rotation is display-only: it never reaches G-code."""
    machine, context = test_machine_and_config
    machine.set_origin(Origin.BOTTOM_LEFT)

    native_gcode = _run_full_pipeline(machine, context, contour_step_class)
    native_coords = _extract_cut_coords(native_gcode)

    machine.set_panel_orientation(orientation)
    rotated_gcode = _run_full_pipeline(machine, context, contour_step_class)
    rotated_coords = _extract_cut_coords(rotated_gcode)

    assert len(native_coords) >= 4
    assert rotated_coords == native_coords


# ----------------------------------------------------------------------
# Stock fold nodes
# ----------------------------------------------------------------------


def _wp_with_geometry(name: str = "wp") -> WorkPiece:
    """A WorkPiece whose world geometry overlaps the stock's
    ``(0,0)-(100,100)`` rect: a 50×50 mm piece at the origin."""
    wp = WorkPiece(name=name)
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 0)
    geo.line_to(1, 1)
    geo.line_to(0, 1)
    geo.close_path()
    # The boundaries are a 1x1 normalized box; scale it to 50×50 mm
    # via the workpiece matrix so its world rect overlaps the stock.
    wp._boundaries_cache = geo
    wp.matrix = Matrix.scale(50.0, 50.0)
    return wp


def _make_doc_with_stock_and_wp(
    step: _TestStep,
    wp: WorkPiece,
    stock_visible: bool = True,
    thickness: float = 18.0,
) -> tuple[Doc, StockItem]:
    """Build a Doc with *step*, *wp*, and a 100×80 mm stock item that
    overlaps the workpiece, returning ``(doc, stock_item)``."""
    doc = _make_doc(step, wp)
    asset = StockAsset(name="sheet", geometry=None)
    asset.set_thickness(thickness)
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
    return doc, item


def test_stock_key_roundtrip():
    """``stock_key``/``parse_stock_key`` are inverse."""
    key = stock_key("abc-123")
    assert key == "stock:abc-123"
    assert parse_stock_key(key) == "abc-123"
    assert parse_stock_key("workpiece:x:y") is None
    assert parse_stock_key("stock:") is None
    assert parse_stock_key("stock:a:b") is None


def test_builds_one_fold_node_per_visible_stock(isolated_machine):
    """Two visible stock items → two ``stock:{uid}`` fold nodes."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp)
    # Add a second stock item.
    asset2 = StockAsset(name="sheet2")
    asset2.set_thickness(5.0)
    geo2 = Geometry()
    geo2.move_to(0, 0)
    geo2.line_to(50, 0)
    geo2.line_to(50, 50)
    geo2.line_to(0, 50)
    geo2.close_path()
    asset2.geometry = geo2
    doc.add_asset(asset2)
    item2 = StockItem(stock_asset_uid=asset2.uid, name="sheet2")
    doc.add_child(item2)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert stock_key(item.uid) in keys
    assert stock_key(item2.uid) in keys


def test_no_fold_node_for_invisible_stock(isolated_machine):
    """An invisible stock item emits no fold node."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp, stock_visible=False)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert stock_key(item.uid) not in keys


def test_fold_node_token_stability(isolated_machine):
    """Building twice with unchanged inputs yields equal fold tokens."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    after = IntentBuilder(machine=isolated_machine).build(doc)
    sk = stock_key(item.uid)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t == after_t


def test_fold_node_token_changes_on_stock_move(isolated_machine):
    """Moving the stock changes its fold node token."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp)
    sk = stock_key(item.uid)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    # Small move keeps the stock overlapping the workpiece so the fold
    # node is still emitted, while changing the token.
    item.matrix = Matrix.translation(10.0, 0.0)
    after = IntentBuilder(machine=isolated_machine).build(doc)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_fold_node_token_changes_on_thickness_change(isolated_machine):
    """Changing the stock thickness changes its fold node token."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp, thickness=18.0)
    sk = stock_key(item.uid)

    before = IntentBuilder(machine=isolated_machine).build(doc)
    asset = item.stock_asset
    assert asset is not None
    asset.set_thickness(5.0)
    after = IntentBuilder(machine=isolated_machine).build(doc)
    before_t = next(n.version_token for n in before if n.key == sk)
    after_t = next(n.version_token for n in after if n.key == sk)
    assert before_t != after_t


def test_fold_node_no_intersecting_workpieces_skipped(isolated_machine):
    """A stock item with no intersecting workpiece emits no fold node."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp)
    # Move the stock far away from the workpiece.
    item.matrix = Matrix.translation(10000.0, 10000.0)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    keys = [n.key for n in nodes]
    assert stock_key(item.uid) not in keys


def test_fold_node_is_material_fold_stage(isolated_machine):
    """The fold node's stage is a ``MaterialFoldSpec`` (passed directly,
    not wrapped in a ``StageSpec`` — pipeline is domain-free)."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc, item = _make_doc_with_stock_and_wp(step, wp)
    sk = stock_key(item.uid)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    fold_node = next(n for n in nodes if n.key == sk)
    assert isinstance(fold_node.stage, MaterialFoldSpec)


def test_fold_node_entries_cover_intersecting_workpieces(isolated_machine):
    """The fold spec's entries list every intersecting workpiece compute
    node; non-intersecting workpieces are absent."""
    step = _TestStep(name="s1")
    wp_far = _wp_with_geometry("wp_far")
    wp_far.matrix = Matrix.translation(10000.0, 10000.0)
    wp_near = _wp_with_geometry("wp_near")
    doc, item = _make_doc_with_stock_and_wp(step, wp_near)
    doc.active_layer.add_child(wp_far)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    sk = stock_key(item.uid)
    fold_node = next(n for n in nodes if n.key == sk)
    spec = fold_node.stage
    source_keys = {e.source_key for e in spec.entries}
    near_key = workpiece_key(wp_near.uid, step.uid)
    far_key = workpiece_key(wp_far.uid, step.uid)
    assert near_key in source_keys
    assert far_key not in source_keys


def test_existing_node_count_unchanged_with_stock(isolated_machine):
    """Adding a stock item does not change the count of workpiece/step/
    job nodes — fold nodes are additive."""
    step = _TestStep(name="s1")
    wp1 = _wp_with_geometry("wp1")
    wp2 = _wp_with_geometry("wp2")
    doc_no_stock = _make_doc(step, wp1, wp2)

    # Build a separate doc with its own workpieces + a stock item, so
    # workpieces are not re-parented between docs.
    step2 = _TestStep(name="s1")
    wp1b = _wp_with_geometry("wp1")
    wp2b = _wp_with_geometry("wp2")
    doc_with_stock, _item = _make_doc_with_stock_and_wp(step2, wp1b)
    doc_with_stock.active_layer.add_child(wp2b)

    no_stock = IntentBuilder(machine=isolated_machine).build(doc_no_stock)
    with_stock = IntentBuilder(machine=isolated_machine).build(doc_with_stock)

    non_stock_keys = [
        n.key for n in with_stock if not n.key.startswith("stock:")
    ]
    assert len(non_stock_keys) == len(no_stock)


# ----------------------------------------------------------------------
# Rotary layer fold nodes
# ----------------------------------------------------------------------


def _with_default_rotary(machine) -> RotaryModule:
    module = RotaryModule()
    module.max_workpiece_length = 300.0
    machine.rotary_modules[module.uid] = module
    machine.default_rotary_module_uid = module.uid
    return module


def _make_doc_with_rotary_wp(step: _TestStep, wp: WorkPiece) -> Doc:
    """A Doc whose active layer is rotary-enabled with a workpiece in
    the unrolled domain."""
    doc = _make_doc(step, wp)
    doc.active_layer.rotary_enabled = True
    doc.active_layer.rotary_diameter = 50.0
    return doc


def test_rotary_layer_emits_fold_node(isolated_machine):
    """A rotary layer with an intersecting workpiece emits a fold node
    keyed by the layer's uid."""
    _with_default_rotary(isolated_machine)
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc = _make_doc_with_rotary_wp(step, wp)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    sk = stock_key(doc.active_layer.uid)
    assert sk in [n.key for n in nodes]


def test_rotary_fold_node_uses_cylinder_stock(isolated_machine):
    """The rotary fold node's stage is a ``MaterialFoldSpec`` against
    a ``CylinderStock`` spanning the module's max workpiece length."""
    module = _with_default_rotary(isolated_machine)
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc = _make_doc_with_rotary_wp(step, wp)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    sk = stock_key(doc.active_layer.uid)
    spec = next(n for n in nodes if n.key == sk).stage
    assert isinstance(spec, MaterialFoldSpec)
    assert isinstance(spec.stock, CylinderStock)
    assert spec.stock.diameter == 50.0
    assert spec.stock.length == module.max_workpiece_length
    near_key = workpiece_key(wp.uid, step.uid)
    assert {e.source_key for e in spec.entries} == {near_key}


def test_rotary_fold_node_skipped_without_module(isolated_machine):
    """Without a default rotary module, no rotary fold node is
    emitted even when a layer is rotary-enabled."""
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc = _make_doc_with_rotary_wp(step, wp)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    sk = stock_key(doc.active_layer.uid)
    assert sk not in [n.key for n in nodes]


def test_rotary_fold_node_includes_far_workpieces(isolated_machine):
    """Workpieces far from the unrolled domain still contribute fold
    entries: image workpieces may not resolve world geometry, and the
    fold's raster sampling is harmless for misses."""
    _with_default_rotary(isolated_machine)
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    wp.matrix = Matrix.translation(10000.0, 10000.0)
    doc = _make_doc_with_rotary_wp(step, wp)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    sk = stock_key(doc.active_layer.uid)
    spec = next(n for n in nodes if n.key == sk).stage
    assert {e.source_key for e in spec.entries} == {
        workpiece_key(wp.uid, step.uid)
    }


def test_flat_stock_node_not_emitted_for_rotary_layer(isolated_machine):
    """Rotary layers don't emit flat-stock keys and vice versa."""
    _with_default_rotary(isolated_machine)
    step = _TestStep(name="s1")
    wp = _wp_with_geometry()
    doc = _make_doc_with_rotary_wp(step, wp)

    nodes = IntentBuilder(machine=isolated_machine).build(doc)
    stock_keys = [n.key for n in nodes if n.key.startswith("stock:")]
    assert stock_keys == [stock_key(doc.active_layer.uid)]


def test_rotary_fold_executes_end_to_end(
    contour_step_class, test_machine_and_config
):
    """A rotary layer's fold node runs through ``execute_stages`` and
    produces a cylindrical material state."""
    machine, context = test_machine_and_config
    module = RotaryModule()
    module.max_workpiece_length = 300.0
    machine.rotary_modules[module.uid] = module
    machine.default_rotary_module_uid = module.uid

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
    doc.active_layer.rotary_enabled = True
    doc.active_layer.rotary_diameter = 50.0

    nodes = IntentBuilder(machine=machine, generation_id=1).build(doc)
    sk = stock_key(doc.active_layer.uid)

    completed = []

    def on_completed(node):
        completed.append(node)

    assert sk in [n.key for n in nodes], [n.key for n in nodes]

    execute_stages(nodes, on_completed)

    fold_result = next(c for c in completed if c.key == sk)
    assert fold_result.error is None, fold_result.error
    state = fold_result.output
    assert state.profile == "cylindrical"
    assert state.surface_map is not None
    assert state.grid is not None
    # Arc axis centered on the machine origin (diameter 50).
    assert state.grid.origin_mm[0] == 0.0
    assert state.grid.origin_mm[1] == pytest.approx(-np.pi * 25.0)
