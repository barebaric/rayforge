"""
Tests for :mod:`rayforge.pipeline.intent_builder`.

These tests build a small :class:`~rayforge.core.doc.Doc` with a
single step (and two workpieces) and verify the keys, version tokens,
and the position-sensitive folding rule.
"""

from raygeo.cnc.execution.intent import create_intent_from_nodes
from raygeo.cnc.execution.specs import EncodeSpec
from raygeo.geo import Geometry
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.pipeline.execute import execute_stages
from raygeo.pipeline.stage import StageSpec

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
    job_encode_key,
    job_key,
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
    assert encode_nodes[0].stage.source_key == job_key()


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
