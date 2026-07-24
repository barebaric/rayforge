"""
Tests for :mod:`rayforge.pipeline.intent_builder`.

These tests build a small :class:`~rayforge.core.doc.Doc` with a
single step (and two workpieces) and verify the keys, version tokens,
and the position-sensitive folding rule.
"""

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
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
    from raygeo.cnc.execution.intent import create_intent_from_nodes
    from raygeo.pipeline.stage import StageSpec

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
    from raygeo.pipeline.stage import StageSpec

    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    wp_node = next(
        n for n in nodes if n.key == workpiece_key(wp1.uid, step.uid)
    )
    assert isinstance(wp_node.stage, StageSpec.Compute)


def test_stage_step_node_is_aggregate():
    from raygeo.pipeline.stage import StageSpec

    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    st_node = next(n for n in nodes if n.key == step_key(step.uid))
    assert isinstance(st_node.stage, StageSpec.Aggregate)


def test_stage_job_node_is_aggregate():
    from raygeo.pipeline.stage import StageSpec

    step = _TestStep(name="s1")
    wp1 = WorkPiece(name="wp1")
    doc = _make_doc(step, wp1)

    nodes = IntentBuilder().build(doc)
    job_node = next(n for n in nodes if n.key == job_key())
    assert isinstance(job_node.stage, StageSpec.Aggregate)
