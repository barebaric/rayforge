"""Tests for pipeline invalidation logic and signal handlers."""

import pytest
from unittest.mock import MagicMock, patch
from rayforge.core.doc import Doc
from rayforge.core.group import Group
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.core.matrix import Matrix
from rayforge.pipeline.pipeline import Pipeline, InvalidationScope
from rayforge.pipeline.artifact import ArtifactKey


@pytest.fixture
def mock_task_mgr():
    """Creates a MagicMock for the TaskManager."""
    mock_mgr = MagicMock()

    def schedule_awarely(callback, *args, **kwargs):
        callback(*args, **kwargs)

    mock_mgr.schedule_on_main_thread = MagicMock(side_effect=schedule_awarely)
    mock_mgr.run_process = MagicMock()
    return mock_mgr


@pytest.fixture
def mock_machine():
    """Creates a mock machine for testing."""
    machine = MagicMock()
    machine.uid = "test-machine-uid"
    return machine


@pytest.fixture
def mock_artifact_store():
    """Creates a mock artifact store for testing."""
    store = MagicMock()
    return store


@pytest.fixture
def pipeline(mock_task_mgr, mock_machine, mock_artifact_store):
    """Creates a Pipeline instance for testing."""
    doc = Doc()
    return Pipeline(
        doc,
        mock_task_mgr,
        mock_artifact_store,
        mock_machine,
    )


@pytest.fixture
def doc_with_workpieces():
    """Creates a doc with workpieces and a step for testing."""
    doc = Doc()
    layer = doc.active_layer

    wp1 = WorkPiece(name="wp1.svg")
    wp2 = WorkPiece(name="wp2.svg")

    layer.add_workpiece(wp1)
    layer.add_workpiece(wp2)

    step = Step(typelabel="contour")
    assert layer.workflow is not None
    layer.workflow.add_step(step)

    return doc, layer, wp1, wp2, step


@pytest.fixture
def doc_with_group():
    """Creates a doc with a group containing workpieces."""
    doc = Doc()
    layer = doc.active_layer

    wp1 = WorkPiece(name="wp1.svg")
    wp2 = WorkPiece(name="wp2.svg")

    group = Group(name="test_group")
    group.add_child(wp1)
    group.add_child(wp2)

    layer.add_child(group)

    return doc, layer, group, wp1, wp2


def test_invalidation_scope_full_reproduction_exists():
    """Test FULL_REPRODUCTION enum value exists."""
    assert InvalidationScope.FULL_REPRODUCTION.value == "full_reproduction"


def test_invalidation_scope_step_only_exists():
    """Test STEP_ONLY enum value exists."""
    assert InvalidationScope.STEP_ONLY.value == "step_only"


def test_collect_single_workpiece(doc_with_workpieces, pipeline):
    """Test collecting a single workpiece."""
    _, _, wp1, _, _ = doc_with_workpieces
    result = pipeline._collect_affected_workpieces(wp1)
    assert len(result) == 1
    assert result[0] == wp1


def test_collect_from_group(doc_with_group, pipeline):
    """Test collecting workpieces from a group."""
    _, _, group, wp1, wp2 = doc_with_group
    result = pipeline._collect_affected_workpieces(group)
    assert len(result) == 2
    assert wp1 in result
    assert wp2 in result


def test_collect_from_layer(doc_with_workpieces, pipeline):
    """Test collecting workpieces from a layer."""
    doc, layer, _, _, _ = doc_with_workpieces
    result = pipeline._collect_affected_workpieces(layer)
    assert len(result) == 2


def test_workpiece_update_invalidates_workpiece(doc_with_workpieces, pipeline):
    """Test that workpiece update invalidates workpiece-step pairs."""
    _, _, wp1, _, step = doc_with_workpieces

    with patch.object(pipeline, "_invalidate_node") as mock_invalidate:
        with patch.object(
            pipeline, "_schedule_reconciliation"
        ) as mock_reconcile:
            pipeline._on_descendant_updated(
                MagicMock(),
                origin=wp1,
                parent_of_origin=wp1.parent,
            )

            wp_step_key = ArtifactKey.for_workpiece(wp1.uid, step.uid)
            mock_invalidate.assert_called_once_with(wp_step_key)
            mock_reconcile.assert_called_once()


def test_workpiece_position_only_invalidates_steps(
    doc_with_workpieces, pipeline
):
    """Test that position-only change invalidates steps, not workpiece."""
    doc, layer, wp1, _, _ = doc_with_workpieces

    old_matrix = Matrix()
    wp1.matrix = Matrix().set_translation(100, 200)

    with patch.object(pipeline, "_invalidate_node") as mock_invalidate:
        with patch.object(
            pipeline, "_schedule_reconciliation"
        ) as mock_reconcile:
            pipeline._on_descendant_transform_changed(
                MagicMock(),
                origin=wp1,
                parent_of_origin=wp1.parent,
                old_matrix=old_matrix,
            )

            step_key = ArtifactKey.for_step(list(layer.workflow.steps)[0].uid)
            mock_invalidate.assert_called_once_with(step_key)
            mock_reconcile.assert_called_once()


def test_workpiece_scale_change_invalidates_workpiece(
    doc_with_workpieces, pipeline
):
    """Test that scale change invalidates the workpiece-step pair."""
    doc, layer, wp1, _, step = doc_with_workpieces

    old_matrix = Matrix()
    wp1.matrix = Matrix.scale(2.0, 2.0)

    with patch.object(pipeline, "_invalidate_node") as mock_invalidate:
        with patch.object(
            pipeline, "_schedule_reconciliation"
        ) as mock_reconcile:
            pipeline._on_descendant_transform_changed(
                MagicMock(),
                origin=wp1,
                parent_of_origin=wp1.parent,
                old_matrix=old_matrix,
            )

            wp_step_key = ArtifactKey.for_workpiece(wp1.uid, step.uid)
            assert mock_invalidate.call_count == 2
            mock_invalidate.assert_any_call(wp_step_key)
            mock_reconcile.assert_called_once()


def test_group_transform_no_invalidations_without_old_matrix(
    doc_with_group, pipeline
):
    """Test that group transform without old_matrix doesn't invalidate."""
    doc, layer, group, wp1, wp2 = doc_with_group

    with patch.object(pipeline, "_invalidate_node") as mock_invalidate:
        with patch.object(
            pipeline, "_schedule_reconciliation"
        ) as mock_reconcile:
            pipeline._on_descendant_transform_changed(
                MagicMock(),
                origin=group,
                parent_of_origin=group.parent,
            )

            mock_invalidate.assert_not_called()
            mock_reconcile.assert_called_once()


def test_layer_transform_no_invalidations_without_old_matrix(
    doc_with_workpieces, pipeline
):
    """Test that layer transform without old_matrix invalidates steps."""
    doc, layer, wp1, wp2, _ = doc_with_workpieces

    with patch.object(pipeline, "_invalidate_node") as mock_invalidate:
        with patch.object(
            pipeline, "_schedule_reconciliation"
        ) as mock_reconcile:
            pipeline._on_descendant_transform_changed(
                MagicMock(),
                origin=layer,
                parent_of_origin=layer.parent,
            )

            step_uid = list(layer.workflow.steps)[0].uid
            step_key = ArtifactKey.for_step(step_uid)
            assert mock_invalidate.call_count == 2
            mock_invalidate.assert_any_call(step_key)
            mock_reconcile.assert_called_once()


def test_workpiece_rotation_only_invalidates_steps(
    doc_with_workpieces, pipeline
):
    """Test that rotation-only change invalidates steps, not workpiece."""
    doc, layer, wp1, _, _ = doc_with_workpieces

    old_matrix = Matrix()
    wp1.matrix = Matrix.rotation(45.0)

    with patch.object(pipeline, "_invalidate_node") as mock_invalidate:
        with patch.object(
            pipeline, "_schedule_reconciliation"
        ) as mock_reconcile:
            pipeline._on_descendant_transform_changed(
                MagicMock(),
                origin=wp1,
                parent_of_origin=wp1.parent,
                old_matrix=old_matrix,
            )

            step_uid = list(layer.workflow.steps)[0].uid
            step_key = ArtifactKey.for_step(step_uid)
            mock_invalidate.assert_called_once_with(step_key)
            mock_reconcile.assert_called_once()
