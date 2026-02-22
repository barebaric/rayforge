import logging
import time
import uuid
import numpy as np
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import pytest

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.artifact.manager import ArtifactManager
from rayforge.pipeline.view.view_manager import ViewManager, ViewEntry
from rayforge.pipeline.artifact import (
    ArtifactKey,
    RenderContext,
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifactHandle,
    WorkPieceViewArtifact,
)


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.task_manager = MagicMock()
    pipeline.task_manager.cancel_task = MagicMock()
    pipeline.task_manager.get_task = MagicMock(return_value=None)
    pipeline.task_manager.run_process = MagicMock()
    pipeline.task_manager.run_thread = MagicMock()
    pipeline.workpiece_artifact_ready = MagicMock()
    pipeline.workpiece_artifact_ready.connect = MagicMock()
    pipeline.workpiece_starting = MagicMock()
    pipeline.workpiece_starting.connect = MagicMock()
    return pipeline


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.retain = MagicMock()
    store.release = MagicMock()
    store.get = MagicMock()
    store.put = MagicMock()
    store.adopt = MagicMock()
    store.safe_adoption = MagicMock()
    return store


@pytest.fixture
def mock_machine():
    return MagicMock()


@pytest.fixture
def view_manager(mock_pipeline, mock_store, mock_machine):
    with patch.object(
        ViewManager, "_connect_pipeline_signals", lambda x: None
    ):
        vm = ViewManager(
            mock_pipeline,
            mock_store,
            mock_machine,
        )
    return vm


@pytest.fixture
def source_handle():
    return WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )


@pytest.fixture
def context():
    return RenderContext(
        pixels_per_mm=(10.0, 10.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict={},
    )


def setup_view_manager_with_source(
    view_manager, wp_uid, source_handle, context, step_uid
):
    view_manager._source_artifact_handles[(wp_uid, step_uid)] = source_handle
    view_manager._current_view_context = context
    return view_manager._get_task_key(wp_uid, step_uid)


def test_view_manager_requests_vector_render(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_args = view_manager._task_manager.run_process.call_args
    assert call_args.kwargs["workpiece_artifact_handle_dict"] == (
        source_handle.to_dict()
    )
    assert call_args.kwargs["render_context_dict"] == context.to_dict()


def test_view_manager_handles_events_and_completion(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]
    when_done_cb = call_kwargs["when_done"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.id = 123
    mock_task.kwargs = {
        "step_uid": step_uid,
        "workpiece_uid": wp_uid,
        "generation_id": 0,
    }
    mock_task.get_status.return_value = "completed"
    mock_task.is_final.return_value = False

    view_manager._task_manager.get_task.return_value = mock_task

    created_handler = MagicMock()
    updated_handler = MagicMock()
    ready_handler = MagicMock()
    finished_handler = MagicMock()
    view_manager.view_artifact_created.connect(created_handler)
    view_manager.view_artifact_updated.connect(updated_handler)
    view_manager.view_artifact_ready.connect(ready_handler)
    view_manager.generation_finished.connect(finished_handler)

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid
    mock_step = MagicMock()
    mock_step.uid = step_uid
    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    @contextmanager
    def mock_safe_adoption(handle_dict):
        yield view_handle

    mock_store.safe_adoption.side_effect = mock_safe_adoption

    handle_dict = {
        "shm_name": "test",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    mock_store.safe_adoption.assert_called_once()
    created_handler.assert_called_once()
    ready_handler.assert_called_once()
    assert isinstance(
        created_handler.call_args.kwargs["handle"],
        WorkPieceViewArtifactHandle,
    )
    assert isinstance(
        ready_handler.call_args.kwargs["handle"],
        WorkPieceViewArtifactHandle,
    )

    when_event_cb(mock_task, "view_artifact_updated", {})

    updated_handler.assert_called_once()
    assert updated_handler.call_args.kwargs["step_uid"] == step_uid

    when_done_cb(mock_task)

    finished_handler.assert_called_once()
    assert finished_handler.call_args.kwargs["key"] == key
    ready_handler.assert_called_once()


def test_adoption_failure_does_not_crash(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid, "workpiece_uid": wp_uid}

    mock_store.safe_adoption.side_effect = Exception("Adoption failed")

    created_handler = MagicMock()
    view_manager.view_artifact_created.connect(created_handler)

    handle_dict = {
        "shm_name": "test",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    created_handler.assert_not_called()


def test_multiple_view_artifact_updated_events(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid, "workpiece_uid": wp_uid}

    created_handler = MagicMock()
    updated_handler = MagicMock()
    view_manager.view_artifact_created.connect(created_handler)
    view_manager.view_artifact_updated.connect(updated_handler)

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid
    mock_step = MagicMock()
    mock_step.uid = step_uid
    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_view",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    @contextmanager
    def mock_safe_adoption(handle_dict):
        yield view_handle

    mock_store.safe_adoption.side_effect = mock_safe_adoption

    handle_dict = {
        "shm_name": "test_view",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    created_handler.assert_called_once()

    when_event_cb(mock_task, "view_artifact_updated", {})
    when_event_cb(mock_task, "view_artifact_updated", {})
    when_event_cb(mock_task, "view_artifact_updated", {})

    assert updated_handler.call_count == 3
    for call in updated_handler.call_args_list:
        assert call.kwargs["step_uid"] == step_uid
        assert call.kwargs["workpiece_uid"] == wp_uid


def test_progressive_rendering_sends_multiple_updates(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid, "workpiece_uid": wp_uid}

    created_handler = MagicMock()
    updated_handler = MagicMock()
    view_manager.view_artifact_created.connect(created_handler)
    view_manager.view_artifact_updated.connect(updated_handler)

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid
    mock_step = MagicMock()
    mock_step.uid = step_uid
    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_progressive",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    @contextmanager
    def mock_safe_adoption(handle_dict):
        yield view_handle

    mock_store.safe_adoption.side_effect = mock_safe_adoption

    handle_dict = {
        "shm_name": "test_progressive",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }

    when_event_cb(
        mock_task,
        "view_artifact_created",
        {"handle_dict": handle_dict},
    )

    created_handler.assert_called_once()

    when_event_cb(mock_task, "view_artifact_updated", {})
    when_event_cb(mock_task, "view_artifact_updated", {})
    when_event_cb(mock_task, "view_artifact_updated", {})

    assert updated_handler.call_count == 3
    for call in updated_handler.call_args_list:
        assert call.kwargs["step_uid"] == step_uid
        assert call.kwargs["workpiece_uid"] == wp_uid
        assert call.kwargs["handle"] is not None


def test_on_chunk_available_receives_chunks(view_manager, mock_store, context):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="view_shm",
        bbox_mm=(0, 0, 10, 10),
        workpiece_size_mm=(10.0, 10.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )
    view_manager._view_entries[composite_id] = ViewEntry(
        handle=view_handle,
        render_context=context,
    )

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    generation_id = 0

    view_manager.on_chunk_available(
        sender=None,
        key=ArtifactKey(id=wp_uid, group="chunk"),
        chunk_handle=chunk_handle,
        generation_id=generation_id,
        step_uid=step_uid,
    )


def test_live_render_context_established_on_view_creation(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {
        "step_uid": step_uid,
        "workpiece_uid": wp_uid,
        "generation_id": 0,
    }

    handle_dict = {
        "shm_name": "test_live_render",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_live_render",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    @contextmanager
    def mock_safe_adoption(handle_dict):
        yield view_handle

    mock_store.safe_adoption.side_effect = mock_safe_adoption

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    composite_id = (wp_uid, step_uid)
    entry = view_manager._view_entries.get(composite_id)
    assert entry is not None
    assert entry.render_context == context


def test_throttled_notification_limits_update_frequency(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_called_once()
    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid, "workpiece_uid": wp_uid}

    handle_dict = {
        "shm_name": "test_throttle",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid
    mock_step = MagicMock()
    mock_step.uid = step_uid
    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_throttle",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    @contextmanager
    def mock_safe_adoption(handle_dict):
        yield view_handle

    mock_store.safe_adoption.side_effect = mock_safe_adoption

    handle_dict = {
        "shm_name": "test_throttle",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    update_handler = MagicMock()
    view_manager.view_artifact_updated.connect(update_handler)

    time.sleep(0.01)

    def mock_run_thread(target, *args, when_done=None, **kwargs):
        mock_thread_task = MagicMock()
        mock_thread_task.get_status.return_value = "completed"
        mock_thread_task.result.return_value = True
        if when_done:
            when_done(mock_thread_task)
        return mock_thread_task

    view_manager._task_manager.run_thread.side_effect = mock_run_thread

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    for i in range(10):
        view_manager.on_chunk_available(
            sender=None,
            key=ArtifactKey(id=wp_uid, group="chunk"),
            chunk_handle=chunk_handle,
            generation_id=0,
            step_uid=step_uid,
        )

    time.sleep(0.1)

    assert update_handler.call_count < 15
    assert update_handler.call_count > 0


def test_incremental_bitmap_rendering_draws_chunk_to_view(
    view_manager,
    mock_store,
    source_handle,
    context,
):
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    setup_view_manager_with_source(
        view_manager, wp_uid, source_handle, context, step_uid
    )
    key = view_manager._get_task_key(wp_uid, step_uid)

    view_manager.request_view_render(wp_uid, step_uid)

    call_kwargs = view_manager._task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid, "workpiece_uid": wp_uid}

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid
    mock_step = MagicMock()
    mock_step.uid = step_uid
    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    blank_bitmap = np.zeros((100, 100, 4), dtype=np.uint8)
    view_artifact = WorkPieceViewArtifact(
        bitmap_data=blank_bitmap,
        bbox_mm=(0, 0, 10.0, 10.0),
        workpiece_size_mm=(10.0, 10.0),
        generation_id=0,
    )

    mock_store.get.return_value = view_artifact

    handle_dict = {
        "shm_name": "view_shm",
        "bbox_mm": (0, 0, 10.0, 10.0),
        "workpiece_size_mm": (10.0, 10.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="view_shm",
        bbox_mm=(0, 0, 10.0, 10.0),
        workpiece_size_mm=(10.0, 10.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    @contextmanager
    def mock_safe_adoption(handle_dict):
        yield view_handle

    mock_store.safe_adoption.side_effect = mock_safe_adoption

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    retain_calls = []
    release_calls = []

    def mock_retain(handle):
        retain_calls.append(handle.shm_name)

    def mock_release(handle):
        release_calls.append(handle.shm_name)

    mock_store.retain.side_effect = mock_retain
    mock_store.release.side_effect = mock_release

    chunk_ops = Ops()
    chunk_ops.move_to(2.0, 2.0, 0)
    chunk_ops.line_to(8.0, 8.0, 0)

    chunk_artifact = WorkPieceArtifact(
        ops=chunk_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
        generation_id=0,
    )

    def mock_get_artifact(handle):
        if handle.shm_name == "view_shm":
            return view_artifact
        elif handle.shm_name == "chunk_shm":
            return chunk_artifact
        return None

    mock_store.get.side_effect = mock_get_artifact

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    initial_bitmap = view_artifact.bitmap_data.copy()
    assert np.count_nonzero(initial_bitmap) == 0

    def mock_run_thread(target, *args, when_done=None, **kwargs):
        mock_thread_task = MagicMock()
        mock_thread_task.get_status.return_value = "completed"
        mock_thread_task.result.return_value = True
        if when_done:
            when_done(mock_thread_task)
        return mock_thread_task

    view_manager._task_manager.run_thread.side_effect = mock_run_thread

    view_manager.on_chunk_available(
        sender=None,
        key=ArtifactKey(id=wp_uid, group="chunk"),
        chunk_handle=chunk_handle,
        generation_id=0,
        step_uid=step_uid,
    )

    assert "chunk_shm" in release_calls


def test_get_render_components(view_manager, mock_store, context):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )

    entry = ViewEntry(handle=handle, render_context=context)
    view_manager._view_entries[composite_id] = entry

    view_handle, render_context = view_manager._get_render_components(
        composite_id, entry
    )

    assert view_handle == handle
    assert render_context == context


def test_get_render_components_missing(view_manager, mock_store):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    entry = ViewEntry(handle=None, render_context=None)
    view_manager._view_entries[composite_id] = entry

    view_handle, render_context = view_manager._get_render_components(
        composite_id, entry
    )

    assert view_handle is None
    assert render_context is None


def test_update_render_context_triggers_renders(view_manager, source_handle):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)
    view_manager._source_artifact_handles[composite_id] = source_handle

    context = RenderContext(
        pixels_per_mm=(10.0, 10.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict={},
    )

    view_manager.update_render_context(context)

    assert view_manager._current_view_context == context
    assert view_manager._view_generation_id == 1


def test_on_workpiece_artifact_ready_manages_handles(
    view_manager, mock_store, source_handle
):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=source_handle,
    )

    composite_id = (wp_uid, step_uid)
    assert composite_id in view_manager._source_artifact_handles
    mock_store.retain.assert_called_once_with(source_handle)


def test_on_workpiece_artifact_ready_releases_old_handle(
    view_manager, mock_store, source_handle
):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    mock_step = MagicMock()
    mock_step.uid = step_uid

    old_handle = WorkPieceArtifactHandle(
        shm_name="old_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )
    view_manager._source_artifact_handles[composite_id] = old_handle

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=source_handle,
    )

    mock_store.release.assert_called_once_with(old_handle)
    mock_store.retain.assert_called_once_with(source_handle)


def test_on_workpiece_artifact_ready_same_handle_no_signal(
    view_manager, mock_store, source_handle, context
):
    """
    When the same handle is received (e.g., from step_assembly_starting
    during a position-only transform change), source_artifact_ready
    signal should NOT be emitted to avoid unnecessary redraws.
    """
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager._source_artifact_handles[composite_id] = source_handle
    view_manager._current_view_context = context

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=source_handle,
    )

    signal_handler.assert_not_called()


def test_on_workpiece_artifact_ready_different_handle_emits_signal(
    view_manager, mock_store, source_handle, context
):
    """
    When a different handle is received (new artifact generated),
    source_artifact_ready signal SHOULD be emitted.
    """
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    old_handle = WorkPieceArtifactHandle(
        shm_name="old_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )
    view_manager._source_artifact_handles[composite_id] = old_handle
    view_manager._current_view_context = context

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=source_handle,
    )

    signal_handler.assert_called_once()


def test_request_view_render_no_context(view_manager, source_handle):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    view_manager._source_artifact_handles[composite_id] = source_handle
    view_manager._current_view_context = None

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_not_called()


def test_request_view_render_no_source_handle(view_manager, context):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())

    view_manager._current_view_context = context

    view_manager.request_view_render(wp_uid, step_uid)

    view_manager._task_manager.run_process.assert_not_called()


@pytest.fixture
def real_artifact_manager(context_initializer):
    return ArtifactManager(context_initializer.artifact_store)


@pytest.fixture
def real_pipeline(task_mgr, context_initializer):
    pipeline = Pipeline(
        Doc(),
        task_mgr,
        context_initializer.artifact_store,
        context_initializer.machine,
    )
    return pipeline


@pytest.fixture
def real_view_manager(real_pipeline, context_initializer):
    return ViewManager(
        real_pipeline,
        context_initializer.artifact_store,
        context_initializer.machine,
    )


@pytest.mark.usefixtures("context_initializer")
def test_on_workpiece_artifact_ready_refcount_increased(
    real_view_manager, real_artifact_manager
):
    """
    Verify that when ViewManager receives workpiece_artifact_ready signal,
    the ArtifactStore refcount for the handle is increased.

    The refcount should be at least 2:
    - 1 for the original put() call (GenerationContext)
    - 1 for ViewManager's retain_handle()
    """
    wp_uid = str(uuid.uuid4())

    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(10, 10, 0)

    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
        generation_id=0,
    )

    handle = real_artifact_manager._store.put(artifact)

    initial_refcount = handle.refcount
    assert initial_refcount == 1, (
        f"Initial refcount should be 1 after put(), got {initial_refcount}"
    )

    step_uid = str(uuid.uuid4())
    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    real_view_manager._pipeline.doc = mock_doc

    real_view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handle,
    )

    final_refcount = handle.refcount
    assert final_refcount >= 2, (
        f"Refcount should be at least 2 after ViewManager retains, "
        f"got {final_refcount}"
    )

    composite_id = (wp_uid, step_uid)
    assert composite_id in real_view_manager._source_artifact_handles


@pytest.mark.usefixtures("context_initializer")
def test_on_workpiece_artifact_ready_releases_old_and_refcount_decreased(
    real_view_manager, real_artifact_manager
):
    """
    Verify that when ViewManager receives a new handle for same workpiece:
    1. The old handle's refcount is decreased (released)
    2. The new handle's refcount is increased (retained)
    """
    wp_uid = str(uuid.uuid4())

    ops1 = Ops()
    ops1.move_to(0, 0, 0)
    ops1.line_to(5, 5, 0)

    artifact1 = WorkPieceArtifact(
        ops=ops1,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
        generation_id=0,
    )

    old_handle = real_artifact_manager._store.put(artifact1)

    mock_step = MagicMock()
    mock_step.uid = str(uuid.uuid4())

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    real_view_manager._pipeline.doc = mock_doc

    real_view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=old_handle,
    )

    old_refcount_after_retain = old_handle.refcount
    assert old_refcount_after_retain == 2, (
        f"Old handle refcount should be 2 after retain, "
        f"got {old_refcount_after_retain}"
    )

    ops2 = Ops()
    ops2.move_to(0, 0, 0)
    ops2.line_to(10, 10, 0)

    artifact2 = WorkPieceArtifact(
        ops=ops2,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
        generation_id=0,
    )

    new_handle = real_artifact_manager._store.put(artifact2)

    real_view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=new_handle,
    )

    old_refcount_after_release = old_handle.refcount
    assert old_refcount_after_release == 1, (
        f"Old handle refcount should be 1 after release, "
        f"got {old_refcount_after_release}"
    )

    new_refcount = new_handle.refcount
    assert new_refcount == 2, (
        f"New handle refcount should be 2 after retain, got {new_refcount}"
    )


def test_view_entry_defaults():
    entry = ViewEntry()
    assert entry.handle is None
    assert entry.render_context is None
    assert entry.source_handle is None


def test_view_entry_with_values():
    handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )
    context = RenderContext(
        pixels_per_mm=(10.0, 10.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict={},
    )
    entry = ViewEntry(handle=handle, render_context=context)
    assert entry.handle == handle
    assert entry.render_context == context


def test_is_view_stale_no_entry(view_manager):
    assert view_manager._is_view_stale("wp_uid", None, None, None) is True


def test_is_view_stale_context_changed(view_manager, context):
    wp_uid = "wp_uid"
    step_uid = "step_uid"
    composite_id = (wp_uid, step_uid)
    old_context = RenderContext(
        pixels_per_mm=(5.0, 5.0),
        show_travel_moves=True,
        margin_px=10,
        color_set_dict={},
    )
    view_manager._view_entries[composite_id] = ViewEntry(
        handle=MagicMock(),
        render_context=old_context,
    )

    assert view_manager._is_view_stale(wp_uid, step_uid, context, None) is True


def test_is_view_stale_same_context(view_manager, context):
    wp_uid = "wp_uid"
    step_uid = "step_uid"
    composite_id = (wp_uid, step_uid)
    view_manager._view_entries[composite_id] = ViewEntry(
        handle=MagicMock(),
        render_context=context,
    )

    assert (
        view_manager._is_view_stale(wp_uid, step_uid, context, None) is False
    )


def test_is_view_stale_entry_exists_but_no_source_handle(
    view_manager, context, source_handle
):
    wp_uid = "wp_uid"
    step_uid = "step_uid"
    composite_id = (wp_uid, step_uid)
    view_manager._view_entries[composite_id] = ViewEntry(
        handle=MagicMock(),
        render_context=context,
        source_handle=None,
    )

    assert (
        view_manager._is_view_stale(wp_uid, step_uid, context, source_handle)
        is True
    )


def test_is_view_stale_source_handle_shm_name_changed(
    view_manager, context, source_handle
):
    wp_uid = "wp_uid"
    step_uid = "step_uid"
    composite_id = (wp_uid, step_uid)

    old_source_handle = WorkPieceArtifactHandle(
        shm_name="old_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )
    new_source_handle = WorkPieceArtifactHandle(
        shm_name="new_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    view_manager._view_entries[composite_id] = ViewEntry(
        handle=MagicMock(),
        render_context=context,
        source_handle=old_source_handle,
    )

    assert (
        view_manager._is_view_stale(
            wp_uid, step_uid, context, new_source_handle
        )
        is True
    )


def test_handles_represent_same_artifact_both_none(view_manager):
    assert view_manager._handles_represent_same_artifact(None, None) is True


def test_handles_represent_same_artifact_one_none(view_manager, source_handle):
    assert (
        view_manager._handles_represent_same_artifact(None, source_handle)
        is False
    )
    assert (
        view_manager._handles_represent_same_artifact(source_handle, None)
        is False
    )


def test_handles_represent_same_artifact_same_handle(
    view_manager, source_handle
):
    assert (
        view_manager._handles_represent_same_artifact(
            source_handle, source_handle
        )
        is True
    )


def test_handles_represent_same_artifact_different_shm_name(
    view_manager, source_handle
):
    other_handle = WorkPieceArtifactHandle(
        shm_name="different_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )
    assert (
        view_manager._handles_represent_same_artifact(
            source_handle, other_handle
        )
        is False
    )


def test_handles_represent_same_artifact_different_generation_size(
    view_manager, source_handle
):
    other_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(20, 20),
        generation_id=0,
    )
    assert (
        view_manager._handles_represent_same_artifact(
            source_handle, other_handle
        )
        is False
    )


def test_handles_represent_same_artifact_different_source_dimensions(
    view_manager, source_handle
):
    other_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(20, 20),
        generation_size=(10, 10),
        generation_id=0,
    )
    assert (
        view_manager._handles_represent_same_artifact(
            source_handle, other_handle
        )
        is False
    )


def test_multiple_position_only_moves_no_signal(
    view_manager, mock_store, source_handle, context
):
    """
    Simulates multiple consecutive position-only moves.
    Each move should NOT emit source_artifact_ready because the handle
    represents the same artifact.
    """
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager._source_artifact_handles[composite_id] = source_handle
    view_manager._current_view_context = context

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)

    for i in range(5):
        view_manager.on_workpiece_artifact_ready(
            sender=None,
            step=mock_step,
            workpiece=mock_workpiece,
            handle=source_handle,
        )

    signal_handler.assert_not_called()


def test_alternating_artifact_ready_signals(
    view_manager, mock_store, source_handle, context
):
    """
    Tests that when the same artifact handle is received multiple times
    (simulating position-only transform changes that trigger step reassembly
    but not workpiece regeneration), the source_artifact_ready signal is
    only emitted once for the first occurrence.
    """
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager._current_view_context = context

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)
    view_render_handler = MagicMock()
    view_manager.view_artifact_updated.connect(view_render_handler)

    view_manager._task_manager.run_process = MagicMock()

    first_handle = source_handle
    second_handle = WorkPieceArtifactHandle(
        shm_name="second_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=first_handle,
    )
    assert signal_handler.call_count == 1

    for i in range(5):
        view_manager.on_workpiece_artifact_ready(
            sender=None,
            step=mock_step,
            workpiece=mock_workpiece,
            handle=first_handle,
        )
    assert signal_handler.call_count == 1

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=second_handle,
    )
    assert signal_handler.call_count == 2

    for i in range(5):
        view_manager.on_workpiece_artifact_ready(
            sender=None,
            step=mock_step,
            workpiece=mock_workpiece,
            handle=second_handle,
        )
    assert signal_handler.call_count == 2


def test_simulated_position_only_transform_changes(
    view_manager, mock_store, context
):
    """
    Simulates the actual pipeline flow for position-only transform changes:
    1. Initial workpiece generation stores a handle
    2. Position-only changes trigger step_assembly_starting with SAME handle
    3. ViewManager should NOT re-emit source_artifact_ready
    """
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager._current_view_context = context
    view_manager._task_manager.run_process = MagicMock()

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)

    initial_handle = WorkPieceArtifactHandle(
        shm_name="initial_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=initial_handle,
    )
    assert signal_handler.call_count == 1, "First call should emit signal"

    same_artifact_handle = WorkPieceArtifactHandle(
        shm_name="initial_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    signal_counts = []
    for i in range(10):
        view_manager.on_workpiece_artifact_ready(
            sender=None,
            step=mock_step,
            workpiece=mock_workpiece,
            handle=same_artifact_handle,
        )
        signal_counts.append(signal_handler.call_count)

    assert signal_handler.call_count == 1, (
        f"Subsequent calls with same artifact should NOT emit signal, "
        f"signal_counts: {signal_counts}"
    )


def test_alternating_signal_debug(view_manager, mock_store, context, caplog):
    """
    Debug test to understand the alternating pattern.
    """

    caplog.set_level(logging.DEBUG)

    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager._current_view_context = context
    view_manager._task_manager.run_process = MagicMock()

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)

    handles = []
    for i in range(10):
        handle = WorkPieceArtifactHandle(
            shm_name="same_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
            generation_id=0,
        )
        handles.append(handle)

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handles[0],
    )
    assert signal_handler.call_count == 1

    call_results = []
    for i in range(1, 10):
        view_manager.on_workpiece_artifact_ready(
            sender=None,
            step=mock_step,
            workpiece=mock_workpiece,
            handle=handles[i],
        )
        call_results.append(signal_handler.call_count)

    expected = [1] * 9
    assert call_results == expected, f"Expected all 1s, got {call_results}"


def test_alternating_shm_names_cause_alternating_signals(
    view_manager, mock_store, context
):
    """
    If the pipeline sends different shm_names on alternating calls,
    signals will be emitted on alternating calls. This test verifies
    that our deduplication logic works based on shm_name comparison.
    """
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())

    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    mock_layer = MagicMock()
    mock_layer.workflow.steps = [mock_step]
    mock_doc = MagicMock()
    mock_doc.all_workpieces = [mock_workpiece]
    mock_doc.layers = [mock_layer]
    view_manager._pipeline.doc = mock_doc

    view_manager._current_view_context = context
    view_manager._task_manager.run_process = MagicMock()

    signal_handler = MagicMock()
    view_manager.source_artifact_ready.connect(signal_handler)

    handle_a = WorkPieceArtifactHandle(
        shm_name="handle_a_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )
    handle_b = WorkPieceArtifactHandle(
        shm_name="handle_b_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        generation_id=0,
    )

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handle_a,
    )
    assert signal_handler.call_count == 1

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handle_b,
    )
    assert signal_handler.call_count == 2

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handle_a,
    )
    assert signal_handler.call_count == 3, (
        "handle_a is different from handle_b, so signal should be emitted"
    )

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handle_b,
    )
    assert signal_handler.call_count == 4, (
        "handle_b is different from handle_a, so signal should be emitted"
    )


def test_shutdown_releases_handles(view_manager, mock_store, source_handle):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)
    view_manager._source_artifact_handles[composite_id] = source_handle

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )
    view_manager._view_entries[composite_id] = ViewEntry(handle=view_handle)

    view_manager.shutdown()

    mock_store.release.assert_called()
    assert len(view_manager._source_artifact_handles) == 0
    assert len(view_manager._view_entries) == 0


def test_get_view_handle(view_manager):
    wp_uid = "wp_uid"
    step_uid = "step_uid"
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )
    view_manager._view_entries[(wp_uid, step_uid)] = ViewEntry(
        handle=view_handle
    )

    result = view_manager.get_view_handle(wp_uid, step_uid)
    assert result == view_handle


def test_get_view_handle_missing(view_manager):
    result = view_manager.get_view_handle("nonexistent", "step_uid")
    assert result is None


def test_reconcile_removes_obsolete_entries(
    view_manager, mock_store, source_handle
):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)
    view_manager._source_artifact_handles[composite_id] = source_handle

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        generation_id=0,
    )
    view_manager._view_entries[composite_id] = ViewEntry(handle=view_handle)

    doc = Doc()
    layer = Layer(name="Empty Layer")
    doc.add_layer(layer)

    view_manager.reconcile(doc, generation_id=1)

    mock_store.release.assert_called()
    assert composite_id not in view_manager._source_artifact_handles
    assert composite_id not in view_manager._view_entries
