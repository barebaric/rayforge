import time
import uuid
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from rayforge.pipeline.view.view_manager import ViewManager, ViewEntry
from rayforge.pipeline.artifact import (
    ArtifactKey,
    RenderContext,
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifactHandle,
    WorkPieceViewArtifact,
)
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem


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

    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_store.adopt.return_value = view_handle

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

    mock_store.adopt.assert_called_once()
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

    mock_store.adopt.side_effect = Exception("Adoption failed")

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

    handle_dict = {
        "shm_name": "test_view",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_view",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_store.adopt.return_value = view_handle
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

    handle_dict = {
        "shm_name": "test_progressive",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_progressive",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_store.adopt.return_value = view_handle
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
    )
    mock_store.adopt.return_value = view_handle

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
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_throttle",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_store.adopt.return_value = view_handle

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

    blank_bitmap = np.zeros((100, 100, 4), dtype=np.uint8)
    view_artifact = WorkPieceViewArtifact(
        bitmap_data=blank_bitmap,
        bbox_mm=(0, 0, 10.0, 10.0),
        workpiece_size_mm=(10.0, 10.0),
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
    )
    mock_store.adopt.return_value = view_handle

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


def test_adopt_view_handle(view_manager, mock_store):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    key = view_manager._get_task_key(wp_uid, step_uid)

    handle_dict = {
        "shm_name": "test_adopt",
        "bbox_mm": (0, 0, 1, 1),
        "workpiece_size_mm": (1.0, 1.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_adopt",
        bbox_mm=(0, 0, 1, 1),
        workpiece_size_mm=(1.0, 1.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_store.adopt.return_value = view_handle

    result = view_manager._adopt_view_handle(key, {"handle_dict": handle_dict})

    assert result is not None
    assert isinstance(result, WorkPieceViewArtifactHandle)
    mock_store.adopt.assert_called_once()


def test_adopt_view_handle_wrong_type(view_manager, mock_store):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    key = view_manager._get_task_key(wp_uid, step_uid)

    handle_dict = {
        "shm_name": "test_wrong",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceArtifactHandle",
        "artifact_type_name": "WorkPieceArtifact",
    }
    wrong_handle = WorkPieceArtifactHandle(
        shm_name="test_wrong",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_store.adopt.return_value = wrong_handle

    with pytest.raises(TypeError):
        view_manager._adopt_view_handle(key, {"handle_dict": handle_dict})


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
    )

    entry = ViewEntry(handle=handle, render_context=context)
    view_manager._view_entries[composite_id] = entry

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    view_handle, render_context = view_manager._get_render_components(
        composite_id, entry, chunk_handle
    )

    assert view_handle == handle
    assert render_context == context


def test_get_render_components_missing(view_manager, mock_store):
    wp_uid = str(uuid.uuid4())
    step_uid = str(uuid.uuid4())
    composite_id = (wp_uid, step_uid)

    entry = ViewEntry(handle=None, render_context=None)
    view_manager._view_entries[composite_id] = entry

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    view_handle, render_context = view_manager._get_render_components(
        composite_id, entry, chunk_handle
    )

    assert view_handle is None
    assert render_context is None
    mock_store.release.assert_called_once_with(chunk_handle)


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
    )
    view_manager._source_artifact_handles[composite_id] = old_handle

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=source_handle,
    )

    mock_store.release.assert_called_once_with(old_handle)
    mock_store.retain.assert_called_once_with(source_handle)


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
    from rayforge.pipeline.artifact.manager import ArtifactManager

    return ArtifactManager(context_initializer.artifact_store)


@pytest.fixture
def real_pipeline(task_mgr, context_initializer):
    from rayforge.core.doc import Doc
    from rayforge.pipeline.pipeline import Pipeline

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
    from rayforge.core.ops import Ops
    from rayforge.pipeline import CoordinateSystem
    from rayforge.pipeline.artifact import WorkPieceArtifact

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
    )

    handle = real_artifact_manager._store.put(artifact)
    shm_name = handle.shm_name

    initial_refcount = real_artifact_manager._store._refcounts.get(shm_name, 1)
    assert initial_refcount == 1, (
        f"Initial refcount should be 1 after put(), got {initial_refcount}"
    )

    step_uid = str(uuid.uuid4())
    mock_step = MagicMock()
    mock_step.uid = step_uid

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    real_view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=handle,
    )

    final_refcount = real_artifact_manager._store._refcounts.get(shm_name, 0)
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
    from rayforge.core.ops import Ops
    from rayforge.pipeline import CoordinateSystem
    from rayforge.pipeline.artifact import WorkPieceArtifact

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
    )

    old_handle = real_artifact_manager._store.put(artifact1)
    old_shm_name = old_handle.shm_name

    mock_step = MagicMock()
    mock_step.uid = str(uuid.uuid4())

    mock_workpiece = MagicMock()
    mock_workpiece.uid = wp_uid

    real_view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=old_handle,
    )

    old_refcount_after_retain = real_artifact_manager._store._refcounts.get(
        old_shm_name, 0
    )
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
    )

    new_handle = real_artifact_manager._store.put(artifact2)
    new_shm_name = new_handle.shm_name

    real_view_manager.on_workpiece_artifact_ready(
        sender=None,
        step=mock_step,
        workpiece=mock_workpiece,
        handle=new_handle,
    )

    old_refcount_after_release = real_artifact_manager._store._refcounts.get(
        old_shm_name, 0
    )
    assert old_refcount_after_release == 1, (
        f"Old handle refcount should be 1 after release, "
        f"got {old_refcount_after_release}"
    )

    new_refcount = real_artifact_manager._store._refcounts.get(new_shm_name, 0)
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
    from rayforge.core.doc import Doc
    from rayforge.core.layer import Layer

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
    )
    view_manager._view_entries[composite_id] = ViewEntry(handle=view_handle)

    doc = Doc()
    layer = Layer(name="Empty Layer")
    doc.add_layer(layer)

    view_manager.reconcile(doc, generation_id=1)

    mock_store.release.assert_called()
    assert composite_id not in view_manager._source_artifact_handles
    assert composite_id not in view_manager._view_entries
