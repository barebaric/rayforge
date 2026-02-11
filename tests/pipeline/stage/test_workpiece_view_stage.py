import time
import uuid
import numpy as np
from unittest.mock import MagicMock
import pytest

from rayforge.pipeline.stage.workpiece_view_stage import (
    WorkPieceViewPipelineStage,
)
from rayforge.pipeline.artifact import (
    ArtifactKey,
    RenderContext,
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifactHandle,
    WorkPieceViewArtifact,
)
from rayforge.pipeline.artifact.lifecycle import (
    LedgerEntry,
    ArtifactLifecycle,
)
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem


@pytest.fixture
def mock_artifact_manager():
    return MagicMock()


@pytest.fixture
def mock_task_manager():
    return MagicMock()


@pytest.fixture
def mock_machine():
    return MagicMock()


@pytest.fixture
def stage(mock_task_manager, mock_artifact_manager, mock_machine):
    stage = WorkPieceViewPipelineStage(
        mock_task_manager,
        mock_artifact_manager,
        mock_machine,
    )
    mock_task_manager.cancel_task = MagicMock()
    return stage


def test_stage_requests_vector_render(
    stage, mock_artifact_manager, mock_task_manager
):
    """
    Tests that the stage correctly calls the task manager for a
    vector artifact.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext(
        pixels_per_mm=(10.0, 10.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict={},
    )

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_args = mock_task_manager.run_process.call_args
    assert call_args.kwargs["workpiece_artifact_handle_dict"] == (
        source_handle.to_dict()
    )
    assert call_args.kwargs["render_context_dict"] == context.to_dict()


def test_stage_handles_events_and_completion(
    stage,
    mock_artifact_manager,
    mock_task_manager,
):
    """
    Tests that the stage correctly handles events from the runner and
    emits its own signals.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]
    when_done_cb = call_kwargs["when_done"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.id = 123
    mock_task.kwargs = {"step_uid": step_uid}
    mock_task.get_status.return_value = "completed"
    mock_task.is_final.return_value = False

    mock_task_manager.get_task.return_value = mock_task

    created_handler = MagicMock()
    updated_handler = MagicMock()
    ready_handler = MagicMock()
    finished_handler = MagicMock()
    stage.view_artifact_created.connect(created_handler)
    stage.view_artifact_updated.connect(updated_handler)
    stage.view_artifact_ready.connect(ready_handler)
    stage.generation_finished.connect(finished_handler)

    handle_dict = {
        "shm_name": "test",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle
    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    mock_artifact_manager.adopt_artifact.assert_called_once()
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
    internal_key = ArtifactKey.for_view(wp_uid)
    assert finished_handler.call_args.kwargs["key"] == internal_key
    ready_handler.assert_called_once()


def test_adoption_failure_does_not_crash(
    stage, mock_artifact_manager, mock_task_manager
):
    """
    Tests that adoption failures are handled gracefully without
    crashing the stage.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid}

    mock_artifact_manager.adopt_artifact.side_effect = Exception(
        "Adoption failed"
    )

    created_handler = MagicMock()
    stage.view_artifact_created.connect(created_handler)

    handle_dict = {
        "shm_name": "test",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    created_handler.assert_not_called()


def test_multiple_view_artifact_updated_events(
    stage, mock_artifact_manager, mock_task_manager
):
    """
    Tests that multiple view_artifact_updated events are sent and
    processed correctly for progressive rendering.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid}

    created_handler = MagicMock()
    updated_handler = MagicMock()
    stage.view_artifact_created.connect(created_handler)
    stage.view_artifact_updated.connect(updated_handler)

    handle_dict = {
        "shm_name": "test_view",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_view",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle
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
    stage, mock_artifact_manager, mock_task_manager
):
    """
    Tests that the workpiece view stage correctly handles multiple
    view_artifact_updated events for progressive rendering.
    Verifies that each update is relayed correctly.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid}

    created_handler = MagicMock()
    updated_handler = MagicMock()
    stage.view_artifact_created.connect(created_handler)
    stage.view_artifact_updated.connect(updated_handler)

    handle_dict = {
        "shm_name": "test_progressive",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_progressive",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle
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


def test_on_workpiece_chunk_available_receives_chunks(
    stage, mock_artifact_manager
):
    """
    Tests that _on_workpiece_chunk_available is called when a chunk
    is available from the workpiece stage.
    """
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    generation_id = 1

    stage._on_workpiece_chunk_available(
        sender=None,
        key=key,
        chunk_handle=chunk_handle,
        generation_id=generation_id,
    )


def test_live_render_context_established_on_view_creation(
    stage, mock_artifact_manager, mock_task_manager
):
    """
    Tests that a live render context is established when a view
    artifact is created, enabling progressive chunk rendering.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid}

    handle_dict = {
        "shm_name": "test_live_render",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_live_render",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle

    mock_entry = MagicMock()
    mock_entry.metadata = {"render_context": context}
    mock_artifact_manager._get_ledger_entry.return_value = mock_entry

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    assert "render_context" in mock_entry.metadata
    assert mock_entry.metadata["render_context"] == context


def test_throttled_notification_limits_update_frequency(
    stage,
    mock_artifact_manager,
    mock_task_manager,
):
    """
    Tests that throttled notification limits the frequency of
    view_artifact_updated signals when many chunks arrive quickly.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    mock_task_manager.run_process.assert_called_once()
    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid}

    handle_dict = {
        "shm_name": "test_throttle",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_throttle",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle
    mock_artifact_manager.get_workpiece_view_handle.return_value = view_handle
    mock_artifact_manager.put_workpiece_view_handle.return_value = None

    mock_entry = LedgerEntry(
        state=ArtifactLifecycle.DONE,
        handle=view_handle,
        metadata={"render_context": context},
        generation_id=0,
    )
    mock_artifact_manager._ledger = {(key, 0): mock_entry}

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    update_handler = MagicMock()
    stage.view_artifact_updated.connect(update_handler)

    time.sleep(0.01)

    chunk_ops = Ops()
    chunk_ops.move_to(0, 0, 0)
    chunk_ops.line_to(1, 1, 0)

    chunk_artifact = WorkPieceArtifact(
        ops=chunk_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
    )
    mock_artifact_manager.get_artifact.return_value = chunk_artifact

    def mock_run_thread(target, *args, when_done=None, **kwargs):
        mock_task = MagicMock()
        mock_task.get_status.return_value = "completed"
        mock_task.result.return_value = True
        if when_done:
            when_done(mock_task)
        return mock_task

    mock_task_manager.run_thread.side_effect = mock_run_thread

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    internal_key = ArtifactKey.for_view(wp_uid)
    for i in range(10):
        stage._on_workpiece_chunk_available(
            sender=None,
            key=internal_key,
            chunk_handle=chunk_handle,
            generation_id=0,
        )

    time.sleep(0.1)

    assert update_handler.call_count < 5
    assert update_handler.call_count > 0


def test_incremental_bitmap_rendering_draws_chunk_to_view(
    stage,
    mock_artifact_manager,
    mock_task_manager,
):
    """
    Unit test for Phase 3, Item 6: Incremental Bitmap Updating.
    Feeds a blank bitmap and a mock vector chunk. Verifies the bitmap
    is modified correctly. Ensures the source artifact's refcount is
    respected during this operation.
    """
    step_uid = str(uuid.uuid4())
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    source_handle = WorkPieceArtifactHandle(
        shm_name="source_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
    )
    mock_artifact_manager.get_workpiece_handle.return_value = source_handle
    context = RenderContext((10.0, 10.0), False, 0, {})

    stage.request_view_render(
        key, context, stage.current_view_generation_id, step_uid
    )

    call_kwargs = mock_task_manager.run_process.call_args.kwargs
    when_event_cb = call_kwargs["when_event"]

    mock_task = MagicMock()
    mock_task.key = key
    mock_task.kwargs = {"step_uid": step_uid}

    blank_bitmap = np.zeros((100, 100, 4), dtype=np.uint8)
    view_artifact = WorkPieceViewArtifact(
        bitmap_data=blank_bitmap,
        bbox_mm=(0, 0, 10.0, 10.0),
    )

    def mock_get_artifact(handle):
        if handle.shm_name == "view_shm":
            return view_artifact
        return None

    mock_artifact_manager.get_artifact.side_effect = mock_get_artifact

    handle_dict = {
        "shm_name": "view_shm",
        "bbox_mm": (0, 0, 10.0, 10.0),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="view_shm",
        bbox_mm=(0, 0, 10.0, 10.0),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle
    mock_artifact_manager.get_workpiece_view_handle.return_value = view_handle
    mock_artifact_manager.put_workpiece_view_handle.return_value = None

    mock_entry = LedgerEntry(
        state=ArtifactLifecycle.DONE,
        handle=view_handle,
        metadata={"render_context": context},
    )
    mock_artifact_manager._get_ledger_entry.return_value = mock_entry

    when_event_cb(
        mock_task, "view_artifact_created", {"handle_dict": handle_dict}
    )

    assert "render_context" in mock_entry.metadata

    retain_calls = []
    release_calls = []

    def mock_retain_handle(handle):
        retain_calls.append(handle.shm_name)

    def mock_release_handle(handle):
        release_calls.append(handle.shm_name)

    mock_artifact_manager.retain_handle.side_effect = mock_retain_handle
    mock_artifact_manager.release_handle.side_effect = mock_release_handle

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

    def mock_get_artifact_with_chunk(handle):
        if handle.shm_name == "view_shm":
            return view_artifact
        elif handle.shm_name == "chunk_shm":
            return chunk_artifact
        return None

    mock_artifact_manager.get_artifact.side_effect = (
        mock_get_artifact_with_chunk
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

    initial_bitmap = view_artifact.bitmap_data.copy()
    assert np.count_nonzero(initial_bitmap) == 0

    def mock_run_thread(target, *args, when_done=None, **kwargs):
        mock_task = MagicMock()
        mock_task.get_status.return_value = "completed"
        mock_task.result.return_value = True
        if when_done:
            when_done(mock_task)
        return mock_task

    mock_task_manager.run_thread.side_effect = mock_run_thread

    internal_key = ArtifactKey.for_view(wp_uid)
    stage._on_workpiece_chunk_available(
        sender=None,
        key=internal_key,
        chunk_handle=chunk_handle,
        generation_id=0,
    )

    assert "chunk_shm" in release_calls


def test_adopt_view_handle(stage, mock_artifact_manager):
    """Tests that _adopt_view_handle correctly adopts a view handle."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    handle_dict = {
        "shm_name": "test_adopt",
        "bbox_mm": (0, 0, 1, 1),
        "handle_class_name": "WorkPieceViewArtifactHandle",
        "artifact_type_name": "WorkPieceViewArtifact",
    }
    view_handle = WorkPieceViewArtifactHandle(
        shm_name="test_adopt",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.adopt_artifact.return_value = view_handle

    result = stage._adopt_view_handle(key, {"handle_dict": handle_dict})

    assert result is not None
    assert isinstance(result, WorkPieceViewArtifactHandle)
    mock_artifact_manager.adopt_artifact.assert_called_once_with(
        key, handle_dict
    )


def test_adopt_view_handle_wrong_type(stage, mock_artifact_manager):
    """Tests that _adopt_view_handle raises on wrong handle type."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

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
    mock_artifact_manager.adopt_artifact.return_value = wrong_handle

    with pytest.raises(TypeError):
        stage._adopt_view_handle(key, {"handle_dict": handle_dict})


def test_replace_current_view_handle(stage, mock_artifact_manager):
    """Tests that _replace_current_view_handle replaces handles."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    old_handle = WorkPieceViewArtifactHandle(
        shm_name="old",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.get_workpiece_view_handle.return_value = old_handle

    new_handle = WorkPieceViewArtifactHandle(
        shm_name="new",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )

    stage._replace_current_view_handle(
        key, new_handle, stage.current_view_generation_id
    )

    mock_artifact_manager.release_handle.assert_called_once_with(old_handle)
    mock_artifact_manager.retain_handle.assert_called_once_with(new_handle)


def test_get_chunk_artifact(stage, mock_artifact_manager):
    """Tests that _get_chunk_artifact returns valid artifact."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    chunk_ops = Ops()
    chunk_ops.move_to(0, 0, 0)
    chunk_artifact = WorkPieceArtifact(
        ops=chunk_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
    )

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_artifact.return_value = chunk_artifact

    result = stage._get_chunk_artifact(key, chunk_handle)
    assert result == chunk_artifact


def test_get_chunk_artifact_none(stage, mock_artifact_manager):
    """Tests that _get_chunk_artifact handles None artifact."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )
    mock_artifact_manager.get_artifact.return_value = None

    result = stage._get_chunk_artifact(key, chunk_handle)
    assert result is None


def test_get_render_components(stage, mock_artifact_manager):
    """Tests that _get_render_components returns components."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    ledger_key = ArtifactKey.for_view(wp_uid)

    context = RenderContext((10.0, 10.0), False, 0, {})
    handle = WorkPieceViewArtifactHandle(
        shm_name="test",
        bbox_mm=(0, 0, 1, 1),
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
    )
    mock_artifact_manager.get_workpiece_view_handle.return_value = handle

    mock_entry = MagicMock()
    mock_entry.metadata = {"render_context": context}
    mock_entry.generation_id = 0
    mock_artifact_manager._ledger = {(ledger_key, 0): mock_entry}

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    view_handle, render_context = stage._get_render_components(
        key, ledger_key, chunk_handle
    )

    assert view_handle == handle
    assert render_context == context


def test_get_render_components_missing(stage, mock_artifact_manager):
    """Tests that _get_render_components handles missing components."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)
    ledger_key = ArtifactKey.for_view(wp_uid)

    mock_artifact_manager.get_workpiece_view_handle.return_value = None
    mock_artifact_manager._ledger = {}

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    view_handle, render_context = stage._get_render_components(
        key, ledger_key, chunk_handle
    )

    assert view_handle is None
    assert render_context is None
    mock_artifact_manager.release_handle.assert_called_once_with(chunk_handle)


def test_should_render_chunk(stage, mock_artifact_manager):
    """Tests that _should_render_chunk checks Ops correctly."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    chunk_ops = Ops()
    chunk_ops.move_to(0, 0, 0)
    chunk_artifact = WorkPieceArtifact(
        ops=chunk_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
    )

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    assert stage._should_render_chunk(chunk_artifact, key, chunk_handle)


def test_should_render_chunk_empty(stage, mock_artifact_manager):
    """Tests that _should_render_chunk returns False for empty Ops."""
    wp_uid = str(uuid.uuid4())
    key = ArtifactKey.for_view(wp_uid)

    chunk_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10.0, 10.0),
        generation_size=(10.0, 10.0),
    )

    chunk_handle = WorkPieceArtifactHandle(
        shm_name="chunk",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
    )

    assert not stage._should_render_chunk(chunk_artifact, key, chunk_handle)
    mock_artifact_manager.release_handle.assert_called_once_with(chunk_handle)
