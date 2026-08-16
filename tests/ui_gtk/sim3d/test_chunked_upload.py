# flake8: noqa: E402
"""UI tests for the ChunkedUploadController."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from rayforge.ui_gtk.sim3d.chunked_upload import (
    ChunkedUploadController,
    _UploadState,
)

if TYPE_CHECKING:
    from rayforge.ui_gtk.sim3d.renderer.scene_renderer import UploadItem


def _make_controller(artifact=None, gl_initialized=True):
    scene = MagicMock()
    scene.prepare_chunked_upload.return_value = []
    rendered = []
    luts_called = []
    uploads = []
    made_current = []

    def _on_upload_complete(sender):
        uploads.append(True)

    controller = ChunkedUploadController(
        scene,
        get_artifact=lambda: artifact,
        get_show_travel_moves=lambda: False,
        get_gl_initialized=lambda: gl_initialized,
        make_current=lambda: made_current.append(True),
        request_render=lambda: rendered.append(True),
        on_luts_required=lambda: luts_called.append(True),
    )
    controller.upload_complete.connect(_on_upload_complete, weak=False)
    return (
        controller,
        scene,
        rendered,
        luts_called,
        uploads,
        made_current,
    )


@pytest.mark.ui
def test_start_without_artifact_clears_renderers(ui_context_initializer):
    controller, scene, rendered, _, _, _ = _make_controller(artifact=None)
    controller.start()
    scene.prepare_chunked_upload.assert_not_called()
    scene.clear_layers.assert_called_once()
    assert rendered == [True]


@pytest.mark.ui
def test_start_without_gl_skips(ui_context_initializer):
    controller, scene, _rendered, _, _, made_current = _make_controller(
        artifact=MagicMock(), gl_initialized=False
    )
    controller.start()
    scene.prepare_chunked_upload.assert_not_called()
    assert made_current == []


@pytest.mark.ui
def test_start_prepares_upload_and_schedules_idle(ui_context_initializer):
    controller, scene, _, _, _, made_current = _make_controller(
        artifact=MagicMock()
    )
    controller.start()
    scene.prepare_chunked_upload.assert_called_once()
    assert made_current == [True]
    assert controller._upload_state == _UploadState(items=[], index=0)
    assert controller._idle_source_id is not None


@pytest.mark.ui
def test_process_pending_starts_when_dirty(ui_context_initializer):
    controller, scene, _, _, _, _ = _make_controller(artifact=MagicMock())
    controller.process_pending()
    scene.prepare_chunked_upload.assert_not_called()

    controller.mark_artifact_dirty()
    controller.process_pending()
    scene.prepare_chunked_upload.assert_called_once()


@pytest.mark.ui
def test_cancel_clears_state(ui_context_initializer):
    controller, scene, _, _, _, _ = _make_controller(artifact=MagicMock())
    controller.mark_artifact_dirty()
    controller.cancel()
    assert controller._upload_state is None
    assert controller._idle_source_id is None
    controller.process_pending()
    scene.prepare_chunked_upload.assert_not_called()


@pytest.mark.ui
@patch("rayforge.ui_gtk.sim3d.chunked_upload.task_mgr")
def test_step_schedules_worker_prepare(mock_task_mgr, ui_context_initializer):
    controller, scene, _, _, _, _ = _make_controller(artifact=MagicMock())
    item = MagicMock()
    state = _UploadState(items=[item], index=0)
    controller._upload_state = state
    assert controller._step(state) is False
    assert state.index == 1
    mock_task_mgr.run_thread.assert_called_once()
    scene.upload_chunk.assert_not_called()
    assert controller._upload_state is state


@pytest.mark.ui
@patch("rayforge.ui_gtk.sim3d.chunked_upload.task_mgr")
def test_step_stale_state_is_noop(mock_task_mgr, ui_context_initializer):
    controller, _scene, _, _, _, _ = _make_controller(artifact=MagicMock())
    stale_item = MagicMock()
    stale_state = _UploadState(items=[stale_item], index=0)
    current_item = MagicMock()
    current_state = _UploadState(items=[current_item], index=0)
    controller._upload_state = current_state

    assert controller._step(stale_state) is False

    mock_task_mgr.run_thread.assert_not_called()
    assert stale_state.index == 0
    assert current_state.index == 0


@pytest.mark.ui
def test_on_item_prepared_uploads_item(ui_context_initializer):
    controller, scene, _, _, _, made_current = _make_controller(
        artifact=MagicMock()
    )
    item = MagicMock()
    state = _UploadState(items=[item], index=1)
    controller._upload_state = state
    task = MagicMock()
    task.get_status.return_value = "completed"
    controller._on_item_prepared(state, item, task)
    scene.upload_chunk.assert_called_once_with(item)
    assert made_current == [True]
    assert controller._idle_source_id is not None
    assert controller._upload_state is state


@pytest.mark.ui
def test_on_item_prepared_failed_task_aborts(ui_context_initializer):
    controller, scene, _, _, _, _ = _make_controller(artifact=MagicMock())
    item = MagicMock()
    state = _UploadState(items=[item], index=1)
    controller._upload_state = state
    task = MagicMock()
    task.get_status.return_value = "failed"
    controller._on_item_prepared(state, item, task)
    scene.upload_chunk.assert_not_called()
    assert controller._upload_state is None


@pytest.mark.ui
def test_on_item_prepared_stale_state_ignored(ui_context_initializer):
    controller, scene, _, _, _, _ = _make_controller(artifact=MagicMock())
    item = MagicMock()
    state = _UploadState(items=[item], index=1)
    controller._upload_state = None
    task = MagicMock()
    task.get_status.return_value = "completed"
    controller._on_item_prepared(state, item, task)
    scene.upload_chunk.assert_not_called()


@pytest.mark.ui
def test_start_uploads_luts_after_prepare(ui_context_initializer):
    controller, scene, _, luts_called, _, _ = _make_controller(
        artifact=MagicMock()
    )
    controller.start()
    assert luts_called == [True]
    scene.prepare_chunked_upload.assert_called_once()


@pytest.mark.ui
def test_step_no_state_returns_false(ui_context_initializer):
    controller, _, _, _, _, _ = _make_controller()
    assert controller._step(_UploadState(items=[], index=0)) is False


@pytest.mark.ui
def test_is_dirty_tracks_pending_upload(ui_context_initializer):
    controller, _, _, _, _, _ = _make_controller()
    assert controller.is_dirty is False
    controller.mark_artifact_dirty()
    assert controller.is_dirty is True
    controller.cancel()
    assert controller.is_dirty is False


@pytest.mark.ui
@patch("rayforge.ui_gtk.sim3d.chunked_upload.task_mgr")
def test_stale_chain_idle_cannot_skip_or_duplicate_items(
    mock_task_mgr, ui_context_initializer
):
    """Regression test for the rotary first-3D-entry bug.

    A replaced chain leaves behind a pending idle (scheduled by its
    last successful item) and an in-flight worker.  Both must become
    no-ops against the new chain: the stale idle must not advance the
    new chain's dispatch index, and the stale worker's payload must
    not be uploaded into a different item's slot.  Before the fix the
    interleaving double-advanced the index, uploaded one item twice,
    silently discarded another, and completed with items missing.
    """
    controller, scene, _, _, uploads, _ = _make_controller(
        artifact=MagicMock()
    )

    dispatched = []

    def _fake_run_thread(func, key=None, when_done=None, **kwargs):
        dispatched.append((func, when_done))

    mock_task_mgr.run_thread.side_effect = _fake_run_thread

    old_items: list[UploadItem] = [
        MagicMock(name="old_ops"),
        MagicMock(name="old_overlay"),
    ]
    new_items: list[UploadItem] = [
        MagicMock(name="new_ops"),
        MagicMock(name="new_overlay"),
        MagicMock(name="new_texture"),
        MagicMock(name="new_stock"),
    ]
    old_state = _UploadState(items=old_items, index=1)
    new_state = _UploadState(items=new_items, index=0)

    # The old chain has one worker in flight (old_items[0]) when a new
    # artifact replaces it.
    controller._upload_state = old_state

    # A new artifact arrives; its chain becomes current and dispatches
    # its first item.
    controller._upload_state = new_state
    controller._step(new_state)
    assert new_state.index == 1
    assert len(dispatched) == 1

    # The stale success idle from the old chain fires now: with the
    # pre-fix index-based stepping it would advance the new chain's
    # dispatch index and re-order the chain.  It must be a no-op.
    controller._step(old_state)
    assert new_state.index == 1
    assert len(dispatched) == 1

    task = MagicMock()
    task.get_status.return_value = "completed"

    # The old chain's worker completes: its payload must NOT be
    # uploaded (its renderers were already replaced by the new chain's
    # prepare_chunked_upload).
    controller._on_item_prepared(old_state, old_items[0], task)
    scene.upload_chunk.assert_not_called()

    # The new chain's worker completes: its payload uploads its OWN
    # item, addressed explicitly rather than through the shared index.
    controller._on_item_prepared(new_state, new_items[0], task)
    scene.upload_chunk.assert_called_once_with(new_items[0])
    assert uploads == []  # neither chain completed via _step
