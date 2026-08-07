"""UI tests for the ChunkedUploadController."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from unittest.mock import MagicMock

import pytest

from rayforge.ui_gtk.sim3d.chunked_upload import ChunkedUploadController


def _make_controller(artifact=None, gl_initialized=True):
    scene = MagicMock()
    scene.prepare_chunked_upload.return_value = []
    rendered = []
    luts_called = []
    op_player_called = []
    made_current = []

    controller = ChunkedUploadController(
        scene,
        get_artifact=lambda: artifact,
        get_show_travel_moves=lambda: False,
        get_gl_initialized=lambda: gl_initialized,
        make_current=lambda: made_current.append(True),
        request_render=lambda: rendered.append(True),
        on_luts_required=lambda: luts_called.append(True),
        on_op_player_required=lambda: op_player_called.append(True),
    )
    return (
        controller,
        scene,
        rendered,
        luts_called,
        op_player_called,
        made_current,
    )


@pytest.mark.ui
def test_start_without_artifact_clears_renderers(ui_context_initializer):
    controller, scene, rendered, _, _, _ = _make_controller(artifact=None)
    controller.start()
    scene.prepare_chunked_upload.assert_not_called()
    assert rendered == [True]


@pytest.mark.ui
def test_start_without_gl_skips(ui_context_initializer):
    controller, scene, rendered, _, _, made_current = _make_controller(
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
    assert controller._upload_state == {"items": [], "index": 0}
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
def test_step_dispatch_kinds(ui_context_initializer):
    controller, scene, _, luts_called, op_player_called, _ = _make_controller(
        artifact=MagicMock()
    )
    controller._upload_state = {
        "items": [
            ("color_luts",),
            ("op_player",),
        ],
        "index": 0,
    }
    controller._step()
    controller._step()
    controller._step()
    assert luts_called == [True]
    assert op_player_called == [True]
    assert controller._upload_state is None


@pytest.mark.ui
def test_step_no_state_returns_false(ui_context_initializer):
    controller, _, _, _, _, _ = _make_controller()
    assert controller._step() is False
