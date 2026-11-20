# flake8: noqa: E402
"""UI tests for the async camera picker (CameraSelectionDialog)."""

import time
from unittest.mock import patch

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import GLib, Gtk

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.camera.remote import RemoteCamera
from rayforge.ui_gtk.camera.selection_dialog import CameraSelectionDialog


def _pump(deadline: float) -> None:
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        time.sleep(0.01)


def _pump_until(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class _FakeActiveController(CameraController):
    """A CameraController that looks like it owns an open local device,
    without starting a real capture thread."""

    def __init__(self, camera: Camera, frame):
        super().__init__(camera)
        self._frame = frame

    @property
    def has_active_source(self) -> bool:
        return True

    @property
    def raw_image_data(self):
        return self._frame


def _type_labels(dialog) -> list[str]:
    model = dialog.type_row.get_model()
    assert isinstance(model, Gtk.StringList)
    return [
        label
        for i in range(model.get_n_items())
        if (label := model.get_string(i)) is not None
    ]


@pytest.mark.ui
def test_local_machine_has_no_daemon_source(ui_context_initializer):
    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
        return_value=[],
    ):
        dialog = CameraSelectionDialog(None)
        assert "Daemon camera" not in _type_labels(dialog)

    dialog.destroy()
    _pump(time.monotonic() + 0.2)


@pytest.mark.ui
def test_dialog_opens_instantly_with_scanning_placeholder(
    ui_context_initializer,
):
    """Constructing the dialog must never block on a hardware scan: the
    device dropdown starts disabled with a placeholder and is only
    populated once the background scan finishes.
    """
    scan_started = time.monotonic()
    scan_may_return = False

    def slow_scan():
        # Simulate a slow hardware probe; if __init__ blocked on this,
        # the assertions right after construction would fail.
        deadline = time.monotonic() + 2.0
        while not scan_may_return and time.monotonic() < deadline:
            time.sleep(0.01)
        return ["/dev/video0"]

    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
        side_effect=slow_scan,
    ):
        dialog = CameraSelectionDialog(None)

        assert time.monotonic() - scan_started < 1.0
        assert dialog.device_row.get_sensitive() is False
        assert dialog._available_devices == []

        scan_may_return = True
        assert _pump_until(lambda: dialog._available_devices != [])

        assert dialog._available_devices == ["/dev/video0"]
        assert dialog.device_row.get_sensitive() is True

    dialog.destroy()
    _pump(time.monotonic() + 0.2)


@pytest.mark.ui
def test_scan_result_arriving_after_close_is_ignored(ui_context_initializer):
    """A scan that completes after the dialog has already been closed
    must not touch any of its widgets."""
    release_scan = False

    def slow_scan():
        deadline = time.monotonic() + 2.0
        while not release_scan and time.monotonic() < deadline:
            time.sleep(0.01)
        return ["/dev/video0"]

    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
        side_effect=slow_scan,
    ):
        dialog = CameraSelectionDialog(None)
        # Mirrors the real shutdown path (camera_preferences_page.py
        # handles "response" and only then calls destroy()); the
        # "response" signal is what actually marks the dialog closed.
        dialog.response("cancel")
        _pump(time.monotonic() + 0.2)
        assert dialog._closed is True

        release_scan = True
        # Give the worker thread's idle_add callback a chance to run; it
        # must see _closed and return without raising or mutating state.
        _pump(time.monotonic() + 1.0)

    assert dialog._available_devices == []


@pytest.mark.ui
def test_preview_reuses_active_controller_frame_without_reopening_device(
    ui_context_initializer,
):
    """Selecting a device that's already active elsewhere in the app must
    reuse that controller's current frame instead of opening a second
    capture on the same hardware.
    """
    import numpy as np

    device_id = "/dev/video0"
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    active_camera = Camera(
        name="Active",
        source_type=CameraSourceType.LOCAL_DEVICE,
        source_config={"device_id": device_id},
    )
    active_controller = _FakeActiveController(active_camera, frame)

    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
        return_value=[device_id],
    ):
        dialog = CameraSelectionDialog(
            None, active_controllers=[active_controller]
        )
        assert _pump_until(lambda: dialog._available_devices != [])

        with patch.object(
            CameraController, "capture_image"
        ) as mock_capture_image:
            dialog.type_row.set_selected(1)  # LOCAL_DEVICE
            dialog.device_row.set_selected(1)  # first (only) real device
            _pump(time.monotonic() + 0.3)

            # No new controller/device open should have happened.
            mock_capture_image.assert_not_called()

        assert dialog._preview_pixbuf is not None

    dialog.destroy()
    _pump(time.monotonic() + 0.2)


@pytest.mark.ui
def test_new_camera_defaults_to_local_and_selects_first_device(
    ui_context_initializer,
):
    device_id = "/dev/video0"

    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
        return_value=[device_id],
    ):
        dialog = CameraSelectionDialog(None)
        assert _pump_until(lambda: dialog.camera_payload is not None)

        assert dialog.type_row.get_selected() == 1
        assert dialog.device_row.get_selected() == 1
        payload = dialog.camera_payload
        assert payload is not None
        assert payload["source_config"] == {"device_id": device_id}
        assert not hasattr(dialog, "name_entry")

    dialog.destroy()
    _pump(time.monotonic() + 0.2)


@pytest.mark.ui
def test_configured_mode_selects_existing_camera(
    ui_context_initializer,
):
    camera = Camera(
        name="Configured camera",
        source_type=CameraSourceType.LOCAL_DEVICE,
        source_config={"device_id": "/dev/video0"},
    )
    controller = _FakeActiveController(camera, None)
    manager = type("CameraManager", (), {"controllers": [controller]})()
    context = type("Context", (), {"camera_mgr": manager})()

    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.get_context",
        return_value=context,
    ):
        dialog = CameraSelectionDialog(None, mode="configured")

        assert dialog.type_row.get_visible() is False
        assert dialog.device_row.get_selected() == 0
        assert dialog.selected_device_id == camera.id
        assert dialog.camera_payload == {"camera_id": camera.id}
        assert dialog.get_response_enabled("select") is True

    dialog.destroy()
    _pump(time.monotonic() + 0.2)


@pytest.mark.ui
def test_daemon_camera_prefills_network_source(ui_context_initializer):
    camera = RemoteCamera(
        name="Enclosure",
        uri="http://pi.local:8080/stream",
        source_type=CameraSourceType.HTTP_STREAM,
    )
    with patch(
        "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
        return_value=[],
    ):
        dialog = CameraSelectionDialog(None, remote_cameras=[camera])
        daemon_index = dialog.type_values.index("daemon")
        assert "Daemon camera" in _type_labels(dialog)
        assert dialog.daemon_row.get_visible() is False

        dialog.type_row.set_selected(daemon_index)
        assert _pump_until(lambda: dialog.daemon_row.get_visible())
        assert dialog.daemon_row.get_selected() == 0

        dialog.daemon_row.set_selected(1)

        assert _pump_until(lambda: dialog.uri_entry.get_text() == camera.uri)
        assert dialog.type_row.get_selected() == daemon_index
        assert dialog.uri_entry.get_editable() is False
        assert dialog.camera_payload == {
            "name": camera.name,
            "source_type": "http_stream",
            "source_config": {"uri": camera.uri},
        }

    dialog.destroy()
    _pump(time.monotonic() + 0.2)


@pytest.mark.ui
def test_daemon_camera_scan_reads_driver_args(ui_context_initializer):
    camera = RemoteCamera(
        name="Bed",
        uri="http://pi.local:8080/bed",
        source_type=CameraSourceType.HTTP_SNAPSHOT,
    )

    async def fake_fetch(driver_args):
        assert driver_args["remote_host"] == "pi.local"
        return [camera]

    with (
        patch(
            "rayforge.ui_gtk.camera.selection_dialog.list_local_device_ids",
            return_value=[],
        ),
        patch(
            "rayforge.ui_gtk.camera.selection_dialog.fetch_remote_cameras",
            fake_fetch,
        ),
    ):
        dialog = CameraSelectionDialog(
            None, driver_args={"remote_host": "pi.local"}
        )
        daemon_index = dialog.type_values.index("daemon")
        dialog.type_row.set_selected(daemon_index)
        assert _pump_until(lambda: dialog.daemon_row.get_visible())

    dialog.destroy()
    _pump(time.monotonic() + 0.2)
