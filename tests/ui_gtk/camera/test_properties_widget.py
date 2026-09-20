# flake8: noqa: E402
"""UI tests for the CameraProperties widget's local-device dropdown."""

from unittest.mock import patch

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.ui_gtk.camera.properties_widget import CameraProperties


def _make_camera(device_id: str) -> Camera:
    return Camera(
        name="Test Camera",
        source_type=CameraSourceType.LOCAL_DEVICE,
        source_config={"device_id": device_id},
    )


class _FakeController(CameraController):
    """A CameraController whose has_active_source is always True.

    Avoids starting a real capture thread while still satisfying type
    checks that require an actual CameraController instance.
    """

    def __init__(self, camera: Camera):
        super().__init__(camera)

    @property
    def has_active_source(self) -> bool:
        return True


@pytest.mark.ui
def test_selecting_local_device_does_not_rescan_hardware(
    ui_context_initializer,
):
    """Selecting a device from the dropdown must not trigger a fresh
    hardware scan.

    A live scan racing with the capture thread's own (asynchronous)
    open() of the newly selected device can crash the V4L2/DirectShow
    driver, so the widget must not re-probe hardware as a reaction to
    a device_id change it just made itself.
    """
    camera = _make_camera("/dev/video0")
    controller = _FakeController(camera)
    widget = CameraProperties(controller)
    widget.set_controller(controller)

    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0", "/dev/video1"],
    ) as mock_scan:
        widget._local_device_ids = ["/dev/video0", "/dev/video1"]
        widget.source_combo.set_selected(1)
        assert mock_scan.call_count == 0
        assert camera.device_id == "/dev/video1"


@pytest.mark.ui
def test_unrelated_camera_change_does_not_rescan_hardware(
    ui_context_initializer,
):
    """A change that cannot affect the device list (e.g. renaming the
    camera, or editing a distortion coefficient) must not re-probe the
    hardware.

    Scanning opens every V4L2 device in turn and blocks the UI thread,
    so reacting to unrelated model changes made the UI stutter and
    risked colliding with a capture stream reopening its device.
    """
    camera = _make_camera("/dev/video0")
    controller = _FakeController(camera)

    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0"],
    ) as mock_scan:
        widget = CameraProperties(controller)
        widget.set_controller(controller)
        mock_scan.reset_mock()

        camera.name = "Renamed Camera"
        camera.distortion_k1 = 0.25
        assert mock_scan.call_count == 0


@pytest.mark.ui
def test_external_device_change_rescans_hardware(ui_context_initializer):
    """A device_id change the widget did not make itself must refresh
    the dropdown so it reflects the new source."""
    camera = _make_camera("/dev/video0")
    controller = _FakeController(camera)

    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0", "/dev/video1"],
    ) as mock_scan:
        widget = CameraProperties(controller)
        widget.set_controller(controller)
        mock_scan.reset_mock()

        camera.device_id = "/dev/video1"
        assert mock_scan.call_count >= 1
