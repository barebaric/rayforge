"""UI tests for machine settings dialog helpers."""

from unittest.mock import patch

import pytest

from rayforge.camera.models.camera import Camera
from rayforge.ui_gtk.machine.settings_dialog import MachineSettingsDialog


@pytest.mark.ui
def test_adopt_current_local_camera_settings_disposes_temp_controller(
    ui_context_initializer,
):
    camera = Camera("Local Camera", "0")
    disposed = []

    class FakeController:
        def __init__(self, config):
            self.config = config

        def read_current_source_settings(self):
            return {
                "contrast": 12.5,
                "brightness": -3.0,
                "white_balance": 5100.0,
            }

        def dispose(self):
            disposed.append(self.config)

    with patch(
        "rayforge.ui_gtk.machine.settings_dialog.CameraController",
        FakeController,
    ):
        dialog = MachineSettingsDialog.__new__(MachineSettingsDialog)
        dialog._adopt_current_local_camera_settings(camera)

    assert camera.contrast == 12.5
    assert camera.brightness == -3.0
    assert camera.white_balance == 5100.0
    assert disposed == [camera]


@pytest.mark.ui
def test_adopt_current_local_camera_settings_disposes_after_error(
    ui_context_initializer,
):
    camera = Camera("Local Camera", "0")
    disposed = []

    class FakeController:
        def __init__(self, config):
            self.config = config

        def read_current_source_settings(self):
            raise OSError("offline")

        def dispose(self):
            disposed.append(self.config)

    with patch(
        "rayforge.ui_gtk.machine.settings_dialog.CameraController",
        FakeController,
    ):
        dialog = MachineSettingsDialog.__new__(MachineSettingsDialog)
        dialog._adopt_current_local_camera_settings(camera)

    assert disposed == [camera]
