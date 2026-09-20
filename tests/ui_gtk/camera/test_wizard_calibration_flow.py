# flake8: noqa: E402
"""UI tests for the lens-calibration step of the camera wizard.

Saving a calibration used to close the whole wizard, making lens
calibration and image/world alignment mutually exclusive in a single
pass. Conversely, Next was always enabled on the capture page, so
captured frames were silently discarded without ever solving the lens
model.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import numpy as np
import pytest
from gi.repository import Gtk

from rayforge.camera.calibration.result import CalibrationResult
from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.ui_gtk.camera.wizard.capture_page import CapturePage
from rayforge.ui_gtk.camera.wizard.lens_calibration_choice_page import (
    LensCalibrationChoicePage,
)
from rayforge.ui_gtk.camera.wizard.wizard import CameraWizard


def _make_camera() -> Camera:
    return Camera(
        name="Test Camera",
        source_type=CameraSourceType.LOCAL_DEVICE,
        source_config={"device_id": "/dev/video0"},
    )


class _FakeController(CameraController):
    """Controller that never starts a real capture thread."""

    @property
    def has_active_source(self) -> bool:
        return True

    def _start_capture_stream(self):
        pass

    def _stop_capture_stream(self):
        pass


def _make_result() -> CalibrationResult:
    return CalibrationResult(
        camera_matrix=np.array(
            [[900.0, 0.0, 320.0], [0.0, 910.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coeffs=np.array([-0.31, 0.12, 0.001, 0.002, -0.05]),
        rms_error=0.42,
        image_size=(640, 480),
        num_frames_used=9,
        calibration_date=datetime.now(timezone.utc),
    )


def _open_wizard_on_capture(controller):
    parent = Gtk.Window()
    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0"],
    ):
        wizard = CameraWizard(parent, controller)
        wizard._navigate_to("lens_choice")
        wizard.on_lens_branch_chosen(
            LensCalibrationChoicePage.BRANCH_AUTOMATIC
        )
        while wizard._current != "capture":
            wizard.advance()
    return wizard


def _capture_page(wizard) -> CapturePage:
    page = wizard._pages["capture"]
    assert isinstance(page, CapturePage)
    return page


@pytest.mark.ui
def test_saving_calibration_stores_result_and_continues(
    ui_context_initializer,
):
    """Saving must persist the lens model and advance to alignment."""
    camera = _make_camera()
    controller = _FakeController(camera)
    wizard = _open_wizard_on_capture(controller)
    page = _capture_page(wizard)
    page._calibration_result = _make_result()

    page._on_result_dialog_response(Gtk.Window(), "save")

    assert camera.has_calibration
    assert camera.distortion_k1 == pytest.approx(-0.31)
    matrix = camera.get_camera_matrix()
    assert matrix is not None
    assert matrix[0][0] == pytest.approx(900.0)
    assert wizard._current == "alignment"


@pytest.mark.ui
def test_discarding_calibration_keeps_the_user_on_the_page(
    ui_context_initializer,
):
    camera = _make_camera()
    controller = _FakeController(camera)
    wizard = _open_wizard_on_capture(controller)
    page = _capture_page(wizard)
    page._calibration_result = _make_result()

    page._on_result_dialog_response(Gtk.Window(), "discard")

    assert not camera.has_calibration
    assert wizard._current == "capture"


@pytest.mark.ui
def test_next_is_vetoed_when_frames_are_uncalibrated(ui_context_initializer):
    """Next must confirm before discarding captured frames."""
    controller = _FakeController(_make_camera())
    wizard = _open_wizard_on_capture(controller)
    page = _capture_page(wizard)

    with patch.object(
        type(page.calibrator),
        "frame_count",
        property(lambda self: 4),
    ):
        wizard._on_next_clicked(None)

    assert wizard._current == "capture"


@pytest.mark.ui
def test_next_proceeds_when_no_frames_were_captured(ui_context_initializer):
    controller = _FakeController(_make_camera())
    wizard = _open_wizard_on_capture(controller)

    wizard._on_next_clicked(None)

    assert wizard._current == "alignment"
