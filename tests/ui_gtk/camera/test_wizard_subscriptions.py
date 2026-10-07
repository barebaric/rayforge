# flake8: noqa: E402
"""UI tests for camera-subscription balance in the camera wizard.

Cancelling the wizard used to release camera subscriptions the wizard
did not own, which dropped the controller's subscriber count to zero
and killed the live feed for the rest of the application.
"""

from unittest.mock import patch

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Gtk

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
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

    def __init__(self, camera: Camera):
        super().__init__(camera)

    @property
    def has_active_source(self) -> bool:
        return True

    def _start_capture_stream(self):
        pass

    def _stop_capture_stream(self):
        pass


def _run_wizard_flow(branch: str | None) -> int:
    """Walk the wizard and cancel it, returning the subscriber count.

    An external subscriber (mimicking the main canvas preview) holds one
    subscription for the whole flow.
    """
    camera = _make_camera()
    controller = _FakeController(camera)
    controller.subscribe()
    baseline = controller._active_subscribers

    parent = Gtk.Window()
    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0"],
    ):
        wizard = CameraWizard(parent, controller)
        wizard._navigate_to("lens_choice")
        if branch is not None:
            wizard.on_lens_branch_chosen(branch)
            while wizard._current != "alignment":
                wizard._on_next_clicked(None)
        wizard.close()

    assert baseline == 1
    return controller._active_subscribers


@pytest.mark.ui
@pytest.mark.parametrize(
    "branch",
    [
        None,
        LensCalibrationChoicePage.BRANCH_AUTOMATIC,
        LensCalibrationChoicePage.BRANCH_MANUAL,
        LensCalibrationChoicePage.BRANCH_SKIPPED,
    ],
)
def test_cancelling_wizard_keeps_external_subscription(
    ui_context_initializer, branch
):
    assert _run_wizard_flow(branch) == 1


@pytest.mark.ui
def test_double_leave_does_not_steal_other_subscriptions(
    ui_context_initializer,
):
    """Releasing a page twice must not drop a foreign subscription.

    The wizard leaves a page during navigation and again on close, so
    a non-idempotent release would decrement the controller past the
    wizard's own share and stop the stream for other consumers.
    """
    camera = _make_camera()
    controller = _FakeController(camera)
    controller.subscribe()

    parent = Gtk.Window()
    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0"],
    ):
        wizard = CameraWizard(parent, controller)
        image_page = wizard._pages["image"]
        image_page.leave()
        image_page.leave()
        image_page.leave()

    assert controller._active_subscribers == 1


@pytest.mark.ui
def test_revisiting_a_page_restores_its_subscription(ui_context_initializer):
    """Navigating back to a page must re-subscribe to the stream."""
    camera = _make_camera()
    controller = _FakeController(camera)

    parent = Gtk.Window()
    with patch.object(
        CameraController,
        "list_available_devices",
        return_value=["/dev/video0"],
    ):
        wizard = CameraWizard(parent, controller)
        assert controller._active_subscribers == 1
        wizard._navigate_to("lens_choice")
        assert controller._active_subscribers == 0
        wizard._navigate_to("image")
        assert controller._active_subscribers == 1
        wizard.close()

    assert controller._active_subscribers == 0
