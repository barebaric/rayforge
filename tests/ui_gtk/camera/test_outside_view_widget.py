# flake8: noqa: E402
"""UI tests for the camera outside view settings group."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera
from rayforge.ui_gtk.camera.outside_view_widget import (
    CameraOutsideViewGroup,
)

pytestmark = pytest.mark.ui


def _controller(name: str) -> CameraController:
    return CameraController(Camera(name, "0"))


def test_controls_follow_toggle_and_preserve_values(ui_context_initializer):
    controller = _controller("Camera")
    camera = controller.config
    camera.outside_view_margin_mm = 40
    camera.outside_view_transparency = 0.4
    group = CameraOutsideViewGroup()
    group.set_controller(controller)

    assert not group.enabled_switch.get_active()
    assert not group.margin_row.get_sensitive()
    assert not group.transparency_row.get_sensitive()
    assert group.margin_row.get_value_in_base_units() == 40
    assert group.transparency_adjustment.get_value() == 0.4

    group.enabled_switch.set_active(True)

    assert camera.outside_view_enabled
    assert group.margin_row.get_sensitive()
    assert group.transparency_row.get_sensitive()

    group.enabled_switch.set_active(False)
    assert camera.outside_view_margin_mm == 40
    assert camera.outside_view_transparency == 0.4


def test_controls_write_to_the_camera(ui_context_initializer):
    controller = _controller("Camera")
    group = CameraOutsideViewGroup()
    group.set_controller(controller)
    group.enabled_switch.set_active(True)

    group.transparency_adjustment.set_value(0.25)
    group.margin_row.get_spin_button().set_value(55)

    assert controller.config.outside_view_transparency == 0.25
    assert controller.config.outside_view_margin_mm == 55


def test_switching_controllers_keeps_settings_independent(
    ui_context_initializer,
):
    first = _controller("First")
    second = _controller("Second")
    first.config.outside_view_enabled = True
    first.config.outside_view_margin_mm = 70
    group = CameraOutsideViewGroup()

    group.set_controller(first)
    group.set_controller(second)

    assert not group.enabled_switch.get_active()
    assert group.margin_row.get_value_in_base_units() == 30
    assert second.config.outside_view_margin_mm == 30
    assert first.config.outside_view_margin_mm == 70

    first.config.outside_view_margin_mm = 80
    assert group.margin_row.get_value_in_base_units() == 30

    group.set_controller(None)
    assert not group.get_sensitive()


def test_alignment_hint(ui_context_initializer):
    controller = _controller("Camera")
    group = CameraOutsideViewGroup()
    group.set_controller(controller)
    unaligned = group.enabled_row.get_subtitle()

    controller.config.image_to_world = (
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )

    assert group.enabled_row.get_subtitle() != unaligned
