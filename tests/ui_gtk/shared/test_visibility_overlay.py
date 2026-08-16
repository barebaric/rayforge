# flake8: noqa: E402
"""UI tests for the canvas VisibilityOverlay widget."""

import gi

gi.require_version("Gtk", "4.0")

import pytest

from rayforge.ui_gtk.shared.visibility_overlay import VisibilityOverlay


@pytest.mark.ui
def test_2d_overlay_buttons(ui_context_initializer):
    overlay = VisibilityOverlay(
        show_workpiece=True,
        show_camera=True,
        show_tabs=True,
        show_stock=True,
    )
    assert overlay.workpiece_button.get_action_name() == "win.show_workpieces"
    assert overlay.camera_button.get_action_name() == "win.toggle_camera_view"
    assert overlay.tabs_button.get_action_name() == "win.show_tabs"
    assert overlay.stock_button.get_action_name() == "win.show_stock"
    assert overlay.travel_button.get_action_name() == "win.toggle_travel_view"
    assert overlay.nogo_button.get_action_name() == "win.show_nogo_zones"
    assert overlay.workpiece_image_button is None


@pytest.mark.ui
def test_3d_overlay_buttons(ui_context_initializer):
    overlay = VisibilityOverlay(
        show_workpiece=False,
        show_models=True,
        show_grid=True,
        show_ops_underlay=True,
        show_stock=True,
        show_workpiece_image=True,
    )
    assert getattr(overlay, "workpiece_button", None) is None
    assert overlay.models_button.get_action_name() == "win.show_models"
    assert overlay.grid_button.get_action_name() == "win.show_grid"
    assert overlay.underlay_button is not None
    assert overlay.underlay_button.get_action_name() == "win.show_ops_underlay"
    assert overlay.stock_button.get_action_name() == "win.show_stock"
    assert overlay.workpiece_image_button is not None
    assert (
        overlay.workpiece_image_button.get_action_name()
        == "win.show_workpiece_image"
    )
    assert overlay.travel_button.get_action_name() == "win.toggle_travel_view"
    assert overlay.nogo_button.get_action_name() == "win.show_nogo_zones"


@pytest.mark.ui
def test_nogo_button_hidden_by_default_when_disabled(ui_context_initializer):
    overlay = VisibilityOverlay(show_nogo_zones=False)
    assert overlay.nogo_button.get_action_name() == "win.show_nogo_zones"
    assert not overlay.nogo_button.get_visible()


@pytest.mark.ui
def test_set_nogo_visible(ui_context_initializer):
    overlay = VisibilityOverlay()
    overlay.set_nogo_visible(False)
    assert not overlay.nogo_button.get_visible()
    overlay.set_nogo_visible(True)
    assert overlay.nogo_button.get_visible()
