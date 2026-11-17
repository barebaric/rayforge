"""Tests for the shop-mode control surface widgets."""

from unittest.mock import MagicMock

import pytest
from raygeo.ops.axis import Axis

from rayforge.machine.driver.driver import DeviceStatus
from rayforge.machine.models.machine import JogDirection
from rayforge.ui_gtk.machine.shop_mode_widget import (
    ShopControlPanel,
    ShopJogPanel,
)


@pytest.mark.ui
def test_control_panel_shows_wcs_position(sync_machine):
    panel = ShopControlPanel()
    panel.set_machine(sync_machine, None)
    sync_machine.device_state.machine_pos = (110.0, 120.0, 8.0)
    sync_machine.update_wcs_offset("G54", (10.0, 20.0, 3.0))

    panel.refresh()

    assert panel._axis_values[Axis.X].get_label() == "100.000"
    assert panel._axis_values[Axis.Y].get_label() == "100.000"
    assert panel._axis_values[Axis.Z].get_label() == "5.000"
    assert panel._axis_zero_btns[Axis.X].get_sensitive()
    assert panel._offset_labels[Axis.X].get_label() == "X: 10.000"


@pytest.mark.ui
def test_control_panel_machine_coordinates(sync_machine):
    panel = ShopControlPanel()
    panel.set_machine(sync_machine, None)
    sync_machine.device_state.machine_pos = (110.0, 120.0, 8.0)
    sync_machine.active_wcs = sync_machine.machine_space_wcs

    panel.refresh()

    assert panel._axis_values[Axis.X].get_label() == "110.000"
    # Zeroing is not possible in machine coordinate mode.
    assert not panel._axis_zero_btns[Axis.X].get_sensitive()


@pytest.mark.ui
def test_control_panel_without_machine(sync_machine):
    panel = ShopControlPanel()
    panel.set_machine(None, None)
    assert panel._axis_values[Axis.X].get_label() == "---"
    assert not panel._axis_zero_btns[Axis.X].get_sensitive()


@pytest.mark.ui
def test_jog_panel_dispatches_jog(sync_machine):
    cmd = MagicMock()
    panel = ShopJogPanel()
    panel.set_machine(sync_machine, cmd)
    panel.jog_distance = 10.0
    panel.jog_speed = 1000

    panel._on_jog_clicked(None, (JogDirection.EAST,))

    cmd.jog.assert_called_once()
    machine, deltas, speed = cmd.jog.call_args.args
    assert machine is sync_machine
    assert speed == 1000
    assert deltas


@pytest.mark.ui
def test_jog_panel_wcs_selection_switches(sync_machine, mocker):
    task_mgr = mocker.patch(
        "rayforge.ui_gtk.machine.shop_mode_widget.task_mgr"
    )
    panel = ShopJogPanel()
    panel.set_machine(sync_machine, MagicMock())

    panel.wcs_dropdown.set_selected(1)

    task_mgr.add_coroutine.assert_called_once()
    assert task_mgr.add_coroutine.call_args.kwargs["key"] == (
        sync_machine.id,
        "select-wcs",
    )


@pytest.mark.ui
def test_jog_panel_initial_wcs_sync_does_not_switch_wcs(sync_machine, mocker):
    task_mgr = mocker.patch(
        "rayforge.ui_gtk.machine.shop_mode_widget.task_mgr"
    )
    wcs_list = list(sync_machine.supported_wcs)
    assert len(wcs_list) > 1
    sync_machine.active_wcs = wcs_list[1]

    panel = ShopJogPanel()
    panel.set_machine(sync_machine, MagicMock())

    assert panel.wcs_dropdown.get_selected() == 1
    assert sync_machine.active_wcs == wcs_list[1]
    task_mgr.add_coroutine.assert_not_called()


@pytest.mark.ui
def test_jog_panel_hold_icon_follows_state(sync_machine):
    panel = ShopJogPanel()
    panel.set_machine(sync_machine, MagicMock())

    sync_machine.device_state.status = DeviceStatus.HOLD
    panel._sync_hold_icon()
    holding_icon = panel.hold_btn.get_child()

    sync_machine.device_state.status = DeviceStatus.IDLE
    panel._sync_hold_icon()
    idle_icon = panel.hold_btn.get_child()

    assert holding_icon is not idle_icon


@pytest.mark.ui
def test_jog_panel_hides_z_controls_without_z_axis(sync_machine, mocker):
    """The whole Z button (including its axis subscript) must hide
    on machines without a Z axis."""
    from rayforge.machine.models.machine import Machine

    panel = ShopJogPanel()
    panel.set_machine(sync_machine, MagicMock())
    up = panel._direction_widgets[(JogDirection.UP,)]
    down = panel._direction_widgets[(JogDirection.DOWN,)]

    mocker.patch.object(
        Machine,
        "has_z_axis",
        new_callable=mocker.PropertyMock,
        return_value=False,
    )
    panel._update_sensitivity()
    assert not up.get_visible()
    assert not down.get_visible()

    mocker.patch.object(
        Machine,
        "has_z_axis",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )
    panel._update_sensitivity()
    assert up.get_visible()
    assert down.get_visible()
