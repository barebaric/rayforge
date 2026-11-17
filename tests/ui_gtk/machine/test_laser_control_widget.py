from unittest.mock import MagicMock

import pytest

from rayforge.machine.models.machine import Machine


@pytest.fixture
def connected_machine(ui_context_initializer, mocker):
    machine = Machine(ui_context_initializer)
    ui_context_initializer.machine_mgr.add_machine(machine)
    mocker.patch.object(machine, "is_connected", return_value=True)
    return machine


def _make_widget(connected_machine):
    from rayforge.ui_gtk.machine.laser_control_widget import (
        LaserControlWidget,
    )

    widget = LaserControlWidget()
    machine_cmd = MagicMock()
    widget.set_machine(connected_machine, machine_cmd)
    return widget, machine_cmd


def _task_with_status(status):
    task = MagicMock()
    task.get_status.return_value = status
    return task


@pytest.mark.ui
def test_toggle_reverts_when_turn_on_fails(connected_machine):
    widget, machine_cmd = _make_widget(connected_machine)

    widget._turn_on()
    assert widget._is_on

    when_done = machine_cmd.set_focus_power.call_args.kwargs["when_done"]
    when_done(_task_with_status("failed"))

    assert widget._is_on is False


@pytest.mark.ui
def test_toggle_stays_on_when_turn_on_succeeds(connected_machine):
    widget, machine_cmd = _make_widget(connected_machine)

    widget._turn_on()
    when_done = machine_cmd.set_focus_power.call_args.kwargs["when_done"]
    when_done(_task_with_status("completed"))

    assert widget._is_on is True


@pytest.mark.ui
def test_toggle_restores_on_when_turn_off_fails(connected_machine):
    widget, machine_cmd = _make_widget(connected_machine)

    widget._turn_on()
    machine_cmd.set_focus_power.call_args.kwargs["when_done"](
        _task_with_status("completed")
    )

    widget._turn_off()
    assert widget._is_on is False

    when_done = machine_cmd.set_focus_power.call_args.kwargs["when_done"]
    when_done(_task_with_status("failed"))

    assert widget._is_on is True
