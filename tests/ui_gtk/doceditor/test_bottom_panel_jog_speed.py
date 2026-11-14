"""Tests for jog speed persistence in the bottom panel controls."""

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def jog_panel(sync_machine):
    """A BottomPanel with only the jog speed wiring, Gtk bypassed."""
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    panel: Any = BottomPanel.__new__(BottomPanel)
    panel.machine = sync_machine
    panel.machine_cmd = None
    panel.speed_row = None
    panel.jog_widget = SimpleNamespace(jog_speed=0)
    panel.console = SimpleNamespace(set_machine=lambda machine: None)
    return panel


@pytest.mark.ui
def test_speed_change_updates_widget_and_machine(
    jog_panel, sync_machine, mocker
):
    jog_panel.speed_row = mocker.Mock()
    jog_panel.speed_row.get_value_in_base_units.return_value = 3000.0

    jog_panel._on_speed_changed(jog_panel.speed_row)

    assert jog_panel.jog_widget.jog_speed == 3000
    assert sync_machine.jog_speed == 3000


@pytest.mark.ui
def test_set_machine_shows_stored_jog_speed(jog_panel, sync_machine):
    """Switching machines shows the new machine's stored jog speed."""
    from rayforge.ui_gtk.shared.pref_rows.speed_spin_row import SpeedSpinRow

    sync_machine.set_jog_speed(4500)
    jog_panel.machine = None
    jog_panel.speed_row = SpeedSpinRow(
        "Jog Speed",
        "Speed",
        lower=1,
        upper=60000,
    )
    jog_panel._connect_machine_signals = lambda: None
    jog_panel._update_wcs_ui = lambda: None

    jog_panel.set_machine(sync_machine)

    assert jog_panel.speed_row.get_value_in_base_units() == 4500
    assert jog_panel.jog_widget.jog_speed == 4500
