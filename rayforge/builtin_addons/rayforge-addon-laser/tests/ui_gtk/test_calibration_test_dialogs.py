# flake8: noqa: E402
"""UI tests for the Interval Test and Focus Test dialogs."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from laser_essentials.calibration_tests import (
    FocusMode,
    FocusTestParams,
    IntervalTestParams,
)
from laser_essentials.widgets.calibration_test_dialogs import (
    FocusTestDialog,
    IntervalTestDialog,
)


@pytest.mark.ui
def test_interval_dialog_defaults_and_create(ui_context):
    created: list[IntervalTestParams] = []
    dialog = IntervalTestDialog(None, created.append, max_speed=3000)
    assert dialog.params() == IntervalTestParams()

    dialog.count_row._spin_button.set_value(7)
    dialog.create_button.emit("clicked")
    assert len(created) == 1
    assert created[0].count == 7


@pytest.mark.ui
def test_interval_dialog_shows_errors(ui_context):
    def refuse(params):
        raise ValueError("bad range")

    dialog = IntervalTestDialog(None, refuse)
    dialog.create_button.emit("clicked")
    assert dialog.banner.get_revealed()
    assert dialog.banner.get_title() == "bad range"


@pytest.mark.ui
def test_focus_dialog_modes(ui_context):
    created: list[FocusTestParams] = []
    dialog = FocusTestDialog(
        None, created.append, [FocusMode.MANUAL, FocusMode.RAMP]
    )
    assert dialog.mode == FocusMode.MANUAL
    assert dialog.step_row.get_visible()
    assert not dialog.ramp_row.get_visible()

    dialog.mode_row.set_selected(1)
    assert dialog.mode == FocusMode.RAMP
    assert not dialog.step_row.get_visible()
    assert dialog.ramp_row.get_visible()

    dialog.create_button.emit("clicked")
    assert created[0].mode == FocusMode.RAMP


@pytest.mark.ui
def test_focus_dialog_without_manual_mode(ui_context):
    dialog = FocusTestDialog(None, lambda p: None, [FocusMode.RAMP])
    assert dialog.mode == FocusMode.RAMP
