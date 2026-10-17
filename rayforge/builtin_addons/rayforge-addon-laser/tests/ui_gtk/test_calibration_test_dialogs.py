# flake8: noqa: E402
"""UI tests for the Interval Test and Focus Test dialogs."""

from typing import cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw
from laser_essentials.calibration_tests import (
    FocusMode,
    FocusTestParams,
    IntervalTestParams,
)
from laser_essentials.widgets.calibration_test_dialogs import (
    DEFAULT_WIDTH,
    FocusTestDialog,
    IntervalTestDialog,
)


def _row(dialog, key):
    for widget in dialog.varset_widgets:
        row = widget.row_for(key)
        if row is not None:
            return row
    raise KeyError(key)


@pytest.mark.ui
def test_interval_dialog_defaults_and_create(laser_machine):
    created: list[IntervalTestParams] = []
    dialog = IntervalTestDialog(None, created.append)
    assert dialog.params() == IntervalTestParams()
    assert dialog.get_default_size()[0] == DEFAULT_WIDTH

    dialog.varset_widgets[0].set_values({"count": 7})
    dialog.create_button.emit("clicked")
    assert len(created) == 1
    assert created[0].count == 7


@pytest.mark.ui
def test_label_rows_follow_include_labels(laser_machine):
    dialog = IntervalTestDialog(None, lambda p: None)
    labels = dialog.varset_widgets[2]
    labels.set_values({"include_labels": False})
    assert not _row(dialog, "label_power_percent").get_sensitive()
    assert not _row(dialog, "label_speed").get_sensitive()
    assert dialog.params().include_labels is False


@pytest.mark.ui
def test_interval_dialog_shows_errors(laser_machine):
    def refuse(params):
        raise ValueError("bad range")

    dialog = IntervalTestDialog(None, refuse)
    dialog.create_button.emit("clicked")
    assert dialog.banner.get_revealed()
    assert dialog.banner.get_title() == "bad range"


@pytest.mark.ui
def test_focus_dialog_modes(laser_machine):
    created: list[FocusTestParams] = []
    dialog = FocusTestDialog(
        None, created.append, [FocusMode.MANUAL, FocusMode.RAMP]
    )
    assert dialog.mode == FocusMode.MANUAL
    assert _row(dialog, "step").get_visible()
    assert not _row(dialog, "ramp_length").get_visible()
    manual_help = dialog.steps_widget.get_description()

    cast(Adw.ComboRow, _row(dialog, "mode")).set_selected(1)
    assert dialog.mode == FocusMode.RAMP
    assert not _row(dialog, "step").get_visible()
    assert _row(dialog, "ramp_length").get_visible()
    assert dialog.steps_widget.get_description() != manual_help

    dialog.create_button.emit("clicked")
    assert created[0].mode == FocusMode.RAMP


@pytest.mark.ui
def test_focus_dialog_without_manual_mode(laser_machine):
    dialog = FocusTestDialog(None, lambda p: None, [FocusMode.RAMP])
    assert dialog.mode == FocusMode.RAMP


@pytest.mark.ui
def test_focus_dialog_defaults_to_z_axis_when_available(laser_machine):
    dialog = FocusTestDialog(
        None,
        lambda p: None,
        [FocusMode.Z_AXIS, FocusMode.MANUAL, FocusMode.RAMP],
    )
    assert dialog.mode == FocusMode.Z_AXIS
    assert _row(dialog, "step").get_visible()
