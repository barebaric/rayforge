# flake8: noqa: E402
"""UI tests for the Test Settings of the calibration tests."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from laser_essentials import frontend
from laser_essentials.commands import CalibrationCmd
from laser_essentials.frontend import _create_test
from laser_essentials.widgets.calibration_provider import (
    CalibrationPropertyProvider,
)

from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.property_providers import (
    property_provider_registry,
)
from rayforge.ui_gtk.shared.pref_rows import SpeedSpinRow


@pytest.fixture
def cmd(editor, laser_machine):
    return CalibrationCmd(editor)


@pytest.fixture
def provider():
    provider = CalibrationPropertyProvider()
    provider.create_widgets()
    return provider


def _show(provider, editor, cmd, generator_id, params=None):
    record = cmd.create(generator_id, params)
    workpieces = cmd.workpieces(record)
    assert provider.can_handle(workpieces)
    provider.update_widgets(editor, workpieces)
    return record


def _row(provider, key):
    row = provider.varset_widget.row_for(key)
    assert row is not None, key
    return row


def _set(provider, key, value):
    """Changes a row the way a user does, so the row emits a change."""
    row = provider.varset_widget.row_for(key)
    adapter = provider.varset_widget.adapter_for(key)
    assert row is not None and adapter is not None, key
    if isinstance(value, tuple):
        rows = [row, *adapter.extra_rows()]
        for spin_row, item in zip(rows, value):
            spin_row.get_spin_button().set_value(item)
    elif hasattr(row, "get_spin_button"):
        row.get_spin_button().set_value(value)
    else:
        adapter.set_value(value)


def _step(editor, record):
    layer = editor.doc.find_descendant_by_uid(record.layer_uid)
    (step,) = layer.workflow.steps
    return step


@pytest.mark.ui
def test_only_handles_workpieces_of_one_test(provider, editor, cmd):
    interval = cmd.create("interval_test")
    grid = cmd.create("material_test_grid")
    other = WorkPiece("Other")
    editor.doc.active_layer.add_child(other)

    assert provider.can_handle(cmd.workpieces(interval))
    assert provider.can_handle(cmd.workpieces(grid))
    assert not provider.can_handle([other])
    assert not provider.can_handle(
        cmd.workpieces(interval) + cmd.workpieces(grid)
    )
    assert not provider.can_handle([])


@pytest.mark.ui
def test_shows_the_parameters_of_the_test(provider, editor, cmd):
    record = _show(provider, editor, cmd, "interval_test", {"count": 7})
    assert provider.varset_widget.get_values()["count"] == 7
    assert provider.group_subtitle == record.name
    assert provider.help_row.get_visible()
    assert not provider.error_row.get_visible()


@pytest.mark.ui
def test_change_rebuilds_the_test_in_one_undo_step(provider, editor, cmd):
    record = _show(provider, editor, cmd, "interval_test", {"count": 3})
    depth = len(editor.history_manager.undo_stack)

    _set(provider, "count", 5)
    provider.apply_now()

    assert record.params["count"] == 5
    assert len(cmd.workpieces(record)) == 6
    assert len(editor.history_manager.undo_stack) == depth + 1


@pytest.mark.ui
def test_invalid_values_show_an_error(provider, editor, cmd):
    record = _show(provider, editor, cmd, "interval_test")
    state = record.state

    _set(provider, "min_interval", 0.3)
    provider.apply_now()

    assert provider.error_row.get_visible()
    assert "minimum line interval" in provider.error_row.get_title()
    assert record.state == state

    _set(provider, "min_interval", 0.1)
    provider.apply_now()
    assert not provider.error_row.get_visible()
    assert record.params["min_interval"] == pytest.approx(0.1)


@pytest.mark.ui
def test_rows_follow_the_selected_test(provider, editor, cmd):
    _show(provider, editor, cmd, "interval_test")
    assert provider.varset_widget.row_for("cell_size") is not None

    _show(provider, editor, cmd, "focus_test")
    assert provider.varset_widget.row_for("cell_size") is None
    assert provider.varset_widget.row_for("mode") is not None


@pytest.mark.ui
def test_focus_help_follows_the_method(provider, editor, cmd):
    _show(provider, editor, cmd, "focus_test")
    help_before = provider.help_row.get_title()
    assert _row(provider, "ramp_length").get_visible() is False

    _set(provider, "mode", "ramp")
    provider.apply_now()

    assert provider.help_row.get_title() != help_before
    assert "Prop up" in provider.help_row.get_title()
    assert _row(provider, "ramp_length").get_visible() is True
    assert _row(provider, "count").get_visible() is False


@pytest.mark.ui
def test_label_rows_follow_include_labels(provider, editor, cmd):
    _show(provider, editor, cmd, "interval_test")
    assert _row(provider, "label_speed").get_sensitive() is True
    _set(provider, "include_labels", False)
    assert _row(provider, "label_power_percent").get_sensitive() is False
    assert _row(provider, "label_speed").get_sensitive() is False


@pytest.mark.ui
def test_grid_mode_visibility(provider, editor, cmd):
    _show(provider, editor, cmd, "material_test_grid")
    power = _row(provider, "power_range")
    passes = _row(provider, "passes_range")
    offset = _row(provider, "offset_range")
    assert power.get_visible() is True
    assert passes.get_visible() is False
    assert offset.get_visible() is False

    _set(provider, "grid_mode", "Speed vs Offset")
    assert power.get_visible() is False
    assert offset.get_visible() is True
    assert _row(provider, "fixed_power").get_visible() is True


@pytest.mark.ui
def test_speed_vs_offset_defaults(provider, editor, cmd):
    record = _show(provider, editor, cmd, "material_test_grid")
    _set(provider, "grid_mode", "Speed vs Offset")
    provider.apply_now()

    values = provider.varset_widget.get_values()
    assert values["test_type"] == "Engrave"
    assert values["line_interval_mm"] == pytest.approx(0.5)
    step = _step(editor, record)
    assert step.test_type == "Engrave"
    assert step.line_interval_mm == pytest.approx(0.5)


@pytest.mark.ui
def test_dimension_labels_follow_grid_mode(provider, editor, cmd):
    _show(provider, editor, cmd, "material_test_grid")
    row = _row(provider, "grid_dimensions")
    assert row.get_title() == "Columns (Power Steps)"

    _set(provider, "grid_mode", "Speed vs Passes")
    provider.apply_now()
    row = _row(provider, "grid_dimensions")
    assert row.get_title() == "Columns (Speed Steps)"


@pytest.mark.ui
def test_preset_applies_values_and_clamps(provider, editor, cmd):
    record = _show(provider, editor, cmd, "material_test_grid")
    _set(provider, "preset", "Diode Engrave")
    provider.apply_now()

    step = _step(editor, record)
    assert step.test_type == "Engrave"
    assert step.power_range == (10.0, 100.0)
    assert step.speed_range == (1000.0, 5000.0)
    values = provider.varset_widget.get_values()
    assert values["speed_range"] == (1000.0, 5000.0)


@pytest.mark.ui
def test_speed_bound_follows_the_machine(provider, editor, cmd, laser_machine):
    record = _show(provider, editor, cmd, "material_test_grid")
    adapter = provider.varset_widget.adapter_for("speed_range")
    assert adapter is not None
    max_row = cast(SpeedSpinRow, adapter.extra_rows()[0])
    assert max_row._upper == pytest.approx(5000.0)

    laser_machine.set_max_cut_speed(8000)
    provider.update_widgets(editor, cmd.workpieces(record))
    adapter = provider.varset_widget.adapter_for("speed_range")
    assert adapter is not None
    max_row = cast(SpeedSpinRow, adapter.extra_rows()[0])
    assert max_row._upper == pytest.approx(8000.0)


@pytest.mark.ui
def test_speed_rows_are_unit_aware(provider, editor, cmd, ui_context):
    _show(provider, editor, cmd, "material_test_grid")
    label = cast(SpeedSpinRow, _row(provider, "label_speed"))
    assert isinstance(label, SpeedSpinRow)

    ui_context.config.set_unit_preference("speed", "mm/s")
    assert label.get_spin_button().get_text() == "16.7"


@pytest.mark.ui
def test_tuple_labels_survive_a_refresh(provider, editor, cmd, ui_context):
    record = _show(provider, editor, cmd, "material_test_grid")
    primary = _row(provider, "power_range")
    adapter = provider.varset_widget.adapter_for("power_range")
    assert adapter is not None
    extra = adapter.extra_rows()[0]

    ui_context.config.set_unit_preference("speed", "mm/s")
    provider.update_widgets(editor, cmd.workpieces(record))

    assert primary.get_title() == "Minimum Power (%)"
    assert extra.get_title() == "Maximum Power (%)"


@pytest.mark.ui
def test_legacy_grid_can_be_changed(provider, editor, cmd):
    record = cmd.create("material_test_grid")
    editor.doc.remove_asset(record)
    workpieces = cmd.workpieces(record)
    assert provider.can_handle(workpieces)
    provider.update_widgets(editor, workpieces)

    _set(provider, "grid_dimensions", (3, 3))
    provider.apply_now()

    (new_wp,) = [
        wp
        for wp in editor.doc.all_workpieces
        if wp.uid not in {w.uid for w in workpieces}
    ]
    adopted = cmd.record_for_items([new_wp])
    assert adopted is not None
    assert adopted.uid in editor.doc.assets
    assert adopted.params["grid_dimensions"] == (3, 3)


@pytest.mark.ui
def test_debounced_change(provider, editor, cmd):
    record = _show(provider, editor, cmd, "interval_test", {"count": 3})
    _set(provider, "count", 4)
    assert provider._timer_id
    assert record.params["count"] == 3
    provider.apply_now()
    assert provider._timer_id == 0
    assert record.params["count"] == 4


@pytest.mark.ui
def test_without_a_test_selected(provider, editor, cmd):
    other = WorkPiece("Other")
    editor.doc.active_layer.add_child(other)
    provider.update_widgets(editor, [other])
    assert provider.generator is None
    provider.apply_now()
    provider.scroll_into_view()
    assert provider._scroll_handlers == []


@pytest.mark.ui
def test_apply_without_changes_does_nothing(provider, editor, cmd):
    record = _show(provider, editor, cmd, "interval_test")
    depth = len(editor.history_manager.undo_stack)
    provider.apply_now()
    assert len(editor.history_manager.undo_stack) == depth
    assert record.params["count"] == 5


@pytest.mark.ui
def test_create_failure_is_reported(editor):
    sent = []
    editor.notification_requested.connect(
        lambda sender, message: sent.append(message), weak=False
    )
    calibration = MagicMock()
    calibration.create.side_effect = ValueError("Select a machine first.")
    window = SimpleNamespace(
        doc_editor=SimpleNamespace(
            calibration=calibration,
            notification_requested=editor.notification_requested,
        )
    )
    _create_test(SimpleNamespace(window=window), "interval_test")
    assert sent == ["Select a machine first."]


@pytest.mark.ui
def test_changes_while_updating_are_ignored(provider, editor, cmd):
    _show(provider, editor, cmd, "interval_test")
    provider._in_update = True
    _set(provider, "count", 9)
    provider._in_update = False
    assert provider._timer_id == 0
    assert not provider._pending_keys


@pytest.mark.ui
def test_unavailable_method_is_kept_and_explained(
    provider, editor, cmd, laser_machine, monkeypatch
):
    """A Focus Test saved on a machine with a Z axis keeps its method on
    a machine without one: the panel says so, and an unrelated change
    does not switch the method."""
    monkeypatch.setattr(
        type(laser_machine), "has_z_axis", property(lambda self: False)
    )
    record = cmd.create("focus_test", {"mode": "ramp"})
    record.set_state(
        replace(record.state, params={**record.params, "mode": "z_axis"})
    )
    provider.update_widgets(editor, cmd.workpieces(record))

    assert provider.varset_widget.get_values()["mode"] == "z_axis"
    assert provider.warning_row.get_visible()
    assert "not available" in provider.warning_row.get_title()

    _set(provider, "ramp_length", 50.0)
    provider.apply_now()
    assert record.params["mode"] == "z_axis"
    assert provider.error_row.get_visible()


@pytest.mark.ui
def test_provider_is_registered_once_at_init():
    """Importing the frontend registers nothing; the init hook registers
    the Test Settings once, however often it runs."""
    registry = property_provider_registry
    assert CalibrationPropertyProvider not in registry.all()
    try:
        frontend.rayforge_init(context=None)
        frontend.rayforge_init(context=None)
        assert registry.all().count(CalibrationPropertyProvider) == 1
    finally:
        registry.unregister(CalibrationPropertyProvider)


@pytest.mark.ui
def test_rejected_value_is_kept_for_the_next_change(provider, editor, cmd):
    """A value that made the test invalid is not dropped when another
    setting then makes it valid."""
    record = _show(provider, editor, cmd, "interval_test")
    _set(provider, "min_interval", 0.3)
    provider.apply_now()
    assert provider.error_row.get_visible()

    _set(provider, "max_interval", 0.5)
    provider.apply_now()

    assert not provider.error_row.get_visible()
    assert record.params["min_interval"] == pytest.approx(0.3)
    assert record.params["max_interval"] == pytest.approx(0.5)
