# flake8: noqa: E402
"""UI tests for the Test Settings of the calibration tests.

The editing lives in the settings controller, which the Test Settings
section of the layer settings dialog hosts. The Test Settings card in
the item properties panel only points there.
"""

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
from laser_essentials.widgets.calibration_layer_section import (
    CalibrationLayerSettingsSection,
)
from laser_essentials.widgets.calibration_provider import (
    CalibrationPropertyProvider,
)
from laser_essentials.widgets.calibration_settings import (
    CalibrationSettingsController,
)

from rayforge.core.layer import Layer
from rayforge.core.layer_types import layer_is_locked
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.layer_settings_sections import (
    layer_settings_section_registry,
)
from rayforge.ui_gtk.doceditor.property_providers import (
    property_provider_registry,
)
from rayforge.ui_gtk.shared.pref_rows import SpeedSpinRow
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget


@pytest.fixture
def cmd(editor, laser_machine):
    return CalibrationCmd(editor)


@pytest.fixture
def provider():
    provider = CalibrationPropertyProvider()
    provider.create_widgets()
    return provider


def _controller(editor, cmd, generator_id, params=None):
    """A new test and the settings controller that edits it."""
    record = cmd.create(generator_id, params)
    return record, CalibrationSettingsController(editor, record)


def _row(controller, key):
    row = controller.group.row_for(key)
    assert row is not None, key
    return row


def _set(controller, key, value):
    """Changes a row the way a user does, so the row emits a change."""
    row = controller.group.row_for(key)
    adapter = controller.group.adapter_for(key)
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
def test_card_points_to_the_layer_settings(provider, editor, cmd):
    record = cmd.create("interval_test")
    workpieces = cmd.workpieces(record)
    assert provider.can_handle(workpieces)
    provider.update_widgets(editor, workpieces)

    assert provider.help_row.get_visible()
    assert "Interval Test" in provider.help_row.get_title()
    assert "layer" in provider.help_row.get_title()
    assert provider.open_button.get_label() == "Open Layer Settings"


@pytest.mark.ui
def test_card_shows_the_name_of_the_test(provider, editor, cmd):
    record = cmd.create("interval_test")
    provider.update_widgets(editor, cmd.workpieces(record))

    assert provider.group_subtitle == record.name


@pytest.mark.ui
def test_controller_shows_the_parameters_of_the_test(editor, cmd):
    record, controller = _controller(
        editor, cmd, "interval_test", {"count": 7}
    )
    assert controller.group.get_values()["count"] == 7
    assert controller.group.get_description() == record.name
    assert controller.help_row.get_visible()
    assert not controller.error_row.get_visible()


@pytest.mark.ui
def test_change_rebuilds_the_test_in_one_undo_step(editor, cmd):
    record, controller = _controller(
        editor, cmd, "interval_test", {"count": 3}
    )
    depth = len(editor.history_manager.undo_stack)

    _set(controller, "count", 5)
    controller.apply_now()

    assert record.params["count"] == 5
    assert len(cmd.workpieces(record)) == 6
    assert len(editor.history_manager.undo_stack) == depth + 1


@pytest.mark.ui
def test_invalid_values_show_an_error(editor, cmd):
    record, controller = _controller(editor, cmd, "interval_test")
    state = record.state

    _set(controller, "min_interval", 0.3)
    controller.apply_now()

    assert controller.error_row.get_visible()
    assert "minimum line interval" in controller.error_row.get_title()
    assert record.state == state

    _set(controller, "min_interval", 0.1)
    controller.apply_now()
    assert not controller.error_row.get_visible()
    assert record.params["min_interval"] == pytest.approx(0.1)


@pytest.mark.ui
def test_rows_follow_the_test_of_the_controller(editor, cmd):
    _, interval = _controller(editor, cmd, "interval_test")
    assert interval.group.row_for("cell_size") is not None

    _, focus = _controller(editor, cmd, "focus_test")
    assert focus.group.row_for("cell_size") is None
    assert focus.group.row_for("mode") is not None


@pytest.mark.ui
def test_focus_help_follows_the_method(editor, cmd):
    _, controller = _controller(editor, cmd, "focus_test")
    help_before = controller.help_row.get_title()
    assert _row(controller, "ramp_length").get_visible() is False

    _set(controller, "mode", "ramp")
    controller.apply_now()

    assert controller.help_row.get_title() != help_before
    assert "Prop up" in controller.help_row.get_title()
    assert _row(controller, "ramp_length").get_visible() is True
    assert _row(controller, "count").get_visible() is False


@pytest.mark.ui
def test_label_rows_follow_include_labels(editor, cmd):
    _, controller = _controller(editor, cmd, "interval_test")
    assert _row(controller, "label_speed").get_sensitive() is True
    _set(controller, "include_labels", False)
    assert _row(controller, "label_power_percent").get_sensitive() is False
    assert _row(controller, "label_speed").get_sensitive() is False


@pytest.mark.ui
def test_grid_mode_visibility(editor, cmd):
    _, controller = _controller(editor, cmd, "material_test_grid")
    power = _row(controller, "power_range")
    passes = _row(controller, "passes_range")
    offset = _row(controller, "offset_range")
    assert power.get_visible() is True
    assert passes.get_visible() is False
    assert offset.get_visible() is False

    _set(controller, "grid_mode", "Speed vs Offset")
    assert power.get_visible() is False
    assert offset.get_visible() is True
    assert _row(controller, "fixed_power").get_visible() is True


@pytest.mark.ui
def test_speed_vs_offset_defaults(editor, cmd):
    record, controller = _controller(editor, cmd, "material_test_grid")
    _set(controller, "grid_mode", "Speed vs Offset")
    controller.apply_now()

    values = controller.group.get_values()
    assert values["test_type"] == "Engrave"
    assert values["line_interval_mm"] == pytest.approx(0.5)
    step = _step(editor, record)
    assert step.test_type == "Engrave"
    assert step.line_interval_mm == pytest.approx(0.5)


@pytest.mark.ui
def test_dimension_labels_follow_grid_mode(editor, cmd):
    _, controller = _controller(editor, cmd, "material_test_grid")
    row = _row(controller, "grid_dimensions")
    assert row.get_title() == "Columns (Power Steps)"

    _set(controller, "grid_mode", "Speed vs Passes")
    controller.apply_now()
    row = _row(controller, "grid_dimensions")
    assert row.get_title() == "Columns (Speed Steps)"


@pytest.mark.ui
def test_preset_applies_values_and_clamps(editor, cmd):
    record, controller = _controller(editor, cmd, "material_test_grid")
    _set(controller, "preset", "Diode Engrave")
    controller.apply_now()

    step = _step(editor, record)
    assert step.test_type == "Engrave"
    assert step.power_range == (10.0, 100.0)
    assert step.speed_range == (1000.0, 5000.0)
    values = controller.group.get_values()
    assert values["speed_range"] == (1000.0, 5000.0)


@pytest.mark.ui
def test_speed_bound_follows_the_machine(editor, cmd, laser_machine):
    record, controller = _controller(editor, cmd, "material_test_grid")
    adapter = controller.group.adapter_for("speed_range")
    assert adapter is not None
    max_row = cast(SpeedSpinRow, adapter.extra_rows()[0])
    assert max_row._upper == pytest.approx(5000.0)

    laser_machine.set_max_cut_speed(8000)
    refreshed = CalibrationSettingsController(editor, record)
    adapter = refreshed.group.adapter_for("speed_range")
    assert adapter is not None
    max_row = cast(SpeedSpinRow, adapter.extra_rows()[0])
    assert max_row._upper == pytest.approx(8000.0)


@pytest.mark.ui
def test_speed_rows_are_unit_aware(editor, cmd, ui_context):
    _, controller = _controller(editor, cmd, "material_test_grid")
    label = cast(SpeedSpinRow, _row(controller, "label_speed"))
    assert isinstance(label, SpeedSpinRow)

    ui_context.config.set_unit_preference("speed", "mm/s")
    assert label.get_spin_button().get_text() == "16.7"


@pytest.mark.ui
def test_tuple_labels_survive_a_refresh(editor, cmd, ui_context):
    _, controller = _controller(editor, cmd, "material_test_grid")
    primary = _row(controller, "power_range")
    adapter = controller.group.adapter_for("power_range")
    assert adapter is not None
    extra = adapter.extra_rows()[0]

    ui_context.config.set_unit_preference("speed", "mm/s")
    controller.refresh()

    assert primary.get_title() == "Minimum Power (%)"
    assert extra.get_title() == "Maximum Power (%)"


@pytest.mark.ui
def test_legacy_grid_can_be_changed(provider, editor, cmd):
    record = cmd.create("material_test_grid")
    editor.doc.remove_asset(record)
    workpieces = cmd.workpieces(record)
    assert provider.can_handle(workpieces)
    provider.update_widgets(editor, workpieces)
    legacy = cmd.record_for_items(workpieces)
    assert legacy is not None
    controller = CalibrationSettingsController(editor, legacy)

    _set(controller, "grid_dimensions", (3, 3))
    controller.apply_now()

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
def test_debounced_change(editor, cmd):
    record, controller = _controller(
        editor, cmd, "interval_test", {"count": 3}
    )
    _set(controller, "count", 4)
    assert controller._timer_id
    assert record.params["count"] == 3
    controller.apply_now()
    assert controller._timer_id == 0
    assert record.params["count"] == 4


@pytest.mark.ui
def test_without_a_test_selected(provider, editor, cmd):
    other = WorkPiece("Other")
    editor.doc.active_layer.add_child(other)
    provider.update_widgets(editor, [other])
    assert provider.generator is None
    provider.scroll_into_view()
    assert provider._scroll_handlers == []


@pytest.mark.ui
def test_apply_without_changes_does_nothing(editor, cmd):
    record, controller = _controller(editor, cmd, "interval_test")
    depth = len(editor.history_manager.undo_stack)
    controller.apply_now()
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
def test_changes_while_updating_are_ignored(editor, cmd):
    _, controller = _controller(editor, cmd, "interval_test")
    controller._in_update = True
    _set(controller, "count", 9)
    controller._in_update = False
    assert controller._timer_id == 0
    assert not controller._pending_keys


@pytest.mark.ui
def test_unavailable_method_is_kept_and_explained(
    editor, cmd, laser_machine, monkeypatch
):
    """A Focus Test saved on a machine with a Z axis keeps its method on
    a machine without one: the settings say so, and an unrelated change
    does not switch the method."""
    monkeypatch.setattr(
        type(laser_machine), "has_z_axis", property(lambda self: False)
    )
    record = cmd.create("focus_test", {"mode": "ramp"})
    record.set_state(
        replace(record.state, params={**record.params, "mode": "z_axis"})
    )
    controller = CalibrationSettingsController(editor, record)

    assert controller.group.get_values()["mode"] == "z_axis"
    assert controller.warning_row.get_visible()
    assert "not available" in controller.warning_row.get_title()

    _set(controller, "ramp_length", 50.0)
    controller.apply_now()
    assert record.params["mode"] == "z_axis"
    assert controller.error_row.get_visible()


@pytest.mark.ui
def test_provider_is_registered_once_at_init():
    """Importing the frontend registers nothing; the init hook registers
    the Test Settings, the dialog section and the layer type once,
    however often it runs."""
    provider_registry = property_provider_registry
    section_registry = layer_settings_section_registry
    assert CalibrationPropertyProvider not in provider_registry.all()
    assert CalibrationLayerSettingsSection not in section_registry.all()
    try:
        frontend.rayforge_init(context=None)
        frontend.rayforge_init(context=None)
        assert provider_registry.all().count(CalibrationPropertyProvider) == 1
        assert (
            section_registry.all().count(CalibrationLayerSettingsSection) == 1
        )
    finally:
        provider_registry.unregister(CalibrationPropertyProvider)
        section_registry.unregister(CalibrationLayerSettingsSection)
        frontend.layer_type_registry.unregister_all_from_addon(
            frontend.ADDON_NAME
        )


@pytest.mark.ui
def test_rejected_value_is_kept_for_the_next_change(editor, cmd):
    """A value that made the test invalid is not dropped when another
    setting then makes it valid."""
    record, controller = _controller(editor, cmd, "interval_test")
    _set(controller, "min_interval", 0.3)
    controller.apply_now()
    assert controller.error_row.get_visible()

    _set(controller, "max_interval", 0.5)
    controller.apply_now()

    assert not controller.error_row.get_visible()
    assert record.params["min_interval"] == pytest.approx(0.3)
    assert record.params["max_interval"] == pytest.approx(0.5)


@pytest.mark.ui
def test_section_handles_test_layers(editor, cmd):
    record = cmd.create("interval_test")
    layer = editor.doc.find_descendant_by_uid(record.layer_uid)
    section = CalibrationLayerSettingsSection()

    assert section.can_handle(layer, editor)
    groups = section.create_widgets(layer, editor)
    try:
        (group,) = groups
        assert isinstance(group, VarSetWidget)
        assert group.get_title() == "Test Settings"
        assert group.get_description() == record.name
        assert group.get_values()["count"] == 5
    finally:
        section.close()


@pytest.mark.ui
def test_section_ignores_ordinary_layers(editor, cmd):
    other = Layer(name="Other")
    editor.doc.add_child(other)
    section = CalibrationLayerSettingsSection()
    assert not section.can_handle(other, editor)


@pytest.mark.ui
def test_test_layers_are_locked(editor, cmd):
    frontend.rayforge_init(context=None)
    try:
        record = cmd.create("interval_test")
        layer = editor.doc.find_descendant_by_uid(record.layer_uid)
        assert layer.layer_type == frontend.LAYER_TYPE_ID
        assert layer_is_locked(layer)
    finally:
        frontend.layer_type_registry.unregister_all_from_addon(
            frontend.ADDON_NAME
        )
