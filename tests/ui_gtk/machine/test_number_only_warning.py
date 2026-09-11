"""Tests for number-only warnings in the G-code editing dialogs."""

import copy

import pytest
from gi.repository import Adw, Gtk

from rayforge.machine.models.dialect import GRBL_DIALECT
from rayforge.machine.models.dialect.grbl_raster import GRBL_RASTER_DIALECT
from rayforge.machine.models.macro import Macro
from rayforge.ui_gtk.machine.dialect_editor import DialectEditorDialog
from rayforge.ui_gtk.machine.gcode_editor import GcodeEditorDialog
from rayforge.ui_gtk.machine.validation import (
    POWER_MOVE_TEMPLATE_KEYS,
    find_number_only_line,
    format_continuous_mode_toggle_warning,
    format_continuous_mode_warning,
    format_number_only_warning,
    has_s_command,
    is_number_only,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("6000", True),
        (" 6000 ", True),
        ("-296.429", True),
        ("+3", True),
        ("3.", True),
        (".5", True),
        ("", False),
        ("   ", False),
        ("G1 X0", False),
        ("6000 rpm", False),
        (";6000", False),
        ("M4", False),
    ],
)
def test_is_number_only(text, expected):
    assert is_number_only(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("G21\n6000\nG90", (2, "6000")),
        ("6000\nG90", (1, "6000")),
        ("G21\nG90", None),
        ("-1.5\n\nG90", (1, "-1.5")),
        ("", None),
    ],
)
def test_find_number_only_line(text, expected):
    assert find_number_only_line(text) == expected


def test_format_number_only_warning():
    assert "42" in format_number_only_warning("42")
    assert "Line 3" in format_number_only_warning("42", lineno=3)


@pytest.mark.parametrize(
    "template,expected",
    [
        ("G1{x_cmd}{y_cmd}{s_command}", True),
        ("G1{f_command}{s_command:.0f}", True),
        ("G1{x_cmd}{y_cmd}{f_command}", False),
        ("G1{s_vel}", False),
        ("", False),
    ],
)
def test_has_s_command(template, expected):
    assert has_s_command(template) is expected


def test_format_continuous_mode_warnings():
    warning = format_continuous_mode_warning()
    assert "{s_command}" in warning
    assert "silently" in warning

    toggle = format_continuous_mode_toggle_warning(["Linear Move"])
    assert "{s_command}" in toggle
    assert "Linear Move" in toggle


@pytest.fixture
def macro() -> Macro:
    return Macro(name="test", code=["G21", "G90"])


@pytest.fixture
def parent() -> Gtk.Window:
    return Gtk.Window()


@pytest.mark.ui
def test_macro_editor_warns_for_number_only_line(macro, parent):
    dialog = GcodeEditorDialog(parent, macro)
    assert dialog.warning_banner.get_revealed() is False

    buffer = dialog.text_view.get_buffer()
    buffer.set_text("G21\n6000\nG90", -1)

    assert dialog.warning_banner.get_revealed() is True
    assert "6000" in dialog.warning_banner.get_title()
    assert "Line 2" in dialog.warning_banner.get_title()
    assert dialog.save_button.get_sensitive() is True


@pytest.mark.ui
def test_macro_editor_clears_warning(macro, parent):
    dialog = GcodeEditorDialog(parent, macro)
    buffer = dialog.text_view.get_buffer()
    buffer.set_text("6000", -1)
    assert dialog.warning_banner.get_revealed() is True

    buffer.set_text("M4 S6000", -1)
    assert dialog.warning_banner.get_revealed() is False


@pytest.mark.ui
def test_macro_editor_warns_on_constructed_content(parent):
    macro = Macro(name="test", code=["G21", "6000"])
    dialog = GcodeEditorDialog(parent, macro)
    assert dialog.warning_banner.get_revealed() is True


@pytest.mark.ui
def test_macro_editor_ignores_comments_and_inline_numbers(macro, parent):
    dialog = GcodeEditorDialog(parent, macro)
    buffer = dialog.text_view.get_buffer()
    buffer.set_text("G21 ; set units\n;6000\nG1 X10 F6000", -1)
    assert dialog.warning_banner.get_revealed() is False


@pytest.fixture
def dialect():
    return copy.deepcopy(GRBL_DIALECT)


def _get_script_text_view(dialog, key) -> Gtk.TextView:
    row, _var = dialog.scripts_widget.widget_map[key]
    text_view = getattr(row, "core_widget", None)
    assert isinstance(text_view, Gtk.TextView)
    return text_view


@pytest.mark.ui
def test_dialect_editor_warns_for_number_only_script_line(dialect, parent):
    dialog = DialectEditorDialog(parent, dialect)
    assert dialog.warning_banner.get_revealed() is False

    text_view = _get_script_text_view(dialog, "preamble")
    text_view.get_buffer().set_text("G21\n6000\nG90", -1)

    assert dialog.warning_banner.get_revealed() is True
    assert "6000" in dialog.warning_banner.get_title()
    assert dialog.save_button.get_sensitive() is True


@pytest.mark.ui
def test_dialect_editor_script_warning_row_state(dialect, parent):
    dialog = DialectEditorDialog(parent, dialect)
    row, _var = dialog.scripts_widget.widget_map["preamble"]
    text_view = _get_script_text_view(dialog, "preamble")

    text_view.get_buffer().set_text("6000", -1)
    assert row.has_css_class("warning")
    icon = getattr(row, "_warning_icon_widget", None)
    assert icon is not None
    assert icon.get_visible()
    assert "6000" in icon.get_tooltip_text()

    text_view.get_buffer().set_text("G21", -1)
    assert not row.has_css_class("warning")
    assert not icon.get_visible()
    assert dialog.warning_banner.get_revealed() is False


@pytest.mark.ui
def test_dialect_editor_warns_for_number_only_template(dialect, parent):
    dialog = DialectEditorDialog(parent, dialect)
    row, _var = dialog.templates_widget.widget_map["laser_on"]
    assert isinstance(row, Adw.EntryRow)

    row.set_text("6000")
    assert row.has_css_class("warning")
    assert dialog.warning_banner.get_revealed() is True
    assert dialog.save_button.get_sensitive() is True

    row.set_text("M4 S{power}")
    assert not row.has_css_class("warning")
    assert dialog.warning_banner.get_revealed() is False


@pytest.mark.ui
def test_dialect_editor_error_still_blocks_save(dialect, parent):
    dialog = DialectEditorDialog(parent, dialect)
    row, _var = dialog.templates_widget.widget_map["laser_on"]
    assert isinstance(row, Adw.EntryRow)
    row.set_text("M4 S{unknown_var}")

    assert row.has_css_class("error")
    assert dialog.save_button.get_sensitive() is False


def _get_continuous_toggle(dialog) -> Adw.SwitchRow:
    row, _var = dialog.settings_widget.widget_map["continuous_laser_mode"]
    assert isinstance(row, Adw.SwitchRow)
    return row


@pytest.mark.ui
def test_dialect_editor_warns_continuous_mode_without_s_command(
    dialect, parent
):
    dialog = DialectEditorDialog(parent, dialect)
    assert dialog.warning_banner.get_revealed() is False

    toggle_row = _get_continuous_toggle(dialog)
    toggle_row.set_active(True)

    linear_row, _var = dialog.templates_widget.widget_map["linear_move"]
    assert linear_row.has_css_class("warning")
    icon = getattr(linear_row, "_warning_icon_widget", None)
    assert icon is not None
    assert icon.get_visible()
    assert "{s_command}" in icon.get_tooltip_text()

    travel_row, _var = dialog.templates_widget.widget_map["travel_move"]
    assert travel_row.has_css_class("warning")
    for key in POWER_MOVE_TEMPLATE_KEYS:
        row, _var = dialog.templates_widget.widget_map[key]
        if _var.value:
            assert row.has_css_class("warning"), key

    assert toggle_row.has_css_class("warning")
    toggle_icon = getattr(toggle_row, "_warning_icon_widget", None)
    assert toggle_icon is not None
    assert "Linear Move" in toggle_icon.get_tooltip_text()

    assert dialog.warning_banner.get_revealed() is True
    assert "{s_command}" in dialog.warning_banner.get_title()
    assert dialog.save_button.get_sensitive() is True


@pytest.mark.ui
def test_dialect_editor_clears_continuous_warning_on_s_command(
    dialect, parent
):
    dialog = DialectEditorDialog(parent, dialect)
    linear_row, _var = dialog.templates_widget.widget_map["linear_move"]
    assert isinstance(linear_row, Adw.EntryRow)
    _get_continuous_toggle(dialog).set_active(True)
    assert linear_row.has_css_class("warning")

    linear_row.set_text("G1{x_cmd}{y_cmd}{z_cmd}{f_command}{s_command}")
    assert not linear_row.has_css_class("warning")


@pytest.mark.ui
def test_dialect_editor_clears_continuous_warning_when_toggled_off(
    dialect, parent
):
    dialog = DialectEditorDialog(parent, dialect)
    linear_row, _var = dialog.templates_widget.widget_map["linear_move"]
    toggle_row = _get_continuous_toggle(dialog)

    toggle_row.set_active(True)
    assert linear_row.has_css_class("warning")
    assert dialog.warning_banner.get_revealed() is True

    toggle_row.set_active(False)
    assert not linear_row.has_css_class("warning")
    assert not toggle_row.has_css_class("warning")
    assert dialog.warning_banner.get_revealed() is False


@pytest.mark.ui
def test_dialect_editor_continuous_mode_with_s_command_is_quiet(parent):
    dialog = DialectEditorDialog(parent, copy.deepcopy(GRBL_RASTER_DIALECT))
    assert dialog.warning_banner.get_revealed() is False
    for key in POWER_MOVE_TEMPLATE_KEYS:
        row, _var = dialog.templates_widget.widget_map[key]
        assert not row.has_css_class("warning"), key
    toggle_row = _get_continuous_toggle(dialog)
    assert not toggle_row.has_css_class("warning")
