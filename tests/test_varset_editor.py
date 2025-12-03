# flake8: noqa: E402
import os
import sys
import pytest
from unittest.mock import Mock

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk, GLib
import time

from rayforge.core.varset import VarSet, IntVar, FloatVar
from rayforge.shared.ui.varset_editor import (
    VarSetEditorWidget,
    VarDefinitionRowWidget,
)


def process_events(duration_sec: float = 0.1):
    """Processes all pending GTK events for a short duration without blocking."""
    end_time = time.monotonic() + duration_sec
    context = GLib.main_context_default()
    while time.monotonic() < end_time:
        if context.pending():
            context.iteration(False)
        # A small sleep prevents pegging the CPU
        time.sleep(0.001)


@pytest.fixture
def var_set() -> VarSet:
    """Provides a simple VarSet with an Int and Float Var for testing."""
    vs = VarSet(title="Test Set")
    vs.add(IntVar(key="int_key", label="Integer Value", default=10))
    vs.add(FloatVar(key="float_key", label="Float Value", default=42.5))
    return vs


@pytest.fixture
def editor_widget_in_window(ui_context_initializer, var_set):
    """
    Creates a VarSetEditorWidget, populates it, and places it in a visible
    Adw.Window for testing.
    """
    window = Adw.Window()
    editor = VarSetEditorWidget()
    editor.populate(var_set)

    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(editor)
    window.set_content(box)
    window.present()

    # Allow GTK to draw and settle
    process_events()

    yield editor, window, var_set

    # Teardown
    window.close()
    process_events()


@pytest.mark.ui
def test_editor_int_var_spin_button_triggers_signal(editor_widget_in_window):
    """
    Tests that changing the 'Default Value' spin button for an IntVar in the
    VarSetEditorWidget correctly triggers the `var_definition_changed` signal
    on the underlying VarSet.
    """
    # --- Arrange ---
    editor, _window, var_set = editor_widget_in_window
    int_var = var_set.get("int_key")
    assert isinstance(int_var, IntVar)
    assert int_var.default == 10

    listener = Mock()
    # The editor modifies the *definition* (the default), not the live value.
    var_set.var_definition_changed.connect(listener)

    # Find the specific Adw.SpinRow for the default value
    list_box_row = editor.list_box.get_row_at_index(0)
    var_def_widget = list_box_row.get_child()
    assert isinstance(var_def_widget, VarDefinitionRowWidget)
    spin_row = var_def_widget.default_row
    assert isinstance(spin_row, Adw.SpinRow)
    assert spin_row.get_value() == 10

    # --- Act ---
    # Simulate the user changing the value in the spin button
    spin_row.get_adjustment().set_value(99)
    # Allow the 'changed' signal from the spin row to be processed by GTK,
    # which then triggers the editor's internal callback.
    process_events()

    # --- Assert ---
    # 1. The signal was emitted correctly
    listener.assert_called_once_with(var_set, var=int_var, property="default")

    # 2. The underlying Var object's default value was updated
    assert int_var.default == 99


@pytest.mark.ui
def test_editor_float_var_spin_button_triggers_signal(editor_widget_in_window):
    """
    Tests that changing the 'Default Value' spin button for a FloatVar in the
    VarSetEditorWidget correctly triggers the `var_definition_changed` signal
    on the underlying VarSet.
    """
    # --- Arrange ---
    editor, _window, var_set = editor_widget_in_window
    float_var = var_set.get("float_key")
    assert isinstance(float_var, FloatVar)
    assert float_var.default == 42.5

    listener = Mock()
    var_set.var_definition_changed.connect(listener)

    # Find the specific Adw.SpinRow for the default value
    list_box_row = editor.list_box.get_row_at_index(1)
    var_def_widget = list_box_row.get_child()
    assert isinstance(var_def_widget, VarDefinitionRowWidget)
    spin_row = var_def_widget.default_row
    assert isinstance(spin_row, Adw.SpinRow)
    assert spin_row.get_value() == 42.5

    # --- Act ---
    # Simulate the user changing the value in the spin button
    spin_row.get_adjustment().set_value(55.5)
    process_events()

    # --- Assert ---
    # 1. The signal was emitted correctly
    listener.assert_called_once_with(
        var_set, var=float_var, property="default"
    )

    # 2. The underlying Var object's default value was updated
    assert float_var.default == 55.5
