# flake8: noqa: E402
"""UI tests for the ComboAdapter, incl. per-var "no selection" labels."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk

from rayforge.core.varset import ChoiceVar
from rayforge.ui_gtk.varset.adapter import create_row_for_var


def _model_strings(row: Adw.ComboRow) -> list:
    model = row.get_model()
    assert isinstance(model, Gtk.StringList)
    return [model.get_string(i) for i in range(model.get_n_items())]


def test_default_null_label_is_none_selected(ui_context_initializer):
    var = ChoiceVar(
        key="protocol",
        label="Protocol variant",
        choices=["ESP3D", "Longer"],
        allow_none=True,
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ComboRow)
    assert adapter is not None
    assert _model_strings(row) == ["None Selected", "ESP3D", "Longer"]
    row.set_selected(0)
    assert adapter.get_value() is None


def test_custom_null_label_shown_and_maps_to_none(ui_context_initializer):
    var = ChoiceVar(
        key="protocol",
        label="Protocol variant",
        choices=["ESP3D", "Longer"],
        allow_none=True,
        null_label="Standard",
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ComboRow)
    assert adapter is not None
    assert _model_strings(row) == ["Standard", "ESP3D", "Longer"]

    row.set_selected(0)
    assert adapter.get_value() is None
    row.set_selected(1)
    assert adapter.get_value() == "ESP3D"

    # set_value() back to "none" lands on the custom label.
    adapter.set_value(None)
    assert row.get_selected() == 0


def test_custom_null_label_round_trips_through_set_value(
    ui_context_initializer,
):
    var = ChoiceVar(
        key="protocol",
        label="Protocol variant",
        choices=["ESP3D", "Longer"],
        allow_none=True,
        null_label="Standard",
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ComboRow)
    assert adapter is not None
    adapter.set_value("Longer")
    assert row.get_selected() == 2
    assert adapter.get_value() == "Longer"
