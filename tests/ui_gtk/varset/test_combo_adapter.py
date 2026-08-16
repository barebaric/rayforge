# flake8: noqa: E402
"""UI tests for the ComboAdapter, incl. per-var "no selection" labels."""

import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Adw, GLib, Gtk

from rayforge.core.varset import ChoiceVar
from rayforge.core.varset.labeledchoicevar import LabeledChoiceVar
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


def test_labeled_choice_var_creates_combo_row(ui_context_initializer):
    """LabeledChoiceVar renders as an Adw.ComboRow with display labels."""
    var = LabeledChoiceVar(
        key="cut_side",
        label="Cut Side",
        choices=[
            ("Inside", "INSIDE"),
            ("Outside", "OUTSIDE"),
            ("Centerline", "CENTERLINE"),
        ],
        default="OUTSIDE",
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ComboRow)
    assert adapter is not None

    # The stored value is the internal name, not the display label.
    assert adapter.get_value() == "OUTSIDE"
    adapter.set_value("INSIDE")
    assert adapter.get_value() == "INSIDE"


def _walk(widget, depth=0):
    """Yield widget and its descendants in tree order."""
    yield widget
    child = widget.get_first_child()
    while child is not None:
        yield from _walk(child, depth + 1)
        child = child.get_next_sibling()


def _find_inline_label(row: Adw.ComboRow) -> Gtk.Label:
    """Find the label inside the inline (collapsed) list view."""
    for w in _walk(row):
        if isinstance(w, Gtk.ListView) and "inline" in w.get_css_classes():
            # The inline list view has one list item whose child is the
            # box built by our factory; the label is its first child.
            item_widget = w.get_first_child()
            box = item_widget.get_first_child() if item_widget else None
            label = box.get_first_child() if box else None
            if isinstance(label, Gtk.Label):
                return label
    raise AssertionError("No inline listview label found")


def test_new_combo_row_right_aligns_inline_label(ui_context_initializer):
    row = Adw.ComboRow(
        title="Unit",
        model=Gtk.StringList.new(["mm", "in"]),
    )
    assert isinstance(row, Adw.ComboRow)
    # A list factory is installed so the popover stays left-aligned.
    assert row.get_list_factory() is not None

    window = Adw.Window()
    window.set_content(row)
    window.present()

    deadline = time.monotonic() + 2.0
    label = None
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        try:
            label = _find_inline_label(row)
            break
        except AssertionError:
            time.sleep(0.02)

    window.destroy()
    if label is None:
        pytest.fail("Inline listview label never appeared")
    assert label.get_xalign() == pytest.approx(1.0)


def test_combo_row_custom_factory_is_preserved(ui_context_initializer):
    """A custom factory + use_subtitle must keep its own rendering."""
    custom_factory = Gtk.SignalListItemFactory()

    def on_setup(_factory, list_item):
        label = Gtk.Label(xalign=0.0)
        list_item.set_child(label)

    def on_bind(_factory, list_item):
        item = list_item.get_item()
        text = ""
        if isinstance(item, Gtk.StringObject):
            text = item.get_string()
        list_item.get_child().set_label(f"custom:{text}")

    custom_factory.connect("setup", on_setup)
    custom_factory.connect("bind", on_bind)

    row = Adw.ComboRow(
        title="Driver",
        model=Gtk.StringList.new(["GRBL"]),
        use_subtitle=True,
        factory=custom_factory,
    )
    assert row.get_factory() is custom_factory
    # The monkey patch must not install its own list factory either.
    assert row.get_list_factory() is None
