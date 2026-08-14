# flake8: noqa: E402
"""Tests for the varset row manager's recipe mode and multi-key adapters.

Phase 1 of recipe2.md: the varset machinery gains per-row apply toggles
(recipe mode via :class:`RecipeVarSetWidget`), composite adapter
support (``related_keys``), and group descriptions. None of these
features are consumed by the recipe editor yet — they are tested here
in isolation.
"""

import os
import sys
from unittest.mock import Mock

import pytest

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests.",
            allow_module_level=True,
        )

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk

from rayforge.core.varset import BoolVar, IntVar, SliderFloatVar, VarSet
from rayforge.ui_gtk.doceditor.recipes.varset_widget import RecipeVarSetWidget
from rayforge.ui_gtk.varset.adapter import RowAdapter, register_adapter
from rayforge.ui_gtk.varset.adapter.base import Var
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

pytestmark = pytest.mark.ui


# ---- Helpers --------------------------------------------------------------


def _make_varset() -> VarSet:
    return VarSet(
        title="Test Group",
        description="A test description.",
        vars=[
            IntVar(key="speed", label="Speed", default=100, min_val=1),
            SliderFloatVar(
                key="power",
                label="Power",
                default=0.8,
                min_val=0.0,
                max_val=1.0,
                show_value=True,
                format_suffix="%",
            ),
            BoolVar(key="enabled", label="Enabled", default=True),
        ],
    )


# ---- Recipe mode: apply toggles -------------------------------------------


def test_recipe_mode_adds_apply_toggle_prefix(ui_context_initializer):
    """Each row gets a toggle button as a native prefix."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    for key in ("speed", "power", "enabled"):
        assert key in widget._apply_toggles
        toggle = widget._apply_toggles[key]
        assert isinstance(toggle, Gtk.ToggleButton)
        assert toggle.get_active() is False


def test_recipe_mode_dims_row_when_toggle_off(ui_context_initializer):
    """While the toggle is off the row is dimmed (opacity 0.5)."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    row, _ = widget.widget_map["speed"]
    assert row.get_opacity() == pytest.approx(0.5, abs=0.01)

    widget.set_apply_state("speed", True)
    assert row.get_opacity() == pytest.approx(1.0)

    widget.set_apply_state("speed", False)
    assert row.get_opacity() == pytest.approx(0.5, abs=0.01)


def test_apply_changed_signal_fires_on_toggle(ui_context_initializer):
    """Toggling fires apply_changed with the key and state."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    listener = Mock()
    widget.apply_changed.connect(listener)

    widget._apply_toggles["speed"].set_active(True)
    listener.assert_called_once_with(widget, key="speed", state=True)

    listener.reset_mock()
    widget._apply_toggles["speed"].set_active(False)
    listener.assert_called_once_with(widget, key="speed", state=False)


def test_recipe_mode_gates_data_changed(ui_context_initializer):
    """data_changed is suppressed while the apply toggle is off."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    listener = Mock()
    widget.data_changed.connect(listener)

    # Toggle off: changing the row value does NOT fire data_changed.
    adapter = widget._adapters["speed"]
    adapter.set_value(200)
    adapter.changed.send(adapter)
    listener.assert_not_called()

    # Toggle on: the toggle itself fires data_changed (announcing the
    # setting is now applied), then a value change fires it again.
    widget.set_apply_state("speed", True)
    listener.assert_called_once_with(widget, key="speed")

    listener.reset_mock()
    adapter.set_value(300)
    adapter.changed.send(adapter)
    listener.assert_called_once_with(widget, key="speed")


def test_set_get_apply_state_round_trip(ui_context_initializer):
    """Apply state can be set and read back per key."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    assert widget.get_apply_state("speed") is False
    widget.set_apply_state("speed", True)
    assert widget.get_apply_state("speed") is True
    widget.set_apply_state("speed", False)
    assert widget.get_apply_state("speed") is False


def test_setting_dicts_round_trip(ui_context_initializer):
    """set_setting_dicts / get_setting_dicts round-trip values and
    apply states."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    widget.set_setting_dicts(
        [
            {"name": "speed", "value": 250, "recipe_apply": True},
            {"name": "power", "value": 0.5, "recipe_apply": False},
            {"name": "enabled", "value": True, "recipe_apply": True},
        ]
    )

    assert widget.get_apply_state("speed") is True
    assert widget.get_apply_state("power") is False
    assert widget.get_apply_state("enabled") is True
    assert widget._adapters["speed"].get_value() == 250

    dicts = widget.get_setting_dicts()
    by_name = {d["name"]: d for d in dicts}
    assert by_name["speed"]["recipe_apply"] is True
    assert by_name["speed"]["value"] == 250
    assert by_name["power"]["recipe_apply"] is False
    assert by_name["power"]["value"] == 0.5
    assert by_name["enabled"]["recipe_apply"] is True


def test_setting_dicts_defaults_toggle_off(ui_context_initializer):
    """Keys without an explicit entry default to recipe_apply=False."""
    vs = _make_varset()
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    widget.set_setting_dicts(
        [{"name": "speed", "value": 250, "recipe_apply": True}]
    )
    assert widget.get_apply_state("speed") is True
    assert widget.get_apply_state("power") is False
    assert widget.get_apply_state("enabled") is False


# ---- Generic VarSetWidget has no recipe knowledge -------------------------


def test_generic_widget_has_no_apply_toggles(ui_context_initializer):
    """VarSetWidget never creates apply toggles."""
    vs = _make_varset()
    widget = VarSetWidget()
    widget.populate(vs)
    assert not hasattr(widget, "_apply_toggles")


# ---- Multi-key adapters (related_keys) ------------------------------------


class _CompositeVar(IntVar):
    """A Var whose adapter also manages a related key."""


@register_adapter(_CompositeVar)
class _CompositeAdapter(RowAdapter):
    """Test adapter that owns a primary key + one related key."""

    related_keys = ("related_key",)

    def __init__(self, row, spin, primary_key):
        super().__init__()
        self._row = row
        self._spin = spin
        self._primary_key = primary_key
        self._related_value: int | None = None
        spin.connect("notify::value", lambda s, p: self.changed.send(self))

    @classmethod
    def create(cls, var: Var, target_property: str):
        row = Adw.SpinRow(
            title=var.label,
            adjustment=Gtk.Adjustment(
                value=var.default or 0,
                lower=0,
                upper=999,
                step_increment=1,
            ),
        )
        return row, cls(row, row, var.key)

    def get_value(self):
        return int(self._spin.get_value())

    def set_value(self, value):
        self._spin.set_value(int(value))

    def get_value_for_key(self, key: str):
        if key == "related_key":
            return self._related_value
        return self.get_value()

    def set_value_for_key(self, key: str, value):
        if key == "related_key":
            self._related_value = int(value) if value is not None else None
        else:
            self.set_value(value)


def test_multi_key_adapter_skips_related_key_row(ui_context_initializer):
    """The related key gets no row of its own; only the primary does."""
    vs = VarSet(
        vars=[
            _CompositeVar(key="primary", label="Primary", default=10),
            IntVar(key="related_key", label="Related", default=5),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    assert "primary" in widget.widget_map
    assert "related_key" in widget._related_keys
    assert "related_key" not in widget._adapters


def test_multi_key_adapter_get_values_includes_related(
    ui_context_initializer,
):
    """get_values() returns both the primary and related key values."""
    vs = VarSet(
        vars=[
            _CompositeVar(key="primary", label="Primary", default=10),
            IntVar(key="related_key", label="Related", default=5),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    widget.set_values({"primary": 20, "related_key": 99})
    values = widget.get_values()
    assert values["primary"] == 20
    assert values["related_key"] == 99


def test_multi_key_adapter_data_changed_fires_for_all_keys(
    ui_context_initializer,
):
    """When a composite adapter fires, data_changed emits for every
    key it manages."""
    vs = VarSet(
        vars=[
            _CompositeVar(key="primary", label="Primary", default=10),
            IntVar(key="related_key", label="Related", default=5),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    listener = Mock()
    widget.data_changed.connect(listener)

    adapter = widget._adapters["primary"]
    adapter.changed.send(adapter)

    keys_emitted = [call.kwargs.get("key") for call in listener.call_args_list]
    assert "primary" in keys_emitted
    assert "related_key" in keys_emitted


def test_multi_key_recipe_mode_apply_toggle_on_primary(
    ui_context_initializer,
):
    """In recipe mode the toggle lives on the primary key; related keys
    share the same toggle."""
    vs = VarSet(
        vars=[
            _CompositeVar(key="primary", label="Primary", default=10),
            IntVar(key="related_key", label="Related", default=5),
        ]
    )
    widget = RecipeVarSetWidget()
    widget.populate(vs)

    assert "primary" in widget._apply_toggles
    assert "related_key" not in widget._apply_toggles

    widget.set_apply_state("related_key", True)
    assert widget.get_apply_state("primary") is True
    assert widget.get_apply_state("related_key") is True


# ---- Group descriptions ---------------------------------------------------


def test_varset_description_displayed(ui_context_initializer):
    """The VarSet description is propagated to the widget."""
    vs = VarSet(
        title="My Group",
        description="Some description text.",
        vars=[IntVar(key="x", label="X", default=1)],
    )
    widget = VarSetWidget()
    widget.populate(vs)
    assert widget.get_description() == "Some description text."
