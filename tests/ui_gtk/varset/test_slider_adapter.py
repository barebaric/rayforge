# flake8: noqa: E402
"""UI tests for the SliderAdapter (SliderFloatVar and SliderIntVar)."""

import os
import sys

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

from rayforge.core.varset import (
    SliderFloatVar,
    SliderIntVar,
    VarSet,
)
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

pytestmark = pytest.mark.ui


def test_slider_float_var_creates_slider_row(ui_context_initializer):
    """SliderFloatVar with format_suffix='%' renders as a slider."""
    var = SliderFloatVar(
        key="power",
        label="Power",
        default=0.8,
        min_val=0.0,
        max_val=1.0,
        show_value=True,
        format_suffix="%",
    )
    row, adapter = create_row_for_var(var, "value")
    assert row is not None
    assert adapter is not None

    # The slider normalizes 0-1 to 0-100%.
    assert adapter.get_value() == 0.8
    adapter.set_value(0.5)
    assert adapter.get_value() == 0.5


def test_slider_float_var_in_varset_widget(ui_context_initializer):
    """SliderFloatVar works inside a VarSetWidget."""
    vs = VarSet(
        vars=[
            SliderFloatVar(
                key="power",
                label="Power",
                default=0.8,
                min_val=0.0,
                max_val=1.0,
                show_value=True,
                format_suffix="%",
            ),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    assert "power" in widget.widget_map
    widget.set_values({"power": 0.5})
    assert widget.get_values()["power"] == 0.5


def test_slider_int_var_renders_int_slider(ui_context_initializer):
    """SliderIntVar renders as a slider returning integer values."""
    var = SliderIntVar(
        key="threshold",
        label="Threshold",
        default=128,
        min_val=0,
        max_val=255,
        show_value=True,
    )
    row, adapter = create_row_for_var(var, "value")
    assert row is not None
    assert adapter is not None

    assert adapter.get_value() == 128
    adapter.set_value(200)
    assert adapter.get_value() == 200
    assert isinstance(adapter.get_value(), int)


def test_slider_int_var_round_trip_through_widget(ui_context_initializer):
    """SliderIntVar values round-trip through a VarSetWidget."""
    vs = VarSet(
        vars=[
            SliderIntVar(
                key="threshold",
                label="Threshold",
                default=128,
                min_val=0,
                max_val=255,
            ),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    widget.set_values({"threshold": 200})
    assert widget.get_values()["threshold"] == 200
