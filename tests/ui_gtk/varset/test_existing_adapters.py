# flake8: noqa: E402
"""Verify that existing var types render correctly via their adapters.

These are the mappings the migration relies on: SliderFloatVar renders
as a percent slider, LaserHeadVar renders as a combo row, and
LabeledChoiceVar renders as a combo with display/value mapping.
"""

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

from gi.repository import Adw

from rayforge.core.varset import (
    SliderFloatVar,
    VarSet,
)
from rayforge.core.varset.labeledchoicevar import LabeledChoiceVar
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
