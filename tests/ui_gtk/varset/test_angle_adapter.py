# flake8: noqa: E402
"""Tests for the AngleVar + AngleRowAdapter."""

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

from rayforge.core.varset import AngleVar, VarSet
from rayforge.ui_gtk.shared.pref_rows.angle_spin_row import AngleSpinRow
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

pytestmark = pytest.mark.ui


def test_angle_var_creates_angle_spin_row(ui_context_initializer):
    var = AngleVar(
        key="angle", label="Angle", default=45.0, min_val=0, max_val=180
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, AngleSpinRow)
    assert adapter is not None
    assert adapter.get_value() == 45.0


def test_angle_adapter_round_trip(ui_context_initializer):
    var = AngleVar(
        key="angle", label="Angle", default=45.0, min_val=-90, max_val=90
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, AngleSpinRow)
    assert adapter is not None

    adapter.set_value(30.0)
    assert adapter.get_value() == 30.0

    adapter.set_value(-45.0)
    assert adapter.get_value() == -45.0


def test_angle_var_in_varset_widget(ui_context_initializer):
    vs = VarSet(
        vars=[
            AngleVar(
                key="scan_angle",
                label="Scan Angle",
                default=0.0,
                min_val=0,
                max_val=180,
            ),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    assert "scan_angle" in widget.widget_map
    row, _ = widget.widget_map["scan_angle"]
    assert isinstance(row, AngleSpinRow)

    widget.set_values({"scan_angle": 90.0})
    assert widget.get_values()["scan_angle"] == 90.0


def test_angle_var_defaults_to_full_rotation_range(
    ui_context_initializer,
):
    var = AngleVar(key="angle", label="Angle", default=0.0)
    row, _adapter = create_row_for_var(var, "value")
    assert isinstance(row, AngleSpinRow)
    adj = row.get_adjustment()
    assert adj.get_lower() == -360.0
    assert adj.get_upper() == 360.0
