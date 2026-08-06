"""UI tests for the LengthRowAdapter (mm-base unit conversion)."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.core.varset import LengthVar
from rayforge.ui_gtk.shared.pref_rows.length_spin_row import LengthSpinRow
from rayforge.ui_gtk.varset.adapter import create_row_for_var


@pytest.mark.ui
def test_length_adapter_mm_display(ui_context_initializer):
    var = LengthVar(key="offset_mm", label="Offset", default=12.5)
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, LengthSpinRow)
    assert adapter is not None

    assert row.get_value() == pytest.approx(12.5)
    assert adapter.get_value() == pytest.approx(12.5)

    adapter.set_value(5.0)
    assert row.get_value() == pytest.approx(5.0)
    assert adapter.get_value() == pytest.approx(5.0)


@pytest.mark.ui
def test_length_adapter_imperial_conversion(ui_context_initializer):
    context = ui_context_initializer
    context.config.unit_preferences["length"] = "in"

    var = LengthVar(key="offset_mm", label="Offset", default=25.4)
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, LengthSpinRow)
    assert adapter is not None

    assert row.get_value() == pytest.approx(1.0)
    assert adapter.get_value() == pytest.approx(25.4)

    adapter.set_value(50.8)
    assert row.get_value() == pytest.approx(2.0)
    assert adapter.get_value() == pytest.approx(50.8)


@pytest.mark.ui
def test_length_adapter_respects_bounds(ui_context_initializer):
    var = LengthVar(
        key="step_over_mm", label="Step Over", min_val=0.0, max_val=10.0
    )
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, LengthSpinRow)
    assert adapter is not None

    adj = row.get_adjustment()
    assert adj.get_lower() == pytest.approx(0.0)
    assert adj.get_upper() == pytest.approx(10.0)
