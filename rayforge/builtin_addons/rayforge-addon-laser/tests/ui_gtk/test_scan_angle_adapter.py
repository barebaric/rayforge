# flake8: noqa: E402
"""UI tests for the ScanAngleAdapter (raster angle row + preview)."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw
from laser_essentials.scan_angle_var import ScanAngleVar
from laser_essentials.widgets.scan_angle_adapter import ScanAngleAdapter

from rayforge.core.varset import BoolVar, VarSet
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

pytestmark = pytest.mark.ui


def test_scan_angle_var_creates_row_with_preview(ui_context):
    """ScanAngleVar renders a row with a DirectionPreview suffix."""
    var = ScanAngleVar()
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ActionRow)
    assert isinstance(adapter, ScanAngleAdapter)
    assert adapter.preview is not None
    assert adapter.get_value() == 0.0


def test_scan_angle_round_trip(ui_context):
    """ScanAngleAdapter round-trips the angle value."""
    var = ScanAngleVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, ScanAngleAdapter)

    adapter.set_value(90.0)
    assert adapter.get_value() == 90.0

    adapter.set_value(180.0)
    assert adapter.get_value() == 180.0


def test_scan_angle_preview_updates_with_cross_hatch(ui_context):
    """update_from_values feeds cross_hatch into the preview."""
    var = ScanAngleVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, ScanAngleAdapter)

    preview = adapter.preview
    assert preview.cross_hatch is False

    adapter.update_from_values({"scan_angle": 0.0, "cross_hatch": True})
    assert preview.cross_hatch is True

    adapter.update_from_values({"scan_angle": 0.0, "cross_hatch": False})
    assert preview.cross_hatch is False


def test_scan_angle_in_varset_widget(ui_context):
    """ScanAngleVar works inside a VarSetWidget with cross_hatch."""
    vs = VarSet(
        vars=[
            ScanAngleVar(),
            BoolVar(key="cross_hatch", label="Cross-Hatch", default=False),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    assert "scan_angle" in widget.widget_map
    adapter = widget._adapters["scan_angle"]
    assert isinstance(adapter, ScanAngleAdapter)

    widget.set_values({"scan_angle": 45.0, "cross_hatch": True})
    assert widget.get_values()["scan_angle"] == 45.0
    assert adapter.preview.cross_hatch is True
