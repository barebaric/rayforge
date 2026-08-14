# flake8: noqa: E402
"""UI tests for the LevelsAdapter (brightness-range histogram row)."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk
from laser_essentials.levels_range_var import LevelsRangeVar
from laser_essentials.widgets.levels_adapter import LevelsAdapter

from rayforge.core.varset import BoolVar, IntVar, VarSet
from rayforge.ui_gtk.shared.histogram_preview import HistogramPreview
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.adapter.switch import SwitchAdapter
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

pytestmark = pytest.mark.ui


def test_levels_range_var_creates_histogram_row(ui_context):
    """LevelsRangeVar renders an ActionRow with a HistogramPreview."""
    var = LevelsRangeVar()
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ActionRow)
    assert isinstance(adapter, LevelsAdapter)
    assert isinstance(adapter.preview, HistogramPreview)
    assert adapter.get_value() == 0


def test_levels_round_trip(ui_context):
    """LevelsAdapter round-trips the two keys it manages."""
    var = LevelsRangeVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, LevelsAdapter)

    adapter.set_value(10)
    adapter.set_value_for_key("white_point", 200)
    assert adapter.get_value() == 10
    assert adapter.get_value_for_key("white_point") == 200


def test_levels_auto_mode_synced_from_sibling(ui_context):
    """update_from_values flips the preview's auto mode from
    auto_levels."""
    var = LevelsRangeVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, LevelsAdapter)

    assert adapter.preview.auto_mode is True
    adapter.update_from_values({"auto_levels": False})
    assert adapter.preview.auto_mode is False
    adapter.update_from_values({"auto_levels": True})
    assert adapter.preview.auto_mode is True


def test_levels_in_varset_widget(ui_context):
    """auto_levels gets its own row; black/white points share the
    histogram row."""
    vs = VarSet(
        vars=[
            BoolVar(key="auto_levels", label="Auto Levels", default=True),
            LevelsRangeVar(),
            IntVar(key="white_point", label="White Point", default=255),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    # auto_levels is its own row; black_point is the histogram row.
    assert "auto_levels" in widget.widget_map
    assert "black_point" in widget.widget_map
    assert "white_point" in widget._related_keys
    assert "white_point" not in widget._adapters

    # The auto-levels row uses the plain SwitchAdapter.
    assert isinstance(widget._adapters["auto_levels"], SwitchAdapter)
    assert isinstance(widget._adapters["black_point"], LevelsAdapter)

    widget.set_values(
        {
            "auto_levels": False,
            "black_point": 30,
            "white_point": 220,
        }
    )
    values = widget.get_values()
    assert values["auto_levels"] is False
    assert values["black_point"] == 30
    assert values["white_point"] == 220


def test_levels_data_changed_fires_for_all_keys(ui_context):
    """Changing any part of the levels row emits for both keys."""
    vs = VarSet(
        vars=[
            BoolVar(key="auto_levels", label="Auto Levels", default=True),
            LevelsRangeVar(),
            IntVar(key="white_point", label="White Point", default=255),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    emitted = []
    widget.data_changed.connect(
        lambda w, key=None: emitted.append(key), weak=False
    )

    adapter = widget._adapters["black_point"]
    adapter.changed.send(adapter)

    assert "black_point" in emitted
    assert "white_point" in emitted


def test_levels_cursor_changes_over_marker(ui_context):
    """Hovering over a draggable marker shows a horizontal-drag cursor."""
    var = LevelsRangeVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, LevelsAdapter)
    preview = adapter.preview

    # Auto levels off so the markers are draggable.
    preview.auto_mode = False

    # Initially no custom cursor.
    assert preview.get_cursor() is None

    # Hover over the black marker (near x=5) -> drag cursor.
    ctrl = Gtk.EventControllerMotion.new()
    preview.add_controller(ctrl)
    preview._on_motion(ctrl, 5.0, 50.0)
    assert preview._hovering == "black"
    assert preview.get_cursor() is not None

    # Hover over empty space -> cursor resets.
    preview._on_motion(ctrl, 100.0, 50.0)
    assert preview._hovering is None
    assert preview.get_cursor() is None


def test_levels_cursor_during_drag(ui_context):
    """The drag cursor stays while dragging a marker."""
    var = LevelsRangeVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, LevelsAdapter)
    preview = adapter.preview
    preview.auto_mode = False

    gesture = Gtk.GestureClick.new()
    preview.add_controller(gesture)
    preview._on_pressed(gesture, 1, 5.0, 50.0)
    assert preview._dragging == "black"
    assert preview.get_cursor() is not None

    preview._on_released(gesture, 1, 55.0, 50.0)
    assert preview.get_cursor() is None
