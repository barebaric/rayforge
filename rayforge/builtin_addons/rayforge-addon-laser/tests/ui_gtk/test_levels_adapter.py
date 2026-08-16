# flake8: noqa: E402
"""UI tests for the LevelsAdapter (brightness-range histogram row)."""

from typing import cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk
from laser_essentials.levels_range_var import LevelsRangeVar
from laser_essentials.steps import EngraveStep
from laser_essentials.widgets.levels_adapter import LevelsAdapter
from laser_essentials.widgets.raster_page import RasterSettingsPage

from rayforge.core.step_registry import step_registry
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


def test_levels_compute_histogram_noop_without_source(ui_context):
    """Without a histogram source the adapter does not crash."""
    var = LevelsRangeVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, LevelsAdapter)
    adapter.compute_histogram()


def test_levels_compute_histogram_updates_preview(ui_context, laser_machine):
    """compute_histogram pushes histogram data into the preview."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = step_cls.create(ui_context)
    # The layer has no workpieces in this fixture, so the preview is
    # cleared rather than filled, but the adapter must handle it.
    var = LevelsRangeVar()
    _row, adapter = create_row_for_var(var, "value")
    assert isinstance(adapter, LevelsAdapter)
    adapter.set_histogram_source(step)
    adapter.compute_histogram()


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


def test_levels_drag_fires_adapter_changed(ui_context):
    """Dragging a marker must fire the adapter's ``changed`` signal on
    release, not during the drag.

    The preview's drag signals are plain blinker Signals; the adapter
    connects inline lambdas to them, so the connection must be held
    strongly or the handler is garbage-collected and drags silently
    never commit. Emissions are deferred to mouse release so the
    pipeline does not restart on every motion event.
    """
    vs = VarSet(
        vars=[
            BoolVar(key="auto_levels", label="Auto Levels", default=True),
            LevelsRangeVar(),
            IntVar(key="white_point", label="White Point", default=255),
        ]
    )
    widget = VarSetWidget()
    widget.populate(vs)

    adapter = widget._adapters["black_point"]
    assert isinstance(adapter, LevelsAdapter)
    adapter.update_from_values({"auto_levels": False})
    preview = adapter.preview
    assert preview.auto_mode is False

    preview.set_size_request(preview.WIDTH, preview.HEIGHT)
    preview.allocate(preview.WIDTH, preview.HEIGHT, -1, None)

    fired = []
    adapter.changed.connect(lambda s: fired.append(s), weak=False)

    ctrl = Gtk.EventControllerMotion.new()
    click = Gtk.GestureClick.new()
    preview.add_controller(ctrl)
    preview.add_controller(click)
    preview._dragging = "black"
    preview._on_motion(ctrl, 100.0, 50.0)

    assert preview.black_point == 128
    assert not fired, "adapter.changed fired during the drag"

    preview._on_released(click, 1, 100.0, 50.0)
    assert fired, "releasing the marker did not fire adapter.changed"


def _drag_black_to(page, x):
    """Drag the histogram's black marker to widget x and release it,
    returning the preview."""
    levels = page.power_widget.adapter_for("black_point")
    preview = levels.preview
    preview.set_size_request(preview.WIDTH, preview.HEIGHT)
    preview.allocate(preview.WIDTH, preview.HEIGHT, -1, None)
    ctrl = Gtk.EventControllerMotion.new()
    click = Gtk.GestureClick.new()
    preview.add_controller(ctrl)
    preview.add_controller(click)
    preview._dragging = "black"
    preview._on_motion(ctrl, float(x), 50.0)
    preview._on_released(click, 1, float(x), 50.0)
    return preview


def test_raster_page_drag_commits_black_point(
    editor, laser_machine, ui_context
):
    """Dragging the histogram bar must commit to the step and survive
    later resyncs triggered by other settings."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    auto = page.power_widget.adapter_for("auto_levels")
    assert auto is not None
    auto.set_value(False)
    auto.changed.send(auto)
    page.power_widget.cancel_pending()

    preview = _drag_black_to(page, 100)
    assert preview.black_point == 128

    page.power_widget.flush_pending()
    assert step.black_point == 128

    # Changing an unrelated setting resyncs rows from the model; the
    # committed black point must not be overwritten by stale data.
    offset = page.engrave_widget.adapter_for("bidir_x_offset_mm")
    assert offset is not None
    offset.set_value(0.5)
    offset.changed.send(offset)
    page.engrave_widget.flush_pending()

    assert step.bidir_x_offset_mm == 0.5
    assert preview.black_point == 128
