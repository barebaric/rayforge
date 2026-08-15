# flake8: noqa: E402
"""UI tests for the laser step settings pages."""

from typing import Any, cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw
from laser_essentials.steps import EngraveStep
from laser_essentials.widgets.contour_page import ContourStepSettingsPage
from laser_essentials.widgets.material_test_grid_page import (
    MaterialTestGridSettingsPage,
)
from laser_essentials.widgets.raster_page import RasterSettingsPage
from laser_essentials.widgets.raster_power_widget import RasterPowerWidget

from rayforge.core.step_registry import step_registry
from rayforge.ui_gtk.doceditor.step_settings.dialog import StepSettingsDialog
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow, SpeedSpinRow


def _row(page, key):
    for widget, _ in page._varset_widgets:
        row = widget.row_for(key)
        if row is not None:
            return row
    raise AssertionError(f"row {key} not found in page")


def _contour_step(ui_context) -> Any:
    step_cls = step_registry.get("ContourStep")
    assert step_cls is not None
    return step_cls.create(ui_context)


@pytest.mark.ui
def test_contour_page_composes_step_and_laser_rows(
    editor, laser_machine, ui_context
):
    step = _contour_step(ui_context)
    page = ContourStepSettingsPage(editor, step)
    laser_page = page.laser_page()

    assert isinstance(page, StepSettingsPage)
    assert isinstance(page, Adw.PreferencesPage)
    assert isinstance(laser_page, StepSettingsPage)

    for key in ("cut_side", "offset_mm", "cut_order", "overcut"):
        _row(page, key)
    for key in (
        "selected_head_uid",
        "power",
        "cut_speed",
        "travel_speed",
        "air_assist",
    ):
        _row(laser_page, key)


@pytest.mark.ui
def test_speed_rows_show_integer_digits(editor, laser_machine, ui_context):
    """Speed rows must not show fractional digits (500, not 500.00)."""
    step = _contour_step(ui_context)
    page = ContourStepSettingsPage(editor, step)
    laser_page = page.laser_page()

    for key in ("cut_speed", "travel_speed"):
        row = cast(SpeedSpinRow, _row(laser_page, key))
        assert row.get_digits() == 0, f"{key} shows fractional digits"


@pytest.mark.ui
def test_path_offset_insensitive_on_centerline(
    editor, laser_machine, ui_context
):
    step = _contour_step(ui_context)
    page = ContourStepSettingsPage(editor, step)
    offset = _row(page, "offset_mm")

    assert step.cut_side == "CENTERLINE"
    assert offset.get_sensitive() is False

    step.cut_side = "OUTSIDE"
    step.updated.send(step)
    assert offset.get_sensitive() is True


@pytest.mark.ui
def test_threshold_visible_only_when_rescanning(
    editor, laser_machine, ui_context
):
    step = _contour_step(ui_context)
    page = ContourStepSettingsPage(editor, step)
    threshold = _row(page, "threshold")

    step.override_threshold = False
    step.updated.send(step)
    assert threshold.get_visible() is False

    step.override_threshold = True
    step.updated.send(step)
    assert threshold.get_visible() is True


@pytest.mark.ui
def test_head_change_does_not_touch_offset(editor, laser_machine, ui_context):
    step = _contour_step(ui_context)
    page = ContourStepSettingsPage(editor, step)
    laser_page = page.laser_page()
    offset_before = step.offset_mm

    target = laser_machine.heads[1]
    laser_page._on_head_changed(target.uid)
    assert step.selected_head_uid == target.uid
    assert step.offset_mm == offset_before


@pytest.mark.ui
def test_offset_row_uses_user_units(editor, laser_machine, ui_context):
    ui_context.config.unit_preferences["length"] = "in"
    step = _contour_step(ui_context)
    page = ContourStepSettingsPage(editor, step)
    offset = _row(page, "offset_mm")

    step.offset_mm = 25.4
    step.updated.send(step)

    offset_row = cast(LengthSpinRow, offset)
    assert offset_row is not None
    assert offset_row.get_value_in_base_units() == pytest.approx(25.4)
    assert offset_row.get_value() == pytest.approx(1.0, abs=1e-2)


@pytest.mark.ui
def test_material_test_page_builds(editor, laser_machine, ui_context):
    step_cls = step_registry.get("MaterialTestStep")
    assert step_cls is not None
    page = MaterialTestGridSettingsPage(editor, step_cls.create(ui_context))
    assert isinstance(page, StepSettingsPage)


@pytest.mark.ui
def test_material_test_tuple_rows_round_trip(
    editor, laser_machine, ui_context
):
    """Tuple rows (ranges, grid dimensions) read and write tuples."""
    from laser_essentials.steps.material_test import MaterialTestStep

    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)

    for key in ("power_range", "speed_range", "passes_range", "offset_range"):
        row = _row(page, key)
        assert row is not None, f"{key} row missing"
    _row(page, "grid_dimensions")

    # Programmatic push lands in the widgets.
    page.params_widget.set_values(
        {
            "power_range": (20.0, 80.0),
            "speed_range": (200.0, 900.0),
        }
    )
    values = page.params_widget.get_values()
    assert values["power_range"] == (20.0, 80.0)
    assert values["speed_range"] == (200.0, 900.0)


@pytest.mark.ui
def test_material_test_grid_mode_visibility(editor, laser_machine, ui_context):
    """Parameters rows follow the grid-mode visible_when predicates."""
    from laser_essentials.steps.material_test import MaterialTestStep

    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)

    power = _row(page, "power_range")
    speed = _row(page, "speed_range")
    passes = _row(page, "passes_range")
    offset = _row(page, "offset_range")
    fixed_speed = _row(page, "fixed_speed")
    fixed_power = _row(page, "fixed_power")

    assert power.get_visible() is True  # Power vs Speed default
    assert speed.get_visible() is True
    assert passes.get_visible() is False
    assert offset.get_visible() is False

    step.grid_mode = "Speed vs Offset"
    step.updated.send(step)
    assert power.get_visible() is False
    assert speed.get_visible() is True
    assert passes.get_visible() is False
    assert offset.get_visible() is True
    assert fixed_speed.get_visible() is False
    assert fixed_power.get_visible() is True


@pytest.mark.ui
def test_material_test_labels_sensitivity(editor, laser_machine, ui_context):
    """Label rows are insensitive while labels are disabled."""
    from laser_essentials.steps.material_test import MaterialTestStep

    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)

    label_power = _row(page, "label_power_percent")
    label_speed = _row(page, "label_speed")
    assert label_power.get_sensitive() is True

    step.include_labels = False
    step.updated.send(step)
    assert label_power.get_sensitive() is False
    assert label_speed.get_sensitive() is False


@pytest.mark.ui
def test_material_test_preset_applies_values(
    editor, laser_machine, ui_context
):
    """Selecting a preset commits speed/power ranges and test type."""
    from laser_essentials.steps.material_test import MaterialTestStep

    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)
    page.preset_row.set_selected(2)  # Diode Cut

    assert step.test_type == "Cut"
    assert step.power_range == (50.0, 100.0)
    assert step.speed_range[0] >= 100.0


@pytest.mark.ui
def test_material_test_speed_vs_offset_defaults(
    editor, laser_machine, ui_context
):
    """Switching to Speed vs Offset applies engrave defaults."""
    from laser_essentials.steps.material_test import MaterialTestStep

    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)

    mode_adapter = page.grid_widget.adapter_for("grid_mode")
    assert mode_adapter is not None
    mode_adapter.set_value("Speed vs Offset")
    page._sync_grid_context()
    page._apply_speed_vs_offset_defaults()

    assert step.test_type == "Engrave"
    assert step.line_interval_mm == 0.5


@pytest.mark.ui
def test_raster_page_builds(editor, laser_machine, ui_context):
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    page = RasterSettingsPage(editor, step_cls.create(ui_context))
    assert isinstance(page, StepSettingsPage)


@pytest.mark.ui
def test_raster_page_mode_visibility(editor, laser_machine, ui_context):
    """Raster rows react to the depth-mode visible_when predicates."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    threshold = _row(page, "threshold")
    sample_interval = _row(page, "sample_interval_mm")

    step.depth_mode = "CONSTANT_POWER"
    step.updated.send(step)
    assert threshold.get_visible() is True
    assert sample_interval.get_visible() is False

    step.depth_mode = "POWER_MODULATION"
    step.updated.send(step)
    assert threshold.get_visible() is False
    assert sample_interval.get_visible() is True


@pytest.mark.ui
def test_raster_page_power_section_visible_by_default(
    editor, laser_machine, ui_context
):
    """The Power section rows must be visible for the default mode.

    Their visible_when predicates key off depth_mode, which lives in
    the Engrave section; the page must feed it in as context.
    """
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    assert step.depth_mode == "POWER_MODULATION"
    for key in (
        "auto_levels",
        "black_point",
        "min_power_level",
        "max_power_level",
        "num_power_levels",
    ):
        row = _row(page, key)
        assert row.get_visible() is True, f"{key} hidden in Power section"


@pytest.mark.ui
def test_raster_page_power_section_hidden_outside_grayscale(
    editor, laser_machine, ui_context
):
    """Switching to a non-grayscale mode hides the Power section rows."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    mode_adapter = page.engrave_widget.adapter_for("depth_mode")
    assert mode_adapter is not None
    mode_adapter.set_value("DITHER")
    page._sync_power_context()

    for key in ("auto_levels", "black_point"):
        row = _row(page, key)
        assert row.get_visible() is False, f"{key} visible in DITHER mode"


@pytest.mark.ui
def test_raster_power_widget_flips_labels_on_invert(
    editor, laser_machine, ui_context
):
    """The Power widget re-titles min/max rows when invert changes."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    assert isinstance(page.power_widget, RasterPowerWidget)

    min_row = _row(page, "min_power_level")
    max_row = _row(page, "max_power_level")
    assert min_row.get_title() == "Min Power (White)"
    assert max_row.get_title() == "Max Power (Black)"

    invert_adapter = page.engrave_widget.adapter_for("invert")
    assert invert_adapter is not None
    invert_adapter.set_value(True)
    page._sync_power_context()

    assert min_row.get_title() == "Min Power (Black)"
    assert max_row.get_title() == "Max Power (White)"


@pytest.mark.ui
def test_raster_page_commits_dither_algorithm(
    editor, laser_machine, ui_context
):
    """dither_algorithm commits the enum through the step setter."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    page.set_step_property("dither_algorithm", "BAYER4")
    assert step.dither_algorithm is not None
    assert step.dither_algorithm.name == "BAYER4"


@pytest.mark.ui
def test_raster_page_power_range_keeps_min_under_max(
    editor, laser_machine, ui_context
):
    """Committing a power range keeps min <= max in one transaction."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    min_adapter = page.power_widget.adapter_for("min_power_level")
    max_adapter = page.power_widget.adapter_for("max_power_level")
    assert min_adapter is not None
    assert max_adapter is not None
    max_adapter.set_value(0.8)
    min_adapter.set_value(0.9)
    page._on_varset_data_changed(page.power_widget, "min_power_level")

    # Max follows min up so the range stays valid.
    assert step.min_power_level == pytest.approx(0.9)
    assert step.max_power_level == pytest.approx(0.9)


@pytest.mark.ui
def test_raster_page_shows_head_derived_interval_defaults(
    editor, laser_machine, ui_context
):
    """Auto (None) interval rows display the laser spot-size default
    and use 3-digit precision, like the old dialog."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    assert step.line_interval_mm is None
    page = RasterSettingsPage(editor, step)

    head = laser_machine.heads[0]
    spot_x, spot_y = head.spot_size_mm

    line = cast(LengthSpinRow, _row(page, "line_interval_mm"))
    sample = cast(LengthSpinRow, _row(page, "sample_interval_mm"))
    dot = cast(LengthSpinRow, _row(page, "dot_width_correction_mm"))

    # The step stays None (auto) until the user edits the row.
    assert step.line_interval_mm is None
    assert step.sample_interval_mm is None
    assert step.dot_width_correction_mm is None

    assert line.get_value_in_base_units() == pytest.approx(spot_y)
    assert sample.get_value_in_base_units() == pytest.approx(spot_x / 2.0)
    assert dot.get_value_in_base_units() == pytest.approx(spot_x / 2.0)

    assert line.get_digits() == 3
    assert sample.get_digits() == 3
    assert dot.get_digits() == 3


@pytest.mark.ui
def test_raster_page_keeps_user_interval_values(
    editor, laser_machine, ui_context
):
    """Explicit interval values are not replaced by head defaults."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.line_interval_mm = 1.5
    page = RasterSettingsPage(editor, step)

    line = cast(LengthSpinRow, _row(page, "line_interval_mm"))
    assert line.get_value_in_base_units() == pytest.approx(1.5)


@pytest.mark.ui
def test_dialog_uses_contour_page(editor, laser_machine, ui_context):
    dialog = StepSettingsDialog(editor, _contour_step(ui_context))
    assert type(dialog.general_view).__name__ == "ContourStepSettingsPage"
    assert [title for title, _, _ in dialog._extra_pages] == ["Laser"]
    assert len(dialog._extra_buttons) == 1
    dialog.close()


@pytest.mark.ui
def test_dialog_initial_laser_page(editor, laser_machine, ui_context):
    dialog = StepSettingsDialog(editor, _contour_step(ui_context))
    dialog.set_initial_page("laser")
    assert dialog._extra_buttons[0].get_active() is True
    assert dialog.btn_step_settings.get_active() is False
    dialog.close()
