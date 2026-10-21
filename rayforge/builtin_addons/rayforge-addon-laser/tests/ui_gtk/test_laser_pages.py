# flake8: noqa: E402
"""UI tests for the laser step settings pages."""

from typing import Any, cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk
from laser_essentials.steps import (
    EngraveStep,
    MaterialTestStep,
    OffsetFillStep,
)
from laser_essentials.widgets import ASSEMBLER_WIDGETS
from laser_essentials.widgets.contour_page import ContourStepSettingsPage
from laser_essentials.widgets.material_test_grid_page import (
    MaterialTestGridSettingsPage,
)
from laser_essentials.widgets.offset_fill_page import (
    OffsetFillStepSettingsPage,
)
from laser_essentials.widgets.raster_page import RasterSettingsPage
from laser_essentials.widgets.raster_power_widget import RasterPowerWidget

from rayforge.core.step_registry import step_registry
from rayforge.image.dither import DitherAlgorithm
from rayforge.pipeline.stage.assembler_helpers import DepthMode
from rayforge.ui_gtk.doceditor.step_settings.dialog import StepSettingsDialog
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow, SpeedSpinRow
from rayforge.ui_gtk.varset.adapter.combo import ComboAdapter


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
        "power_mode",
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
def test_material_test_laser_page_hides_travel_speed_without_support(
    editor, laser_machine, ui_context
):
    """GRBL can't emit travel speed, so the material-test laser page
    hides the travel speed row."""
    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)
    laser_page = page.laser_page()

    assert not laser_machine.supports_travel_speed()
    row = _row(laser_page, "travel_speed")
    assert row.get_visible() is False


@pytest.mark.ui
def test_material_test_laser_page_shows_travel_speed_with_support(
    editor, laser_machine, ui_context
):
    """A machine that can emit travel speed shows the material-test
    travel speed row."""
    laser_machine.set_dialect_uid("smoothieware")
    step = MaterialTestStep()
    page = MaterialTestGridSettingsPage(editor, step)
    laser_page = page.laser_page()

    assert laser_machine.supports_travel_speed()
    row = _row(laser_page, "travel_speed")
    assert row.get_visible() is True


@pytest.mark.ui
def test_material_test_page_leaves_the_grid_to_the_test_settings(
    editor, laser_machine, ui_context
):
    """The grid is set in the Test Settings panel, so the step page
    has no grid, label or range rows."""
    page = MaterialTestGridSettingsPage(editor, MaterialTestStep())
    for key in ("grid_mode", "grid_dimensions", "speed_range"):
        with pytest.raises(AssertionError):
            _row(page, key)


@pytest.mark.ui
def test_raster_page_builds(editor, laser_machine, ui_context):
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    page = RasterSettingsPage(editor, step_cls.create(ui_context))
    assert isinstance(page, StepSettingsPage)


@pytest.mark.ui
def test_speed_above_machine_max_shows_warning(
    editor, laser_machine, ui_context
):
    """A loaded speed above the machine's max is shown, not clamped.

    The step's real value is displayed (matching the step box) and the
    row is flagged with a warning explaining it exceeds the machine.
    """
    laser_machine.max_cut_speed = 1000
    ui_context.config.set_machine(laser_machine)

    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.cut_speed = 24000
    data = step.to_dict()
    data["max_cut_speed"] = 24000
    loaded = step_cls.from_dict(data)

    page = RasterSettingsPage(editor, loaded)
    laser_page = page.laser_page()
    row = cast(SpeedSpinRow, _row(laser_page, "cut_speed"))
    # The loaded value is displayed above the machine's max ...
    assert row.get_spin_button().get_text() == "24000"
    # ... while a read-back clamps to the machine's ceiling (an edit
    # would commit the clamped value, not the stale loaded one).
    assert row.get_value_in_base_units() == pytest.approx(1000.0)

    icon = laser_page._speed_warning_icons.get(row)
    assert icon is not None
    assert icon.get_visible() is True
    assert icon.get_tooltip_text() != ""


@pytest.mark.ui
def test_speed_within_machine_max_has_no_warning(
    editor, laser_machine, ui_context
):
    """A speed within the machine's max shows no warning icon."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.cut_speed = 1000  # within laser_machine's default max of 5000
    page = RasterSettingsPage(editor, step)
    laser_page = page.laser_page()
    row = cast(SpeedSpinRow, _row(laser_page, "cut_speed"))

    icon = laser_page._speed_warning_icons.get(row)
    assert icon is None or icon.get_visible() is False


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
def test_raster_page_dither_option_visibility(
    editor, laser_machine, ui_context
):
    """Serpentine follows the error-diffusion algorithms, the halftone
    rows only show for Halftone."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.depth_mode = "DITHER"
    page = RasterSettingsPage(editor, step)

    serpentine = _row(page, "dither_serpentine")
    cell = _row(page, "halftone_cell_mm")
    angle = _row(page, "halftone_angle")

    step.set_dither_algorithm("STUCKI")
    assert serpentine.get_visible() is True
    assert cell.get_visible() is False
    assert angle.get_visible() is False

    step.set_dither_algorithm("HALFTONE")
    assert serpentine.get_visible() is False
    assert cell.get_visible() is True
    assert angle.get_visible() is True

    step.depth_mode = "POWER_MODULATION"
    step.updated.send(step)
    assert serpentine.get_visible() is False
    assert cell.get_visible() is False


@pytest.mark.ui
def test_raster_page_shows_the_steps_dither_algorithm(
    editor, laser_machine, ui_context
):
    """The step stores the algorithm as an enum; the combo row must
    still select it, on open and after an external change."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.depth_mode = "DITHER"
    step.dither_algorithm = DitherAlgorithm.BAYER4
    page = RasterSettingsPage(editor, step)

    adapter = page.engrave_widget.adapter_for("dither_algorithm")
    assert adapter is not None
    assert adapter.get_value() == "BAYER4"

    step.set_dither_algorithm(DitherAlgorithm.STUCKI)
    assert adapter.get_value() == "STUCKI"


@pytest.mark.ui
def test_raster_page_dither_algorithm_lists_new_methods(
    editor, laser_machine, ui_context
):
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.depth_mode = "DITHER"
    page = RasterSettingsPage(editor, step)

    adapter = page.engrave_widget.adapter_for("dither_algorithm")
    assert isinstance(adapter, ComboAdapter)
    row = cast(Adw.ComboRow, _row(page, "dither_algorithm"))
    model = row.get_model()
    assert model is not None
    labels = [
        cast(Gtk.StringObject, model.get_item(i)).get_string()
        for i in range(model.get_n_items())
    ]
    for algo in DitherAlgorithm:
        assert algo.display_name in labels


def _depth_mode_adapter(page: RasterSettingsPage) -> ComboAdapter:
    adapter = page.engrave_widget.adapter_for("depth_mode")
    assert isinstance(adapter, ComboAdapter)
    return adapter


@pytest.mark.ui
def test_raster_page_multi_depth_available_by_default(
    editor, laser_machine, ui_context
):
    """G-code drivers keep the multi-depth entry selectable."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.depth_mode = "MULTI_PASS"
    page = RasterSettingsPage(editor, step)

    assert _depth_mode_adapter(page)._disabled_displays == set()

    row = _row(page, "depth_mode")
    icon = page._speed_warning_icons.get(row)
    assert icon is None or icon.get_visible() is False


@pytest.mark.ui
def test_raster_page_multi_depth_disabled_on_ruida(
    editor, laser_machine, ui_context
):
    """Ruida machines grey out multi-depth and flag existing usage."""
    laser_machine.auto_connect = False
    laser_machine.driver_name = "RuidaRPAAdapter"
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    step.depth_mode = "MULTI_PASS"
    page = RasterSettingsPage(editor, step)

    disabled = _depth_mode_adapter(page)._disabled_displays
    assert DepthMode.MULTI_PASS.display_name in disabled

    row = _row(page, "depth_mode")
    icon = page._speed_warning_icons.get(row)
    assert icon is not None
    assert icon.get_visible() is True
    tooltip = icon.get_tooltip_text() or ""
    assert "Multiple Depths" in tooltip


@pytest.mark.ui
def test_raster_page_multi_depth_no_warning_when_not_selected(
    editor, laser_machine, ui_context
):
    """A Ruida machine only warns when the step uses multi-depth."""
    laser_machine.auto_connect = False
    laser_machine.driver_name = "RuidaRPAAdapter"
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    assert step.depth_mode == "POWER_MODULATION"
    page = RasterSettingsPage(editor, step)

    disabled = _depth_mode_adapter(page)._disabled_displays
    assert DepthMode.MULTI_PASS.display_name in disabled

    row = _row(page, "depth_mode")
    icon = page._speed_warning_icons.get(row)
    assert icon is None or icon.get_visible() is False


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
    """Constant-power mode hides the Power section's level rows.

    With no applicable rows left, the empty section hides entirely.
    """
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    mode_adapter = page.engrave_widget.adapter_for("depth_mode")
    assert mode_adapter is not None
    mode_adapter.set_value("CONSTANT_POWER")
    page._sync_power_context()

    for key in ("auto_levels", "black_point"):
        row = _row(page, key)
        assert row.get_visible() is False, f"{key} visible in CONSTANT_POWER"
    assert page.power_widget.get_visible() is False


@pytest.mark.ui
def test_raster_page_power_section_levels_visible_in_dither(
    editor, laser_machine, ui_context
):
    """Dither mode shows the brightness-range rows so users can
    adapt the dither input via the histogram, but hides the
    variable-power rows."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    mode_adapter = page.engrave_widget.adapter_for("depth_mode")
    assert mode_adapter is not None
    mode_adapter.set_value("DITHER")
    page._sync_power_context()

    assert page.power_widget.get_visible() is True
    for key in ("auto_levels", "black_point", "white_point"):
        row = _row(page, key)
        assert row.get_visible() is True, f"{key} hidden in DITHER mode"
    for key in ("min_power_level", "max_power_level", "num_power_levels"):
        row = _row(page, key)
        assert row.get_visible() is False, f"{key} visible in DITHER mode"


@pytest.mark.ui
def test_raster_power_widget_titles_rows_by_area(
    editor, laser_machine, ui_context
):
    """The min/max rows are titled after the areas they drive, and
    the Engrave section no longer offers Invert."""
    step_cls = step_registry.get("EngraveStep")
    assert step_cls is not None
    step = cast(EngraveStep, step_cls.create(ui_context))
    page = RasterSettingsPage(editor, step)

    assert isinstance(page.power_widget, RasterPowerWidget)
    assert _row(page, "min_power_level").get_title() == "Min Power (White)"
    assert _row(page, "max_power_level").get_title() == "Max Power (Black)"
    assert page.engrave_widget.adapter_for("invert") is None


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


@pytest.mark.ui
def test_offset_fill_page_shows_fill_rows(editor, laser_machine, ui_context):
    assert ASSEMBLER_WIDGETS["offset_fill"] is OffsetFillStepSettingsPage
    step = OffsetFillStep.create(ui_context)
    page = OffsetFillStepSettingsPage(editor, step)
    for key in ("line_interval_mm", "offset_mm", "fill_direction"):
        _row(page, key)
    assert isinstance(_row(page, "line_interval_mm"), LengthSpinRow)
