"""
Material Test Grid Settings Widget

Provides UI for configuring material test array parameters.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, GLib, GObject, Gtk

from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.shared.pref_rows import SpinRow
from rayforge.ui_gtk.shared.slider import create_slider_row

from ..material_test_helpers import GridMode
from .rows import LaserStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


logger = logging.getLogger(__name__)

PRESET_KEYS = [
    "Diode Engrave",
    "Diode Cut",
    "CO2 Engrave",
    "CO2 Cut",
]

PRESETS = {
    "Diode Engrave": {
        "test_type": "Engrave",
        "speed_range": (1000.0, 10000.0),
        "power_range": (10.0, 100.0),
    },
    "Diode Cut": {
        "test_type": "Cut",
        "speed_range": (100.0, 5000.0),
        "power_range": (50.0, 100.0),
    },
    "CO2 Engrave": {
        "test_type": "Engrave",
        "speed_range": (3000.0, 20000.0),
        "power_range": (10.0, 50.0),
    },
    "CO2 Cut": {
        "test_type": "Cut",
        "speed_range": (1000.0, 20000.0),
        "power_range": (30.0, 100.0),
    },
}


class MaterialTestGridSettingsPage(LaserStepSettingsPage):
    """Material Test Grid settings widget."""

    include_process = False

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
    ):
        super().__init__(editor, step)
        preset_group = self.add_section(
            _("Preset"),
            description=_("Load common test configurations."),
        )
        grid_group = self.add_section(
            _("Grid"),
            description=_("Test cell dimensions, shape, and spacing."),
        )
        labels_group = self.add_section(
            _("Labels"),
            description=_("Speed/power annotations on the grid."),
        )
        self._param_group = self.add_section(
            _("Parameters"),
            description=_("Define the parameter ranges for the test grid."),
        )

        self._build_preset_selector(preset_group)
        self._build_test_type_selector(grid_group)
        self._build_grid_mode_selector(grid_group)
        self._build_grid_dimensions(grid_group)
        self._build_shape_size(grid_group)
        self._build_spacing(grid_group)
        self._build_label_settings(labels_group)
        self._build_power_and_speed_group(self._param_group)

    def _add(self, group, widget):
        self._rows.append(widget)
        group.add(widget)

    def _build_preset_selector(self, group):
        """Builds the preset dropdown."""
        _PRESET_LABELS = {
            "Diode Engrave": _("Diode Engrave"),
            "Diode Cut": _("Diode Cut"),
            "CO2 Engrave": _("CO2 Engrave"),
            "CO2 Cut": _("CO2 Cut"),
        }
        string_list = Gtk.StringList()
        string_list.append(_("Select"))
        for key in PRESET_KEYS:
            string_list.append(_PRESET_LABELS[key])

        self.preset_row = Adw.ComboRow(
            title=_("Presets"),
            subtitle=_("Load common test configurations"),
            model=string_list,
        )
        self.preset_row.set_selected(0)
        self._add(group, self.preset_row)
        self.preset_row.connect("notify::selected", self._on_preset_changed)

    def _build_test_type_selector(self, group):
        """Builds the test type dropdown (Cut/Engrave)."""
        from ..material_test_helpers import MaterialTestGridType

        self._test_type_values = [m.value for m in MaterialTestGridType]
        test_type_labels = [m.label() for m in MaterialTestGridType]
        string_list = Gtk.StringList.new(test_type_labels)
        self.test_type_row = Adw.ComboRow(
            title=_("Test Type"),
            subtitle=_("Cut: outlines; Engrave: fills with raster lines"),
            model=string_list,
        )
        current_text = self.step.test_type
        for i, val in enumerate(self._test_type_values):
            if val == current_text:
                self.test_type_row.set_selected(i)
                break
        self._add(group, self.test_type_row)
        self.test_type_row.connect(
            "notify::selected", self._on_test_type_changed
        )

    def _build_grid_mode_selector(self, group):
        """Builds the grid mode dropdown."""
        self._grid_mode_values = [m.value for m in GridMode]
        grid_mode_labels = [m.label() for m in GridMode]
        string_list = Gtk.StringList.new(grid_mode_labels)
        self.grid_mode_row = Adw.ComboRow(
            title=_("Grid Mode"),
            subtitle=_("Choose which parameters to vary on axes"),
            model=string_list,
        )
        current_mode = self.step.grid_mode
        for i, val in enumerate(self._grid_mode_values):
            if val == current_mode:
                self.grid_mode_row.set_selected(i)
                break
        self._add(group, self.grid_mode_row)
        self.grid_mode_row.connect(
            "notify::selected", self._on_grid_mode_changed
        )

    def _build_power_and_speed_group(self, group):
        """Builds the group for power and speed settings."""
        machine_max_speed = self.step.max_cut_speed

        # Fixed Speed (used in Power vs Passes mode)
        self.fixed_speed_row = SpinRow(
            _("Fixed Speed"),
            _("Constant speed for all cells (mm/min)"),
            lower=1.0,
            upper=machine_max_speed,
            step_increment=10.0,
            digits=0,
            value=min(self.step.fixed_speed, machine_max_speed),
        )
        self._add(group, self.fixed_speed_row)
        self.fixed_speed_row.value_changed.connect(
            lambda r: self._debounce(self._on_fixed_speed_changed, r),
        )

        # Fixed Power (used in Speed vs Passes mode)
        fixed_power_adj = Gtk.Adjustment(
            lower=1,
            upper=100,
            step_increment=0.1,
            value=self.step.fixed_power,
        )
        self.fixed_power_row, self.fixed_power_scale = create_slider_row(
            title=_("Fixed Power (%)"),
            adjustment=fixed_power_adj,
            subtitle=_("Constant power for all cells"),
            digits=1,
            on_value_changed=lambda s: self._debounce(
                self._on_fixed_power_changed, s
            ),
        )
        self._add(group, self.fixed_power_row)

        # Power Range (used in Power vs Speed and Power vs Passes modes)
        min_power, max_power = self.step.power_range
        self.min_power_adj = Gtk.Adjustment(
            lower=1, upper=100, step_increment=0.1, value=min_power
        )
        min_power_row, self.min_power_scale = create_slider_row(
            title=_("Minimum Power (%)"),
            adjustment=self.min_power_adj,
            subtitle=_("For first column"),
            digits=1,
        )
        self.min_power_row = min_power_row
        self._add(group, self.min_power_row)

        self.max_power_adj = Gtk.Adjustment(
            lower=1, upper=100, step_increment=0.1, value=max_power
        )
        max_power_row, self.max_power_scale = create_slider_row(
            title=_("Maximum Power (%)"),
            adjustment=self.max_power_adj,
            subtitle=_("For last column"),
            digits=1,
        )
        self.max_power_row = max_power_row
        self._add(group, self.max_power_row)

        self.min_power_handler_id = self.min_power_scale.connect(
            "value-changed", self._on_min_power_scale_changed
        )
        self.max_power_handler_id = self.max_power_scale.connect(
            "value-changed", self._on_max_power_scale_changed
        )

        # Speed Range (used in Power vs Speed and Speed vs Passes modes)
        min_speed, max_speed = self.step.speed_range
        machine_max_speed = self.step.max_cut_speed
        min_speed = min(min_speed, machine_max_speed)
        max_speed = min(max_speed, machine_max_speed)
        self.speed_min_row = SpinRow(
            _("Minimum Speed"),
            _("Starting speed (mm/min)"),
            lower=1.0,
            upper=machine_max_speed,
            step_increment=10.0,
            digits=0,
            value=min_speed,
        )
        self._add(group, self.speed_min_row)

        self.speed_max_row = SpinRow(
            _("Maximum Speed"),
            _("Ending speed (mm/min)"),
            lower=1.0,
            upper=machine_max_speed,
            step_increment=10.0,
            digits=0,
            value=max_speed,
        )
        self._add(group, self.speed_max_row)

        self.speed_min_row.value_changed.connect(
            lambda r: self._debounce(self._on_speed_min_changed, r)
        )
        self.speed_max_row.value_changed.connect(
            lambda r: self._debounce(self._on_speed_max_changed, r)
        )

        # Passes Range (used in Power vs Passes and Speed vs Passes modes)
        min_passes, max_passes = self.step.passes_range
        self.passes_min_row = SpinRow(
            _("Minimum Passes"),
            _("Starting number of passes"),
            lower=1,
            upper=50,
            digits=0,
            value=min_passes,
        )
        self._add(group, self.passes_min_row)

        self.passes_max_row = SpinRow(
            _("Maximum Passes"),
            _("Ending number of passes"),
            lower=1,
            upper=50,
            digits=0,
            value=max_passes,
        )
        self._add(group, self.passes_max_row)

        self.passes_min_row.value_changed.connect(
            lambda r: self._debounce(self._on_passes_min_changed, r),
        )
        self.passes_max_row.value_changed.connect(
            lambda r: self._debounce(self._on_passes_max_changed, r),
        )

        # Offset Range (used in Speed vs Offset mode)
        min_offset, max_offset = self.step.offset_range
        self.offset_min_row = SpinRow(
            _("Minimum Offset"),
            _("Bidir scan X-offset for first row (mm)"),
            lower=-10.0,
            upper=10.0,
            step_increment=0.05,
            digits=2,
            value=min_offset,
        )
        self._add(group, self.offset_min_row)

        self.offset_max_row = SpinRow(
            _("Maximum Offset"),
            _("Bidir scan X-offset for last row (mm)"),
            lower=-10.0,
            upper=10.0,
            step_increment=0.05,
            digits=2,
            value=max_offset,
        )
        self._add(group, self.offset_max_row)

        self.offset_min_row.value_changed.connect(
            lambda r: self._debounce(self._on_offset_min_changed, r),
        )
        self.offset_max_row.value_changed.connect(
            lambda r: self._debounce(self._on_offset_max_changed, r),
        )

        # Label settings
        power_adj = Gtk.Adjustment(
            lower=1,
            upper=100,
            step_increment=0.1,
            value=self.step.label_power_percent,
        )
        self.label_power_row, power_scale = create_slider_row(
            title=_("Label Engrave Power (%)"),
            adjustment=power_adj,
            digits=1,
            on_value_changed=lambda s: self._debounce(
                self._on_label_power_changed, s
            ),
        )
        self._add(group, self.label_power_row)

        self.label_speed_row = SpinRow(
            _("Label Engrave Speed"),
            _("Speed for engraving labels (mm/min)"),
            lower=1.0,
            upper=machine_max_speed,
            step_increment=10.0,
            digits=0,
            value=min(self.step.label_speed, machine_max_speed),
        )
        self._add(group, self.label_speed_row)
        self.label_speed_row.value_changed.connect(
            lambda r: self._debounce(self._on_label_speed_changed, r),
        )

        self._on_labels_toggled(
            self.include_labels_switch, self.step.include_labels
        )

        self._update_control_visibility()
        self._update_dimension_labels()

    def _build_grid_dimensions(self, group):
        """Builds grid dimension controls."""
        cols, rows = self.step.grid_dimensions

        self.cols_row = SpinRow(
            _("Columns (Power Steps)"),
            _("Number of power variations"),
            lower=2,
            upper=20,
            digits=0,
            value=cols,
        )
        self._add(group, self.cols_row)

        self.rows_row = SpinRow(
            _("Rows (Speed Steps)"),
            _("Number of speed variations"),
            lower=2,
            upper=20,
            digits=0,
            value=rows,
        )
        self._add(group, self.rows_row)

        self.cols_row.value_changed.connect(
            lambda r: self._debounce(self._on_grid_cols_changed, r)
        )
        self.rows_row.value_changed.connect(
            lambda r: self._debounce(self._on_grid_rows_changed, r)
        )

    def _build_shape_size(self, group):
        """Builds shape size control."""
        self.shape_size_row = SpinRow(
            _("Shape Size"),
            _("Size of each test square (mm)"),
            lower=1,
            upper=100,
            digits=1,
            value=self.step.shape_size,
        )
        self._add(group, self.shape_size_row)
        self.shape_size_row.value_changed.connect(
            lambda r: self._debounce(self._on_shape_size_changed, r)
        )

    def _build_spacing(self, group):
        """Builds spacing control."""
        self.spacing_row = SpinRow(
            _("Spacing"),
            _("Gap between test squares (mm)"),
            upper=50,
            step_increment=0.5,
            digits=1,
            value=self.step.spacing,
        )
        self._add(group, self.spacing_row)
        self.spacing_row.value_changed.connect(
            lambda r: self._debounce(self._on_spacing_changed, r)
        )

        head = self.get_selected_head()
        laser = head if isinstance(head, LaserHead) else None
        default_line_interval_mm = laser.spot_size_mm[1] if laser else 0.1
        self.line_interval_row = SpinRow(
            _("Line Interval"),
            _(
                "Distance between scan lines in machine units "
                "(for Engrave mode). Leave at 0 to use laser spot size."
            ),
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            digits=2,
            value=(
                self.step.line_interval_mm
                if self.step.line_interval_mm is not None
                else default_line_interval_mm
            ),
        )
        self._add(group, self.line_interval_row)
        self.line_interval_row.value_changed.connect(
            lambda r: self._debounce(self._on_line_interval_changed, r),
        )

    def _build_label_settings(self, group):
        """Builds controls for label appearance and behavior."""
        self.include_labels_switch = Gtk.Switch(
            valign=Gtk.Align.CENTER, active=self.step.include_labels
        )
        labels_row = Adw.ActionRow(
            title=_("Include Labels"),
            subtitle=_("Add speed/power annotations to the grid"),
        )
        labels_row.add_suffix(self.include_labels_switch)
        labels_row.set_activatable_widget(self.include_labels_switch)
        self._add(group, labels_row)

        self.include_labels_switch.connect(
            "state-set", self._on_labels_toggled
        )

    # Signal handlers
    def _on_preset_changed(self, row: Adw.ComboRow, _pspec):
        """Loads preset values."""
        selected_idx = row.get_selected()
        if selected_idx == Gtk.INVALID_LIST_POSITION or selected_idx == 0:
            return
        preset_key = PRESET_KEYS[selected_idx - 1]
        preset = PRESETS[preset_key]
        speed_range = preset["speed_range"]
        power_range = preset["power_range"]
        test_type = preset.get("test_type", "Cut")

        machine_max_speed = self.step.max_cut_speed
        min_speed = min(speed_range[0], machine_max_speed)
        max_speed = min(speed_range[1], machine_max_speed)

        self.speed_min_row.set_value(min_speed)
        self.speed_max_row.set_value(max_speed)
        self.min_power_adj.set_value(power_range[0])
        self.max_power_adj.set_value(power_range[1])

        # Cancel any debounced callbacks triggered by set_value() above.
        # DebounceMixin uses a single timer slot, so rapid set_value()
        # calls cause earlier callbacks to be lost. We commit directly
        # below instead.
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0

        self._update_range_param("speed_range", (min_speed, max_speed))
        self._commit_power_range_change()

        for i, val in enumerate(self._test_type_values):
            if val == test_type:
                self.test_type_row.set_selected(i)
                break

    def _on_test_type_changed(self, row: Adw.ComboRow, _pspec):
        """Updates the test type parameter."""
        selected_idx = row.get_selected()
        if selected_idx != Gtk.INVALID_LIST_POSITION:
            test_type_text = self._test_type_values[selected_idx]
            self._update_param("test_type", test_type_text)

    def _on_speed_min_changed(self, spin_row):
        min_speed = spin_row.get_value()
        max_speed = self.speed_max_row.get_value()
        self._update_range_param("speed_range", (min_speed, max_speed))

    def _on_speed_max_changed(self, spin_row):
        min_speed = self.speed_min_row.get_value()
        max_speed = spin_row.get_value()
        self._update_range_param("speed_range", (min_speed, max_speed))

    def _commit_power_range_change(self):
        """Commits the min/max power range to the step."""
        min_p = self.min_power_adj.get_value()
        max_p = self.max_power_adj.get_value()
        new_range = (min_p, max_p)

        if self.step.power_range == new_range:
            return

        self._exit_preview_mode_if_active()
        self.set_step_property("power_range", new_range)

    def _on_min_power_scale_changed(self, scale: Gtk.Scale):
        new_min_value = self.min_power_adj.get_value()
        GObject.signal_handler_block(
            self.max_power_scale, self.max_power_handler_id
        )
        if self.max_power_adj.get_value() < new_min_value:
            self.max_power_adj.set_value(new_min_value)
        GObject.signal_handler_unblock(
            self.max_power_scale, self.max_power_handler_id
        )
        self._debounce(self._commit_power_range_change)

    def _on_max_power_scale_changed(self, scale: Gtk.Scale):
        new_max_value = self.max_power_adj.get_value()
        GObject.signal_handler_block(
            self.min_power_scale, self.min_power_handler_id
        )
        if self.min_power_adj.get_value() > new_max_value:
            self.min_power_adj.set_value(new_max_value)
        GObject.signal_handler_unblock(
            self.min_power_scale, self.min_power_handler_id
        )
        self._debounce(self._commit_power_range_change)

    def _on_grid_cols_changed(self, spin_row):
        cols = spin_row.get_int_value()
        _, rows = self.step.grid_dimensions
        self._update_grid_param((cols, rows))

    def _on_grid_rows_changed(self, spin_row):
        cols, _ = self.step.grid_dimensions
        rows = spin_row.get_int_value()
        self._update_grid_param((cols, rows))

    def _on_shape_size_changed(self, spin_row):
        self._update_param("shape_size", spin_row.get_value())

    def _on_spacing_changed(self, spin_row):
        self._update_param("spacing", spin_row.get_value())

    def _on_line_interval_changed(self, spin_row):
        value = spin_row.get_value()
        if value <= 0:
            value = None
        self._update_param("line_interval_mm", value)

    def _on_labels_toggled(self, switch, state):
        self.label_power_row.set_sensitive(state)
        self.label_speed_row.set_sensitive(state)
        self._update_param("include_labels", state)
        return False

    def _on_label_power_changed(self, scale: Gtk.Scale):
        val = scale.get_value()
        logger.debug("Label power slider changed: %s", val)
        self._update_param("label_power_percent", val)

    def _on_label_speed_changed(self, spin_row):
        self._update_param("label_speed", spin_row.get_value())

    def _on_grid_mode_changed(self, row: Adw.ComboRow, _pspec):
        selected_idx = row.get_selected()
        if selected_idx == Gtk.INVALID_LIST_POSITION:
            return
        mode_value = self._grid_mode_values[selected_idx]
        self._update_param("grid_mode", mode_value)
        self._update_control_visibility()
        self._update_dimension_labels()

        if mode_value == "Speed vs Offset":
            self._apply_speed_vs_offset_defaults()

    def _apply_speed_vs_offset_defaults(self):
        """Bidir scan offset calibration only makes sense for raster
        engraving (Cut has no bidirectional scanning to calibrate), and
        needs wide line spacing to make row-to-row misalignment clearly
        visible by eye. Can't default the preset dropdown too, since
        there are multiple Engrave presets (Diode/CO2) with different
        ranges."""
        for i, val in enumerate(self._test_type_values):
            if val == "Engrave":
                self.test_type_row.set_selected(i)
                break

        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0
        self.line_interval_row.set_value(0.5)
        self._update_param("line_interval_mm", 0.5)

    def _on_fixed_speed_changed(self, spin_row):
        self._update_param("fixed_speed", spin_row.get_value())

    def _on_fixed_power_changed(self, scale: Gtk.Scale):
        self._update_param("fixed_power", scale.get_value())

    def _on_passes_min_changed(self, spin_row):
        min_passes = spin_row.get_int_value()
        _, max_passes = self.step.passes_range
        self._update_range_param("passes_range", (min_passes, max_passes))

    def _on_passes_max_changed(self, spin_row):
        min_passes, _ = self.step.passes_range
        max_passes = spin_row.get_int_value()
        self._update_range_param("passes_range", (min_passes, max_passes))

    def _on_offset_min_changed(self, spin_row):
        min_offset = spin_row.get_value()
        _, max_offset = self.step.offset_range
        self._update_range_param("offset_range", (min_offset, max_offset))

    def _on_offset_max_changed(self, spin_row):
        min_offset, _ = self.step.offset_range
        max_offset = spin_row.get_value()
        self._update_range_param("offset_range", (min_offset, max_offset))

    def _get_current_grid_mode(self) -> str:
        selected_idx = self.grid_mode_row.get_selected()
        if selected_idx == Gtk.INVALID_LIST_POSITION:
            return GridMode.POWER_VS_SPEED.value
        return self._grid_mode_values[selected_idx]

    def _update_control_visibility(self):
        mode = self._get_current_grid_mode()
        show_power_range = mode in ("Power vs Speed", "Power vs Passes")
        show_speed_range = mode in (
            "Power vs Speed",
            "Speed vs Passes",
            "Speed vs Offset",
        )
        show_passes_range = mode in ("Power vs Passes", "Speed vs Passes")
        show_offset_range = mode == "Speed vs Offset"
        show_fixed_speed = mode == "Power vs Passes"
        show_fixed_power = mode in ("Speed vs Passes", "Speed vs Offset")

        self.fixed_speed_row.set_visible(show_fixed_speed)
        self.fixed_power_row.set_visible(show_fixed_power)

        self.min_power_row.set_visible(show_power_range)
        self.max_power_row.set_visible(show_power_range)
        self.min_power_scale.set_visible(show_power_range)
        self.max_power_scale.set_visible(show_power_range)

        self.speed_min_row.set_visible(show_speed_range)
        self.speed_max_row.set_visible(show_speed_range)

        self.passes_min_row.set_visible(show_passes_range)
        self.passes_max_row.set_visible(show_passes_range)

        self.offset_min_row.set_visible(show_offset_range)
        self.offset_max_row.set_visible(show_offset_range)

    def _update_dimension_labels(self):
        mode = self._get_current_grid_mode()
        if mode == "Power vs Passes":
            col_title = _("Columns (Power Steps)")
            col_sub = _("Number of power variations")
            row_title = _("Rows (Passes Steps)")
            row_sub = _("Number of passes variations")
        elif mode == "Speed vs Passes":
            col_title = _("Columns (Speed Steps)")
            col_sub = _("Number of speed variations")
            row_title = _("Rows (Passes Steps)")
            row_sub = _("Number of passes variations")
        elif mode == "Speed vs Offset":
            col_title = _("Columns (Speed Steps)")
            col_sub = _("Number of speed variations")
            row_title = _("Rows (Offset Steps)")
            row_sub = _("Number of offset variations")
        else:
            col_title = _("Columns (Power Steps)")
            col_sub = _("Number of power variations")
            row_title = _("Rows (Speed Steps)")
            row_sub = _("Number of speed variations")
        self.cols_row.set_title(col_title)
        self.cols_row.set_subtitle(col_sub)
        self.rows_row.set_title(row_title)
        self.rows_row.set_subtitle(row_sub)

    # Helper methods
    def _update_param(self, param_name: str, new_value: Any):
        """Updates a simple parameter on the step."""
        current = getattr(self.step, param_name, None)
        if current == new_value:
            return
        self._exit_preview_mode_if_active()
        self.set_step_property(param_name, new_value)

    def _update_range_param(self, param_name: str, new_value: Any):
        """Updates a range tuple parameter on the step."""
        current = getattr(self.step, param_name, None)
        if current == new_value:
            return
        self._exit_preview_mode_if_active()
        self.set_step_property(param_name, new_value)

    def _update_grid_param(self, new_value: Any):
        """Updates grid dimensions on the step."""
        current = self.step.grid_dimensions
        if current == new_value:
            return
        self._exit_preview_mode_if_active()
        self.set_step_property("grid_dimensions", new_value)

    def _exit_preview_mode_if_active(self):
        """Exits execution preview mode if currently active."""
        if not self.step.doc:
            return
        from rayforge.ui_gtk.mainwindow import MainWindow

        root = self.get_root()
        if not isinstance(root, MainWindow):
            return

        action = root.action_manager.get_action("view_mode")
        if not action:
            return

        state = action.get_state()
        if state and state.get_string() == "preview":
            action.change_state(GLib.Variant.new_string("2d"))
