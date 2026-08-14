"""
Material Test Grid Settings Widget

Provides UI for configuring material test array parameters.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, GLib, Gtk

from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.mainwindow import MainWindow

from ..material_test_helpers import GridMode
from .laser_step_page import LaserStepSettingsPage
from .tuple_adapter import TupleAdapter

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

_GRID_KEYS = {
    "test_type",
    "grid_mode",
    "grid_dimensions",
    "shape_size",
    "spacing",
    "line_interval_mm",
}

_LABEL_KEYS = {
    "include_labels",
    "label_power_percent",
    "label_speed",
}

_PARAM_KEYS = {
    "fixed_speed",
    "fixed_power",
    "power_range",
    "speed_range",
    "passes_range",
    "offset_range",
}

#: Dimension row titles and subtitles per grid mode.
_DIMENSION_LABELS = {
    "Power vs Speed": (
        (_("Columns (Power Steps)"), _("Rows (Speed Steps)")),
        (_("Number of power variations"), _("Number of speed variations")),
    ),
    "Power vs Passes": (
        (_("Columns (Power Steps)"), _("Rows (Passes Steps)")),
        (_("Number of power variations"), _("Number of passes variations")),
    ),
    "Speed vs Passes": (
        (_("Columns (Speed Steps)"), _("Rows (Passes Steps)")),
        (_("Number of speed variations"), _("Number of passes variations")),
    ),
    "Speed vs Offset": (
        (_("Columns (Speed Steps)"), _("Rows (Offset Steps)")),
        (_("Number of speed variations"), _("Number of offset variations")),
    ),
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
        self._build_preset_section()
        groups = self.step.recipe_varset_groups()
        step_vars = groups[-1][1] if len(groups) > 1 else None
        if step_vars is None:
            return
        self.grid_widget = self.add_varset_section(
            _("Grid"),
            self._varset_for_keys(step_vars, _GRID_KEYS),
            description=_("Test cell dimensions, shape, and spacing."),
        )
        self.labels_widget = self.add_varset_section(
            _("Labels"),
            self._varset_for_keys(step_vars, _LABEL_KEYS),
            description=_("Speed/power annotations on the grid."),
        )
        self.params_widget = self.add_varset_section(
            _("Parameters"),
            self._varset_for_keys(step_vars, _PARAM_KEYS),
            description=_("Define the parameter ranges for the test grid."),
        )

        # The Parameters section's visible_when predicates key off
        # grid_mode, whose row lives in the Grid section. Feed it in
        # as context and refresh whenever the mode changes.
        self._sync_grid_context()
        mode_adapter = self.grid_widget.adapter_for("grid_mode")
        if mode_adapter is not None:
            mode_adapter.changed.connect(
                lambda sender: self._sync_grid_context(), weak=False
            )
        self._update_dimension_labels()
        self._update_machine_bounds()
        self._push_head_defaults()

    def _push_head_defaults(self):
        """Show the laser spot-width default for the auto line interval.

        The step keeps ``None`` (auto) until the user edits the row;
        the dialog displays the laser spot-width default instead.
        """
        head = self.get_selected_head()
        if not isinstance(head, LaserHead):
            return
        if self.step.line_interval_mm is None:
            self.grid_widget.set_values(
                {"line_interval_mm": head.spot_size_mm[1]}
            )

    def _build_preset_section(self):
        """Builds the preset dropdown (page chrome: a UI action, not a
        setting)."""
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

        group = self.add_section(
            _("Preset"),
            description=_("Load common test configurations."),
        )
        self.preset_row = Adw.ComboRow(
            title=_("Presets"),
            subtitle=_("Load common test configurations"),
            model=string_list,
        )
        self.preset_row.set_selected(0)
        group.add(self.preset_row)
        self.preset_row.connect("notify::selected", self._on_preset_changed)

    def _sync_grid_context(self):
        """Feed the current grid_mode into the Parameters section."""
        mode = self.grid_widget.get_values().get("grid_mode")
        if mode is None:
            mode = self.step.grid_mode
        self.params_widget.set_context_values({"grid_mode": mode})

    def _update_dimension_labels(self):
        """Retitle the cols/rows rows for the current grid mode."""
        adapter = self.grid_widget.adapter_for("grid_dimensions")
        if not isinstance(adapter, TupleAdapter):
            return
        mode = self.grid_widget.get_values().get("grid_mode")
        if mode is None:
            mode = self.step.grid_mode
        labels, subtitles = _DIMENSION_LABELS.get(
            mode, _DIMENSION_LABELS[GridMode.POWER_VS_SPEED.value]
        )
        adapter.set_item_labels(labels)
        adapter.set_item_subtitles(subtitles)

    def _update_machine_bounds(self):
        """Cap the speed range at the step's machine limit."""
        adapter = self.params_widget.adapter_for("speed_range")
        if isinstance(adapter, TupleAdapter):
            adapter.set_bounds(1.0, float(self.step.max_cut_speed))

    def _sync_widgets_to_model(self, *args):
        super()._sync_widgets_to_model(*args)
        self._sync_grid_context()
        self._update_dimension_labels()
        self._update_machine_bounds()

    def _on_varset_data_changed(self, widget, key):
        if key == "grid_mode":
            super()._on_varset_data_changed(widget, key)
            self._sync_grid_context()
            self._update_dimension_labels()
            self._apply_speed_vs_offset_defaults()
            return
        if key in (
            "power_range",
            "speed_range",
            "passes_range",
            "offset_range",
        ):
            self._exit_preview_mode_if_active()
        super()._on_varset_data_changed(widget, key)

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

        self.set_step_property("speed_range", (min_speed, max_speed))
        self.set_step_property("power_range", power_range)
        self.set_step_property("test_type", test_type)
        self.params_widget.set_values(
            {"speed_range": (min_speed, max_speed), "power_range": power_range}
        )
        self.grid_widget.set_values({"test_type": test_type})

    def _apply_speed_vs_offset_defaults(self):
        """Bidir scan offset calibration only makes sense for raster
        engraving (Cut has no bidirectional scanning to calibrate), and
        needs wide line spacing to make row-to-row misalignment clearly
        visible by eye. Can't default the preset dropdown too, since
        there are multiple Engrave presets (Diode/CO2) with different
        ranges."""
        mode = self.grid_widget.get_values().get("grid_mode")
        if mode != GridMode.SPEED_VS_OFFSET.value:
            return
        self.set_step_property("test_type", "Engrave")
        self.grid_widget.set_values({"test_type": "Engrave"})
        self.set_step_property("line_interval_mm", 0.5)
        self.grid_widget.set_values({"line_interval_mm": 0.5})

    def _exit_preview_mode_if_active(self):
        """Exits execution preview mode if currently active."""
        if not self.step.doc:
            return
        root = self.get_root()
        if not isinstance(root, MainWindow):
            return

        action = root.action_manager.get_action("view_mode")
        if not action:
            return

        state = action.get_state()
        if state and state.get_string() == "preview":
            action.change_state(GLib.Variant.new_string("2d"))
