"""
Material Test Grid Settings Widget

Provides UI for configuring material test array parameters.
"""

from typing import Dict, Any, TYPE_CHECKING, cast

from gi.repository import Adw, GLib, GObject, Gtk

from ....core.undo import DictItemCommand
from ....pipeline.producer import MaterialTestGridProducer
from ...shared.adwfix import get_spinrow_float, get_spinrow_int
from ....shared.util.glib import DebounceMixin
from .base import StepComponentSettingsWidget

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


# Preset selector constants
PRESET_NONE = "Select"

# Preset configurations
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


class MaterialTestGridSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """MaterialTestGridProducer UI."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        # Get current params
        producer = MaterialTestGridProducer.from_dict(target_dict)

        super().__init__(
            editor,
            title,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        # Populate the main group (self)
        self._build_preset_selector(producer)
        self._build_test_type_selector(producer)
        self._build_grid_dimensions(producer)
        self._build_shape_size(producer)
        self._build_spacing(producer)
        self._build_label_settings(producer)

        # Schedule the creation of the second group to run after this widget
        # has been added to its parent page, ensuring correct order.
        GLib.idle_add(self._build_power_and_speed_group, producer)

    def _build_preset_selector(self, producer: MaterialTestGridProducer):
        """Builds the preset dropdown."""
        string_list = Gtk.StringList()
        string_list.append(PRESET_NONE)
        for preset_name in PRESETS:
            string_list.append(preset_name)

        self.preset_row = Adw.ComboRow(
            title=_("Presets"),
            subtitle=_("Load common test configurations"),
            model=string_list,
        )
        self.preset_row.set_selected(0)
        self.add(self.preset_row)
        self.preset_row.connect("notify::selected", self._on_preset_changed)

    def _build_test_type_selector(self, producer: MaterialTestGridProducer):
        """Builds the test type dropdown (Cut/Engrave)."""
        string_list = Gtk.StringList.new(["Cut", "Engrave"])
        self.test_type_row = Adw.ComboRow(
            title=_("Test Type"),
            subtitle=_("Cut: outlines; Engrave: fills with raster lines"),
            model=string_list,
        )
        current_text = producer.test_type.value
        if current_text == "Cut":
            self.test_type_row.set_selected(0)
        else:
            self.test_type_row.set_selected(1)
        self.add(self.test_type_row)
        self.test_type_row.connect(
            "notify::selected", self._on_test_type_changed
        )

    def _build_power_and_speed_group(self, producer: MaterialTestGridProducer):
        """Builds the group for power and speed settings."""
        group = Adw.PreferencesGroup(
            title=_("Speed &amp; Power"),
            description=_(
                "Define the range of speeds and powers for the test grid"
            ),
        )
        self.page.add(group)

        # Power Range
        min_power, max_power = producer.power_range
        self.min_power_adj = Gtk.Adjustment(
            lower=1, upper=100, step_increment=1, value=min_power
        )
        self.min_power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.min_power_adj,
            digits=0,
            draw_value=True,
            width_request=200,
        )
        min_power_row = Adw.ActionRow(
            title=_("Minimum Power (%)"), subtitle=_("For first column")
        )
        min_power_row.add_suffix(self.min_power_scale)
        group.add(min_power_row)

        self.max_power_adj = Gtk.Adjustment(
            lower=1, upper=100, step_increment=1, value=max_power
        )
        self.max_power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.max_power_adj,
            digits=0,
            draw_value=True,
            width_request=200,
        )
        max_power_row = Adw.ActionRow(
            title=_("Maximum Power (%)"), subtitle=_("For last column")
        )
        max_power_row.add_suffix(self.max_power_scale)
        group.add(max_power_row)

        self.min_power_handler_id = self.min_power_scale.connect(
            "value-changed", self._on_min_power_scale_changed
        )
        self.max_power_handler_id = self.max_power_scale.connect(
            "value-changed", self._on_max_power_scale_changed
        )

        # Speed Range
        min_speed, max_speed = producer.speed_range
        machine_max_speed = self.step.max_cut_speed
        min_speed = min(min_speed, machine_max_speed)
        max_speed = min(max_speed, machine_max_speed)
        min_adj = Gtk.Adjustment(
            lower=1.0, upper=machine_max_speed, step_increment=10.0
        )
        self.speed_min_row = Adw.SpinRow(
            title=_("Minimum Speed"),
            subtitle=_("Starting speed (mm/min)"),
            adjustment=min_adj,
            digits=0,
            value=min_speed,
        )
        group.add(self.speed_min_row)

        max_adj = Gtk.Adjustment(
            lower=1.0, upper=machine_max_speed, step_increment=10.0
        )
        self.speed_max_row = Adw.SpinRow(
            title=_("Maximum Speed"),
            subtitle=_("Ending speed (mm/min)"),
            adjustment=max_adj,
            digits=0,
            value=max_speed,
        )
        group.add(self.speed_max_row)

        self.speed_min_row.connect(
            "changed", lambda r: self._debounce(self._on_speed_min_changed, r)
        )
        self.speed_max_row.connect(
            "changed", lambda r: self._debounce(self._on_speed_max_changed, r)
        )
        return False  # for GLib.idle_add

    def _build_grid_dimensions(self, producer: MaterialTestGridProducer):
        """Builds grid dimension controls."""
        cols, rows = producer.grid_dimensions

        cols_adj = Gtk.Adjustment(lower=2, upper=20, step_increment=1)
        self.cols_row = Adw.SpinRow(
            title=_("Columns (Power Steps)"),
            subtitle=_("Number of power variations"),
            adjustment=cols_adj,
            digits=0,
            value=cols,
        )
        self.add(self.cols_row)

        rows_adj = Gtk.Adjustment(lower=2, upper=20, step_increment=1)
        self.rows_row = Adw.SpinRow(
            title=_("Rows (Speed Steps)"),
            subtitle=_("Number of speed variations"),
            adjustment=rows_adj,
            digits=0,
            value=rows,
        )
        self.add(self.rows_row)

        self.cols_row.connect(
            "changed", lambda r: self._debounce(self._on_grid_cols_changed, r)
        )
        self.rows_row.connect(
            "changed", lambda r: self._debounce(self._on_grid_rows_changed, r)
        )

    def _build_shape_size(self, producer: MaterialTestGridProducer):
        """Builds shape size control."""
        adj = Gtk.Adjustment(lower=1, upper=100, step_increment=1)
        self.shape_size_row = Adw.SpinRow(
            title=_("Shape Size"),
            subtitle=_("Size of each test square (mm)"),
            adjustment=adj,
            digits=1,
            value=producer.shape_size,
        )
        self.add(self.shape_size_row)
        self.shape_size_row.connect(
            "changed", lambda r: self._debounce(self._on_shape_size_changed, r)
        )

    def _build_spacing(self, producer: MaterialTestGridProducer):
        """Builds spacing control."""
        adj = Gtk.Adjustment(lower=0, upper=50, step_increment=0.5)
        self.spacing_row = Adw.SpinRow(
            title=_("Spacing"),
            subtitle=_("Gap between test squares (mm)"),
            adjustment=adj,
            digits=1,
            value=producer.spacing,
        )
        self.add(self.spacing_row)
        self.spacing_row.connect(
            "changed", lambda r: self._debounce(self._on_spacing_changed, r)
        )

    def _build_label_settings(self, producer: MaterialTestGridProducer):
        """Builds controls for label appearance and behavior."""
        self.include_labels_switch = Gtk.Switch(
            valign=Gtk.Align.CENTER, active=producer.include_labels
        )
        labels_row = Adw.ActionRow(
            title=_("Include Labels"),
            subtitle=_("Add speed/power annotations to the grid"),
        )
        labels_row.add_suffix(self.include_labels_switch)
        labels_row.set_activatable_widget(self.include_labels_switch)
        self.add(labels_row)

        power_adj = Gtk.Adjustment(
            lower=1,
            upper=100,
            step_increment=1,
            value=producer.label_power_percent,
        )
        power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=power_adj,
            digits=0,
            draw_value=True,
            width_request=200,
        )
        self.label_power_row = Adw.ActionRow(
            title=_("Label Engrave Power (%)")
        )
        self.label_power_row.add_suffix(power_scale)
        self.add(self.label_power_row)

        self.include_labels_switch.connect(
            "state-set", self._on_labels_toggled
        )
        power_scale.connect(
            "value-changed",
            lambda s: self._debounce(self._on_label_power_changed, s),
        )

        self._on_labels_toggled(
            self.include_labels_switch, producer.include_labels
        )

    # Signal handlers
    def _on_preset_changed(self, row: Adw.ComboRow, _pspec):
        """Loads preset values."""
        selected_idx = row.get_selected()
        if selected_idx == Gtk.INVALID_LIST_POSITION:
            return
        model = cast(Gtk.StringList, row.get_model())
        preset_name = model.get_string(selected_idx)
        if not preset_name or preset_name == PRESET_NONE:
            return
        if preset_name not in PRESETS:
            return

        preset = PRESETS[preset_name]
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

        model = cast(Gtk.StringList, self.test_type_row.get_model())
        for i in range(model.get_n_items()):
            if model.get_string(i) == test_type:
                self.test_type_row.set_selected(i)
                break

    def _on_test_type_changed(self, row: Adw.ComboRow, _pspec):
        """Updates the test type parameter."""
        selected_idx = row.get_selected()
        if selected_idx != Gtk.INVALID_LIST_POSITION:
            model = cast(Gtk.StringList, row.get_model())
            test_type_text = model.get_string(selected_idx)
            self._update_param("test_type", test_type_text)

    def _on_speed_min_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_range_param("speed_range", 0, new_value)

    def _on_speed_max_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_range_param("speed_range", 1, new_value)

    def _commit_power_range_change(self):
        """Commits the min/max power range to the model via a command."""
        min_p = self.min_power_adj.get_value()
        max_p = self.max_power_adj.get_value()
        new_range = [min_p, max_p]

        params_dict = self.target_dict.setdefault("params", {})
        if params_dict.get("power_range") == new_range:
            return

        def on_change():
            self._exit_preview_mode_if_active()
            self.step.updated.send(self.step)

        command = DictItemCommand(
            target_dict=params_dict,
            key="power_range",
            new_value=new_range,
            name=_("Change Power Range"),
            on_change_callback=on_change,
        )
        self.history_manager.execute(command)

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
        new_value = get_spinrow_int(spin_row)
        self._update_grid_param(0, new_value)

    def _on_grid_rows_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        self._update_grid_param(1, new_value)

    def _on_shape_size_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_param("shape_size", new_value)

    def _on_spacing_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_param("spacing", new_value)

    def _on_labels_toggled(self, switch, state):
        self.label_power_row.set_sensitive(state)
        self._update_param("include_labels", state)
        return False

    def _on_label_power_changed(self, scale: Gtk.Scale):
        self._update_param("label_power_percent", scale.get_value())

    # Helper methods
    def _update_param(self, param_name: str, new_value: Any):
        """Updates a simple parameter in the step's producer dictionary."""
        params_dict = self.target_dict.setdefault("params", {})
        if params_dict.get(param_name) == new_value:
            return

        def on_change():
            self._exit_preview_mode_if_active()
            self.step.updated.send(self.step)

        command = DictItemCommand(
            target_dict=params_dict,
            key=param_name,
            new_value=new_value,
            name=_(f"Change {param_name}"),
            on_change_callback=on_change,
        )
        self.history_manager.execute(command)

    def _update_range_param(
        self, param_name: str, index: int, new_value: float
    ):
        """Updates one element of a range tuple."""
        params_dict = self.target_dict.setdefault("params", {})
        current_range = list(params_dict.get(param_name, [0.0, 0.0]))
        current_range[index] = new_value

        def on_change():
            self._exit_preview_mode_if_active()
            self.step.updated.send(self.step)

        command = DictItemCommand(
            target_dict=params_dict,
            key=param_name,
            new_value=current_range,
            name=_(f"Change {param_name}"),
            on_change_callback=on_change,
        )
        self.history_manager.execute(command)

    def _update_grid_param(self, index: int, new_value: int):
        """Updates grid dimensions and triggers a sync."""
        params_dict = self.target_dict.setdefault("params", {})
        current_grid = list(params_dict.get("grid_dimensions", [5, 5]))
        current_grid[index] = new_value

        def on_change():
            self._exit_preview_mode_if_active()
            self.step.updated.send(self.step)

        command = DictItemCommand(
            target_dict=params_dict,
            key="grid_dimensions",
            new_value=current_grid,
            name=_("Change Grid Dimensions"),
            on_change_callback=on_change,
        )
        self.history_manager.execute(command)

    def _exit_preview_mode_if_active(self):
        """Exits execution preview mode if currently active."""
        if not self.step.doc:
            return
        from ...mainwindow import MainWindow

        root = self.get_root()
        if not isinstance(root, MainWindow):
            return

        action = root.action_manager.get_action("view_mode")
        if not action:
            return

        state = action.get_state()
        if state and state.get_string() == "preview":
            action.change_state(GLib.Variant.new_string("2d"))
