"""
Material Test Grid Settings Widget

Provides UI for configuring material test array parameters.
"""

from typing import Dict, Any, TYPE_CHECKING, Tuple, cast
from gi.repository import Gtk, Adw, GLib  # noqa: F401
from .base import StepComponentSettingsWidget
from ....shared.util.adwfix import get_spinrow_float, get_spinrow_int
from ....shared.util.glib import DebounceMixin
from ....pipeline.producer import MaterialTestGridProducer
from ....pipeline.producer.material_test_grid import MaterialTestGridType
from ....undo import DictItemCommand

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

        self._build_preset_selector()
        self._build_test_type_selector(producer)
        self._build_speed_range(producer)
        self._build_power_range(producer)
        self._build_grid_dimensions(producer)
        self._build_shape_size(producer)
        self._build_spacing(producer)

    def _build_preset_selector(self):
        """Builds the preset dropdown."""
        # Use StringList for Adw.ComboRow
        string_list = Gtk.StringList()
        string_list.append(PRESET_NONE)  # Default option
        for preset_name in PRESETS:
            string_list.append(preset_name)

        preset_row = Adw.ComboRow(
            title=_("Preset"),
            subtitle=_("Load common test configurations"),
            model=string_list,
        )
        preset_row.set_selected(0)  # Start with PRESET_NONE selected

        # Create a separate group for presets
        preset_group = Adw.PreferencesGroup(title=_("Presets"))
        preset_group.add(preset_row)
        self.add(preset_group)

        preset_row.connect("notify::selected", self._on_preset_changed)
        self.preset_row = preset_row

    def _build_test_type_selector(self, producer: MaterialTestGridProducer):
        """Builds the test type dropdown (Cut/Engrave)."""
        # Use StringList for better value management
        string_list = Gtk.StringList()
        string_list.append("Cut")
        string_list.append("Engrave")

        test_type_row = Adw.ComboRow(
            title=_("Test Type"),
            subtitle=_("Cut: outlines; Engrave: fills with raster lines"),
            model=string_list,
        )

        # Set current value by finding matching string
        current_type = producer.test_type
        current_text = (
            current_type.value
            if isinstance(current_type, MaterialTestGridType)
            else current_type
        )
        for i in range(string_list.get_n_items()):
            if string_list.get_string(i) == current_text:
                test_type_row.set_selected(i)
                break

        self.add(test_type_row)
        test_type_row.connect("notify::selected", self._on_test_type_changed)
        self.test_type_row = test_type_row

    def _build_speed_range(self, producer: MaterialTestGridProducer):
        """Builds speed range controls."""
        min_speed, max_speed = producer.speed_range

        # Get max allowed speed from machine settings via step
        machine_max_speed = self.step.max_cut_speed

        # Clamp current values to machine maximum
        min_speed = min(min_speed, machine_max_speed)
        max_speed = min(max_speed, machine_max_speed)

        # Min speed
        min_adj = Gtk.Adjustment(
            lower=1.0,
            upper=machine_max_speed,
            step_increment=10.0,
            page_increment=100.0,
        )
        self.speed_min_row = Adw.SpinRow(
            title=_("Minimum Speed"),
            subtitle=_("Starting speed for test grid (mm/min)"),
            adjustment=min_adj,
            digits=0,
        )
        min_adj.set_value(min_speed)
        self.add(self.speed_min_row)

        # Max speed
        max_adj = Gtk.Adjustment(
            lower=1.0,
            upper=machine_max_speed,
            step_increment=10.0,
            page_increment=100.0,
        )
        self.speed_max_row = Adw.SpinRow(
            title=_("Maximum Speed"),
            subtitle=_("Ending speed for test grid (mm/min)"),
            adjustment=max_adj,
            digits=0,
        )
        max_adj.set_value(max_speed)
        self.add(self.speed_max_row)

        # Connect signals
        self.speed_min_row.connect(
            "changed", lambda r: self._debounce(self._on_speed_min_changed, r)
        )
        self.speed_max_row.connect(
            "changed", lambda r: self._debounce(self._on_speed_max_changed, r)
        )

    def _build_power_range(self, producer: MaterialTestGridProducer):
        """Builds power range controls."""
        min_power, max_power = producer.power_range

        # Min power
        min_adj = Gtk.Adjustment(
            lower=1.0, upper=100.0, step_increment=1.0, page_increment=10.0
        )
        self.power_min_row = Adw.SpinRow(
            title=_("Minimum Power"),
            subtitle=_("Starting power for test grid (%)"),
            adjustment=min_adj,
            digits=0,
        )
        min_adj.set_value(min_power)
        self.add(self.power_min_row)

        # Max power
        max_adj = Gtk.Adjustment(
            lower=1.0, upper=100.0, step_increment=1.0, page_increment=10.0
        )
        self.power_max_row = Adw.SpinRow(
            title=_("Maximum Power"),
            subtitle=_("Ending power for test grid (%)"),
            adjustment=max_adj,
            digits=0,
        )
        max_adj.set_value(max_power)
        self.add(self.power_max_row)

        # Connect signals
        self.power_min_row.connect(
            "changed", lambda r: self._debounce(self._on_power_min_changed, r)
        )
        self.power_max_row.connect(
            "changed", lambda r: self._debounce(self._on_power_max_changed, r)
        )

    def _build_grid_dimensions(self, producer: MaterialTestGridProducer):
        """Builds grid dimension controls."""
        cols, rows = producer.grid_dimensions

        # Columns
        cols_adj = Gtk.Adjustment(
            lower=2.0, upper=20.0, step_increment=1.0, page_increment=5.0
        )
        self.cols_row = Adw.SpinRow(
            title=_("Columns (Speed Steps)"),
            subtitle=_("Number of speed variations"),
            adjustment=cols_adj,
            digits=0,
        )
        cols_adj.set_value(cols)
        self.add(self.cols_row)

        # Rows
        rows_adj = Gtk.Adjustment(
            lower=2.0, upper=20.0, step_increment=1.0, page_increment=5.0
        )
        self.rows_row = Adw.SpinRow(
            title=_("Rows (Power Steps)"),
            subtitle=_("Number of power variations"),
            adjustment=rows_adj,
            digits=0,
        )
        rows_adj.set_value(rows)
        self.add(self.rows_row)

        # Connect signals
        self.cols_row.connect(
            "changed", lambda r: self._debounce(self._on_grid_cols_changed, r)
        )
        self.rows_row.connect(
            "changed", lambda r: self._debounce(self._on_grid_rows_changed, r)
        )

    def _build_shape_size(self, producer: MaterialTestGridProducer):
        """Builds shape size control."""
        adj = Gtk.Adjustment(
            lower=1.0, upper=100.0, step_increment=1.0, page_increment=5.0
        )
        self.shape_size_row = Adw.SpinRow(
            title=_("Shape Size"),
            subtitle=_("Size of each test square (mm)"),
            adjustment=adj,
            digits=1,
        )
        adj.set_value(producer.shape_size)
        self.add(self.shape_size_row)

        self.shape_size_row.connect(
            "changed", lambda r: self._debounce(self._on_shape_size_changed, r)
        )

    def _build_spacing(self, producer: MaterialTestGridProducer):
        """Builds spacing control."""
        adj = Gtk.Adjustment(
            lower=0.0, upper=50.0, step_increment=0.5, page_increment=2.0
        )
        self.spacing_row = Adw.SpinRow(
            title=_("Spacing"),
            subtitle=_("Gap between test squares (mm)"),
            adjustment=adj,
            digits=1,
        )
        adj.set_value(producer.spacing)
        self.add(self.spacing_row)

        self.spacing_row.connect(
            "changed", lambda r: self._debounce(self._on_spacing_changed, r)
        )

    def _build_labels_toggle(self, producer: MaterialTestGridProducer):
        """Builds labels toggle switch."""
        switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        switch.set_active(producer.include_labels)

        labels_row = Adw.ActionRow(
            title=_("Include Labels"),
            subtitle=_("Add speed/power annotations to the grid"),
        )
        labels_row.add_suffix(switch)
        labels_row.set_activatable_widget(switch)
        self.add(labels_row)

        switch.connect("state-set", self._on_labels_toggled)

    # Signal handlers
    def _on_preset_changed(self, row: Adw.ComboRow, _pspec):
        """Loads preset values."""
        selected_idx = row.get_selected()
        if selected_idx == Gtk.INVALID_LIST_POSITION:
            return

        model = row.get_model()
        if model is None:
            return
        string_list = cast(Gtk.StringList, model)
        preset_name = string_list.get_string(selected_idx)

        # Ignore PRESET_NONE selection - it's just a placeholder
        if not preset_name or preset_name == PRESET_NONE:
            return

        # Only apply preset if it exists
        if preset_name not in PRESETS:
            return

        preset = PRESETS[preset_name]

        # Update UI (which will trigger parameter updates)
        speed_range = preset["speed_range"]
        power_range = preset["power_range"]
        test_type = preset.get("test_type", "Cut")

        # Clamp speed values to machine maximum
        machine_max_speed = self.step.max_cut_speed
        min_speed = min(speed_range[0], machine_max_speed)
        max_speed = min(speed_range[1], machine_max_speed)

        self.speed_min_row.get_adjustment().set_value(min_speed)
        self.speed_max_row.get_adjustment().set_value(max_speed)
        self.power_min_row.get_adjustment().set_value(power_range[0])
        self.power_max_row.get_adjustment().set_value(power_range[1])

        # Find and set test type by matching string
        model = self.test_type_row.get_model()
        if model is None:
            return
        string_list = cast(Gtk.StringList, model)
        for i in range(string_list.get_n_items()):
            if string_list.get_string(i) == test_type:
                self.test_type_row.set_selected(i)
                break

    def _on_test_type_changed(self, row: Adw.ComboRow, _pspec):
        """Updates the test type parameter."""
        selected_idx = row.get_selected()
        if selected_idx != Gtk.INVALID_LIST_POSITION:
            model = row.get_model()
            if model is None:
                return
            string_list = cast(Gtk.StringList, model)
            test_type_text = string_list.get_string(selected_idx)
            # Store enum value string for serialization
            self._update_param("test_type", test_type_text)

    def _on_speed_min_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_range_param("speed_range", 0, new_value)

    def _on_speed_max_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_range_param("speed_range", 1, new_value)

    def _on_power_min_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_range_param("power_range", 0, new_value)

    def _on_power_max_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_range_param("power_range", 1, new_value)

    def _on_grid_cols_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        self._update_grid_param(0, new_value)

    def _on_grid_rows_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        self._update_grid_param(1, new_value)

    def _on_shape_size_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_param("shape_size", new_value, update_size=True)

    def _on_spacing_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._update_param("spacing", new_value, update_size=True)

    def _on_labels_toggled(self, switch, state):
        self._update_param("include_labels", state)
        return False  # Allow the toggle to proceed

    # Helper methods
    def _calculate_grid_size(
        self, params_dict: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculates the total grid size based on parameters."""
        cols, rows = params_dict.get("grid_dimensions", [5, 5])
        shape_size = params_dict.get("shape_size", 10.0)
        spacing = params_dict.get("spacing", 2.0)

        width = cols * (shape_size + spacing) - spacing
        height = rows * (shape_size + spacing) - spacing

        return width, height

    def _update_import_source_data(self):
        """Updates the ImportSource data with current parameters."""
        import json

        params_dict = self.target_dict.get("params", {})

        # Find the workpiece associated with this step
        if not self.step.doc:
            return

        for layer in self.step.doc.layers:
            for item in layer.children:
                from ....core.workpiece import WorkPiece

                if isinstance(item, WorkPiece):
                    source = item.source
                    if (
                        source
                        and source.metadata.get("type") == "material_test"
                    ):
                        # Update the import source data
                        source.data = json.dumps(params_dict).encode("utf-8")
                        source.original_data = source.data
                        # Clear render cache so it re-renders
                        item.clear_render_cache()
                        # Signal workpiece updated to trigger redraw
                        item.updated.send(item)
                        break

    def _update_workpiece_size(self):
        """Updates the workpiece size based on current grid parameters."""
        params_dict = self.target_dict.get("params", {})
        width, height = self._calculate_grid_size(params_dict)

        # Find the workpiece associated with this step
        if not self.step.doc:
            return

        for layer in self.step.doc.layers:
            for item in layer.children:
                from ....core.workpiece import WorkPiece

                if isinstance(item, WorkPiece):
                    source = item.source
                    if (
                        source
                        and source.metadata.get("type") == "material_test"
                    ):
                        # Update workpiece size for new grid dimensions
                        item.set_size(width, height)
                        break

    def _update_param(
        self, param_name: str, new_value: Any, update_size: bool = False
    ):
        """Updates a simple parameter."""
        params_dict = self.target_dict.setdefault("params", {})

        if params_dict.get(param_name) == new_value:
            return

        def on_change():
            self._update_import_source_data()
            if update_size:
                self._update_workpiece_size()
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
            self._update_import_source_data()
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
        """Updates grid dimensions and recalculates workpiece size."""
        params_dict = self.target_dict.setdefault("params", {})
        current_grid = list(params_dict.get("grid_dimensions", [5, 5]))
        current_grid[index] = new_value

        def on_change():
            self._update_import_source_data()
            self._update_workpiece_size()
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
        # Find the main window through the document
        from ....mainwindow import MainWindow

        # Get the root window
        root = self.get_root()
        if not isinstance(root, MainWindow):
            return

        action = root.action_manager.get_action("view_mode")
        if not action:
            return

        state = action.get_state()
        if state is None:
            return

        current_mode = state.get_string()
        if current_mode != "preview":
            return

        # Switch back to 2D view
        action.change_state(GLib.Variant.new_string("2d"))
        # Safely call the method if it exists
        view_mode_changed = getattr(root, "on_view_mode_changed", None)
        if callable(view_mode_changed):
            view_mode_changed(action, GLib.Variant.new_string("2d"))
