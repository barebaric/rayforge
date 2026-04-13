from pathlib import Path
from typing import cast
from gettext import gettext as _
from gi.repository import Adw, Gtk, Gdk
from ...context import get_context
from ...core.model import Model
from ...machine.models.laser import Laser
from ...machine.models.machine import Machine
from ..icons import get_icon
from ..shared.adwfix import get_spinrow_int, get_spinrow_float
from ..shared.unit_spin_row import UnitSpinRowHelper
from ..shared.model_selection_dialog import ModelSelectionDialog
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage
from ..sim3d.canvas3d.model_renderer import get_model_extent


class LaserRow(Gtk.Box):
    """A widget representing a single Laser Head in a ListBox."""

    def __init__(self, machine: Machine, laser: Laser):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.machine = machine
        self.laser = laser
        self.delete_button: Gtk.Button
        self.title_label: Gtk.Label
        self.subtitle_label: Gtk.Label
        self._setup_ui()

    def _setup_ui(self):
        """Builds the user interface for the row."""
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        self.title_label = Gtk.Label(
            label=self.laser.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        labels_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(
            label=self._get_subtitle_text(),
            halign=Gtk.Align.START,
            xalign=0,
            wrap=True,
        )
        self.subtitle_label.add_css_class("dim-label")
        labels_box.append(self.subtitle_label)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.add_css_class("flat")
        self.delete_button.connect("clicked", self._on_remove_clicked)
        self.append(self.delete_button)

    def _get_subtitle_text(self) -> str:
        """Generates the subtitle text from laser properties."""
        spot_x, spot_y = self.laser.spot_size_mm
        spot_x_str = f"{spot_x:.2f}".rstrip("0").rstrip(".")
        spot_y_str = f"{spot_y:.2f}".rstrip("0").rstrip(".")

        return _(
            "Tool {tool_number}, max power {max_power}, "
            "spot size {spot_x}x{spot_y}"
        ).format(
            tool_number=self.laser.tool_number,
            max_power=self.laser.max_power,
            spot_x=spot_x_str,
            spot_y=spot_y_str,
        )

    def _on_remove_clicked(self, button: Gtk.Button):
        """Asks the machine to remove the associated laser head."""
        self.machine.remove_head(self.laser)


class LaserListEditor(PreferencesGroupWithButton):
    """
    An Adwaita widget for displaying and managing a list of laser heads.
    """

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(button_label=_("Add New Laser Head"), **kwargs)
        self.machine = machine
        self._setup_ui()
        self.machine.changed.connect(self._on_machine_changed)
        self._on_machine_changed(self.machine)  # Initial population

    def _setup_ui(self):
        """Configures the widget's list box and placeholder."""
        placeholder = Gtk.Label(
            label=_("No laser heads configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.set_show_separators(True)

    def _on_machine_changed(self, sender: Machine, **kwargs):
        """
        Callback to rebuild the list efficiently when the machine model
        changes.
        """
        selected_laser = None
        selected_row = self.list_box.get_selected_row()
        if selected_row:
            laser_row_widget = cast(LaserRow, selected_row.get_child())
            selected_laser = laser_row_widget.laser

        # Get current number of rows
        row_count = 0
        while self.list_box.get_row_at_index(row_count):
            row_count += 1

        # Update or add rows to match machine.heads
        new_selection_index = -1
        for i, head in enumerate(self.machine.heads):
            if head == selected_laser:
                new_selection_index = i

            if i < row_count:
                # Update existing row
                row = self.list_box.get_row_at_index(i)
                if not row:
                    continue
                laser_row = cast(LaserRow, row.get_child())
                laser_row.laser = head  # Re-assign laser object
                laser_row.title_label.set_label(head.name)
                laser_row.subtitle_label.set_label(
                    laser_row._get_subtitle_text()
                )
            else:
                # Add new row
                list_box_row = Gtk.ListBoxRow()
                list_box_row.set_child(self.create_row_widget(head))
                self.list_box.append(list_box_row)

        # Remove extra rows
        while row_count > len(self.machine.heads):
            last_row = self.list_box.get_row_at_index(row_count - 1)
            if last_row:
                self.list_box.remove(last_row)
            row_count -= 1

        # Enforce at least one laser by managing delete button sensitivity.
        can_delete = len(self.machine.heads) > 1
        tooltip = (
            None if can_delete else _("At least one laser head is required")
        )
        current_row_index = 0
        while True:
            row = self.list_box.get_row_at_index(current_row_index)
            if not row:
                break
            laser_row = cast(LaserRow, row.get_child())
            laser_row.delete_button.set_sensitive(can_delete)
            laser_row.delete_button.set_tooltip_text(tooltip)
            current_row_index += 1

        # Restore selection
        if new_selection_index >= 0:
            row = self.list_box.get_row_at_index(new_selection_index)
            self.list_box.select_row(row)
        elif len(self.machine.heads) > 0:
            row = self.list_box.get_row_at_index(0)
            self.list_box.select_row(row)
        else:
            # Manually trigger selection changed handler for empty state
            if self.list_box.get_selected_row():
                self.list_box.unselect_all()
            else:
                self.list_box.emit("row-selected", None)

    def create_row_widget(self, item: Laser) -> Gtk.Widget:
        """Creates a LaserRow for the given laser item."""
        return LaserRow(self.machine, item)

    def _on_add_clicked(self, button: Gtk.Button):
        """Handles the 'Add New Laser Head' button click."""
        new_head = Laser()
        new_head.name = _("New Laser")
        self.machine.add_head(new_head)

        # The machine.changed signal has already run and updated the UI.
        # Now, select the newly added row, which is the last one.
        new_row_index = len(self.machine.heads) - 1
        if new_row_index >= 0:
            row = self.list_box.get_row_at_index(new_row_index)
            self.list_box.select_row(row)


class LaserPreferencesPage(TrackedPreferencesPage):
    key = "laser"
    path_prefix = "/machine-settings/"

    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Laser Heads"),
            icon_name="settings-symbolic",
            **kwargs,
        )
        self.machine = machine
        self.handler_ids = {}

        # List of Lasers, using the new MacroListEditor-style widget
        self.laser_list_editor = LaserListEditor(
            machine=self.machine,
            title=_("Laser Heads"),
            description=_(
                "You can configure multiple laser heads if your machine "
                "supports it."
            ),
        )
        self.add(self.laser_list_editor)

        # Configuration panel for the selected Laser
        self.laserhead_config_group = Adw.PreferencesGroup(
            title=_("Laser Properties"),
            description=_("Configure the selected laser."),
        )
        self.add(self.laserhead_config_group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.handler_ids["name"] = self.name_row.connect(
            "changed", self.on_name_changed
        )
        self.laserhead_config_group.add(self.name_row)

        tool_number_adjustment = Gtk.Adjustment(
            lower=0, upper=255, step_increment=1, page_increment=1
        )
        self.tool_number_row = Adw.SpinRow(
            title=_("Tool Number"),
            subtitle=_("G-code tool number (e.g., T0, T1)"),
            adjustment=tool_number_adjustment,
        )
        tool_number_adjustment.set_value(0)
        self.handler_ids["tool_number"] = self.tool_number_row.connect(
            "changed", self.on_tool_number_changed
        )
        self.laserhead_config_group.add(self.tool_number_row)

        max_power_adjustment = Gtk.Adjustment(
            lower=0, upper=100000, step_increment=1, page_increment=10
        )
        self.max_power_row = Adw.SpinRow(
            title=_("Max Power"),
            subtitle=_("Maximum power value in GCode"),
            adjustment=max_power_adjustment,
        )
        max_power_adjustment.set_value(0)
        self.handler_ids["max_power"] = self.max_power_row.connect(
            "changed", self.on_max_power_changed
        )
        self.laserhead_config_group.add(self.max_power_row)

        focus_power_adjustment = Gtk.Adjustment(
            lower=0, upper=100, step_increment=0.1, page_increment=1
        )
        self.focus_power_row = Adw.SpinRow(
            title=_("Focus Power"),
            subtitle=_(
                "Power value in percent to use when focusing. 0 to disable"
            ),
            adjustment=focus_power_adjustment,
            digits=2,
        )
        focus_power_adjustment.set_value(0)
        self.handler_ids["focus_power"] = self.focus_power_row.connect(
            "changed", self.on_focus_power_changed
        )
        self.laserhead_config_group.add(self.focus_power_row)

        spot_size_x_adjustment = Gtk.Adjustment(
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            page_increment=0.05,
        )
        self.spot_size_x_row = Adw.SpinRow(
            title=_("Spot Size X"),
            subtitle=_("Size of the laser spot in the X direction"),
            digits=3,
            adjustment=spot_size_x_adjustment,
        )
        spot_size_x_adjustment.set_value(0.1)
        self.handler_ids["spot_x"] = self.spot_size_x_row.connect(
            "changed", self.on_spot_size_changed
        )
        self.laserhead_config_group.add(self.spot_size_x_row)

        spot_size_y_adjustment = Gtk.Adjustment(
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            page_increment=0.05,
        )
        self.spot_size_y_row = Adw.SpinRow(
            title=_("Spot Size Y"),
            subtitle=_("Size of the laser spot in the Y direction"),
            digits=3,
            adjustment=spot_size_y_adjustment,
        )
        spot_size_y_adjustment.set_value(0.1)
        self.handler_ids["spot_y"] = self.spot_size_y_row.connect(
            "changed", self.on_spot_size_changed
        )
        self.laserhead_config_group.add(self.spot_size_y_row)

        self.cut_color_button = Gtk.ColorButton()
        self.cut_color_button.set_size_request(32, 32)
        self.cut_color_row = Adw.ActionRow(
            title=_("Cut Color"),
            subtitle=_("Color for cutting operations"),
            activatable_widget=self.cut_color_button,
        )
        self.cut_color_row.add_suffix(self.cut_color_button)
        self.handler_ids["cut_color"] = self.cut_color_button.connect(
            "color-set", self.on_cut_color_changed
        )
        self.laserhead_config_group.add(self.cut_color_row)

        self.raster_color_button = Gtk.ColorButton()
        self.raster_color_button.set_size_request(32, 32)
        self.raster_color_row = Adw.ActionRow(
            title=_("Raster Color"),
            subtitle=_("Color for engraving/raster operations"),
            activatable_widget=self.raster_color_button,
        )
        self.raster_color_row.add_suffix(self.raster_color_button)
        self.handler_ids["raster_color"] = self.raster_color_button.connect(
            "color-set", self.on_raster_color_changed
        )
        self.laserhead_config_group.add(self.raster_color_row)

        # Framing preferences group
        self.frame_group = Adw.PreferencesGroup(
            title=_("Framing"),
            description=_(
                "Settings for the frame outline operation that "
                "traces the job boundary."
            ),
        )
        self.add(self.frame_group)

        frame_power_adjustment = Gtk.Adjustment(
            lower=0, upper=100, step_increment=0.1, page_increment=1
        )
        self.frame_power_row = Adw.SpinRow(
            title=_("Frame Power"),
            subtitle=_(
                "Power value in percent to use when framing. 0 to disable"
            ),
            adjustment=frame_power_adjustment,
            digits=2,
        )
        frame_power_adjustment.set_value(0)
        self.handler_ids["frame_power"] = self.frame_power_row.connect(
            "changed", self.on_frame_power_changed
        )
        self.frame_group.add(self.frame_power_row)

        frame_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=60000,
            step_increment=10,
            page_increment=100,
        )
        frame_speed_row = Adw.SpinRow(
            title=_("Frame Speed"),
            subtitle=_(
                "Speed for frame outline. Leave at 0 to use "
                "the machine's max travel speed"
            ),
            adjustment=frame_speed_adjustment,
        )
        self.frame_speed_helper = UnitSpinRowHelper(
            spin_row=frame_speed_row,
            quantity="speed",
        )
        self.frame_speed_row = frame_speed_row
        self.handler_ids["frame_speed"] = frame_speed_row.connect(
            "changed", self.on_frame_speed_changed
        )
        self.frame_group.add(frame_speed_row)

        frame_repeat_adjustment = Gtk.Adjustment(
            lower=1, upper=100, step_increment=1, page_increment=5
        )
        self.frame_repeat_row = Adw.SpinRow(
            title=_("Repeat Count"),
            subtitle=_("Number of times to trace the frame outline"),
            adjustment=frame_repeat_adjustment,
        )
        frame_repeat_adjustment.set_value(1)
        self.handler_ids["frame_repeat"] = self.frame_repeat_row.connect(
            "changed", self.on_frame_repeat_changed
        )
        self.frame_group.add(self.frame_repeat_row)

        frame_corner_pause_adjustment = Gtk.Adjustment(
            lower=0, upper=10, step_increment=0.1, page_increment=1
        )
        self.frame_corner_pause_row = Adw.SpinRow(
            title=_("Pause at Corners"),
            subtitle=_(
                "Pause duration in seconds at each corner "
                "of the frame outline. 0 to disable"
            ),
            adjustment=frame_corner_pause_adjustment,
            digits=1,
        )
        frame_corner_pause_adjustment.set_value(0)
        self.handler_ids["frame_corner_pause"] = (
            self.frame_corner_pause_row.connect(
                "changed", self.on_frame_corner_pause_changed
            )
        )
        self.frame_group.add(self.frame_corner_pause_row)

        # Model preferences group
        self.model_group = Adw.PreferencesGroup(
            title=_("3D Model"),
            description=_(
                "Select and configure a 3D model for this laser head."
            ),
        )
        self.add(self.model_group)

        self.model_row = Adw.ActionRow(
            title=_("Model"),
            activatable=True,
        )
        self.model_row.connect("activated", self._on_model_activated)
        self.model_row.add_suffix(get_icon("go-next-symbolic"))
        self.model_group.add(self.model_row)

        self.scale_row = Adw.SpinRow(
            title=_("Scale"),
            subtitle=_("Uniform scale factor for the model"),
            adjustment=Gtk.Adjustment(
                lower=0.01, upper=1000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.handler_ids["scale"] = self.scale_row.connect(
            "notify::value", self._on_scale_changed
        )
        self.model_group.add(self.scale_row)

        self.rx_row = Adw.SpinRow(
            title=_("X Rotation"),
            subtitle=_("Degrees around the X axis"),
            adjustment=Gtk.Adjustment(
                lower=-360, upper=360, step_increment=1, page_increment=15
            ),
            digits=1,
        )
        self.handler_ids["rx"] = self.rx_row.connect(
            "notify::value", self._on_rotation_changed
        )
        self.model_group.add(self.rx_row)

        self.ry_row = Adw.SpinRow(
            title=_("Y Rotation"),
            subtitle=_("Degrees around the Y axis"),
            adjustment=Gtk.Adjustment(
                lower=-360, upper=360, step_increment=1, page_increment=15
            ),
            digits=1,
        )
        self.handler_ids["ry"] = self.ry_row.connect(
            "notify::value", self._on_rotation_changed
        )
        self.model_group.add(self.ry_row)

        self.rz_row = Adw.SpinRow(
            title=_("Z Rotation"),
            subtitle=_("Degrees around the Z axis"),
            adjustment=Gtk.Adjustment(
                lower=-360, upper=360, step_increment=1, page_increment=15
            ),
            digits=1,
        )
        self.handler_ids["rz"] = self.rz_row.connect(
            "notify::value", self._on_rotation_changed
        )
        self.model_group.add(self.rz_row)

        focal_distance_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.focal_distance_row = Adw.SpinRow(
            title=_("Focal Distance"),
            subtitle=_(
                "Distance from the laser head to the work surface (Z offset)"
            ),
            adjustment=focal_distance_adj,
            digits=2,
        )
        focal_distance_adj.set_value(0)
        self.handler_ids["focal_distance"] = self.focal_distance_row.connect(
            "notify::value", self._on_focal_distance_changed
        )
        self.model_group.add(self.focal_distance_row)

        # Connect signals
        self.laser_list_editor.list_box.connect(
            "row-selected", self.on_laserhead_selected
        )

        # The initial selection is set inside the LaserListEditor's
        # constructor, which runs before this signal handler is connected.
        # Manually trigger the handler now to sync the UI with the initial
        # state.
        initial_row = self.laser_list_editor.list_box.get_selected_row()
        self.on_laserhead_selected(
            self.laser_list_editor.list_box, initial_row
        )

    def on_laserhead_selected(self, listbox, row):
        """Update the configuration panel when a Laser is selected."""
        has_selection = row is not None
        self.laserhead_config_group.set_visible(has_selection)
        self.model_group.set_visible(has_selection)
        self.frame_group.set_visible(has_selection)
        if has_selection:
            # Block handlers to prevent feedback loop
            self.name_row.handler_block(self.handler_ids["name"])
            self.tool_number_row.handler_block(self.handler_ids["tool_number"])
            self.max_power_row.handler_block(self.handler_ids["max_power"])
            self.focus_power_row.handler_block(self.handler_ids["focus_power"])
            self.spot_size_x_row.handler_block(self.handler_ids["spot_x"])
            self.spot_size_y_row.handler_block(self.handler_ids["spot_y"])
            self.cut_color_button.handler_block(self.handler_ids["cut_color"])
            self.raster_color_button.handler_block(
                self.handler_ids["raster_color"]
            )
            self.scale_row.handler_block(self.handler_ids["scale"])
            self.rx_row.handler_block(self.handler_ids["rx"])
            self.ry_row.handler_block(self.handler_ids["ry"])
            self.rz_row.handler_block(self.handler_ids["rz"])
            self.focal_distance_row.handler_block(
                self.handler_ids["focal_distance"]
            )
            self.frame_power_row.handler_block(self.handler_ids["frame_power"])
            self.frame_speed_row.handler_block(self.handler_ids["frame_speed"])
            self.frame_repeat_row.handler_block(
                self.handler_ids["frame_repeat"]
            )
            self.frame_corner_pause_row.handler_block(
                self.handler_ids["frame_corner_pause"]
            )

            selected_head = self._get_selected_laser()
            if not selected_head:
                return  # Should not happen if row is selected

            self.name_row.set_text(selected_head.name)
            self.tool_number_row.set_value(selected_head.tool_number)
            self.max_power_row.set_value(selected_head.max_power)
            self.focus_power_row.set_value(
                selected_head.focus_power_percent * 100
            )
            spot_x, spot_y = selected_head.spot_size_mm
            self.spot_size_x_row.set_value(spot_x)
            self.spot_size_y_row.set_value(spot_y)
            self._set_color_button(
                self.cut_color_button, selected_head.cut_color
            )
            self._set_color_button(
                self.raster_color_button, selected_head.raster_color
            )
            self._update_model_subtitle(selected_head)
            self.scale_row.set_value(selected_head.get_scale())
            rx, ry, rz = selected_head.get_rotation()
            self.rx_row.set_value(rx)
            self.ry_row.set_value(ry)
            self.rz_row.set_value(rz)
            self.focal_distance_row.set_value(selected_head.focal_distance)
            self.frame_power_row.set_value(
                selected_head.frame_power_percent * 100
            )
            self.frame_speed_helper.set_value_in_base_units(
                selected_head.frame_speed
            )
            self.frame_repeat_row.set_value(selected_head.frame_repeat_count)
            self.frame_corner_pause_row.set_value(
                selected_head.frame_corner_pause
            )

            # Unblock handlers
            self.name_row.handler_unblock(self.handler_ids["name"])
            self.tool_number_row.handler_unblock(
                self.handler_ids["tool_number"]
            )
            self.max_power_row.handler_unblock(self.handler_ids["max_power"])
            self.focus_power_row.handler_unblock(
                self.handler_ids["focus_power"]
            )
            self.spot_size_x_row.handler_unblock(self.handler_ids["spot_x"])
            self.spot_size_y_row.handler_unblock(self.handler_ids["spot_y"])
            self.cut_color_button.handler_unblock(
                self.handler_ids["cut_color"]
            )
            self.raster_color_button.handler_unblock(
                self.handler_ids["raster_color"]
            )
            self.scale_row.handler_unblock(self.handler_ids["scale"])
            self.rx_row.handler_unblock(self.handler_ids["rx"])
            self.ry_row.handler_unblock(self.handler_ids["ry"])
            self.rz_row.handler_unblock(self.handler_ids["rz"])
            self.focal_distance_row.handler_unblock(
                self.handler_ids["focal_distance"]
            )
            self.frame_power_row.handler_unblock(
                self.handler_ids["frame_power"]
            )
            self.frame_speed_row.handler_unblock(
                self.handler_ids["frame_speed"]
            )
            self.frame_repeat_row.handler_unblock(
                self.handler_ids["frame_repeat"]
            )
            self.frame_corner_pause_row.handler_unblock(
                self.handler_ids["frame_corner_pause"]
            )

    def _get_selected_laser(self):
        selected_row = self.laser_list_editor.list_box.get_selected_row()
        if not selected_row:
            return None
        # The child of the ListBoxRow is our custom LaserRow
        laser_row = cast(LaserRow, selected_row.get_child())
        return laser_row.laser

    def on_name_changed(self, entry_row):
        """Update the name of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_name(entry_row.get_text())

    def on_tool_number_changed(self, spinrow):
        """Update the tool number of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_tool_number(get_spinrow_int(spinrow))

    def on_max_power_changed(self, spinrow):
        """Update the max power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_max_power(get_spinrow_int(spinrow))

    def on_frame_power_changed(self, spinrow):
        """Update the frame power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_frame_power(get_spinrow_float(spinrow) / 100)

    def on_focus_power_changed(self, spinrow):
        """Update the focus power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_focus_power(get_spinrow_float(spinrow) / 100)

    def on_spot_size_changed(self, spinrow):
        """Update the spot size of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        x = get_spinrow_float(self.spot_size_x_row)
        y = get_spinrow_float(self.spot_size_y_row)
        selected_laser.set_spot_size(x, y)

    def _set_color_button(self, button: Gtk.ColorButton, hex_color: str):
        """Set the color button from a hex color string."""
        rgba = Gdk.RGBA()
        if not rgba.parse(hex_color):
            rgba.parse("#ff00ff")
        button.set_rgba(rgba)

    def _get_hex_color(self, button: Gtk.ColorButton) -> str:
        """Get the hex color string from a color button."""
        rgba = button.get_rgba()
        r = int(rgba.red * 255)
        g = int(rgba.green * 255)
        b = int(rgba.blue * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def on_cut_color_changed(self, button: Gtk.ColorButton):
        """Update the cut color of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_cut_color(self._get_hex_color(button))

    def on_raster_color_changed(self, button: Gtk.ColorButton):
        """Update the raster color of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_raster_color(self._get_hex_color(button))

    def on_frame_speed_changed(self, spinrow):
        """Update the frame speed of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        value = self.frame_speed_helper.get_value_in_base_units()
        selected_laser.set_frame_speed(int(value))

    def on_frame_repeat_changed(self, spinrow):
        """Update the frame repeat count of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_frame_repeat_count(get_spinrow_int(spinrow))

    def on_frame_corner_pause_changed(self, spinrow):
        """Update the frame corner pause of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_frame_corner_pause(get_spinrow_float(spinrow))

    def _update_model_subtitle(self, laser: Laser):
        if laser.model_path:
            model_mgr = get_context().model_mgr
            model = Model(name="", path=Path(laser.model_path))
            resolved = model_mgr.resolve(model)
            if resolved:
                self.model_row.set_subtitle(resolved.stem)
                return
        self.model_row.set_subtitle(_("None"))

    def _on_model_activated(self, row):
        laser = self._get_selected_laser()
        if not laser:
            return

        root = self.get_root()
        dialog = ModelSelectionDialog(
            current_model_path=laser.model_path,
            transient_for=cast(Gtk.Window, root) if root else None,
        )

        def on_response(d, response_id):
            if response_id != "select":
                d.destroy()
                return
            selected_path = d.get_selected_model_path()
            if selected_path != laser.model_path:
                laser.set_model_path(selected_path)
                if selected_path is not None:
                    self._apply_model_scale(laser, selected_path)
            self._update_model_subtitle(laser)
            self.laser_list_editor._on_machine_changed(self.machine)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _apply_model_scale(self, laser: Laser, model_path: str):
        resolved = get_context().model_mgr.resolve(
            Model(name="", path=Path(model_path))
        )
        if resolved is None:
            return
        extent = get_model_extent(resolved)
        if extent and extent > 1e-6:
            laser.set_scale(40.0 / extent)
            self.scale_row.set_value(laser.get_scale())

    def _on_scale_changed(self, _spinrow, _param):
        selected_laser = self._get_selected_laser()
        if selected_laser:
            selected_laser.set_scale(get_spinrow_float(self.scale_row))

    def _on_rotation_changed(self, _spinrow, _param):
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        rx = get_spinrow_float(self.rx_row)
        ry = get_spinrow_float(self.ry_row)
        rz = get_spinrow_float(self.rz_row)
        selected_laser.set_rotation(rx, ry, rz)

    def _on_focal_distance_changed(self, _spinrow, _param):
        selected_laser = self._get_selected_laser()
        if selected_laser:
            selected_laser.set_focal_distance(
                get_spinrow_float(self.focal_distance_row)
            )
