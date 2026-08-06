from gettext import gettext as _
from pathlib import Path
from typing import List, Optional, cast

from gi.repository import Adw, Gdk, Gtk
from raygeo.ops.state import CoolantMode

from ...context import get_context
from ...core.model import Model
from ...machine.models.head import Head
from ...machine.models.laser import LaserHead, LaserType
from ...machine.models.machine import Machine
from ...machine.models.spindle import SpindleHead
from ...shared.util.glib import DebounceMixin
from ..icons import get_icon
from ..shared.model_selection_dialog import ModelSelectionDialog
from ..shared.pref_rows.angle_spin_row import AngleSpinRow
from ..shared.pref_rows.base import SpinRow
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.pref_rows.speed_spin_row import SpeedSpinRow
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage
from ..sim3d.renderer.model_renderer import get_model_extent


class HeadRow(Gtk.Box):
    """A widget representing a single head in the ListBox."""

    def __init__(self, machine: Machine, head: Head):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.machine = machine
        self.head = head
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
            label=self.head.name,
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
        """Generates the subtitle text from head properties."""
        if isinstance(self.head, SpindleHead):
            return _("Tool {tool_number}, {min_rpm}-{max_rpm} rpm").format(
                tool_number=self.head.tool_number,
                min_rpm=self.head.min_rpm,
                max_rpm=self.head.max_rpm,
            )
        if isinstance(self.head, LaserHead):
            spot_x, spot_y = self.head.spot_size_mm
            spot_x_str = f"{spot_x:.2f}".rstrip("0").rstrip(".")
            spot_y_str = f"{spot_y:.2f}".rstrip("0").rstrip(".")

            return _(
                "Tool {tool_number}, max power {max_power}, "
                "spot size {spot_x}x{spot_y}"
            ).format(
                tool_number=self.head.tool_number,
                max_power=self.head.max_power,
                spot_x=spot_x_str,
                spot_y=spot_y_str,
            )
        return _("Tool {tool_number}").format(
            tool_number=self.head.tool_number
        )

    def _on_remove_clicked(self, button: Gtk.Button):
        """Asks the machine to remove the associated head."""
        self.machine.remove_head(self.head)


class HeadListEditor(PreferencesGroupWithButton):
    """
    An Adwaita widget for displaying and managing the machine's heads.
    """

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            button_label=_("Add New Head"),
            selection_mode=Gtk.SelectionMode.SINGLE,
            **kwargs,
        )
        self.machine = machine
        self._setup_ui()
        self.machine.changed.connect(self._on_machine_changed)
        self._on_machine_changed(self.machine)  # Initial population

    def _setup_ui(self):
        """Configures the widget's list box and placeholder."""
        placeholder = Gtk.Label(
            label=_("No heads configured"),
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
        selected_head = None
        selected_row = self.list_box.get_selected_row()
        if selected_row:
            head_row = cast(HeadRow, selected_row.get_child())
            selected_head = head_row.head

        # Get current number of rows
        row_count = 0
        while self.list_box.get_row_at_index(row_count):
            row_count += 1

        # Update or add rows to match machine.heads
        new_selection_index = -1
        for i, head in enumerate(self.machine.heads):
            if head == selected_head:
                new_selection_index = i

            if i < row_count:
                # Update existing row
                row = self.list_box.get_row_at_index(i)
                if not row:
                    continue
                head_row = cast(HeadRow, row.get_child())
                head_row.head = head
                head_row.title_label.set_label(head.name)
                head_row.subtitle_label.set_label(
                    head_row._get_subtitle_text()
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

        # Enforce at least one head by managing delete button sensitivity.
        can_delete = len(self.machine.heads) > 1
        tooltip = None if can_delete else _("At least one head is required")
        current_row_index = 0
        while True:
            row = self.list_box.get_row_at_index(current_row_index)
            if not row:
                break
            head_row = cast(HeadRow, row.get_child())
            head_row.delete_button.set_sensitive(can_delete)
            head_row.delete_button.set_tooltip_text(tooltip)
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

    def create_row_widget(self, item: Head) -> Gtk.Widget:
        """Creates a HeadRow for the given head item."""
        return HeadRow(self.machine, item)

    def _create_add_button(self, button_label: str) -> Gtk.Widget:
        """Creates a MenuButton with a chooser for the head type."""
        menu_btn = Gtk.MenuButton()
        content = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            halign=Gtk.Align.CENTER,
            margin_top=10,
            margin_end=12,
            margin_bottom=10,
            margin_start=12,
        )
        content.append(get_icon("add-symbolic"))
        content.append(Gtk.Label(label=button_label))
        menu_btn.set_child(content)

        popover = Gtk.Popover()
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        laser_button = Gtk.Button(label=_("Laser"))
        spindle_button = Gtk.Button(label=_("Spindle"))
        laser_button.add_css_class("flat")
        spindle_button.add_css_class("flat")
        laser_button.connect(
            "clicked",
            lambda *args: self._add_head(LaserHead(), _("New Laser")),
        )
        spindle_button.connect(
            "clicked",
            lambda *args: self._add_head(SpindleHead(), _("New Spindle")),
        )
        vbox.append(laser_button)
        vbox.append(spindle_button)
        popover.set_child(vbox)
        menu_btn.set_popover(popover)
        return menu_btn

    def _add_head(self, head: Head, default_name: str):
        """Adds a new head and selects its row."""
        popover = cast(Gtk.MenuButton, self.add_button).get_popover()
        if popover:
            popover.popdown()
        head.name = default_name
        self.machine.add_head(head)

        # The machine.changed signal has already run and updated the UI.
        # Now, select the newly added row, which is the last one.
        new_row_index = len(self.machine.heads) - 1
        if new_row_index >= 0:
            row = self.list_box.get_row_at_index(new_row_index)
            self.list_box.select_row(row)


class HeadModelGroup(Adw.PreferencesGroup):
    """3D model and transform editing shared by all head types."""

    def __init__(self):
        super().__init__(
            title=_("3D Model"),
            description=_("Select and configure a 3D model for this head."),
        )
        self._head: Optional[Head] = None
        self._setup_ui()

    def _setup_ui(self):
        """Builds the model and transform rows."""
        self.model_row = Adw.ActionRow(
            title=_("Model"),
            activatable=True,
        )
        self.model_row.connect("activated", self._on_model_activated)
        self.model_row.add_suffix(get_icon("go-next-symbolic"))
        self.add(self.model_row)

        self.scale_row = SpinRow(
            _("Scale"),
            _("Uniform scale factor for the model"),
            lower=0.01,
            upper=1000,
            digits=2,
        )
        self.scale_row.value_changed.connect(self._on_scale_changed)
        self.add(self.scale_row)

        self.rx_row = AngleSpinRow(
            _("X Rotation"),
            _("Degrees around the X axis"),
        )
        self.rx_row.value_changed.connect(self._on_rotation_changed)
        self.add(self.rx_row)

        self.ry_row = AngleSpinRow(
            _("Y Rotation"),
            _("Degrees around the Y axis"),
        )
        self.ry_row.value_changed.connect(self._on_rotation_changed)
        self.add(self.ry_row)

        self.rz_row = AngleSpinRow(
            _("Z Rotation"),
            _("Degrees around the Z axis"),
        )
        self.rz_row.value_changed.connect(self._on_rotation_changed)
        self.add(self.rz_row)

    def set_head(self, head: Optional[Head]):
        """Syncs the rows with the given head."""
        self._head = head
        if head is None:
            return
        self.scale_row.set_value(head.get_scale())
        rx, ry, rz = head.get_rotation()
        self.rx_row.set_value(rx)
        self.ry_row.set_value(ry)
        self.rz_row.set_value(rz)
        self._update_model_subtitle(head)

    def _update_model_subtitle(self, head: Head):
        if head.model_path:
            model_mgr = get_context().model_mgr
            model = Model.from_path(Path(head.model_path))
            resolved = model_mgr.resolve(model)
            if resolved:
                self.model_row.set_subtitle(resolved.stem)
                return
        self.model_row.set_subtitle(_("None"))

    def _on_model_activated(self, row):
        head = self._head
        if not head:
            return

        root = self.get_root()
        dialog = ModelSelectionDialog(
            current_model_path=head.model_path,
            transient_for=cast(Gtk.Window, root) if root else None,
        )

        def on_response(d, response_id):
            if response_id != "select":
                d.destroy()
                return
            selected_path = d.get_selected_model_path()
            if selected_path != head.model_path:
                head.set_model_path(selected_path)
                if selected_path is not None:
                    self._apply_model_scale(head, selected_path)
            self._update_model_subtitle(head)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _apply_model_scale(self, head: Head, model_path: str):
        resolved = get_context().model_mgr.resolve(
            Model.from_path(Path(model_path))
        )
        if resolved is None:
            return
        extent = get_model_extent(resolved)
        if extent and extent > 1e-6:
            head.set_scale(40.0 / extent)
            self.scale_row.set_value(head.get_scale())

    def _on_scale_changed(self, _spinrow):
        if self._head:
            self._head.set_scale(self.scale_row.get_value())

    def _on_rotation_changed(self, _spinrow):
        if not self._head:
            return
        rx = self.rx_row.get_value()
        ry = self.ry_row.get_value()
        rz = self.rz_row.get_value()
        self._head.set_rotation(rx, ry, rz)


class LaserHeadDetailWidget(DebounceMixin):
    """Owns the PreferencesGroups for editing a LaserHead."""

    def __init__(self):
        super().__init__()
        self._head: Optional[LaserHead] = None
        self._handler_ids = {}
        self._laser_type_values = [
            LaserType.DIODE,
            LaserType.CO2,
            LaserType.FIBER,
        ]

        self.properties_group = Adw.PreferencesGroup(
            title=_("Laser Properties"),
            description=_("Configure the selected laser head."),
        )
        self.pwm_group = Adw.PreferencesGroup(
            title=_("PWM"),
            description=_(
                "Pulse Width Modulation settings for frequency "
                "and pulse width control."
            ),
        )
        self.frame_group = Adw.PreferencesGroup(
            title=_("Framing"),
            description=_(
                "Settings for the frame outline operation that "
                "traces the job boundary."
            ),
        )
        self.model_group = HeadModelGroup()
        self.groups: List[Adw.PreferencesGroup] = [
            self.properties_group,
            self.pwm_group,
            self.frame_group,
            self.model_group,
        ]
        self._build_ui()

    def _build_ui(self):
        """Builds the laser-specific rows."""
        self.name_row = Adw.EntryRow(title=_("Name"))
        self._handler_ids["name"] = self.name_row.connect(
            "changed", self._on_name_changed
        )
        self.properties_group.add(self.name_row)

        self.tool_number_row = SpinRow(
            _("Tool Number"),
            _("G-code tool number (e.g., T0, T1)"),
            lower=-32768,
            upper=65535,
            page_increment=1,
            value=0,
        )
        self.tool_number_row.value_changed.connect(
            self._on_tool_number_changed
        )
        self.properties_group.add(self.tool_number_row)

        laser_type_store = Gtk.StringList()
        laser_type_store.append(_("Diode"))
        laser_type_store.append(_("CO₂"))
        laser_type_store.append(_("Fiber"))
        self.laser_type_row = Adw.ComboRow(
            title=_("Laser Type"),
            subtitle=_("Type of laser tube or diode"),
            model=laser_type_store,
        )
        self._handler_ids["laser_type"] = self.laser_type_row.connect(
            "notify::selected", self._on_laser_type_changed
        )
        self.properties_group.add(self.laser_type_row)

        self.max_power_row = SpinRow(
            _("Max Power"),
            _("Maximum power value in GCode"),
            upper=100000,
            value=0,
        )
        self.max_power_row.value_changed.connect(self._on_max_power_changed)
        self.properties_group.add(self.max_power_row)

        self.focus_power_row = SpinRow(
            _("Focus Power"),
            _("Power value in percent to use when focusing. 0 to disable"),
            upper=100,
            step_increment=0.1,
            digits=2,
            value=0,
        )
        self.focus_power_row.value_changed.connect(
            self._on_focus_power_changed
        )
        self.properties_group.add(self.focus_power_row)

        self.spot_size_x_row = LengthSpinRow(
            _("Spot Size X"),
            _("Size of the laser spot in the X direction"),
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            page_increment=0.05,
            max_value_in_base=10.0,
            value_in_base=0.1,
        )
        self.spot_size_x_row.value_changed.connect(self._on_spot_size_changed)
        self.properties_group.add(self.spot_size_x_row)

        self.spot_size_y_row = LengthSpinRow(
            _("Spot Size Y"),
            _("Size of the laser spot in the Y direction"),
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            page_increment=0.05,
            max_value_in_base=10.0,
            value_in_base=0.1,
        )
        self.spot_size_y_row.value_changed.connect(self._on_spot_size_changed)
        self.properties_group.add(self.spot_size_y_row)

        self.cut_color_button = Gtk.ColorButton()
        self.cut_color_button.set_size_request(32, 32)
        self.cut_color_row = Adw.ActionRow(
            title=_("Cut Color"),
            subtitle=_("Color for cutting operations"),
            activatable_widget=self.cut_color_button,
        )
        self.cut_color_row.add_suffix(self.cut_color_button)
        self._handler_ids["cut_color"] = self.cut_color_button.connect(
            "color-set", self._on_cut_color_changed
        )
        self.properties_group.add(self.cut_color_row)

        self.raster_color_button = Gtk.ColorButton()
        self.raster_color_button.set_size_request(32, 32)
        self.raster_color_row = Adw.ActionRow(
            title=_("Raster Color"),
            subtitle=_("Color for engraving/raster operations"),
            activatable_widget=self.raster_color_button,
        )
        self.raster_color_row.add_suffix(self.raster_color_button)
        self._handler_ids["raster_color"] = self.raster_color_button.connect(
            "color-set", self._on_raster_color_changed
        )
        self.properties_group.add(self.raster_color_row)

        self.focal_distance_row = LengthSpinRow(
            _("Focal Distance"),
            _("Distance from the laser head to the work surface (Z offset)"),
            upper=10000,
            max_value_in_base=10000.0,
            value_in_base=0,
        )
        self.focal_distance_row.value_changed.connect(
            self._on_focal_distance_changed
        )
        self.properties_group.add(self.focal_distance_row)

        self.pwm_frequency_row = SpinRow(
            _("PWM Frequency"),
            _("Default PWM frequency in Hz"),
            lower=1,
            upper=100000,
            step_increment=100,
        )
        self.pwm_frequency_row.value_changed.connect(
            self._on_pwm_frequency_changed
        )
        self.pwm_group.add(self.pwm_frequency_row)

        self.max_pwm_frequency_row = SpinRow(
            _("Max PWM Frequency"),
            _("Maximum supported PWM frequency in Hz"),
            lower=1,
            upper=100000,
            step_increment=100,
        )
        self.max_pwm_frequency_row.value_changed.connect(
            self._on_max_pwm_frequency_changed
        )
        self.pwm_group.add(self.max_pwm_frequency_row)

        self.pulse_width_row = SpinRow(
            _("Pulse Width"),
            _("Default pulse width in µs"),
            lower=1,
            upper=100000,
        )
        self.pulse_width_row.value_changed.connect(
            self._on_pulse_width_changed
        )
        self.pwm_group.add(self.pulse_width_row)

        self.min_pulse_width_row = SpinRow(
            _("Min Pulse Width"),
            _("Minimum pulse width in µs"),
            lower=1,
            upper=100000,
        )
        self.min_pulse_width_row.value_changed.connect(
            self._on_min_pulse_width_changed
        )
        self.pwm_group.add(self.min_pulse_width_row)

        self.max_pulse_width_row = SpinRow(
            _("Max Pulse Width"),
            _("Maximum pulse width in µs"),
            lower=1,
            upper=100000,
        )
        self.max_pulse_width_row.value_changed.connect(
            self._on_max_pulse_width_changed
        )
        self.pwm_group.add(self.max_pulse_width_row)

        self.frame_power_row = SpinRow(
            _("Frame Power"),
            _("Power value in percent to use when framing. 0 to disable"),
            upper=100,
            step_increment=0.1,
            digits=2,
            value=0,
        )
        self.frame_power_row.value_changed.connect(
            self._on_frame_power_changed
        )
        self.frame_group.add(self.frame_power_row)

        self.frame_speed_row = SpeedSpinRow(
            _("Frame Speed"),
            _(
                "Speed for frame outline. Leave at 0 to use "
                "the machine's max travel speed"
            ),
            upper=60000,
            digits=0,
        )
        self.frame_speed_row.value_changed.connect(
            self._on_frame_speed_changed
        )
        self.frame_group.add(self.frame_speed_row)

        self.frame_repeat_row = SpinRow(
            _("Repeat Count"),
            _("Number of times to trace the frame outline"),
            lower=1,
            upper=100,
            page_increment=5,
            value=1,
        )
        self.frame_repeat_row.value_changed.connect(
            self._on_frame_repeat_changed
        )
        self.frame_group.add(self.frame_repeat_row)

        self.frame_corner_pause_row = SpinRow(
            _("Pause at Corners"),
            _(
                "Pause duration in seconds at each corner "
                "of the frame outline. 0 to disable"
            ),
            upper=10,
            step_increment=0.1,
            digits=1,
            value=0,
        )
        self.frame_corner_pause_row.value_changed.connect(
            self._on_frame_corner_pause_changed
        )
        self.frame_group.add(self.frame_corner_pause_row)

    def set_head(self, head: Optional[LaserHead]):
        """Syncs the laser rows with the given head."""
        self._head = head
        self.model_group.set_head(head)
        if head is None:
            for group in self.groups:
                group.set_visible(False)
            return
        for group in self.groups:
            group.set_visible(True)

        # Block handlers to prevent feedback loop
        self.name_row.handler_block(self._handler_ids["name"])
        self.laser_type_row.handler_block(self._handler_ids["laser_type"])
        self.cut_color_button.handler_block(self._handler_ids["cut_color"])
        self.raster_color_button.handler_block(
            self._handler_ids["raster_color"]
        )

        self.name_row.set_text(head.name)
        self.tool_number_row.set_value(head.tool_number)
        self.max_power_row.set_value(head.max_power)
        self.focus_power_row.set_value(head.focus_power_percent * 100)
        spot_x, spot_y = head.spot_size_mm
        self.spot_size_x_row.set_value_in_base_units(spot_x)
        self.spot_size_y_row.set_value_in_base_units(spot_y)
        self._set_color_button(self.cut_color_button, head.cut_color)
        self._set_color_button(self.raster_color_button, head.raster_color)
        self.focal_distance_row.set_value_in_base_units(head.focal_distance)
        self.frame_power_row.set_value(head.frame_power_percent * 100)
        self.frame_speed_row.set_value_in_base_units(head.frame_speed)
        self.frame_repeat_row.set_value(head.frame_repeat_count)
        self.frame_corner_pause_row.set_value(head.frame_corner_pause)

        try:
            type_idx = self._laser_type_values.index(head.laser_type)
        except ValueError:
            type_idx = 0
        self.laser_type_row.set_selected(type_idx)

        self.pwm_frequency_row.set_value(head.pwm_frequency)
        self.max_pwm_frequency_row.set_value(head.max_pwm_frequency)
        self.pulse_width_row.set_value(head.pulse_width)
        self.min_pulse_width_row.set_value(head.min_pulse_width)
        self.max_pulse_width_row.set_value(head.max_pulse_width)
        self._update_pwm_visibility()

        # Unblock handlers
        self.name_row.handler_unblock(self._handler_ids["name"])
        self.laser_type_row.handler_unblock(self._handler_ids["laser_type"])
        self.cut_color_button.handler_unblock(self._handler_ids["cut_color"])
        self.raster_color_button.handler_unblock(
            self._handler_ids["raster_color"]
        )

    def _on_name_changed(self, entry_row):
        """Update the name of the selected laser."""
        if self._head:
            self._head.set_name(entry_row.get_text())

    def _on_tool_number_changed(self, spinrow):
        """Update the tool number of the selected laser."""
        if self._head:
            self._head.set_tool_number(spinrow.get_int_value())

    def _on_max_power_changed(self, spinrow):
        """Update the max power of the selected laser."""
        if self._head:
            self._head.set_max_power(spinrow.get_int_value())

    def _on_frame_power_changed(self, spinrow):
        """Update the frame power of the selected laser."""
        if self._head:
            self._head.set_frame_power(spinrow.get_value() / 100)

    def _on_focus_power_changed(self, spinrow):
        """Update the focus power of the selected laser."""
        if self._head:
            self._head.set_focus_power(spinrow.get_value() / 100)

    def _on_spot_size_changed(self, spinrow):
        """Update the spot size of the selected laser."""
        if not self._head:
            return
        x = self.spot_size_x_row.get_value_in_base_units()
        y = self.spot_size_y_row.get_value_in_base_units()
        self._head.set_spot_size(x, y)

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

    def _on_cut_color_changed(self, button: Gtk.ColorButton):
        """Update the cut color of the selected laser."""
        if self._head:
            self._head.set_cut_color(self._get_hex_color(button))

    def _on_raster_color_changed(self, button: Gtk.ColorButton):
        """Update the raster color of the selected laser."""
        if self._head:
            self._head.set_raster_color(self._get_hex_color(button))

    def _on_frame_speed_changed(self, spinrow):
        """Update the frame speed of the selected laser."""
        if not self._head:
            return
        value = self.frame_speed_row.get_value_in_base_units()
        self._head.set_frame_speed(int(value))

    def _on_frame_repeat_changed(self, spinrow):
        """Update the frame repeat count of the selected laser."""
        if self._head:
            self._head.set_frame_repeat_count(spinrow.get_int_value())

    def _on_frame_corner_pause_changed(self, spinrow):
        """Update the frame corner pause of the selected laser."""
        if self._head:
            self._head.set_frame_corner_pause(spinrow.get_value())

    def _on_focal_distance_changed(self, spinrow):
        if self._head:
            self._head.set_focal_distance(
                self.focal_distance_row.get_value_in_base_units()
            )

    def _on_laser_type_changed(self, row, _param):
        if not self._head:
            return
        selected = row.get_selected()
        if selected < len(self._laser_type_values):
            self._head.set_laser_type(self._laser_type_values[selected])
            self._update_pwm_visibility()

    def _update_pwm_visibility(self):
        if self._head is not None:
            show_pwm = self._head.laser_type.supports_pwm
        else:
            show_pwm = False
        self.pwm_group.set_visible(show_pwm)

    def _apply_pwm_fields(self, laser):
        laser.set_max_pwm_frequency(self.max_pwm_frequency_row.get_int_value())
        laser.set_pwm_frequency(self.pwm_frequency_row.get_int_value())
        laser.set_max_pulse_width(self.max_pulse_width_row.get_int_value())
        laser.set_min_pulse_width(self.min_pulse_width_row.get_int_value())
        laser.set_pulse_width(self.pulse_width_row.get_int_value())

    def _on_pwm_frequency_changed(self, spinrow):
        if not self._head:
            return
        value = spinrow.get_int_value()
        max_val = self.max_pwm_frequency_row.get_int_value()
        if value > max_val:
            self.max_pwm_frequency_row.set_value(value)
        self._debounce(self._apply_pwm_fields, self._head)

    def _on_max_pwm_frequency_changed(self, spinrow):
        if not self._head:
            return
        max_val = spinrow.get_int_value()
        freq_val = self.pwm_frequency_row.get_int_value()
        if freq_val > max_val:
            self.pwm_frequency_row.set_value(max_val)
        self._debounce(self._apply_pwm_fields, self._head)

    def _on_pulse_width_changed(self, spinrow):
        if not self._head:
            return
        value = spinrow.get_int_value()
        min_val = self.min_pulse_width_row.get_int_value()
        max_val = self.max_pulse_width_row.get_int_value()
        if value < min_val:
            self.min_pulse_width_row.set_value(value)
        if value > max_val:
            self.max_pulse_width_row.set_value(value)
        self._debounce(self._apply_pwm_fields, self._head)

    def _on_min_pulse_width_changed(self, spinrow):
        if not self._head:
            return
        min_val = spinrow.get_int_value()
        max_val = self.max_pulse_width_row.get_int_value()
        if min_val > max_val:
            self.max_pulse_width_row.set_value(min_val)
        pw_val = self.pulse_width_row.get_int_value()
        if pw_val < min_val:
            self.pulse_width_row.set_value(min_val)
        self._debounce(self._apply_pwm_fields, self._head)

    def _on_max_pulse_width_changed(self, spinrow):
        if not self._head:
            return
        max_val = spinrow.get_int_value()
        min_val = self.min_pulse_width_row.get_int_value()
        if max_val < min_val:
            self.min_pulse_width_row.set_value(max_val)
        pw_val = self.pulse_width_row.get_int_value()
        if pw_val > max_val:
            self.pulse_width_row.set_value(max_val)
        self._debounce(self._apply_pwm_fields, self._head)


class SpindleHeadDetailWidget:
    """Owns the PreferencesGroups for editing a SpindleHead."""

    def __init__(self):
        self._head: Optional[SpindleHead] = None
        self._handler_ids = {}

        self.properties_group = Adw.PreferencesGroup(
            title=_("Spindle Properties"),
            description=_("Configure the selected spindle head."),
        )
        self.model_group = HeadModelGroup()
        self.groups: List[Adw.PreferencesGroup] = [
            self.properties_group,
            self.model_group,
        ]
        self._build_ui()

    def _build_ui(self):
        """Builds the spindle-specific rows."""
        self.name_row = Adw.EntryRow(title=_("Name"))
        self._handler_ids["name"] = self.name_row.connect(
            "changed", self._on_name_changed
        )
        self.properties_group.add(self.name_row)

        self.tool_number_row = SpinRow(
            _("Tool Number"),
            _("G-code tool number (e.g., T0, T1)"),
            lower=-32768,
            upper=65535,
            page_increment=1,
            value=0,
        )
        self.tool_number_row.value_changed.connect(
            self._on_tool_number_changed
        )
        self.properties_group.add(self.tool_number_row)

        self.min_rpm_row = SpinRow(
            _("Min RPM"),
            _("Minimum spindle speed"),
            upper=100000,
            step_increment=100,
            value=1000,
        )
        self.min_rpm_row.value_changed.connect(self._on_min_rpm_changed)
        self.properties_group.add(self.min_rpm_row)

        self.max_rpm_row = SpinRow(
            _("Max RPM"),
            _("Maximum spindle speed"),
            upper=100000,
            step_increment=100,
            value=20000,
        )
        self.max_rpm_row.value_changed.connect(self._on_max_rpm_changed)
        self.properties_group.add(self.max_rpm_row)

        self.flood_cooling_row = Adw.SwitchRow(
            title=_("Supports Flood Coolant"),
            subtitle=_("Coolant delivered to the workpiece as a flood"),
        )
        self._handler_ids["flood_cooling"] = self.flood_cooling_row.connect(
            "notify::active", self._on_cooling_method_toggled
        )
        self.properties_group.add(self.flood_cooling_row)

        self.mist_cooling_row = Adw.SwitchRow(
            title=_("Supports Mist Coolant"),
            subtitle=_("Coolant delivered to the workpiece as a mist"),
        )
        self._handler_ids["mist_cooling"] = self.mist_cooling_row.connect(
            "notify::active", self._on_cooling_method_toggled
        )
        self.properties_group.add(self.mist_cooling_row)

    def set_head(self, head: Optional[SpindleHead]):
        """Syncs the spindle rows with the given head."""
        self._head = head
        self.model_group.set_head(head)
        if head is None:
            for group in self.groups:
                group.set_visible(False)
            return
        for group in self.groups:
            group.set_visible(True)

        self.name_row.handler_block(self._handler_ids["name"])
        self.flood_cooling_row.handler_block(
            self._handler_ids["flood_cooling"]
        )
        self.mist_cooling_row.handler_block(self._handler_ids["mist_cooling"])

        self.name_row.set_text(head.name)
        self.tool_number_row.set_value(head.tool_number)
        self.min_rpm_row.set_value(head.min_rpm)
        self.max_rpm_row.set_value(head.max_rpm)
        self.flood_cooling_row.set_active(
            CoolantMode.FLOOD in head.cooling_methods
        )
        self.mist_cooling_row.set_active(
            CoolantMode.MIST in head.cooling_methods
        )

        self.name_row.handler_unblock(self._handler_ids["name"])
        self.flood_cooling_row.handler_unblock(
            self._handler_ids["flood_cooling"]
        )
        self.mist_cooling_row.handler_unblock(
            self._handler_ids["mist_cooling"]
        )

    def _on_name_changed(self, entry_row):
        """Update the name of the selected spindle."""
        if self._head:
            self._head.set_name(entry_row.get_text())

    def _on_tool_number_changed(self, spinrow):
        """Update the tool number of the selected spindle."""
        if self._head:
            self._head.set_tool_number(spinrow.get_int_value())

    def _on_max_rpm_changed(self, spinrow):
        """Update the max RPM of the selected spindle."""
        if self._head:
            self._head.set_max_rpm(spinrow.get_int_value())

    def _on_min_rpm_changed(self, spinrow):
        """Update the min RPM of the selected spindle."""
        if self._head:
            self._head.set_min_rpm(spinrow.get_int_value())

    def _on_cooling_method_toggled(self, row, _param):
        """Updates the supported coolant methods of the selected spindle."""
        if not self._head:
            return
        methods = set(self._head.cooling_methods)
        method = (
            CoolantMode.FLOOD
            if row is self.flood_cooling_row
            else CoolantMode.MIST
        )
        if row.get_active():
            methods.add(method)
        else:
            methods.discard(method)
        self._head.set_cooling_methods(methods)


class HeadPreferencesPage(TrackedPreferencesPage):
    """Machine settings page for managing all heads."""

    key = "heads"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Heads"),
            icon_name="settings-symbolic",
            **kwargs,
        )
        self.machine = machine

        self.head_list_editor = HeadListEditor(
            machine=self.machine,
            title=_("Heads"),
            description=_(
                "You can configure multiple lasers or spindles if your machine"
                " supports it."
            ),
        )
        self.add(self.head_list_editor)

        self.laser_widget = LaserHeadDetailWidget()
        for group in self.laser_widget.groups:
            self.add(group)

        self.spindle_widget = SpindleHeadDetailWidget()
        for group in self.spindle_widget.groups:
            self.add(group)

        # Connect signals
        self.head_list_editor.list_box.connect(
            "row-selected", self._on_head_selected
        )

        # The initial selection is set inside the HeadListEditor's
        # constructor, which runs before this signal handler is connected.
        # Manually trigger the handler now to sync the UI with the initial
        # state.
        initial_row = self.head_list_editor.list_box.get_selected_row()
        self._on_head_selected(self.head_list_editor.list_box, initial_row)

        self.connect("destroy", self._on_destroy)

    def _get_selected_head(self) -> Optional[Head]:
        selected_row = self.head_list_editor.list_box.get_selected_row()
        if not selected_row:
            return None
        # The child of the ListBoxRow is our custom HeadRow
        head_row = cast(HeadRow, selected_row.get_child())
        return head_row.head

    def _on_head_selected(self, listbox, row):
        """Swap the detail widget based on the selected head's type."""
        head = self._get_selected_head()
        if isinstance(head, LaserHead):
            self.spindle_widget.set_head(None)
            self.laser_widget.set_head(head)
        elif isinstance(head, SpindleHead):
            self.laser_widget.set_head(None)
            self.spindle_widget.set_head(head)
        else:
            self.laser_widget.set_head(None)
            self.spindle_widget.set_head(None)

    def _on_destroy(self, *args):
        """Disconnects signals to prevent memory leaks."""
        self.machine.changed.disconnect(
            self.head_list_editor._on_machine_changed
        )
