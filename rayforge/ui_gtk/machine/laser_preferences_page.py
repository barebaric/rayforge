from typing import cast
from gi.repository import Gtk, Adw
from ...machine.models.laser import Laser
from ...machine.models.machine import Machine
from ..settings.preferences_group import PreferencesGroupWithButton
from ..shared.adwfix import get_spinrow_int, get_spinrow_float
from ..icons import get_icon


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


class LaserPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Laser Heads"),
            icon_name="preferences-other-symbolic",
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
            description=_("Configure the selected laser"),
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
            lower=0, upper=10000, step_increment=1, page_increment=10
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
        self.laserhead_config_group.add(self.frame_power_row)

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
            upper=0.2,
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
            upper=0.2,
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

        # Connect signals
        self.laser_list_editor.list_box.connect(
            "row-selected", self.on_laserhead_selected
        )
        # Initial state is handled by the initial _on_machine_changed call

    def on_laserhead_selected(self, listbox, row):
        """Update the configuration panel when a Laser is selected."""
        has_selection = row is not None
        self.laserhead_config_group.set_visible(has_selection)
        if has_selection:
            # Block handlers to prevent feedback loop
            self.name_row.handler_block(self.handler_ids["name"])
            self.tool_number_row.handler_block(self.handler_ids["tool_number"])
            self.max_power_row.handler_block(self.handler_ids["max_power"])
            self.frame_power_row.handler_block(self.handler_ids["frame_power"])
            self.focus_power_row.handler_block(self.handler_ids["focus_power"])
            self.spot_size_x_row.handler_block(self.handler_ids["spot_x"])
            self.spot_size_y_row.handler_block(self.handler_ids["spot_y"])

            selected_head = self._get_selected_laser()
            if not selected_head:
                return  # Should not happen if row is selected

            self.name_row.set_text(selected_head.name)
            self.tool_number_row.set_value(selected_head.tool_number)
            self.max_power_row.set_value(selected_head.max_power)
            self.frame_power_row.set_value(
                selected_head.frame_power_percent * 100
            )
            self.focus_power_row.set_value(
                selected_head.focus_power_percent * 100
            )
            spot_x, spot_y = selected_head.spot_size_mm
            self.spot_size_x_row.set_value(spot_x)
            self.spot_size_y_row.set_value(spot_y)

            # Unblock handlers
            self.name_row.handler_unblock(self.handler_ids["name"])
            self.tool_number_row.handler_unblock(
                self.handler_ids["tool_number"]
            )
            self.max_power_row.handler_unblock(self.handler_ids["max_power"])
            self.frame_power_row.handler_unblock(
                self.handler_ids["frame_power"]
            )
            self.focus_power_row.handler_unblock(
                self.handler_ids["focus_power"]
            )
            self.spot_size_x_row.handler_unblock(self.handler_ids["spot_x"])
            self.spot_size_y_row.handler_unblock(self.handler_ids["spot_y"])

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
