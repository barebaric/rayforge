from gi.repository import Gtk, Adw  # type: ignore
from ..models.laser import Laser
from ...shared.util.adwfix import get_spinrow_int, get_spinrow_float


class LaserPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Laser Heads"),
            icon_name="preferences-other-symbolic",
            **kwargs,
        )
        self.machine = machine

        # List of Lasers
        laserhead_list_group = Adw.PreferencesGroup(
            title=_("Laser Heads"),
            description=_(
                "You can configure multiple laser heads, but "
                "only the first in the list will be used because "
                "Rayforge does not support multi-head operations yet."
            ),
        )
        self.add(laserhead_list_group)
        self.laserhead_list = Gtk.ListBox()
        self.laserhead_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.laserhead_list.set_show_separators(True)
        laserhead_list_group.add(self.laserhead_list)

        # Add and Remove buttons (right-aligned)
        button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=5,
            halign=Gtk.Align.END,
        )
        add_button = Gtk.Button(icon_name="list-add-symbolic")
        add_button.connect("clicked", self.on_add_laserhead)
        remove_button = Gtk.Button(icon_name="list-remove-symbolic")
        remove_button.connect("clicked", self.on_remove_laserhead)
        button_box.append(add_button)
        button_box.append(remove_button)
        laserhead_list_group.add(button_box)

        # Configuration panel for the selected Laser
        self.laserhead_config_group = Adw.PreferencesGroup(
            title=_("Laser Properties"),
            description=_("Configure the selected laser"),
        )
        self.add(self.laserhead_config_group)

        max_power_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.max_power_row = Adw.SpinRow(
            title=_("Max Power"),
            subtitle=_("Maximum power value in GCode"),
            adjustment=max_power_adjustment,
        )
        max_power_adjustment.set_value(0)
        self.max_power_row.connect("changed", self.on_max_power_changed)
        self.laserhead_config_group.add(self.max_power_row)

        frame_power_adjustment = Gtk.Adjustment(
            lower=0, upper=100, step_increment=1, page_increment=10
        )
        self.frame_power_row = Adw.SpinRow(
            title=_("Frame Power"),
            subtitle=_(
                "Power value in Gcode to use when framing. 0 to disable"
            ),
            adjustment=frame_power_adjustment,
        )
        frame_power_adjustment.set_value(0)
        self.frame_power_row.connect("changed", self.on_frame_power_changed)
        self.laserhead_config_group.add(self.frame_power_row)

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
        self.spot_size_x_row.connect("changed", self.on_spot_size_changed)
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
        self.spot_size_y_row.connect("changed", self.on_spot_size_changed)
        self.laserhead_config_group.add(self.spot_size_y_row)

        # Connect signals
        self.laserhead_list.connect("row-selected", self.on_laserhead_selected)

        # Populate the list with existing Lasers
        self.refresh_laserhead_list()

    def refresh_laserhead_list(self):
        """
        Refresh the list of Lasers, preserving selection and avoiding flicker.
        """
        selected_index = -1
        selected_row = self.laserhead_list.get_selected_row()
        if selected_row:
            selected_index = selected_row.get_index()

        # Get current number of rows
        row_count = 0
        while self.laserhead_list.get_row_at_index(row_count):
            row_count += 1

        # Update or add rows to match machine.heads
        for i, head in enumerate(self.machine.heads):
            if i < row_count:
                # Update existing row
                row = self.laserhead_list.get_row_at_index(i)
                row.set_title(
                    _("Laser (Max Power: {head.max_power})").format(head=head)
                )
            else:
                # Add new row
                row = Adw.ActionRow(
                    title=_("Laser (Max Power: {head.max_power})").format(
                        head=head
                    )
                )
                row.set_margin_top(5)
                row.set_margin_bottom(5)
                self.laserhead_list.append(row)

        # Remove extra rows
        while row_count > len(self.machine.heads):
            last_row = self.laserhead_list.get_row_at_index(row_count - 1)
            self.laserhead_list.remove(last_row)
            row_count -= 1

        # Restore selection
        if selected_index >= 0 and selected_index < len(self.machine.heads):
            row = self.laserhead_list.get_row_at_index(selected_index)
            self.laserhead_list.select_row(row)
        elif len(self.machine.heads) > 0:
            row = self.laserhead_list.get_row_at_index(0)
            self.laserhead_list.select_row(row)

    def on_add_laserhead(self, button):
        """Add a new Laser to the machine."""
        new_head = Laser()
        self.machine.add_head(new_head)
        self.refresh_laserhead_list()
        # Select the newly added laser
        last_index = len(self.machine.heads) - 1
        last_row = self.laserhead_list.get_row_at_index(last_index)
        self.laserhead_list.select_row(last_row)

    def on_remove_laserhead(self, button):
        """Remove the selected Laser from the machine."""
        selected_row = self.laserhead_list.get_selected_row()
        if selected_row:
            index = selected_row.get_index()
            head = self.machine.heads[index]
            self.machine.remove_head(head)
            self.refresh_laserhead_list()

    def on_laserhead_selected(self, listbox, row):
        """Update the configuration panel when a Laser is selected."""
        if row is not None:
            index = row.get_index()
            selected_head = self.machine.heads[index]
            self.max_power_row.set_value(selected_head.max_power)
            self.frame_power_row.set_value(selected_head.frame_power)
            spot_x, spot_y = selected_head.spot_size_mm
            self.spot_size_x_row.set_value(spot_x)
            self.spot_size_y_row.set_value(spot_y)

    def _get_selected_laser(self):
        selected_row = self.laserhead_list.get_selected_row()
        if not selected_row:
            return None
        index = selected_row.get_index()
        return self.machine.heads[index]

    def on_max_power_changed(self, spinrow):
        """Update the max power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_max_power(get_spinrow_int(spinrow))
        self.refresh_laserhead_list()

    def on_frame_power_changed(self, spinrow):
        """Update the max power of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        selected_laser.set_frame_power(get_spinrow_int(spinrow))
        self.refresh_laserhead_list()

    def on_spot_size_changed(self, spinrow):
        """Update the spot size of the selected Laser."""
        selected_laser = self._get_selected_laser()
        if not selected_laser:
            return
        x = get_spinrow_float(self.spot_size_x_row)
        y = get_spinrow_float(self.spot_size_y_row)
        selected_laser.set_spot_size(x, y)
        self.refresh_laserhead_list()
