from gi.repository import Gtk, Adw, Gdk
from ...machine.models.machine import Machine
from ...machine.cmd import MachineCmd
from .jog_widget import JogWidget
from ..shared.adwfix import get_spinrow_int, get_spinrow_float


class JogDialog(Adw.Window):
    """Dialog for manually jogging the machine."""

    def __init__(self, *, machine: Machine, machine_cmd: MachineCmd, **kwargs):
        super().__init__(**kwargs)
        self.machine = machine
        self.machine_cmd = machine_cmd

        self.set_title(_("Machine Jog Control"))
        self.set_default_size(600, 700)  # Made dialog even taller
        self.set_hide_on_close(False)
        self.connect("close-request", self._on_close_request)
        self.connect("show", self._on_show)

        # Add a key controller to close the dialog on Escape press
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # Connect to machine connection status changes
        if self.machine:
            self.machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.machine.changed.connect(self._on_machine_changed)

        # Create a vertical box to hold the header bar and the content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Add a header bar for title and window controls (like close)
        header = Adw.HeaderBar()
        main_box.append(header)

        # The main content area should be scrollable
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        scrolled_window.set_vexpand(True)  # Allow the scrolled area to grow
        main_box.append(scrolled_window)

        # Create a preferences page and add it to the scrollable area
        page = Adw.PreferencesPage()
        scrolled_window.set_child(page)

        # Homing group
        homing_group = Adw.PreferencesGroup(title=_("Homing"))
        page.add(homing_group)

        # Create a box for home buttons
        home_button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        homing_group.add(home_button_box)

        self.home_x_btn = Gtk.Button(label=_("Home X"))
        self.home_x_btn.add_css_class("pill")
        self.home_x_btn.connect("clicked", self._on_home_x_clicked)
        home_button_box.append(self.home_x_btn)

        self.home_y_btn = Gtk.Button(label=_("Home Y"))
        self.home_y_btn.add_css_class("pill")
        self.home_y_btn.connect("clicked", self._on_home_y_clicked)
        home_button_box.append(self.home_y_btn)

        self.home_z_btn = Gtk.Button(label=_("Home Z"))
        self.home_z_btn.add_css_class("pill")
        self.home_z_btn.connect("clicked", self._on_home_z_clicked)
        home_button_box.append(self.home_z_btn)

        self.home_all_btn = Gtk.Button(label=_("Home All"))
        self.home_all_btn.add_css_class("suggested-action")
        self.home_all_btn.add_css_class("pill")
        self.home_all_btn.connect("clicked", self._on_home_all_clicked)
        home_button_box.append(self.home_all_btn)

        # Create and add the jog widget
        self.jog_widget = JogWidget()
        self.jog_widget.set_machine(machine, machine_cmd)
        page.add(self.jog_widget)

        # Speed control group (moved to bottom)
        speed_group = Adw.PreferencesGroup(title=_("Jog Settings"))
        page.add(speed_group)

        # Speed row
        speed_adjustment = Gtk.Adjustment(
            value=1000, lower=1, upper=10000, step_increment=10
        )
        self.speed_row = Adw.SpinRow(
            title=_("Jog Speed"),
            subtitle=_("Speed in mm/min"),
            adjustment=speed_adjustment,
        )
        self.speed_row.connect("changed", self._on_speed_changed)
        speed_group.add(self.speed_row)

        # Distance row
        distance_adjustment = Gtk.Adjustment(
            value=10.0, lower=0.1, upper=1000, step_increment=1
        )
        self.distance_row = Adw.SpinRow(
            title=_("Jog Distance"),
            subtitle=_("Distance in mm"),
            adjustment=distance_adjustment,
            digits=1,
        )
        self.distance_row.connect("changed", self._on_distance_changed)
        speed_group.add(self.distance_row)

        self._update_button_sensitivity()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events, closing the dialog on Escape or Ctrl+W."""
        has_ctrl = state & Gdk.ModifierType.CONTROL_MASK

        # Gdk.KEY_w covers both lowercase 'w' and uppercase 'W'
        if keyval == Gdk.KEY_Escape or (has_ctrl and keyval == Gdk.KEY_w):
            self.close()
            return True
        return False

    def _on_show(self, widget):
        """Handle dialog show event to set focus to jog widget."""
        self.jog_widget.grab_focus()

    def _on_close_request(self, window):
        """Handle window close request."""
        # Disconnect from machine signals to prevent memory leaks
        if self.machine:
            self.machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self.machine.changed.disconnect(self._on_machine_changed)
        return False  # Allow the window to close

    def _on_speed_changed(self, spin_row):
        """Handle jog speed change."""
        self.jog_widget.jog_speed = get_spinrow_int(spin_row)

    def _on_distance_changed(self, spin_row):
        """Handle jog distance change."""
        self.jog_widget.jog_distance = get_spinrow_float(spin_row)

    def _on_home_x_clicked(self, button):
        """Handle Home X button click."""
        if self.machine and self.machine_cmd:
            from ...machine.driver.driver import Axis

            self.machine_cmd.home(self.machine, Axis.X)

    def _on_home_y_clicked(self, button):
        """Handle Home Y button click."""
        if self.machine and self.machine_cmd:
            from ...machine.driver.driver import Axis

            self.machine_cmd.home(self.machine, Axis.Y)

    def _on_home_z_clicked(self, button):
        """Handle Home Z button click."""
        if self.machine and self.machine_cmd:
            from ...machine.driver.driver import Axis

            self.machine_cmd.home(self.machine, Axis.Z)

    def _on_home_all_clicked(self, button):
        """Handle Home All button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine)

    def _update_button_sensitivity(self):
        """Update button sensitivity based on machine capabilities."""
        has_machine = self.machine is not None
        is_connected = has_machine and self.machine.is_connected()
        single_axis_homing_enabled = (
            has_machine and self.machine.single_axis_homing_enabled
        )

        # Home buttons
        from ...machine.driver.driver import Axis

        self.home_x_btn.set_sensitive(
            is_connected
            and self.machine.can_home(Axis.X)
            and single_axis_homing_enabled
        )
        self.home_y_btn.set_sensitive(
            is_connected
            and self.machine.can_home(Axis.Y)
            and single_axis_homing_enabled
        )
        self.home_z_btn.set_sensitive(
            is_connected
            and self.machine.can_home(Axis.Z)
            and single_axis_homing_enabled
        )
        self.home_all_btn.set_sensitive(is_connected)

        # Update tooltips for single axis home buttons
        tooltip = None
        if single_axis_homing_enabled:
            tooltip = None
        else:
            tooltip = _("Single axis homing is disabled in machine settings")

        self.home_x_btn.set_tooltip_text(tooltip)
        self.home_y_btn.set_tooltip_text(tooltip)
        self.home_z_btn.set_tooltip_text(tooltip)

        # Update jog widget sensitivity
        self.jog_widget._update_button_sensitivity()

    def _on_connection_status_changed(self, machine, status, message=None):
        """
        Handle machine connection status changes to update button
        sensitivity.
        """
        self._update_button_sensitivity()

    def _on_machine_state_changed(self, machine, state):
        """
        Handle machine state changes to update button sensitivity.
        """
        self._update_button_sensitivity()

    def _on_machine_changed(self, machine, **kwargs):
        """
        Handle machine configuration changes to update button sensitivity.
        """
        self._update_button_sensitivity()
