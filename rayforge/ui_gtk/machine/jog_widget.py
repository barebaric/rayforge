from gi.repository import Gtk, Adw, Gdk
from typing import Optional
from ...machine.driver.driver import Axis
from ...machine.models.machine import Machine
from ...machine.cmd import MachineCmd
from ..icons import get_icon


class JogWidget(Adw.PreferencesGroup):
    """Widget for manually jogging the machine."""

    def __init__(self, **kwargs):
        super().__init__(title=_("Manual Jog"), **kwargs)
        self.machine: Optional[Machine] = None
        self.machine_cmd: Optional[MachineCmd] = None
        self.jog_speed = 1000  # Default jog speed in mm/min
        self.jog_distance = 10.0  # Default jog distance in mm

        # Make the widget focusable to receive key events
        self.set_focusable(True)

        # Create grid for jog buttons
        jog_grid = Gtk.Grid()
        jog_grid.set_row_spacing(6)
        jog_grid.set_column_spacing(6)
        jog_grid.set_halign(Gtk.Align.CENTER)
        self.add(jog_grid)

        # Create icon for buttons
        def create_icon_button(icon_name, tooltip):
            button = Gtk.Button()
            button.set_size_request(60, 60)
            button.set_tooltip_text(tooltip)
            icon = get_icon(icon_name)
            button.set_child(icon)
            return button

        # Top-left diagonal button
        self.x_minus_y_plus_btn = create_icon_button(
            "arrow-north-west-symbolic", _("Move North-West")
        )
        self.x_minus_y_plus_btn.connect(
            "clicked", self._on_x_minus_y_plus_clicked
        )
        jog_grid.attach(self.x_minus_y_plus_btn, 0, 0, 1, 1)

        # Away/North button
        self.y_plus_btn = create_icon_button(
            "arrow-north-symbolic", _("Move North")
        )
        self.y_plus_btn.connect("clicked", self._on_y_plus_clicked)
        jog_grid.attach(self.y_plus_btn, 1, 0, 1, 1)

        # Top-right diagonal button
        self.x_plus_y_plus_btn = create_icon_button(
            "arrow-north-east-symbolic", _("Move North-East")
        )
        self.x_plus_y_plus_btn.connect(
            "clicked", self._on_x_plus_y_plus_clicked
        )
        jog_grid.attach(self.x_plus_y_plus_btn, 2, 0, 1, 1)

        # Left/West button
        self.x_minus_btn = create_icon_button(
            "arrow-west-symbolic", _("Move West (Left)")
        )
        self.x_minus_btn.connect("clicked", self._on_x_minus_clicked)
        jog_grid.attach(self.x_minus_btn, 0, 1, 1, 1)

        # Right/East button
        self.x_plus_btn = create_icon_button(
            "arrow-east-symbolic", _("Move East (Right)")
        )
        self.x_plus_btn.connect("clicked", self._on_x_plus_clicked)
        jog_grid.attach(self.x_plus_btn, 2, 1, 1, 1)

        # Bottom-left diagonal button
        self.x_minus_y_minus_btn = create_icon_button(
            "arrow-south-west-symbolic", _("Move South-West")
        )
        self.x_minus_y_minus_btn.connect(
            "clicked", self._on_x_minus_y_minus_clicked
        )
        jog_grid.attach(self.x_minus_y_minus_btn, 0, 2, 1, 1)

        # Toward/South button
        self.y_minus_btn = create_icon_button(
            "arrow-south-symbolic", _("Move South")
        )
        self.y_minus_btn.connect("clicked", self._on_y_minus_clicked)
        jog_grid.attach(self.y_minus_btn, 1, 2, 1, 1)

        # Bottom-right diagonal button
        self.x_plus_y_minus_btn = create_icon_button(
            "arrow-south-east-symbolic", _("Move South-East")
        )
        self.x_plus_y_minus_btn.connect(
            "clicked", self._on_x_plus_y_minus_clicked
        )
        jog_grid.attach(self.x_plus_y_minus_btn, 2, 2, 1, 1)

        # Z buttons to the right
        self.z_plus_btn = create_icon_button(
            "arrow-z-up-symbolic", _("Move Up")
        )
        self.z_plus_btn.set_size_request(60, 60)
        self.z_plus_btn.connect("clicked", self._on_z_plus_clicked)
        jog_grid.attach(self.z_plus_btn, 4, 0, 1, 1)

        self.z_minus_btn = create_icon_button(
            "arrow-z-down-symbolic", _("Move Down")
        )
        self.z_minus_btn.set_size_request(60, 60)
        self.z_minus_btn.connect("clicked", self._on_z_minus_clicked)
        jog_grid.attach(self.z_minus_btn, 4, 2, 1, 1)

        # Add key controller for cursor key support
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # Set initial sensitivity
        self._update_button_sensitivity()

    def set_machine(self, machine: Machine, machine_cmd: MachineCmd):
        """Set the machine this widget controls."""
        # Disconnect from previous machine if any
        if self.machine:
            try:
                self.machine.state_changed.disconnect(
                    self._on_machine_state_changed
                )
            except (TypeError, RuntimeError):
                # Signal might not be connected or already disconnected
                pass

        self.machine = machine
        self.machine_cmd = machine_cmd

        # Connect to state changes
        if self.machine:
            self.machine.state_changed.connect(self._on_machine_state_changed)

        self._update_button_sensitivity()
        self._update_limit_status()

    def _update_button_sensitivity(self):
        """Update button sensitivity based on machine capabilities."""
        # Default all buttons to disabled
        self.x_plus_btn.set_sensitive(False)
        self.x_minus_btn.set_sensitive(False)
        self.y_plus_btn.set_sensitive(False)
        self.y_minus_btn.set_sensitive(False)
        self.x_plus_y_plus_btn.set_sensitive(False)
        self.x_minus_y_plus_btn.set_sensitive(False)
        self.x_plus_y_minus_btn.set_sensitive(False)
        self.x_minus_y_minus_btn.set_sensitive(False)
        self.z_plus_btn.set_sensitive(False)
        self.z_minus_btn.set_sensitive(False)

        # Only enable buttons if machine exists, is connected
        if self.machine is None or not self.machine.is_connected():
            return

        # Type assertion to help Pylance understand machine is not None
        machine: Machine = self.machine  # type: ignore

        # Jog buttons
        self.x_plus_btn.set_sensitive(machine.can_jog(Axis.X))
        self.x_minus_btn.set_sensitive(machine.can_jog(Axis.X))
        self.y_plus_btn.set_sensitive(machine.can_jog(Axis.Y))
        self.y_minus_btn.set_sensitive(machine.can_jog(Axis.Y))

        # Diagonal buttons - need both X and Y axis support
        can_jog_xy = machine.can_jog(Axis.X) and machine.can_jog(Axis.Y)
        self.x_plus_y_plus_btn.set_sensitive(can_jog_xy)
        self.x_minus_y_plus_btn.set_sensitive(can_jog_xy)
        self.x_plus_y_minus_btn.set_sensitive(can_jog_xy)
        self.x_minus_y_minus_btn.set_sensitive(can_jog_xy)

        self.z_plus_btn.set_sensitive(machine.can_jog(Axis.Z))
        self.z_minus_btn.set_sensitive(machine.can_jog(Axis.Z))

        self._update_limit_status()

    def _update_limit_status(self):
        """Update button styling based on whether jog would exceed limits."""
        if not self.machine or not self.machine.is_connected():
            return

        machine = self.machine

        buttons = [
            self.x_plus_btn,
            self.x_minus_btn,
            self.y_plus_btn,
            self.y_minus_btn,
            self.z_plus_btn,
            self.z_minus_btn,
            self.x_plus_y_plus_btn,
            self.x_minus_y_plus_btn,
            self.x_plus_y_minus_btn,
            self.x_minus_y_minus_btn,
        ]
        for button in buttons:
            button.remove_css_class("warning")
            button.remove_css_class("destructive-action")

        if not machine.soft_limits_enabled:
            return

        # Get the signed coordinate deltas for visual directions from the model
        x_east, y_north, z_up = machine.get_visual_jog_deltas(
            self.jog_distance
        )

        # Check limits using the final signed delta that will be commanded
        if machine.would_jog_exceed_limits(Axis.X, x_east):
            self.x_plus_btn.add_css_class("warning")
        if machine.would_jog_exceed_limits(Axis.X, -x_east):
            self.x_minus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.Y, y_north):
            self.y_plus_btn.add_css_class("warning")
        if machine.would_jog_exceed_limits(Axis.Y, -y_north):
            self.y_minus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.Z, z_up):
            self.z_plus_btn.add_css_class("warning")
        if machine.would_jog_exceed_limits(Axis.Z, -z_up):
            self.z_minus_btn.add_css_class("warning")

        # Diagonal buttons
        if machine.would_jog_exceed_limits(
            Axis.X, x_east
        ) or machine.would_jog_exceed_limits(Axis.Y, y_north):
            self.x_plus_y_plus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(
            Axis.X, -x_east
        ) or machine.would_jog_exceed_limits(Axis.Y, y_north):
            self.x_minus_y_plus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(
            Axis.X, x_east
        ) or machine.would_jog_exceed_limits(Axis.Y, -y_north):
            self.x_plus_y_minus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(
            Axis.X, -x_east
        ) or machine.would_jog_exceed_limits(Axis.Y, -y_north):
            self.x_minus_y_minus_btn.add_css_class("warning")

    def _on_machine_state_changed(self, machine, state):
        """Handle machine state changes to update limit status."""
        self._update_limit_status()

    def _jog_xy(self, x_dist: float, y_dist: float):
        """Helper to jog X and Y by sending separate commands."""
        if not self.machine or not self.machine_cmd:
            return

        if x_dist != 0:
            self.machine_cmd.jog(self.machine, Axis.X, x_dist, self.jog_speed)
        if y_dist != 0:
            self.machine_cmd.jog(self.machine, Axis.Y, y_dist, self.jog_speed)

    def _on_x_plus_clicked(self, button):
        """Handle Right (East) button click."""
        if self.machine and self.machine_cmd:
            x_dist, _, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self.machine_cmd.jog(self.machine, Axis.X, x_dist, self.jog_speed)

    def _on_x_minus_clicked(self, button):
        """Handle Left (West) button click."""
        if self.machine and self.machine_cmd:
            x_dist, _, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self.machine_cmd.jog(self.machine, Axis.X, -x_dist, self.jog_speed)

    def _on_y_plus_clicked(self, button):
        """Handle Away (North) button click."""
        if self.machine and self.machine_cmd:
            _, y_dist, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self.machine_cmd.jog(self.machine, Axis.Y, y_dist, self.jog_speed)

    def _on_y_minus_clicked(self, button):
        """Handle Toward (South) button click."""
        if self.machine and self.machine_cmd:
            _, y_dist, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self.machine_cmd.jog(self.machine, Axis.Y, -y_dist, self.jog_speed)

    def _on_z_plus_clicked(self, button):
        """Handle Up button click."""
        if self.machine and self.machine_cmd:
            _, _, z_dist = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self.machine_cmd.jog(self.machine, Axis.Z, z_dist, self.jog_speed)

    def _on_z_minus_clicked(self, button):
        """Handle Down button click."""
        if self.machine and self.machine_cmd:
            _, _, z_dist = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self.machine_cmd.jog(self.machine, Axis.Z, -z_dist, self.jog_speed)

    def _on_x_plus_y_plus_clicked(self, button):
        """Handle Right-Away diagonal button click."""
        if self.machine:
            x_dist, y_dist, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self._jog_xy(x_dist, y_dist)

    def _on_x_minus_y_plus_clicked(self, button):
        """Handle Left-Away diagonal button click."""
        if self.machine:
            x_dist, y_dist, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self._jog_xy(-x_dist, y_dist)

    def _on_x_plus_y_minus_clicked(self, button):
        """Handle Right-Toward diagonal button click."""
        if self.machine:
            x_dist, y_dist, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self._jog_xy(x_dist, -y_dist)

    def _on_x_minus_y_minus_clicked(self, button):
        """Handle Left-Toward diagonal button click."""
        if self.machine:
            x_dist, y_dist, _ = self.machine.get_visual_jog_deltas(
                self.jog_distance
            )
            self._jog_xy(-x_dist, -y_dist)
        return False

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events for cursor key jogging."""
        if not self.machine or not self.machine.is_connected():
            return False

        # Map cursor keys to jog actions
        if keyval == Gdk.KEY_Up:
            self._on_y_plus_clicked(None)  # Away
            return True
        elif keyval == Gdk.KEY_Down:
            self._on_y_minus_clicked(None)  # Toward
            return True
        elif keyval == Gdk.KEY_Left:
            self._on_x_minus_clicked(None)  # Left
            return True
        elif keyval == Gdk.KEY_Right:
            self._on_x_plus_clicked(None)  # Right
            return True
        elif keyval == Gdk.KEY_Page_Up:
            self._on_z_plus_clicked(None)  # Up
            return True
        elif keyval == Gdk.KEY_Page_Down:
            self._on_z_minus_clicked(None)  # Down
            return True

        return False
