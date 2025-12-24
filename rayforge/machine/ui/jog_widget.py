from gi.repository import Gtk, Adw, Gdk
from typing import Optional, Tuple
from ...machine.driver.driver import Axis
from ...machine.models.machine import Machine
from ...machine.cmd import MachineCmd
from ...icons import get_icon


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

        # X-Y+ button (top left diagonal)
        self.x_minus_y_plus_btn = create_icon_button(
            "arrow-north-west", _("X- Y+")
        )
        self.x_minus_y_plus_btn.connect(
            "clicked", self._on_x_minus_y_plus_clicked
        )
        jog_grid.attach(self.x_minus_y_plus_btn, 0, 0, 1, 1)

        # Y+ button (top center) - Icon updated in _update_direction_icons
        self.y_plus_btn = create_icon_button("arrow-north", _("Y+"))
        self.y_plus_btn.connect("clicked", self._on_y_plus_clicked)
        jog_grid.attach(self.y_plus_btn, 1, 0, 1, 1)

        # X+Y+ button (top right diagonal)
        self.x_plus_y_plus_btn = create_icon_button(
            "arrow-north-east", _("X+ Y+")
        )
        self.x_plus_y_plus_btn.connect(
            "clicked", self._on_x_plus_y_plus_clicked
        )
        jog_grid.attach(self.x_plus_y_plus_btn, 2, 0, 1, 1)

        # X- button (middle left) - Icon updated in _update_direction_icons
        self.x_minus_btn = create_icon_button("arrow-west", _("X-"))
        self.x_minus_btn.connect("clicked", self._on_x_minus_clicked)
        jog_grid.attach(self.x_minus_btn, 0, 1, 1, 1)

        # X+ button (middle right) - Icon updated in _update_direction_icons
        self.x_plus_btn = create_icon_button("arrow-east", _("X+"))
        self.x_plus_btn.connect("clicked", self._on_x_plus_clicked)
        jog_grid.attach(self.x_plus_btn, 2, 1, 1, 1)

        # X-Y- button (bottom left diagonal)
        self.x_minus_y_minus_btn = create_icon_button(
            "arrow-south-west", _("X- Y-")
        )
        self.x_minus_y_minus_btn.connect(
            "clicked", self._on_x_minus_y_minus_clicked
        )
        jog_grid.attach(self.x_minus_y_minus_btn, 0, 2, 1, 1)

        # Y- button (bottom center) - Icon updated in _update_direction_icons
        self.y_minus_btn = create_icon_button("arrow-south", _("Y-"))
        self.y_minus_btn.connect("clicked", self._on_y_minus_clicked)
        jog_grid.attach(self.y_minus_btn, 1, 2, 1, 1)

        # X+Y- button (bottom right diagonal)
        self.x_plus_y_minus_btn = create_icon_button(
            "arrow-south-east", _("X+ Y-")
        )
        self.x_plus_y_minus_btn.connect(
            "clicked", self._on_x_plus_y_minus_clicked
        )
        jog_grid.attach(self.x_plus_y_minus_btn, 2, 2, 1, 1)

        # Z buttons to the right
        self.z_plus_btn = Gtk.Button(label=_("Z+"))
        self.z_plus_btn.set_size_request(60, 60)
        self.z_plus_btn.connect("clicked", self._on_z_plus_clicked)
        jog_grid.attach(self.z_plus_btn, 4, 0, 1, 1)

        self.z_minus_btn = Gtk.Button(label=_("Z-"))
        self.z_minus_btn.set_size_request(60, 60)
        self.z_minus_btn.connect("clicked", self._on_z_minus_clicked)
        jog_grid.attach(self.z_minus_btn, 4, 2, 1, 1)  # One row lower

        # Add key controller for cursor key support
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # Set initial icons and sensitivity
        self._update_direction_icons()
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

        self._update_direction_icons()
        self._update_button_sensitivity()
        self._update_limit_status()

    def _get_visual_jog_deltas(self) -> Tuple[float, float]:
        """
        Calculate jog deltas for visual Right (X) and Up (Y) directions.
        Returns:
            Tuple[float, float]: (x_right_delta, y_up_delta)
        """
        # Default standard Cartesian
        x_right = self.jog_distance
        y_up = self.jog_distance

        if self.machine:
            # If X axis is right-positive (standard), Right button moves +X.
            # Otherwise, Right button moves -X.
            x_right = (
                self.jog_distance
                if self.machine.x_axis_right
                else -self.jog_distance
            )

            # If Y axis is down-positive (inverted), Up button moves -Y.
            # Otherwise, Up button moves +Y.
            y_up = (
                -self.jog_distance
                if self.machine.y_axis_down
                else self.jog_distance
            )

        return x_right, y_up

    def _update_direction_icons(self):
        """Update X and Y direction icons based on machine configuration."""
        # Default to standard Cartesian if machine is not set
        y_axis_down = self.machine and self.machine.y_axis_down
        x_axis_right = self.machine.x_axis_right if self.machine else True

        # Determine icons based on configuration
        # If Y is down (positive), Up button (negative Y) is moving 'North'
        y_plus_icon = "arrow-south" if y_axis_down else "arrow-north"
        y_minus_icon = "arrow-north" if y_axis_down else "arrow-south"

        # If X is right (positive), Right button (positive X) is 'East'
        x_plus_icon = "arrow-east" if x_axis_right else "arrow-west"
        x_minus_icon = "arrow-west" if x_axis_right else "arrow-east"

        self.y_plus_btn.set_child(get_icon(y_plus_icon))
        self.y_minus_btn.set_child(get_icon(y_minus_icon))
        self.x_plus_btn.set_child(get_icon(x_plus_icon))
        self.x_minus_btn.set_child(get_icon(x_minus_icon))

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

        # Reset all button styles
        buttons = [
            self.x_plus_btn,
            self.x_minus_btn,
            self.y_plus_btn,
            self.y_minus_btn,
            self.x_plus_y_plus_btn,
            self.x_minus_y_plus_btn,
            self.x_plus_y_minus_btn,
            self.x_minus_y_minus_btn,
            self.z_plus_btn,
            self.z_minus_btn,
        ]

        for button in buttons:
            button.remove_css_class("warning")
            button.remove_css_class("destructive-action")

        if not machine.soft_limits_enabled:
            return

        x_right_dist, y_up_dist = self._get_visual_jog_deltas()

        # Check limits using the actual distance that will be commanded
        if machine.would_jog_exceed_limits(Axis.X, x_right_dist):
            self.x_plus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.X, -x_right_dist):
            self.x_minus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.Y, y_up_dist):
            self.y_plus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.Y, -y_up_dist):
            self.y_minus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.Z, self.jog_distance):
            self.z_plus_btn.add_css_class("warning")

        if machine.would_jog_exceed_limits(Axis.Z, -self.jog_distance):
            self.z_minus_btn.add_css_class("warning")

        # Diagonal buttons
        # Right-Up
        if machine.would_jog_exceed_limits(
            Axis.X, x_right_dist
        ) or machine.would_jog_exceed_limits(Axis.Y, y_up_dist):
            self.x_plus_y_plus_btn.add_css_class("warning")

        # Left-Up
        if machine.would_jog_exceed_limits(
            Axis.X, -x_right_dist
        ) or machine.would_jog_exceed_limits(Axis.Y, y_up_dist):
            self.x_minus_y_plus_btn.add_css_class("warning")

        # Right-Down
        if machine.would_jog_exceed_limits(
            Axis.X, x_right_dist
        ) or machine.would_jog_exceed_limits(Axis.Y, -y_up_dist):
            self.x_plus_y_minus_btn.add_css_class("warning")

        # Left-Down
        if machine.would_jog_exceed_limits(
            Axis.X, -x_right_dist
        ) or machine.would_jog_exceed_limits(Axis.Y, -y_up_dist):
            self.x_minus_y_minus_btn.add_css_class("warning")

    def _on_machine_state_changed(self, machine, state):
        """Handle machine state changes to update limit status."""
        self._update_limit_status()

    def _jog_xy(self, x_dist: float, y_dist: float):
        """Helper to jog X and Y, splitting commands if necessary."""
        if not self.machine or not self.machine_cmd:
            return

        # Send separate commands to ensure correct direction per axis,
        # as mixed signs in a combined bitmask might not be supported
        # or might result in incorrect direction if the driver/firmware
        # expects a single distance value.
        if x_dist != 0:
            self.machine_cmd.jog(self.machine, Axis.X, x_dist, self.jog_speed)
        if y_dist != 0:
            self.machine_cmd.jog(self.machine, Axis.Y, y_dist, self.jog_speed)

    def _on_x_plus_clicked(self, button):
        """Handle Right button click."""
        if self.machine and self.machine_cmd:
            x_dist, _ = self._get_visual_jog_deltas()
            self.machine_cmd.jog(self.machine, Axis.X, x_dist, self.jog_speed)

    def _on_x_minus_clicked(self, button):
        """Handle Left button click."""
        if self.machine and self.machine_cmd:
            x_dist, _ = self._get_visual_jog_deltas()
            self.machine_cmd.jog(self.machine, Axis.X, -x_dist, self.jog_speed)

    def _on_y_plus_clicked(self, button):
        """Handle Up button click."""
        if self.machine and self.machine_cmd:
            _, y_dist = self._get_visual_jog_deltas()
            self.machine_cmd.jog(self.machine, Axis.Y, y_dist, self.jog_speed)

    def _on_y_minus_clicked(self, button):
        """Handle Down button click."""
        if self.machine and self.machine_cmd:
            _, y_dist = self._get_visual_jog_deltas()
            self.machine_cmd.jog(self.machine, Axis.Y, -y_dist, self.jog_speed)

    def _on_z_plus_clicked(self, button):
        """Handle Z+ button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.jog(
                self.machine, Axis.Z, self.jog_distance, self.jog_speed
            )

    def _on_z_minus_clicked(self, button):
        """Handle Z- button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.jog(
                self.machine, Axis.Z, -self.jog_distance, self.jog_speed
            )

    def _on_x_plus_y_plus_clicked(self, button):
        """Handle Right-Up diagonal button click."""
        x_dist, y_dist = self._get_visual_jog_deltas()
        self._jog_xy(x_dist, y_dist)

    def _on_x_minus_y_plus_clicked(self, button):
        """Handle Left-Up diagonal button click."""
        x_dist, y_dist = self._get_visual_jog_deltas()
        self._jog_xy(-x_dist, y_dist)

    def _on_x_plus_y_minus_clicked(self, button):
        """Handle Right-Down diagonal button click."""
        x_dist, y_dist = self._get_visual_jog_deltas()
        self._jog_xy(x_dist, -y_dist)

    def _on_x_minus_y_minus_clicked(self, button):
        """Handle Left-Down diagonal button click."""
        x_dist, y_dist = self._get_visual_jog_deltas()
        self._jog_xy(-x_dist, -y_dist)
        return False

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events for cursor key jogging."""
        # Only process cursor keys if machine is connected and ready
        if not self.machine or not self.machine.is_connected():
            return False

        # Map cursor keys to jog actions
        if keyval == Gdk.KEY_Up:
            self._on_y_plus_clicked(None)
            return True
        elif keyval == Gdk.KEY_Down:
            self._on_y_minus_clicked(None)
            return True
        elif keyval == Gdk.KEY_Left:
            self._on_x_minus_clicked(None)
            return True
        elif keyval == Gdk.KEY_Right:
            self._on_x_plus_clicked(None)
            return True
        elif keyval == Gdk.KEY_Page_Up:
            self._on_z_plus_clicked(None)
            return True
        elif keyval == Gdk.KEY_Page_Down:
            self._on_z_minus_clicked(None)
            return True

        return False
