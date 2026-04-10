from gi.repository import Gdk, Gsk, Graphene, Gtk
from typing import Optional
from gettext import gettext as _
from ...core.ops.axis import Axis
from ...machine.models.machine import JogDirection, Machine
from ...machine.cmd import MachineCmd
from ..icons import get_icon

_GAP = 12
_SPACING = 6
_MAX_HEIGHT = 4 * 60 + 3 * _SPACING


class JogWidget(Gtk.Widget):
    """Widget for manually jogging the machine."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._jog_grid = Gtk.Grid()
        self._jog_grid.set_parent(self)
        self._jog_grid.set_row_spacing(_SPACING)
        self._jog_grid.set_column_spacing(_SPACING)
        self._jog_grid.set_row_homogeneous(True)
        self._jog_grid.set_column_homogeneous(True)

        self._action_grid = Gtk.Grid()
        self._action_grid.set_parent(self)
        self._action_grid.set_row_spacing(_SPACING)
        self._action_grid.set_row_homogeneous(True)

        self.machine: Optional[Machine] = None
        self.machine_cmd: Optional[MachineCmd] = None
        self.jog_speed = 1000
        self.jog_distance = 10.0
        self._buttons = []

        self.set_focusable(True)

        def create_button(icon_name, tooltip):
            button = Gtk.Button()
            button.set_size_request(60, 60)
            button.set_tooltip_text(tooltip)
            icon = get_icon(icon_name)
            button.set_child(icon)
            button.set_hexpand(True)
            button.set_vexpand(True)
            self._buttons.append(button)
            return button

        # Row 0: NW - N - NE
        self.north_west_btn = create_button(
            "arrow-north-west-symbolic", _("Move North-West")
        )
        self.north_west_btn.connect("clicked", self._on_x_minus_y_plus_clicked)
        self._jog_grid.attach(self.north_west_btn, 0, 0, 1, 1)

        self.north_btn = create_button("arrow-north-symbolic", _("Move North"))
        self.north_btn.connect("clicked", self._on_y_plus_clicked)
        self._jog_grid.attach(self.north_btn, 1, 0, 1, 1)

        self.north_east_btn = create_button(
            "arrow-north-east-symbolic", _("Move North-East")
        )
        self.north_east_btn.connect("clicked", self._on_x_plus_y_plus_clicked)
        self._jog_grid.attach(self.north_east_btn, 2, 0, 1, 1)

        # Row 1: W - Home - E
        self.west_btn = create_button(
            "arrow-west-symbolic", _("Move West (Left)")
        )
        self.west_btn.connect("clicked", self._on_x_minus_clicked)
        self._jog_grid.attach(self.west_btn, 0, 1, 1, 1)

        self.home_all_btn = create_button("home-symbolic", _("Home All"))
        self.home_all_btn.connect("clicked", self._on_home_all_clicked)
        self._jog_grid.attach(self.home_all_btn, 1, 1, 1, 1)

        self.east_btn = create_button(
            "arrow-east-symbolic", _("Move East (Right)")
        )
        self.east_btn.connect("clicked", self._on_x_plus_clicked)
        self._jog_grid.attach(self.east_btn, 2, 1, 1, 1)

        # Row 2: SW - S - SE
        self.south_west_btn = create_button(
            "arrow-south-west-symbolic", _("Move South-West")
        )
        self.south_west_btn.connect(
            "clicked", self._on_x_minus_y_minus_clicked
        )
        self._jog_grid.attach(self.south_west_btn, 0, 2, 1, 1)

        self.south_btn = create_button("arrow-south-symbolic", _("Move South"))
        self.south_btn.connect("clicked", self._on_y_minus_clicked)
        self._jog_grid.attach(self.south_btn, 1, 2, 1, 1)

        self.south_east_btn = create_button(
            "arrow-south-east-symbolic", _("Move South-East")
        )
        self.south_east_btn.connect("clicked", self._on_x_plus_y_minus_clicked)
        self._jog_grid.attach(self.south_east_btn, 2, 2, 1, 1)

        # Row 3: home x - home y - home z
        self.home_x_btn = create_button("home-x-symbolic", _("Home X"))
        self.home_x_btn.connect("clicked", self._on_home_x_clicked)
        self._jog_grid.attach(self.home_x_btn, 0, 3, 1, 1)

        self.home_y_btn = create_button("home-y-symbolic", _("Home Y"))
        self.home_y_btn.connect("clicked", self._on_home_y_clicked)
        self._jog_grid.attach(self.home_y_btn, 1, 3, 1, 1)

        self.home_z_btn = create_button("home-z-symbolic", _("Home Z"))
        self.home_z_btn.connect("clicked", self._on_home_z_clicked)
        self._jog_grid.attach(self.home_z_btn, 2, 3, 1, 1)

        # Action column (separate grid for extra gap)
        self.send_btn = create_button("send-symbolic", _("Send to machine"))
        self.send_btn.add_css_class("suggested-action")
        self.send_btn.connect("clicked", self._on_send_clicked)
        self._action_grid.attach(self.send_btn, 0, 0, 1, 1)

        self.z_plus_btn = create_button(
            "arrow-z-up-symbolic", _("Increase Z-Distance")
        )
        self.z_plus_btn.connect("clicked", self._on_z_plus_clicked)
        self._action_grid.attach(self.z_plus_btn, 0, 1, 1, 1)

        self.z_minus_btn = create_button(
            "arrow-z-down-symbolic", _("Decrease Z-Distance")
        )
        self.z_minus_btn.connect("clicked", self._on_z_minus_clicked)
        self._action_grid.attach(self.z_minus_btn, 0, 2, 1, 1)

        self.cancel_btn = create_button(
            "stop-symbolic", _("Cancel running job")
        )
        self.cancel_btn.add_css_class("destructive-action")
        self.cancel_btn.connect("clicked", self._on_cancel_clicked)
        self._action_grid.attach(self.cancel_btn, 0, 3, 1, 1)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        self._update_button_sensitivity()

    @staticmethod
    def _calc_grid_widths(height):
        cell_h = (height - 3 * _SPACING) / 4
        jog_w = 3 * cell_h + 2 * _SPACING
        act_w = cell_h
        return jog_w, act_w

    def do_get_request_mode(self):
        return Gtk.SizeRequestMode.WIDTH_FOR_HEIGHT

    def do_measure(self, orientation, for_size):
        if orientation == Gtk.Orientation.HORIZONTAL:
            h = for_size if for_size > 0 else _MAX_HEIGHT
            jog_w, act_w = self._calc_grid_widths(h)
            total = int(jog_w) + _GAP + int(act_w)
            return (total, total, -1, -1)
        m = self._jog_grid.measure(orientation, for_size)
        return (m[0], min(m[1], _MAX_HEIGHT), -1, -1)

    def do_size_allocate(self, width, height, baseline):
        jog_w, act_w = self._calc_grid_widths(height)

        total_needed = jog_w + _GAP + act_w
        if width > total_needed:
            extra = width - total_needed
            jog_w += extra * 3 / 4
            act_w += extra / 4

        jog_w = int(jog_w)
        act_w = int(act_w)

        self._jog_grid.allocate(jog_w, height, baseline, None)

        transform = Gsk.Transform().translate(
            Graphene.Point().init(jog_w + _GAP, 0)
        )
        self._action_grid.allocate(act_w, height, baseline, transform)

    def set_machine(
        self, machine: Optional[Machine], machine_cmd: Optional[MachineCmd]
    ):
        """Set the machine this widget controls."""
        # Disconnect from previous machine if any
        if self.machine:
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self.machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )

        self.machine = machine
        self.machine_cmd = machine_cmd

        # Connect to state changes
        if self.machine:
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )

        self._update_button_sensitivity()
        self._update_limit_status()

    def _update_button_sensitivity(self):
        """Update button sensitivity based on machine capabilities."""
        # Default all buttons to disabled
        self.east_btn.set_sensitive(False)
        self.west_btn.set_sensitive(False)
        self.north_btn.set_sensitive(False)
        self.south_btn.set_sensitive(False)
        self.north_east_btn.set_sensitive(False)
        self.north_west_btn.set_sensitive(False)
        self.south_east_btn.set_sensitive(False)
        self.south_west_btn.set_sensitive(False)
        self.z_plus_btn.set_sensitive(False)
        self.z_minus_btn.set_sensitive(False)
        self.home_x_btn.set_sensitive(False)
        self.home_y_btn.set_sensitive(False)
        self.home_z_btn.set_sensitive(False)
        self.home_all_btn.set_sensitive(False)
        self.send_btn.set_sensitive(False)
        self.cancel_btn.set_sensitive(False)

        # Only enable buttons if machine exists, is connected
        if self.machine is None or not self.machine.is_connected():
            return

        # Type assertion to help Pylance understand machine is not None
        machine: Machine = self.machine  # type: ignore

        # Jog buttons
        self.east_btn.set_sensitive(machine.can_jog(Axis.X))
        self.west_btn.set_sensitive(machine.can_jog(Axis.X))
        self.north_btn.set_sensitive(machine.can_jog(Axis.Y))
        self.south_btn.set_sensitive(machine.can_jog(Axis.Y))

        # Diagonal buttons - need both X and Y axis support
        can_jog_xy = machine.can_jog(Axis.X) and machine.can_jog(Axis.Y)
        self.north_east_btn.set_sensitive(can_jog_xy)
        self.north_west_btn.set_sensitive(can_jog_xy)
        self.south_east_btn.set_sensitive(can_jog_xy)
        self.south_west_btn.set_sensitive(can_jog_xy)

        self.z_plus_btn.set_sensitive(machine.can_jog(Axis.Z))
        self.z_minus_btn.set_sensitive(machine.can_jog(Axis.Z))

        # Home buttons - only enable if single axis homing is supported
        single_axis_homing = machine.single_axis_homing_enabled
        self.home_x_btn.set_sensitive(
            machine.can_home(Axis.X) and single_axis_homing
        )
        self.home_y_btn.set_sensitive(
            machine.can_home(Axis.Y) and single_axis_homing
        )
        self.home_z_btn.set_sensitive(
            machine.can_home(Axis.Z) and single_axis_homing
        )
        self.home_all_btn.set_sensitive(True)

        # Send and Cancel buttons - always enabled when connected
        self.send_btn.set_sensitive(True)
        self.cancel_btn.set_sensitive(True)

        # Hide home buttons if single axis homing is not supported
        home_visible = single_axis_homing
        self.home_x_btn.set_visible(home_visible)
        self.home_y_btn.set_visible(home_visible)
        self.home_z_btn.set_visible(home_visible)

        self._update_limit_status()

    def _update_limit_status(self):
        """Update button styling based on whether jog would exceed limits."""
        if not self.machine or not self.machine.is_connected():
            return

        machine = self.machine

        buttons = [
            self.east_btn,
            self.west_btn,
            self.north_btn,
            self.south_btn,
            self.z_plus_btn,
            self.z_minus_btn,
            self.north_east_btn,
            self.north_west_btn,
            self.south_east_btn,
            self.south_west_btn,
        ]
        for button in buttons:
            button.remove_css_class("warning")
            button.remove_css_class("destructive-action")

        if not machine.soft_limits_enabled:
            return

        # Get the signed coordinate deltas for each visual direction from the
        # model
        x_east = machine.calculate_jog(JogDirection.EAST, self.jog_distance)
        x_west = machine.calculate_jog(JogDirection.WEST, self.jog_distance)
        y_north = machine.calculate_jog(JogDirection.NORTH, self.jog_distance)
        y_south = machine.calculate_jog(JogDirection.SOUTH, self.jog_distance)
        z_up = machine.calculate_jog(JogDirection.UP, self.jog_distance)
        z_down = machine.calculate_jog(JogDirection.DOWN, self.jog_distance)

        # Check limits using the final signed delta that will be commanded
        exceeds_east = machine.would_jog_exceed_limits(Axis.X, x_east)
        exceeds_west = machine.would_jog_exceed_limits(Axis.X, x_west)
        exceeds_north = machine.would_jog_exceed_limits(Axis.Y, y_north)
        exceeds_south = machine.would_jog_exceed_limits(Axis.Y, y_south)
        exceeds_up = machine.would_jog_exceed_limits(Axis.Z, z_up)
        exceeds_down = machine.would_jog_exceed_limits(Axis.Z, z_down)

        if exceeds_east:
            self.east_btn.add_css_class("warning")
        if exceeds_west:
            self.west_btn.add_css_class("warning")
        if exceeds_north:
            self.north_btn.add_css_class("warning")
        if exceeds_south:
            self.south_btn.add_css_class("warning")
        if exceeds_up:
            self.z_plus_btn.add_css_class("warning")
        if exceeds_down:
            self.z_minus_btn.add_css_class("warning")

        # Diagonal buttons
        if exceeds_east or exceeds_north:
            self.north_east_btn.add_css_class("warning")
        if exceeds_west or exceeds_north:
            self.north_west_btn.add_css_class("warning")
        if exceeds_east or exceeds_south:
            self.south_east_btn.add_css_class("warning")
        if exceeds_west or exceeds_south:
            self.south_west_btn.add_css_class("warning")

    def _on_machine_state_changed(self, machine, state):
        """Handle machine state changes to update limit status."""
        self._update_limit_status()

    def _on_connection_status_changed(self, sender, **kwargs):
        """Handle connection status changes to update button sensitivity."""
        self._update_button_sensitivity()

    def _perform_jog(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Helper to jog multiple axes simultaneously by sending a single
        command dictionary.
        """
        if not self.machine or not self.machine_cmd:
            return

        deltas = {}
        if x != 0:
            deltas[Axis.X] = x
        if y != 0:
            deltas[Axis.Y] = y
        if z != 0:
            deltas[Axis.Z] = z

        if deltas:
            self.machine_cmd.jog(self.machine, deltas, self.jog_speed)

    def _on_x_plus_clicked(self, button):
        """Handle Right (East) button click."""
        if self.machine:
            x_dist = self.machine.calculate_jog(
                JogDirection.EAST, self.jog_distance
            )
            self._perform_jog(x=x_dist)

    def _on_x_minus_clicked(self, button):
        """Handle Left (West) button click."""
        if self.machine:
            x_dist = self.machine.calculate_jog(
                JogDirection.WEST, self.jog_distance
            )
            self._perform_jog(x=x_dist)

    def _on_y_plus_clicked(self, button):
        """Handle Away (North) button click."""
        if self.machine:
            y_dist = self.machine.calculate_jog(
                JogDirection.NORTH, self.jog_distance
            )
            self._perform_jog(y=y_dist)

    def _on_y_minus_clicked(self, button):
        """Handle Toward (South) button click."""
        if self.machine:
            y_dist = self.machine.calculate_jog(
                JogDirection.SOUTH, self.jog_distance
            )
            self._perform_jog(y=y_dist)

    def _on_z_plus_clicked(self, button):
        """Handle Up button click."""
        if self.machine:
            z_dist = self.machine.calculate_jog(
                JogDirection.UP, self.jog_distance
            )
            self._perform_jog(z=z_dist)

    def _on_z_minus_clicked(self, button):
        """Handle Down button click."""
        if self.machine:
            z_dist = self.machine.calculate_jog(
                JogDirection.DOWN, self.jog_distance
            )
            self._perform_jog(z=z_dist)

    def _on_x_plus_y_plus_clicked(self, button):
        """Handle Right-Away diagonal button click."""
        if self.machine:
            x_dist = self.machine.calculate_jog(
                JogDirection.EAST, self.jog_distance
            )
            y_dist = self.machine.calculate_jog(
                JogDirection.NORTH, self.jog_distance
            )
            self._perform_jog(x=x_dist, y=y_dist)

    def _on_x_minus_y_plus_clicked(self, button):
        """Handle Left-Away diagonal button click."""
        if self.machine:
            x_dist = self.machine.calculate_jog(
                JogDirection.WEST, self.jog_distance
            )
            y_dist = self.machine.calculate_jog(
                JogDirection.NORTH, self.jog_distance
            )
            self._perform_jog(x=x_dist, y=y_dist)

    def _on_x_plus_y_minus_clicked(self, button):
        """Handle Right-Toward diagonal button click."""
        if self.machine:
            x_dist = self.machine.calculate_jog(
                JogDirection.EAST, self.jog_distance
            )
            y_dist = self.machine.calculate_jog(
                JogDirection.SOUTH, self.jog_distance
            )
            self._perform_jog(x=x_dist, y=y_dist)

    def _on_x_minus_y_minus_clicked(self, button):
        """Handle Left-Toward diagonal button click."""
        if self.machine:
            x_dist = self.machine.calculate_jog(
                JogDirection.WEST, self.jog_distance
            )
            y_dist = self.machine.calculate_jog(
                JogDirection.SOUTH, self.jog_distance
            )
            self._perform_jog(x=x_dist, y=y_dist)

    def _on_home_all_clicked(self, button):
        """Handle Home All button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine)

    def _on_home_x_clicked(self, button):
        """Handle Home X button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine, Axis.X)

    def _on_home_y_clicked(self, button):
        """Handle Home Y button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine, Axis.Y)

    def _on_home_z_clicked(self, button):
        """Handle Home Z button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine, Axis.Z)

    def _on_send_clicked(self, button):
        """Handle Send button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.run_send_job(self.machine)

    def _on_cancel_clicked(self, button):
        """Handle Cancel button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.cancel_job(self.machine)

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
