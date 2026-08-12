from gettext import gettext as _

from gi.repository import Gdk, Graphene, Gsk, Gtk
from raygeo.ops.axis import Axis

from ...machine.cmd import MachineCmd
from ...machine.models.machine import JogDirection, Machine
from ..icons import get_icon

_GAP = 12
_SPACING = 6
_MAX_HEIGHT = 4 * 60 + 3 * _SPACING


class JogWidget(Gtk.Widget):
    """Widget for manually jogging the machine."""

    def __init__(self, show_actions: bool = True, **kwargs):
        super().__init__(**kwargs)

        self._jog_grid = Gtk.Grid()
        self._jog_grid.set_parent(self)
        self._jog_grid.set_row_spacing(_SPACING)
        self._jog_grid.set_column_spacing(_SPACING)
        self._jog_grid.set_row_homogeneous(True)
        self._jog_grid.set_column_homogeneous(True)

        self._show_actions = show_actions

        self._action_grid = Gtk.Grid()
        self._action_grid.set_parent(self)
        self._action_grid.set_row_spacing(_SPACING)
        self._action_grid.set_row_homogeneous(True)
        self._action_grid.set_visible(show_actions)

        self.machine: Machine | None = None
        self.machine_cmd: MachineCmd | None = None
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
            total = int(jog_w)
            if self._show_actions:
                total += _GAP + int(act_w)
            return (total, total, -1, -1)
        m = self._jog_grid.measure(orientation, for_size)
        return (m[0], min(m[1], _MAX_HEIGHT), -1, -1)

    def do_size_allocate(self, width, height, baseline):
        jog_w, act_w = self._calc_grid_widths(height)

        if self._show_actions:
            total_needed = jog_w + _GAP + act_w
            if width > total_needed:
                extra = width - total_needed
                jog_w += extra * 3 / 4
                act_w += extra / 4
            act_w = int(act_w)
        else:
            jog_w = width

        jog_w = int(jog_w)

        self._jog_grid.allocate(jog_w, height, baseline, None)

        if self._show_actions:
            transform = Gsk.Transform().translate(
                Graphene.Point().init(jog_w + _GAP, 0)
            )
            self._action_grid.allocate(act_w, height, baseline, transform)
        else:
            self._action_grid.allocate(0, 0, -1, None)

    def set_machine(
        self, machine: Machine | None, machine_cmd: MachineCmd | None
    ):
        """Set the machine this widget controls."""
        if self.machine:
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self.machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            self.machine.changed.disconnect(self._on_machine_changed)

        self.machine = machine
        self.machine_cmd = machine_cmd

        if self.machine:
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self.machine.changed.connect(self._on_machine_changed)

        self._update_button_sensitivity()
        self._update_limit_status()

    def _on_machine_changed(self, sender, **kwargs):
        self._update_button_sensitivity()
        self._update_limit_status()

    def _jog_deltas(self, *directions: JogDirection) -> dict[Axis, float]:
        """Aggregate native-axis deltas for one or more visual
        directions."""
        if not self.machine:
            return {}
        deltas: dict[Axis, float] = {}
        for direction in directions:
            for axis, delta in self.machine.panel.calculate_jog(
                direction, self.jog_distance
            ).items():
                deltas[axis] = deltas.get(axis, 0.0) + delta
        return deltas

    def _can_jog_direction(self, direction: JogDirection) -> bool:
        """Whether the machine can jog every axis a direction drives."""
        if not self.machine:
            return False
        return all(
            self.machine.can_jog(axis) for axis in self._jog_deltas(direction)
        )

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

        # Jog buttons - a direction is joggable when every native axis it
        # drives is supported (under rotation a visual axis may map to
        # the orthogonal native axis)
        can_jog_east = self._can_jog_direction(JogDirection.EAST)
        can_jog_west = self._can_jog_direction(JogDirection.WEST)
        can_jog_north = self._can_jog_direction(JogDirection.NORTH)
        can_jog_south = self._can_jog_direction(JogDirection.SOUTH)
        self.east_btn.set_sensitive(can_jog_east)
        self.west_btn.set_sensitive(can_jog_west)
        self.north_btn.set_sensitive(can_jog_north)
        self.south_btn.set_sensitive(can_jog_south)

        # Diagonal buttons - need both cardinal directions
        self.north_east_btn.set_sensitive(can_jog_east and can_jog_north)
        self.north_west_btn.set_sensitive(can_jog_west and can_jog_north)
        self.south_east_btn.set_sensitive(can_jog_east and can_jog_south)
        self.south_west_btn.set_sensitive(can_jog_west and can_jog_south)

        self.z_plus_btn.set_sensitive(self._can_jog_direction(JogDirection.UP))
        self.z_minus_btn.set_sensitive(
            self._can_jog_direction(JogDirection.DOWN)
        )

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

        def exceeds(*directions: JogDirection) -> bool:
            if not self.machine:
                return False
            return any(
                self.machine.would_jog_exceed_limits(axis, delta)
                for axis, delta in self._jog_deltas(*directions).items()
            )

        if exceeds(JogDirection.EAST):
            self.east_btn.add_css_class("warning")
        if exceeds(JogDirection.WEST):
            self.west_btn.add_css_class("warning")
        if exceeds(JogDirection.NORTH):
            self.north_btn.add_css_class("warning")
        if exceeds(JogDirection.SOUTH):
            self.south_btn.add_css_class("warning")
        if exceeds(JogDirection.UP):
            self.z_plus_btn.add_css_class("warning")
        if exceeds(JogDirection.DOWN):
            self.z_minus_btn.add_css_class("warning")

        # Diagonal buttons
        if exceeds(JogDirection.EAST, JogDirection.NORTH):
            self.north_east_btn.add_css_class("warning")
        if exceeds(JogDirection.WEST, JogDirection.NORTH):
            self.north_west_btn.add_css_class("warning")
        if exceeds(JogDirection.EAST, JogDirection.SOUTH):
            self.south_east_btn.add_css_class("warning")
        if exceeds(JogDirection.WEST, JogDirection.SOUTH):
            self.south_west_btn.add_css_class("warning")

    def _on_machine_state_changed(self, machine, state):
        """Handle machine state changes to update limit status."""
        self._update_limit_status()

    def _on_connection_status_changed(self, sender, **kwargs):
        """Handle connection status changes to update button sensitivity."""
        self._update_button_sensitivity()

    def _perform_jog(self, deltas: dict[Axis, float]):
        """
        Helper to jog multiple axes simultaneously by sending a single
        command dictionary.
        """
        if not self.machine or not self.machine_cmd:
            return

        if deltas:
            self.machine_cmd.jog(self.machine, deltas, self.jog_speed)

    def _perform_visual_jog(self, *directions: JogDirection):
        """Jog according to one or more visual directions."""
        if not self.machine:
            return
        self._perform_jog(self._jog_deltas(*directions))

    def _on_x_plus_clicked(self, button):
        """Handle Right (East) button click."""
        self._perform_visual_jog(JogDirection.EAST)

    def _on_x_minus_clicked(self, button):
        """Handle Left (West) button click."""
        self._perform_visual_jog(JogDirection.WEST)

    def _on_y_plus_clicked(self, button):
        """Handle Away (North) button click."""
        self._perform_visual_jog(JogDirection.NORTH)

    def _on_y_minus_clicked(self, button):
        """Handle Toward (South) button click."""
        self._perform_visual_jog(JogDirection.SOUTH)

    def _on_z_plus_clicked(self, button):
        """Handle Up button click."""
        self._perform_visual_jog(JogDirection.UP)

    def _on_z_minus_clicked(self, button):
        """Handle Down button click."""
        self._perform_visual_jog(JogDirection.DOWN)

    def _on_x_plus_y_plus_clicked(self, button):
        """Handle Right-Away diagonal button click."""
        self._perform_visual_jog(JogDirection.EAST, JogDirection.NORTH)

    def _on_x_minus_y_plus_clicked(self, button):
        """Handle Left-Away diagonal button click."""
        self._perform_visual_jog(JogDirection.WEST, JogDirection.NORTH)

    def _on_x_plus_y_minus_clicked(self, button):
        """Handle Right-Toward diagonal button click."""
        self._perform_visual_jog(JogDirection.EAST, JogDirection.SOUTH)

    def _on_x_minus_y_minus_clicked(self, button):
        """Handle Left-Toward diagonal button click."""
        self._perform_visual_jog(JogDirection.WEST, JogDirection.SOUTH)

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
