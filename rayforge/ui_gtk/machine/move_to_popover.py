import logging
from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw, Gtk
from raygeo.ops.axis import Axis

from ...machine.cmd import MachineCmd
from ...machine.driver.dummy import NoDeviceDriver
from ...machine.models.machine import Machine
from ..icons import get_icon
from ..shared.pref_rows.length_spin_row import LengthSpinRow

logger = logging.getLogger(__name__)


class MoveToPopover(Gtk.Popover):
    """Popover for moving the laser head.

    Offers direct X/Y(/Z) coordinate entry plus the selection/workarea
    corner shortcuts and the active-WCS origin shortcut. Values are
    interpreted in the active work coordinate system, matching the
    position readout in the machine panel.

    While pointer alignment is on, all absolute moves issued here are
    shifted so the pointer dot lands on the entered position; the
    coordinate prefill reports the pointer dot's position accordingly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.machine: Machine | None = None
        self.machine_cmd: MachineCmd | None = None
        self._get_bounds_callback = None
        self._get_speed_callback: Callable[[], float | None] | None = None

        coords_group = Adw.PreferencesGroup(title=_("Move to Position"))
        self.set_child(coords_group)

        self.x_row = LengthSpinRow(
            _("X"),
            None,
            lower=0.0,
            upper=200.0,
            value_in_base=0.0,
        )
        coords_group.add(self.x_row)

        self.y_row = LengthSpinRow(
            _("Y"),
            None,
            lower=0.0,
            upper=200.0,
            value_in_base=0.0,
        )
        coords_group.add(self.y_row)

        self.z_row = LengthSpinRow(
            _("Z"),
            None,
            lower=0.0,
            upper=100.0,
            value_in_base=0.0,
        )
        self.z_row.set_visible(False)
        coords_group.add(self.z_row)

        self.alignment_row = Adw.SwitchRow(title=_("Pointer Alignment"))
        self._alignment_handler_id = self.alignment_row.connect(
            "notify::active", self._on_alignment_toggled
        )
        coords_group.add(self.alignment_row)

        self.travel_warning_label = Gtk.Label()
        self.travel_warning_label.set_visible(False)
        self.travel_warning_label.set_wrap(True)
        self.travel_warning_label.set_xalign(0.0)
        self.travel_warning_label.add_css_class("warning")
        coords_group.add(self.travel_warning_label)

        self.move_row = Adw.ActionRow()
        coords_group.add(self.move_row)

        self.ll_btn = self._add_move_button(
            "bottom-left-symbolic",
            _("Move to Lower-Left of Selection or Workarea"),
            self._on_move_to_position,
            "ll",
        )
        self.center_btn = self._add_move_button(
            "center-symbolic",
            _("Move to Center of Selection or Workarea"),
            self._on_move_to_position,
            "center",
        )
        self.ur_btn = self._add_move_button(
            "top-right-symbolic",
            _("Move to Upper-Right of Selection or Workarea"),
            self._on_move_to_position,
            "ur",
        )
        self.origin_btn = self._add_move_button(
            "goto-origin-symbolic",
            _("Move to Origin of Active WCS"),
            self._on_move_to_wcs_zero,
        )

        self.move_btn = Gtk.Button(label=_("Move"))
        self.move_btn.add_css_class("suggested-action")
        self.move_btn.set_valign(Gtk.Align.CENTER)
        self.move_btn.set_margin_start(6)
        self.move_btn.connect("clicked", self._on_move_clicked)
        self.move_row.add_suffix(self.move_btn)

        self.connect("show", self._on_show)

    def _add_move_button(
        self, icon_name: str, tooltip: str, handler, *user_data
    ) -> Gtk.Button:
        btn = Gtk.Button(child=get_icon(icon_name))
        btn.add_css_class("flat")
        btn.set_valign(Gtk.Align.CENTER)
        btn.set_tooltip_text(tooltip)
        btn.connect("clicked", handler, *user_data)
        self.move_row.add_suffix(btn)
        return btn

    def set_machine(
        self, machine: Machine | None, machine_cmd: MachineCmd | None
    ):
        if self.machine:
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self.machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            self.machine.changed.disconnect(self._on_machine_changed)
            self.machine.pointer_alignment_changed.disconnect(
                self._on_pointer_alignment_changed
            )

        self.machine = machine
        self.machine_cmd = machine_cmd

        if self.machine:
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self.machine.changed.connect(self._on_machine_changed)
            self.machine.pointer_alignment_changed.connect(
                self._on_pointer_alignment_changed
            )

        self._update_bounds_and_axes()

    def set_get_bounds_callback(
        self,
        callback: Callable[[], tuple[float, float, float, float] | None]
        | None,
    ):
        self._get_bounds_callback = callback

    def set_speed_getter(self, callback: Callable[[], float | None] | None):
        """Registers a callback supplying the move speed in mm/min."""
        self._get_speed_callback = callback

    def _on_machine_state_changed(self, machine, state):
        self.update_sensitivity()

    def _on_connection_status_changed(self, machine, status, **kwargs):
        self.update_sensitivity()

    def _on_machine_changed(self, machine, **kwargs):
        self._update_bounds_and_axes()

    def _on_pointer_alignment_changed(self, machine):
        """Machine-side alignment toggles resync the switch.

        The coordinate rows are deliberately left untouched: their
        values stay frozen while the popover is open so the user can
        move back to a position that was entered before the toggle.
        """
        self._sync_alignment_row_state()

    def _update_bounds_and_axes(self):
        if not self.machine:
            self.z_row.set_visible(False)
        else:
            width, height = self.machine.axis_extents
            self.x_row.set_range(0.0, width)
            self.y_row.set_range(0.0, height)
            self.z_row.set_visible(self.machine.has_z_axis)
            if self.machine.has_z_axis:
                z_cfg = self.machine.axes.get(Axis.Z)
                z_min, z_max = z_cfg.extents if z_cfg else (-50.0, 50.0)
                self.z_row.set_range(float(z_min), float(z_max))
        self._update_alignment_row()
        self.update_sensitivity()

    def _is_machine_active(self) -> bool:
        if not self.machine:
            return False
        is_dummy = isinstance(self.machine.driver, NoDeviceDriver)
        return self.machine.is_connected() or is_dummy

    def update_sensitivity(self):
        is_active = self._is_machine_active()
        has_bounds = (
            self._get_bounds_callback is not None
            and self._get_bounds_callback() is not None
        )
        self.move_btn.set_sensitive(is_active)
        self.origin_btn.set_sensitive(is_active)
        self.ll_btn.set_sensitive(has_bounds and is_active)
        self.center_btn.set_sensitive(has_bounds and is_active)
        self.ur_btn.set_sensitive(has_bounds and is_active)

    def _on_show(self, widget):
        self.update_sensitivity()
        self._prefill_from_machine()

    # -- Pointer alignment ------------------------------------------

    def _get_pointer_shift(self) -> tuple[float, float]:
        """The (x, y) amount to subtract from absolute aim targets
        while pointer alignment is on, (0, 0) otherwise."""
        if not self.machine or not self.machine.pointer_alignment_enabled:
            return (0.0, 0.0)
        return self.machine.get_pointer_offset()

    def _update_alignment_row(self):
        """Syncs the alignment switch sensitivity and subtitle."""
        if not self.machine:
            self.alignment_row.set_sensitive(False)
            return
        has_offset = self.machine.has_pointer_offset()
        self.alignment_row.set_sensitive(has_offset)
        if has_offset:
            self.alignment_row.set_subtitle(
                _("Aim moves at the pointer dot instead of the beam")
            )
        else:
            self.alignment_row.set_subtitle(
                _("Requires a pointer offset on the laser head")
            )
        self._sync_alignment_row_state()

    def _sync_alignment_row_state(self):
        """Mirrors the machine state into the switch without echo."""
        if not self.machine:
            return
        self.alignment_row.handler_block(self._alignment_handler_id)
        self.alignment_row.set_active(self.machine.pointer_alignment_enabled)
        self.alignment_row.handler_unblock(self._alignment_handler_id)

    def _on_alignment_toggled(self, row, _param):
        if not self.machine:
            return
        self.machine.set_pointer_alignment(row.get_active())
        # The machine may have refused (no offset); re-sync the switch.
        self._sync_alignment_row_state()

    # -- Prefill and move issuance ----------------------------------

    def _prefill_from_machine(self):
        """Prefill the coordinate rows with the current position.

        While pointer alignment is on, the rows show where the pointer
        dot is, so re-issuing the prefilled values does not move the
        head: entered values round-trip through the same shift.
        """
        if not self.machine:
            return
        m_pos = self.machine.device_state.machine_pos
        if not m_pos or len(m_pos) < 3:
            return
        m_x, m_y, m_z = m_pos[0], m_pos[1], m_pos[2]
        if m_x is None or m_y is None or m_z is None:
            return
        if self.machine.active_wcs == self.machine.machine_space_wcs:
            x, y, z = m_x, m_y, m_z
        else:
            off_x, off_y, off_z = self.machine.get_wcs_offset(
                self.machine.active_wcs
            )
            x, y, z = m_x - off_x, m_y - off_y, m_z - off_z
        dx, dy = self._get_pointer_shift()
        self.x_row.set_value_in_base_units(x + dx)
        self.y_row.set_value_in_base_units(y + dy)
        if self.machine.has_z_axis:
            self.z_row.set_value_in_base_units(z)

    def _get_speed(self) -> float | None:
        if self._get_speed_callback is None:
            return None
        return self._get_speed_callback()

    def _clamp_command_to_travel(
        self, cmd_x: float, cmd_y: float
    ) -> tuple[float, float]:
        """Clamps a command-space aim target so the beam stays within
        the machine travel, showing a warning when clamping.

        The device adds the active WCS offset back to the command, so
        the resulting beam position is what gets clamped.
        """
        self.travel_warning_label.set_visible(False)
        if not self.machine:
            return cmd_x, cmd_y
        off_x, off_y, _off_z = self.machine.get_active_wcs_offset()
        width, height = self.machine.axis_extents
        x_min = -width if self.machine.reverse_x_axis else 0.0
        x_max = 0.0 if self.machine.reverse_x_axis else width
        y_min = -height if self.machine.reverse_y_axis else 0.0
        y_max = 0.0 if self.machine.reverse_y_axis else height
        beam_x = min(max(cmd_x + off_x, x_min), x_max)
        beam_y = min(max(cmd_y + off_y, y_min), y_max)
        if (beam_x, beam_y) != (cmd_x + off_x, cmd_y + off_y):
            self.travel_warning_label.set_text(
                _(
                    "The shifted target is outside the machine travel; "
                    "moving to the nearest reachable position instead."
                )
            )
            self.travel_warning_label.set_visible(True)
        return (beam_x - off_x, beam_y - off_y)

    def _on_move_clicked(self, button):
        self._issue_move()

    def _issue_move(self) -> bool:
        """Send the move command from the current row values.

        Returns True when a command was issued.
        """
        if not self.machine or not self.machine_cmd:
            logger.debug("Move skipped: no machine or command handler")
            return False
        x = self.x_row.get_value_in_base_units()
        y = self.y_row.get_value_in_base_units()
        z = (
            self.z_row.get_value_in_base_units()
            if self.machine.has_z_axis
            else None
        )
        dx, dy = self._get_pointer_shift()
        x, y = self._clamp_command_to_travel(x - dx, y - dy)
        logger.info(
            "Moving to X %.2f Y %.2f%s",
            x,
            y,
            f" Z {z:.2f}" if z is not None else "",
        )
        self.machine_cmd.move_to(
            self.machine, x, y, z, speed=self._get_speed()
        )
        return True

    def _on_move_to_position(self, button, position: str):
        if not self.machine or not self.machine_cmd:
            logger.debug(
                "Move to %s skipped: no machine or command handler",
                position,
            )
            return
        if not self._get_bounds_callback:
            logger.debug("Move to %s skipped: no bounds callback", position)
            return

        bounds = self._get_bounds_callback()
        if not bounds:
            logger.debug("Move to %s skipped: no bounds", position)
            return

        min_x, min_y, max_x, max_y = bounds

        # The bounds are in WORLD coordinates while the buttons refer to
        # the presented (PANEL) corners, so project the selection into
        # PANEL space before picking the corner.
        panel = self.machine.panel
        panel_min_x, panel_min_y, panel_max_x, panel_max_y = (
            panel.world_bbox_to_panel((min_x, min_y, max_x, max_y))
        )

        if position == "ll":
            panel_x, panel_y = panel_min_x, panel_min_y
        elif position == "center":
            panel_x, panel_y = (
                (panel_min_x + panel_max_x) / 2,
                (panel_min_y + panel_max_y) / 2,
            )
        elif position == "ur":
            panel_x, panel_y = panel_max_x, panel_max_y
        else:
            return

        machine_x, machine_y = panel.panel_point_to_machine(panel_x, panel_y)
        wcs_offset = self.machine.get_command_wcs_offset()
        x_off, y_off, _ = panel.get_command_offset(
            wcs_offset=wcs_offset,
            wcs_is_workarea_origin=self.machine.wcs_origin_is_workarea_origin,
        )
        cmd_x, cmd_y = self._clamp_command_to_travel(
            machine_x - x_off, machine_y - y_off
        )
        logger.info(
            "Moving to %s at command position (%.2f, %.2f)",
            position,
            cmd_x,
            cmd_y,
        )
        self.machine_cmd.move_to(
            self.machine,
            cmd_x,
            cmd_y,
            speed=self._get_speed(),
        )

    def _on_move_to_wcs_zero(self, button):
        if not self.machine or not self.machine_cmd:
            return
        dx, dy = self._get_pointer_shift()
        x, y = self._clamp_command_to_travel(-dx, -dy)
        self.machine_cmd.move_to(self.machine, x, y, speed=self._get_speed())
