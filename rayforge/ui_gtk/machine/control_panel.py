from typing import Optional
import logging
from gettext import gettext as _
from gi.repository import Gtk, Adw
from blinker import Signal
from ...logging_setup import ui_log_event_received
from ...machine.models.machine import Machine
from ...machine.driver.driver import Axis
from ...machine.driver.dummy import NoDeviceDriver
from ...machine.cmd import MachineCmd
from ...shared.tasker import task_mgr
from ..icons import get_icon
from ..shared.unit_spin_row import UnitSpinRowHelper
from .console import Console
from .jog_widget import JogWidget


logger = logging.getLogger(__name__)


class MachineControlPanel(Gtk.Box):
    notification_requested = Signal()

    def __init__(
        self,
        machine: Optional[Machine],
        machine_cmd: Optional[MachineCmd] = None,
        **kwargs,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self.machine = machine
        self.machine_cmd = machine_cmd
        self._edit_dialog = None

        self.hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.hbox.set_spacing(12)
        self.append(self.hbox)

        self.console = Console()
        self.console.set_hexpand(True)
        self.console.set_vexpand(True)
        if machine:
            self.console.set_machine(machine)
        self.console.command_submitted.connect(self._on_command_submitted)
        self.hbox.append(self.console)

        right_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        right_hbox.set_spacing(12)
        right_hbox.set_hexpand(False)
        self.hbox.append(right_hbox)

        if machine:
            self._setup_wcs_controls(right_hbox)
            self._connect_machine_signals()

        self.jog_widget = JogWidget()
        self.jog_widget.set_size_request(250, -1)
        self.jog_widget.set_hexpand(False)
        self.jog_widget.set_margin_end(12)
        self.jog_widget.set_margin_top(12)
        self.jog_widget.set_margin_bottom(12)
        self.jog_widget.set_valign(Gtk.Align.CENTER)
        right_hbox.append(self.jog_widget)

        if machine and machine_cmd:
            self.jog_widget.set_machine(machine, machine_cmd)

        ui_log_event_received.connect(self.console.on_log_received)

    def _on_command_submitted(self, sender, command: str, machine: Machine):
        async def send_command(ctx):
            try:
                await machine.run_raw(command)
            except Exception as e:
                logger.error(str(e), extra={"log_category": "ERROR"})

        task_mgr.add_coroutine(send_command)

    def _setup_wcs_controls(self, parent):
        """Set up the Work Coordinate System controls."""
        self.wcs_group = Adw.PreferencesGroup()
        self.wcs_group.set_margin_top(12)
        self.wcs_group.set_margin_bottom(12)
        self.wcs_group.set_valign(Gtk.Align.CENTER)
        parent.append(self.wcs_group)

        # Create string list from machine supported WCS
        if self.machine:
            self.wcs_list = self.machine.supported_wcs
        else:
            self.wcs_list = []
        wcs_model = Gtk.StringList.new(self.wcs_list)

        self.wcs_row = Adw.ComboRow(title=_("Active System"), model=wcs_model)
        self.wcs_row.connect(
            "notify::selected", self._on_wcs_selection_changed
        )
        self.wcs_group.add(self.wcs_row)

        self.offsets_row = Adw.ActionRow(title=_("Current Offsets"))

        # Add Edit button to row
        self.edit_offsets_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        self.edit_offsets_btn.set_tooltip_text(_("Edit Offsets Manually"))
        self.edit_offsets_btn.add_css_class("flat")
        self.edit_offsets_btn.connect("clicked", self._on_edit_offsets_clicked)
        self.offsets_row.add_suffix(self.edit_offsets_btn)

        self.wcs_group.add(self.offsets_row)

        self.position_row = Adw.ActionRow(title=_("Current Position"))
        self.wcs_group.add(self.position_row)

        # Zeroing Buttons in one ActionRow
        self.zero_row = Adw.ActionRow(title=_("Zero Axes"))
        self.wcs_group.add(self.zero_row)

        zero_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        zero_button_box.set_spacing(6)
        self.zero_row.add_suffix(zero_button_box)

        self.zero_x_btn = Gtk.Button(label=_("X"))
        self.zero_x_btn.add_css_class("flat")
        self.zero_x_btn.set_size_request(40, -1)
        self.zero_x_btn.connect("clicked", self._on_zero_axis_clicked, Axis.X)
        self.zero_x_btn.set_tooltip_text(
            _("Set current X position as 0 for active WCS")
        )
        zero_button_box.append(self.zero_x_btn)

        self.zero_y_btn = Gtk.Button(label=_("Y"))
        self.zero_y_btn.add_css_class("flat")
        self.zero_y_btn.set_size_request(40, -1)
        self.zero_y_btn.connect("clicked", self._on_zero_axis_clicked, Axis.Y)
        self.zero_y_btn.set_tooltip_text(
            _("Set current Y position as 0 for active WCS")
        )
        zero_button_box.append(self.zero_y_btn)

        self.zero_z_btn = Gtk.Button(label=_("Z"))
        self.zero_z_btn.add_css_class("flat")
        self.zero_z_btn.set_size_request(40, -1)
        self.zero_z_btn.connect("clicked", self._on_zero_axis_clicked, Axis.Z)
        self.zero_z_btn.set_tooltip_text(
            _("Set current Z position as 0 for active WCS")
        )
        zero_button_box.append(self.zero_z_btn)

        # Add Zero Here button to the same row
        self.zero_here_btn = Gtk.Button(child=get_icon("zero-here-symbolic"))
        self.zero_here_btn.set_tooltip_text(
            _("Set Work Zero at Current Position")
        )
        self.zero_here_btn.add_css_class("flat")
        self.zero_here_btn.set_size_request(40, -1)
        self.zero_here_btn.connect(
            "clicked", self._on_zero_axis_clicked, Axis.X | Axis.Y | Axis.Z
        )
        zero_button_box.append(self.zero_here_btn)

        # Jog Speed row
        speed_adjustment = Gtk.Adjustment(
            value=1000, lower=1, upper=60000, step_increment=10
        )
        self.speed_row = Adw.SpinRow(
            title=_("Jog Speed"),
            subtitle=_("Speed"),
            adjustment=speed_adjustment,
        )
        self.speed_helper = UnitSpinRowHelper(
            self.speed_row, quantity="speed", max_value_in_base=60000
        )
        self.speed_helper.set_value_in_base_units(1000)
        self.speed_helper.changed.connect(self._on_speed_changed)
        self.wcs_group.add(self.speed_row)

        # Jog Distance row
        distance_adjustment = Gtk.Adjustment(
            value=10.0, lower=0.1, upper=1000, step_increment=1
        )
        self.distance_row = Adw.SpinRow(
            title=_("Jog Distance"),
            subtitle=_("Distance in machine units"),
            adjustment=distance_adjustment,
            digits=1,
        )
        self.distance_row.connect("changed", self._on_distance_changed)
        self.wcs_group.add(self.distance_row)

        # Initial update
        self._update_wcs_ui()

    def _on_speed_changed(self, helper):
        """Handle jog speed change."""
        speed_mm_min = int(helper.get_value_in_base_units())
        self.jog_widget.jog_speed = speed_mm_min

    def _on_distance_changed(self, spin_row):
        """Handle jog distance change."""
        self.jog_widget.jog_distance = float(spin_row.get_value())

    def _connect_machine_signals(self):
        """Connect to machine signals for WCS updates."""
        if self.machine:
            self.machine.wcs_updated.connect(self._on_wcs_updated)
            self.machine.state_changed.connect(self._on_machine_state_changed)

    def _disconnect_machine_signals(self):
        """Disconnect from machine signals."""
        if self.machine:
            self.machine.wcs_updated.disconnect(self._on_wcs_updated)
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )

    def set_machine(
        self,
        machine: Optional[Machine],
        machine_cmd: Optional[MachineCmd] = None,
    ):
        """Update the machine this panel controls."""
        self._disconnect_machine_signals()

        self.machine = machine
        self.machine_cmd = machine_cmd

        self.console.set_machine(machine)

        if self.machine:
            self._connect_machine_signals()
            self._update_wcs_ui()

        if self.machine and self.machine_cmd:
            self.jog_widget.set_machine(self.machine, self.machine_cmd)

    def _on_wcs_selection_changed(self, combo_row, _pspec):
        """Handle WCS ComboRow selection change."""
        if not self.machine:
            return
        idx = combo_row.get_selected()
        if 0 <= idx < len(self.wcs_list):
            wcs = self.wcs_list[idx]
            if self.machine.active_wcs != wcs:
                self.machine.set_active_wcs(wcs)
        self._update_wcs_ui()

    def _on_zero_axis_clicked(self, button, axis):
        """Handle Zero [Axis] button click."""
        if not self.machine:
            return
        machine = self.machine
        task_mgr.add_coroutine(lambda ctx: machine.set_work_origin_here(axis))

    def _on_edit_offsets_clicked(self, button):
        """Open a dialog to edit WCS offsets manually."""
        if not self.machine:
            return

        machine = self.machine
        off_x, off_y, off_z = machine.get_active_wcs_offset()

        root = self.get_root()
        self._edit_dialog = Adw.MessageDialog(
            heading=_("Edit Work Offsets"),
            body=_(
                "Enter the offset from Machine Zero to Work Zero for "
                "the active WCS."
            ),
            transient_for=root if isinstance(root, Gtk.Window) else None,
        )
        self._edit_dialog.add_response("cancel", _("Cancel"))
        self._edit_dialog.add_response("save", _("Save"))
        self._edit_dialog.set_response_appearance(
            "save", Adw.ResponseAppearance.SUGGESTED
        )
        self._edit_dialog.set_default_response("save")
        self._edit_dialog.set_close_response("cancel")

        # Create inputs
        group = Adw.PreferencesGroup()

        row_x = Adw.SpinRow.new_with_range(-10000, 10000, 0.1)
        row_x.set_title("X Offset")
        row_x.set_value(off_x)
        group.add(row_x)

        row_y = Adw.SpinRow.new_with_range(-10000, 10000, 0.1)
        row_y.set_title("Y Offset")
        row_y.set_value(off_y)
        group.add(row_y)

        row_z = Adw.SpinRow.new_with_range(-10000, 10000, 0.1)
        row_z.set_title("Z Offset")
        row_z.set_value(off_z)
        group.add(row_z)

        self._edit_dialog.set_extra_child(group)

        def on_response(dlg, response):
            if response == "save":
                nx = row_x.get_value()
                ny = row_y.get_value()
                nz = row_z.get_value()
                task_mgr.add_coroutine(
                    lambda ctx: machine.set_work_origin(nx, ny, nz)
                )
            self._edit_dialog = None

        self._edit_dialog.connect("response", on_response)
        self._edit_dialog.present()

    def _on_wcs_updated(self, machine):
        """Handle signals when WCS offsets or active system change."""
        self._update_wcs_ui()

    def _on_machine_state_changed(self, machine, state):
        """Handle machine state changes."""
        self._update_wcs_ui()
        self.console.on_machine_state_changed(machine, state)

    def _update_wcs_ui(self):
        """Update the WCS group widgets based on machine state."""
        if not self.machine:
            return

        # Hide WCS-specific controls when workarea origin is coordinate zero
        hide_wcs_controls = self.machine.wcs_origin_is_workarea_origin
        self.wcs_row.set_visible(not hide_wcs_controls)
        self.offsets_row.set_visible(not hide_wcs_controls)
        self.zero_row.set_visible(not hide_wcs_controls)

        # Update active selection in dropdown
        current_wcs = self.machine.active_wcs
        if current_wcs in self.wcs_list:
            idx = self.wcs_list.index(current_wcs)
            if self.wcs_row.get_selected() != idx:
                self.wcs_row.set_selected(idx)

        # Update Offset Display
        off_x, off_y, off_z = self.machine.get_active_wcs_offset()
        self.offsets_row.set_subtitle(
            f"X: {off_x:.2f}   Y: {off_y:.2f}   Z: {off_z:.2f}"
        )

        # Update Position Display
        is_dummy = isinstance(self.machine.driver, NoDeviceDriver)
        is_connected = self.machine.is_connected()
        is_active = is_connected or is_dummy

        m_pos = self.machine.device_state.machine_pos
        m_x, m_y, m_z = (
            m_pos
            if m_pos and all(p is not None for p in m_pos)
            else (None, None, None)
        )

        selected_idx = self.wcs_row.get_selected()
        if 0 <= selected_idx < len(self.wcs_list):
            selected_wcs_ui = self.wcs_list[selected_idx]
        else:
            selected_wcs_ui = self.machine.active_wcs

        pos_x, pos_y, pos_z = (None, None, None)
        if m_x is not None and m_y is not None and m_z is not None:
            if selected_wcs_ui == self.machine.machine_space_wcs:
                pos_x, pos_y, pos_z = m_x, m_y, m_z
            else:
                offset = self.machine.wcs_offsets.get(
                    selected_wcs_ui, (0.0, 0.0, 0.0)
                )
                pos_x = m_x - offset[0]
                pos_y = m_y - offset[1]
                pos_z = m_z - offset[2]

        pos_str = ""
        if pos_x is not None:
            pos_str += f"X: {pos_x:.2f}   "
        if pos_y is not None:
            pos_str += f"Y: {pos_y:.2f}   "
        if pos_z is not None:
            pos_str += f"Z: {pos_z:.2f}"

        if not is_active:
            self.position_row.set_subtitle(_("Offline - Position Unknown"))
        else:
            self.position_row.set_subtitle(pos_str if pos_str else "---")

        # Update Button Sensitivity
        is_mcs = current_wcs == self.machine.machine_space_wcs
        can_zero = is_active and not is_mcs
        can_manual = not is_mcs

        self.zero_x_btn.set_sensitive(can_zero)
        self.zero_y_btn.set_sensitive(can_zero)
        self.zero_z_btn.set_sensitive(can_zero)
        self.zero_here_btn.set_sensitive(can_zero)
        self.edit_offsets_btn.set_sensitive(can_manual)

        if is_mcs:
            msg = _(
                "Offsets cannot be set in Machine Coordinate Mode ({wcs})"
            ).format(wcs=self.machine.machine_space_wcs_display_name)
        elif not is_active:
            msg = _("Machine must be connected to set Zero Here")
        else:
            msg = _("Set current position as 0")

        self.zero_here_btn.set_tooltip_text(msg)
