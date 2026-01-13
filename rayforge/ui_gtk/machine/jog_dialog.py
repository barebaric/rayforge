from gi.repository import Adw, Gdk, Gtk

from ...machine.cmd import MachineCmd
from ...machine.driver.driver import Axis
from ...machine.driver.dummy import NoDeviceDriver
from ...machine.models.machine import Machine
from ...shared.tasker import task_mgr
from ..shared.adwfix import get_spinrow_float, get_spinrow_int
from ..shared.patched_dialog_window import PatchedDialogWindow
from .jog_widget import JogWidget
from ..icons import get_icon


class JogDialog(PatchedDialogWindow):
    """Dialog for manually jogging the machine."""

    def __init__(self, *, machine: Machine, machine_cmd: MachineCmd, **kwargs):
        super().__init__(**kwargs)
        self.machine = machine
        self.machine_cmd = machine_cmd
        self._edit_dialog = None  # Reference to keep dialog alive

        self.set_title(_("Machine Jog Control"))
        self.set_default_size(600, 800)
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
            self.machine.wcs_updated.connect(self._on_wcs_updated)

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

        # --- Work Coordinate System Group ---
        wcs_group = Adw.PreferencesGroup(title=_("Work Coordinates"))
        page.add(wcs_group)

        # Create string list from machine supported WCS
        self.wcs_list = self.machine.supported_wcs
        wcs_model = Gtk.StringList.new(self.wcs_list)

        self.wcs_row = Adw.ComboRow(title=_("Active System"), model=wcs_model)
        self.wcs_row.connect(
            "notify::selected", self._on_wcs_selection_changed
        )
        wcs_group.add(self.wcs_row)

        self.offsets_row = Adw.ActionRow(title=_("Current Offsets"))

        # Add Zero Here button to row
        self.zero_here_btn = Gtk.Button(child=get_icon("zero-here-symbolic"))
        self.zero_here_btn.set_tooltip_text(
            _("Set Work Zero at Current Position")
        )
        self.zero_here_btn.add_css_class("flat")
        self.zero_here_btn.connect(
            "clicked", self._on_zero_axis_clicked, Axis.X | Axis.Y | Axis.Z
        )
        self.offsets_row.add_suffix(self.zero_here_btn)

        # Add Edit button to row
        self.edit_offsets_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        self.edit_offsets_btn.set_tooltip_text(_("Edit Offsets Manually"))
        self.edit_offsets_btn.add_css_class("flat")
        self.edit_offsets_btn.connect("clicked", self._on_edit_offsets_clicked)
        self.offsets_row.add_suffix(self.edit_offsets_btn)

        wcs_group.add(self.offsets_row)

        self.position_row = Adw.ActionRow(title=_("Current Position"))
        wcs_group.add(self.position_row)

        # Zeroing Buttons (Individual Axes)
        zero_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        zero_box.set_margin_top(12)
        wcs_group.add(zero_box)

        self.zero_x_btn = Gtk.Button(label=_("Zero X"))
        self.zero_x_btn.add_css_class("pill")
        self.zero_x_btn.connect("clicked", self._on_zero_axis_clicked, Axis.X)
        self.zero_x_btn.set_tooltip_text(
            _("Set current X position as 0 for active WCS")
        )
        zero_box.append(self.zero_x_btn)

        self.zero_y_btn = Gtk.Button(label=_("Zero Y"))
        self.zero_y_btn.add_css_class("pill")
        self.zero_y_btn.connect("clicked", self._on_zero_axis_clicked, Axis.Y)
        self.zero_y_btn.set_tooltip_text(
            _("Set current Y position as 0 for active WCS")
        )
        zero_box.append(self.zero_y_btn)

        self.zero_z_btn = Gtk.Button(label=_("Zero Z"))
        self.zero_z_btn.add_css_class("pill")
        self.zero_z_btn.connect("clicked", self._on_zero_axis_clicked, Axis.Z)
        self.zero_z_btn.set_tooltip_text(
            _("Set current Z position as 0 for active WCS")
        )
        zero_box.append(self.zero_z_btn)

        # --- Jog Widget ---
        self.jog_widget = JogWidget()
        self.jog_widget.set_machine(machine, machine_cmd)
        page.add(self.jog_widget)

        # --- Jog Settings ---
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

        # Initial Update
        self._update_button_sensitivity()
        self._update_wcs_ui()

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
            self.machine.wcs_updated.disconnect(self._on_wcs_updated)
        return False  # Allow the window to close

    def _on_speed_changed(self, spin_row):
        """Handle jog speed change."""
        self.jog_widget.jog_speed = get_spinrow_int(spin_row)

    def _on_distance_changed(self, spin_row):
        """Handle jog distance change."""
        self.jog_widget.jog_distance = get_spinrow_float(spin_row)

    def _on_edit_offsets_clicked(self, button):
        """Open a dialog to edit WCS offsets manually."""
        if not self.machine:
            return

        off_x, off_y, off_z = self.machine.get_active_wcs_offset()

        self._edit_dialog = Adw.MessageDialog(
            heading=_("Edit Work Offsets"),
            body=_(
                "Enter the offset from Machine Zero to Work Zero for "
                "the active WCS."
            ),
            transient_for=self,
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
                    lambda ctx: self.machine.set_work_origin(nx, ny, nz)
                )
            self._edit_dialog = None

        self._edit_dialog.connect("response", on_response)
        self._edit_dialog.present()

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

    def _on_home_all_clicked(self, button):
        """Handle Home All button click."""
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine)

    def _on_wcs_selection_changed(self, combo_row, _pspec):
        """Handle WCS ComboRow selection change."""
        if not self.machine:
            return
        idx = combo_row.get_selected()
        if 0 <= idx < len(self.wcs_list):
            wcs = self.wcs_list[idx]
            if self.machine.active_wcs != wcs:
                self.machine.set_active_wcs(wcs)
        # Also update the position display immediately
        self._update_wcs_ui()

    def _on_zero_axis_clicked(self, button, axis):
        """Handle Zero [Axis] button click."""
        if not self.machine:
            return
        # Schedule the async machine operation
        task_mgr.add_coroutine(
            lambda ctx: self.machine.set_work_origin_here(axis)
        )

    def _on_wcs_updated(self, machine):
        """Handle signals when WCS offsets or active system change."""
        self._update_wcs_ui()

    def _update_wcs_ui(self):
        """Update the WCS group widgets based on machine state."""
        if not self.machine:
            return

        # 1. Update active selection in dropdown
        current_wcs = self.machine.active_wcs
        if current_wcs in self.wcs_list:
            idx = self.wcs_list.index(current_wcs)
            if self.wcs_row.get_selected() != idx:
                # Block handler to prevent loop if desired, though
                # _on_wcs_selection_changed checks for equality
                self.wcs_row.set_selected(idx)

        # 2. Update Offset Display
        off_x, off_y, off_z = self.machine.get_active_wcs_offset()
        self.offsets_row.set_subtitle(
            f"X: {off_x:.2f}   Y: {off_y:.2f}   Z: {off_z:.2f}"
        )

        # 3. Update Position Display (Calculated for selected WCS)
        is_dummy = isinstance(self.machine.driver, NoDeviceDriver)
        is_connected = self.machine.is_connected()
        # Treat Dummy as effectively connected for WCS/Position display
        # purposes because it maintains state even if strictly
        # 'disconnected' in driver status
        is_active = is_connected or is_dummy

        m_pos = self.machine.device_state.machine_pos
        m_x, m_y, m_z = (
            m_pos
            if m_pos and all(p is not None for p in m_pos)
            else (None, None, None)
        )

        # Check for unlisted WCS (e.g. machine space or unexpected)
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

        # 4. Update Button Sensitivity
        # Cannot set offsets for machine space WCS
        is_mcs = current_wcs == self.machine.machine_space_wcs

        # Zero Here requires machine position (Active) and writable WCS
        # (not machine space)
        can_zero = is_active and not is_mcs

        # Manual entry requires writable WCS, works offline
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

        # Update tooltips to reflect status
        self.zero_here_btn.set_tooltip_text(msg)

    def _update_button_sensitivity(self):
        """Update button sensitivity based on machine capabilities."""
        has_machine = self.machine is not None
        is_connected = has_machine and self.machine.is_connected()
        single_axis_homing_enabled = (
            has_machine and self.machine.single_axis_homing_enabled
        )

        # Home buttons
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

        # Tooltips for home buttons
        tooltip = (
            None
            if single_axis_homing_enabled
            else _("Single axis homing is disabled in machine settings")
        )
        self.home_x_btn.set_tooltip_text(tooltip)
        self.home_y_btn.set_tooltip_text(tooltip)
        self.home_z_btn.set_tooltip_text(tooltip)

        # Update jog widget sensitivity
        self.jog_widget._update_button_sensitivity()

        # Update WCS UI (buttons depend on connection)
        self._update_wcs_ui()

    def _on_connection_status_changed(self, machine, status, message=None):
        """Handle machine connection status changes."""
        self._update_button_sensitivity()

    def _on_machine_state_changed(self, machine, state):
        """Handle machine state changes."""
        self._update_button_sensitivity()
        self._update_wcs_ui()

    def _on_machine_changed(self, machine, **kwargs):
        """Handle machine configuration changes."""
        self._update_button_sensitivity()
