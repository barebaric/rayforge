from typing import Optional, Callable, Tuple, TYPE_CHECKING
import logging
from gettext import gettext as _
from gi.repository import Gtk, Adw
from blinker import Signal
from ...logging_setup import ui_log_event_received
from ...machine.models.machine import Machine
from ...machine.driver.driver import Axis
from ...machine.driver.dummy import NoDeviceDriver
from ...machine.cmd import MachineCmd
from ...shared.gcodeedit.viewer import GcodeViewer
from ...shared.tasker import task_mgr
from ..icons import get_icon
from ..machine.console import Console
from ..machine.jog_widget import JogWidget
from ..shared.icon_tab_widget import IconTabWidget
from ..shared.unit_spin_row import UnitSpinRowHelper
from .asset_browser import AssetBrowser

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class BottomPanel(Gtk.Box):
    def __init__(
        self,
        machine: Optional[Machine],
        doc_editor: "DocEditor",
        machine_cmd: Optional[MachineCmd] = None,
        **kwargs,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self.notification_requested = Signal()
        self.click_to_zero_mode_changed = Signal()
        self.tab_changed = Signal()
        self.tab_order_changed = Signal()
        self.machine = machine
        self.machine_cmd = machine_cmd
        self._edit_dialog = None
        self._click_to_zero_mode = False
        self._get_bounds_callback: Optional[
            Callable[[], Optional[Tuple[float, float, float, float]]]
        ] = None

        self.hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.hbox.set_spacing(12)
        self.append(self.hbox)

        self.tab_widget = IconTabWidget()
        self.tab_widget.set_hexpand(True)
        self.tab_widget.set_vexpand(True)

        self.hbox.append(self.tab_widget)

        right_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        right_hbox.set_spacing(12)
        right_hbox.set_hexpand(False)
        self.hbox.append(right_hbox)

        if machine:
            self._setup_wcs_controls(right_hbox)
            self._connect_machine_signals()

        self.jog_widget = JogWidget()
        self.jog_widget.set_vexpand(True)
        self.jog_widget.set_valign(Gtk.Align.FILL)
        self.jog_widget.set_hexpand(False)
        self.jog_widget.set_margin_end(15)
        self.jog_widget.set_margin_top(9)
        self.jog_widget.set_margin_bottom(9)
        right_hbox.append(self.jog_widget)

        if machine and machine_cmd:
            self.jog_widget.set_machine(machine, machine_cmd)

        self.console = Console()
        self.console.set_hexpand(True)
        self.console.set_vexpand(True)
        if machine:
            self.console.set_machine(machine)
        self.console.command_submitted.connect(self._on_command_submitted)

        self.asset_browser = AssetBrowser(doc_editor)

        self.gcode_viewer = GcodeViewer()

        self.tab_widget.add_tab(
            "assets",
            "image-x-generic-symbolic",
            self.asset_browser,
            _("Assets"),
        )
        self.tab_widget.add_tab(
            "gcode",
            "gcode-symbolic",
            self.gcode_viewer,
            _("G-code Viewer"),
        )
        self.tab_widget.add_tab(
            "console", "terminal-symbolic", self.console, _("Console")
        )

        ui_log_event_received.connect(self.console.on_log_received)

        self.tab_widget.tab_changed.connect(self._on_tab_changed)
        self.tab_widget.tab_order_changed.connect(self._on_tab_order_changed)

    def apply_saved_state(self, tab_order, active_tab):
        known = self.tab_widget.get_tab_order()
        known_set = set(known)
        if tab_order:
            filtered = [n for n in tab_order if n in known_set]
            self.tab_widget.set_tab_order(filtered)

        if active_tab and active_tab in known_set:
            self.tab_widget.set_current_tab(active_tab)
        elif known:
            self.tab_widget.set_current_tab(known[0])

    def _on_tab_changed(self, sender, *, name: str):
        self.tab_changed.send(self, name=name)

    def _on_tab_order_changed(self, sender):
        self.tab_order_changed.send(self)

    def set_doc(self, doc):
        self.asset_browser.set_doc(doc)

    def _on_command_submitted(self, sender, command: str, machine: Machine):
        async def send_command(ctx):
            try:
                await machine.run_raw(command)
            except Exception as e:
                logger.error(str(e), extra={"log_category": "ERROR"})

        task_mgr.add_coroutine(send_command)

    def _setup_wcs_controls(self, parent):
        self.wcs_group = Adw.PreferencesGroup()
        self.wcs_group.set_margin_top(9)
        self.wcs_group.set_margin_bottom(9)
        self.wcs_group.set_valign(Gtk.Align.CENTER)
        parent.append(self.wcs_group)

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

        self.edit_offsets_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        self.edit_offsets_btn.set_tooltip_text(_("Edit Offsets Manually"))
        self.edit_offsets_btn.add_css_class("flat")
        self.edit_offsets_btn.connect("clicked", self._on_edit_offsets_clicked)
        self.offsets_row.add_suffix(self.edit_offsets_btn)

        self.wcs_group.add(self.offsets_row)

        self.position_row = Adw.ActionRow(title=_("Current Position"))
        self.wcs_group.add(self.position_row)

        position_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        position_button_box.set_spacing(6)
        self.position_row.add_suffix(position_button_box)

        self.move_ll_btn = Gtk.Button(child=get_icon("bottom-left-symbolic"))
        self.move_ll_btn.add_css_class("flat")
        self.move_ll_btn.set_size_request(40, -1)
        self.move_ll_btn.connect("clicked", self._on_move_to_position, "ll")
        self.move_ll_btn.set_tooltip_text(
            _("Move to Lower-Left of Selection or Workarea")
        )
        position_button_box.append(self.move_ll_btn)

        self.move_center_btn = Gtk.Button(child=get_icon("center-symbolic"))
        self.move_center_btn.add_css_class("flat")
        self.move_center_btn.set_size_request(40, -1)
        self.move_center_btn.connect(
            "clicked", self._on_move_to_position, "center"
        )
        self.move_center_btn.set_tooltip_text(
            _("Move to Center of Selection or Workarea")
        )
        position_button_box.append(self.move_center_btn)

        self.move_ur_btn = Gtk.Button(child=get_icon("top-right-symbolic"))
        self.move_ur_btn.add_css_class("flat")
        self.move_ur_btn.set_size_request(40, -1)
        self.move_ur_btn.connect("clicked", self._on_move_to_position, "ur")
        self.move_ur_btn.set_tooltip_text(
            _("Move to Upper-Right of Selection or Workarea")
        )
        position_button_box.append(self.move_ur_btn)

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

        self._click_to_zero_icon = get_icon("crosshairs-symbolic")
        self.click_to_zero_btn = Gtk.Button(child=self._click_to_zero_icon)
        self.click_to_zero_btn.set_tooltip_text(
            _("Click Canvas to Set Work Zero")
        )
        self.click_to_zero_btn.add_css_class("flat")
        self.click_to_zero_btn.set_size_request(40, -1)
        self.click_to_zero_btn.connect(
            "clicked", self._on_click_to_zero_toggled
        )
        zero_button_box.append(self.click_to_zero_btn)
        self.click_to_zero_btn.set_tooltip_text(
            _("Click on canvas to set work zero")
        )

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

        self._update_wcs_ui()

    def _on_speed_changed(self, helper):
        speed_mm_min = int(helper.get_value_in_base_units())
        self.jog_widget.jog_speed = speed_mm_min

    def _on_distance_changed(self, spin_row):
        self.jog_widget.jog_distance = float(spin_row.get_value())

    def _connect_machine_signals(self):
        if self.machine:
            self.machine.wcs_updated.connect(self._on_wcs_updated)
            self.machine.state_changed.connect(self._on_machine_state_changed)

    def _disconnect_machine_signals(self):
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
        if not self.machine:
            return
        idx = combo_row.get_selected()
        if 0 <= idx < len(self.wcs_list):
            wcs = self.wcs_list[idx]
            if self.machine.active_wcs != wcs:
                self.machine.set_active_wcs(wcs)
        self._update_wcs_ui()

    def _on_zero_axis_clicked(self, button, axis):
        if not self.machine:
            return
        machine = self.machine
        task_mgr.add_coroutine(lambda ctx: machine.set_work_origin_here(axis))

    def set_click_to_zero_mode(self, active: bool):
        if self._click_to_zero_mode != active:
            self._click_to_zero_mode = active
            self._update_wcs_ui()
            self.click_to_zero_mode_changed.send(self, active=active)

    def set_get_bounds_callback(
        self,
        callback: Optional[
            Callable[[], Optional[Tuple[float, float, float, float]]]
        ],
    ):
        self._get_bounds_callback = callback

    def update_position_menu_sensitivity(self):
        if not self.machine:
            return
        is_dummy = isinstance(self.machine.driver, NoDeviceDriver)
        is_connected = self.machine.is_connected()
        is_active = is_connected or is_dummy

        has_bounds = (
            self._get_bounds_callback is not None
            and self._get_bounds_callback() is not None
        )
        self.move_ll_btn.set_sensitive(has_bounds and is_active)
        self.move_center_btn.set_sensitive(has_bounds and is_active)
        self.move_ur_btn.set_sensitive(has_bounds and is_active)

    def _on_move_to_position(self, button, position: str):
        if not self.machine or not self.machine_cmd:
            return
        if not self._get_bounds_callback:
            return

        bounds = self._get_bounds_callback()
        if not bounds:
            return

        min_x, min_y, max_x, max_y = bounds

        if position == "ll":
            world_x, world_y = min_x, min_y
        elif position == "center":
            world_x, world_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        elif position == "ur":
            world_x, world_y = max_x, max_y
        else:
            return

        space = self.machine.get_coordinate_space()
        machine_x, machine_y = space.world_point_to_machine(world_x, world_y)
        self.machine_cmd.move_to(self.machine, machine_x, machine_y)

    def _on_click_to_zero_toggled(self, button):
        self.set_click_to_zero_mode(not self._click_to_zero_mode)

    def _on_edit_offsets_clicked(self, button):
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
        self._update_wcs_ui()

    def _on_machine_state_changed(self, machine, state):
        self._update_wcs_ui()
        self.console.on_machine_state_changed(machine, state)

    def _update_wcs_ui(self):
        if not self.machine:
            return

        hide_wcs_controls = self.machine.wcs_origin_is_workarea_origin
        self.wcs_row.set_visible(not hide_wcs_controls)
        self.offsets_row.set_visible(not hide_wcs_controls)
        self.zero_row.set_visible(not hide_wcs_controls)

        current_wcs = self.machine.active_wcs
        if current_wcs in self.wcs_list:
            idx = self.wcs_list.index(current_wcs)
            if self.wcs_row.get_selected() != idx:
                self.wcs_row.set_selected(idx)

        off_x, off_y, off_z = self.machine.get_active_wcs_offset()
        self.offsets_row.set_subtitle(
            f"X: {off_x:.2f}   Y: {off_y:.2f}   Z: {off_z:.2f}"
        )

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

        is_mcs = current_wcs == self.machine.machine_space_wcs
        can_zero = is_active and not is_mcs
        can_manual = not is_mcs

        self.zero_x_btn.set_sensitive(can_zero)
        self.zero_y_btn.set_sensitive(can_zero)
        self.zero_z_btn.set_sensitive(can_zero)
        self.zero_here_btn.set_sensitive(can_zero)
        self.edit_offsets_btn.set_sensitive(can_manual)

        self.update_position_menu_sensitivity()

        if is_mcs:
            msg = _(
                "Offsets cannot be set in Machine Coordinate Mode ({wcs})"
            ).format(wcs=self.machine.machine_space_wcs_display_name)
        elif not is_active:
            msg = _("Machine must be connected to set Zero Here")
        else:
            msg = _("Set current position as 0")

        self.zero_here_btn.set_tooltip_text(msg)
