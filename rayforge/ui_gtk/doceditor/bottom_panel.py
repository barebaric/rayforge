from typing import Optional, Callable, Tuple, TYPE_CHECKING
import logging
from gettext import gettext as _
from gi.repository import Gtk, Adw
from blinker import Signal
from ...core.ops.axis import Axis
from ...logging_setup import ui_log_event_received
from ...machine.models.machine import Machine
from ...machine.driver.dummy import NoDeviceDriver
from ...machine.cmd import MachineCmd
from ..shared.adwfix import get_spinrow_float
from ...shared.gcodeedit.viewer import GcodeViewer
from ...shared.tasker import task_mgr
from ..doceditor.layers_tab import LayersTab
from ..icons import get_icon
from ..machine.console import Console
from ..machine.jog_widget import JogWidget
from ..machine.wcs_dialog import WcsDialog
from ..shared.dock_item import DockItem
from ..shared.dock_layout import DockLayout
from ..shared.gtk import apply_css
from ..shared.responsive_box import ResponsiveBox
from ..shared.unit_spin_row import UnitSpinRowHelper
from .asset_browser import AssetBrowser

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

controls_css = """
preferencesgroup.compact list {
    margin-left: 0;
    margin-right: 0;
}
"""

apply_css(controls_css)


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
        self.layout_changed = Signal()
        self.edit_item_requested = Signal()
        self.machine = machine
        self.machine_cmd = machine_cmd
        self.doc = None
        self._edit_dialog = None
        self._click_to_zero_mode = False
        self._updating_wcs_ui = False
        self._active_layer = None
        self._get_bounds_callback: Optional[
            Callable[[], Optional[Tuple[float, float, float, float]]]
        ] = None

        self.console = Console()
        self.console.set_hexpand(True)
        self.console.set_vexpand(True)
        if machine:
            self.console.set_machine(machine)
        self.console.command_submitted.connect(self._on_command_submitted)

        ui_log_event_received.connect(self.console.on_log_received)

        self.layers_tab = LayersTab(doc_editor)
        self.layers_tab.edit_item_requested.connect(
            self._on_layers_tab_edit_item
        )

        self.asset_browser = AssetBrowser(doc_editor)

        self.gcode_viewer = GcodeViewer()
        self.gcode_viewer.set_margin_start(0)
        self.gcode_viewer.set_margin_end(0)
        self.gcode_viewer.set_margin_top(9)
        self.gcode_viewer.set_margin_bottom(9)

        self.jog_widget = JogWidget()
        if machine and machine_cmd:
            self.jog_widget.set_machine(machine, machine_cmd)

        self._controls_widget = ResponsiveBox()
        self._controls_widget.set_halign(Gtk.Align.FILL)
        self._controls_widget.set_vexpand(True)
        self._controls_widget.set_valign(Gtk.Align.FILL)
        self._controls_widget.set_margin_end(9)
        self._controls_widget.set_margin_top(9)
        self._controls_widget.set_margin_bottom(9)

        if machine:
            self._setup_wcs_controls()
            self._connect_machine_signals()
            self._controls_widget.set_children(self.wcs_group, self.jog_widget)
        else:
            self._controls_widget.set_children(self.jog_widget)

        self.dock_layout = DockLayout(orientation=Gtk.Orientation.HORIZONTAL)
        self.dock_layout.layout_changed.connect(self._on_dock_layout_changed)
        self.dock_layout.tab_changed.connect(self._on_dock_tab_changed)

        self._register_items()
        self._build_default_layout()
        self.append(self.dock_layout)

    def _register_items(self):
        self.dock_layout.register_item(
            DockItem(
                name="layers",
                icon_name="layers-symbolic",
                widget=self.layers_tab,
                label=_("Layers"),
            )
        )
        self.dock_layout.register_item(
            DockItem(
                name="assets",
                icon_name="image-x-generic-symbolic",
                widget=self.asset_browser,
                label=_("Assets"),
            )
        )
        self.dock_layout.register_item(
            DockItem(
                name="gcode",
                icon_name="gcode-symbolic",
                widget=self.gcode_viewer,
                label=_("G-code Viewer"),
            )
        )
        self.dock_layout.register_item(
            DockItem(
                name="console",
                icon_name="terminal-symbolic",
                widget=self.console,
                label=_("Console"),
            )
        )
        self.dock_layout.register_item(
            DockItem(
                name="controls",
                icon_name="jog-symbolic",
                widget=self._controls_widget,
                label=_("Controls"),
                expands=False,
            )
        )

    def _build_default_layout(self):
        tabs_area = self.dock_layout.add_area()
        tabs_area.add_item(self.dock_layout.get_item("layers"))
        tabs_area.add_item(self.dock_layout.get_item("assets"))
        tabs_area.add_item(self.dock_layout.get_item("gcode"))
        tabs_area.add_item(self.dock_layout.get_item("console"))

        controls_area = self.dock_layout.add_area()
        controls_area.add_item(self.dock_layout.get_item("controls"))

    def to_dict(self):
        return {
            "visible": self.get_visible(),
            "areas": self.dock_layout.get_layout()["areas"],
        }

    def from_dict(self, data):
        if not data:
            return
        visible = data.get("visible", False)
        self.set_visible(visible)
        areas = data.get("areas")
        if areas:
            self.dock_layout.apply_layout({"areas": areas})

    def is_item_visible(self, name):
        area = self.dock_layout.find_item_area(name)
        if area is None:
            return False
        active = area.get_active_item()
        return active == name

    def _on_dock_layout_changed(self, sender):
        self.layout_changed.send(self)

    def _on_dock_tab_changed(self, sender, *, name):
        self.tab_changed.send(self, name=name)

    def set_doc(self, doc):
        self._disconnect_layer_signals()
        self.doc = doc
        self.asset_browser.set_doc(doc)
        self.layers_tab.set_doc(doc)
        if doc:
            doc.active_layer_changed.connect(self._on_active_layer_changed)
            self._connect_layer_signals()
        if self.machine:
            self._update_wcs_ui()

    def _on_layers_tab_edit_item(self, sender, **kwargs):
        self.edit_item_requested.send(sender, **kwargs)

    def _on_active_layer_changed(self, sender):
        self._disconnect_layer_signals()
        self._connect_layer_signals()
        if self.machine:
            self._update_wcs_ui()

    def _connect_layer_signals(self):
        if self.doc and self.doc.active_layer:
            self._active_layer = self.doc.active_layer
            self._active_layer.updated.connect(self._on_layer_updated)

    def _disconnect_layer_signals(self):
        if self._active_layer:
            self._active_layer.updated.disconnect(self._on_layer_updated)
            self._active_layer = None

    def _on_layer_updated(self, sender):
        if self.machine:
            self._update_wcs_ui()

    def _on_command_submitted(self, sender, command: str, machine: Machine):
        async def send_command(ctx):
            try:
                await machine.run_raw(command)
            except Exception as e:
                logger.error(str(e), extra={"log_category": "ERROR"})

        task_mgr.add_coroutine(send_command)

    def _setup_wcs_controls(self):
        self.wcs_group = Adw.PreferencesGroup()
        self.wcs_group.add_css_class("compact")

        if self.machine:
            self.wcs_list = self.machine.supported_wcs
        else:
            self.wcs_list = []
        wcs_model = Gtk.StringList.new(self.wcs_list)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_wcs_factory_setup)
        factory.connect("bind", self._on_wcs_factory_bind)

        self.wcs_row = Adw.ComboRow(
            model=wcs_model,
            factory=factory,
            use_subtitle=True,
        )
        self.wcs_row.connect(
            "notify::selected", self._on_wcs_selection_changed
        )
        self.wcs_group.add(self.wcs_row)

        self.offsets_row = Adw.ActionRow(title=_("Current Offsets"))

        self.edit_offsets_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        self.edit_offsets_btn.set_tooltip_text(_("Edit Offsets Manually"))
        self.edit_offsets_btn.add_css_class("flat")
        self.edit_offsets_btn.set_valign(Gtk.Align.CENTER)
        self.edit_offsets_btn.connect("clicked", self._on_edit_offsets_clicked)
        self.wcs_row.add_suffix(self.edit_offsets_btn)

        self.wcs_group.add(self.wcs_row)

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
        self.jog_widget.jog_distance = get_spinrow_float(spin_row)

    def _connect_machine_signals(self):
        if self.machine:
            self.machine.wcs_updated.connect(self._on_wcs_updated)
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.machine.changed.connect(self._on_wcs_updated)

    def _disconnect_machine_signals(self):
        if self.machine:
            self.machine.wcs_updated.disconnect(self._on_wcs_updated)
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self.machine.changed.disconnect(self._on_wcs_updated)

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
        if self._updating_wcs_ui:
            return
        if not self.machine:
            return
        machine = self.machine
        idx = combo_row.get_selected()
        if 0 <= idx < len(self.wcs_list):
            wcs = self.wcs_list[idx]
            if machine.active_wcs != wcs:
                task_mgr.add_coroutine(
                    lambda ctx, w=wcs: machine.switch_active_wcs(w),
                    key=(machine.id, "select-wcs"),
                )

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

        root = self.get_root()
        self._edit_dialog = WcsDialog(
            machine=self.machine,
            transient_for=root if isinstance(root, Gtk.Window) else None,
        )
        self._edit_dialog.connect(
            "destroy", lambda *_: setattr(self, "_edit_dialog", None)
        )
        self._edit_dialog.present()

    def _on_wcs_updated(self, machine):
        self._update_wcs_ui()

    def _on_machine_state_changed(self, machine, state):
        self._update_wcs_ui()
        self.console.on_machine_state_changed(machine, state)

    def _on_wcs_factory_setup(self, factory, list_item):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        name_label = Gtk.Label(xalign=0)
        subtitle_label = Gtk.Label(xalign=0)
        subtitle_label.add_css_class("dim-label")
        box.append(name_label)
        box.append(subtitle_label)
        list_item.set_child(box)

    def _on_wcs_factory_bind(self, factory, list_item):
        idx = list_item.get_position()
        if idx < 0 or idx >= len(self.wcs_list):
            return
        wcs_name = self.wcs_list[idx]
        box = list_item.get_child()
        name_label = box.get_first_child()
        subtitle_label = name_label.get_next_sibling()
        if self.machine:
            label = self.machine.get_wcs_label(wcs_name)
            if label:
                name_label.set_label(f"{wcs_name} ({label})")
            else:
                name_label.set_label(wcs_name)
            off = self.machine.get_wcs_offset(wcs_name)
            subtitle_label.set_label(
                f"X: {off[0]:.2f} Y: {off[1]:.2f} Z: {off[2]:.2f}"
            )
            subtitle_label.set_visible(True)
        else:
            name_label.set_label(wcs_name)
            subtitle_label.set_visible(False)

    def _update_wcs_ui(self):
        if not self.machine:
            return

        hide_wcs_controls = self.machine.wcs_origin_is_workarea_origin
        self.wcs_row.set_visible(not hide_wcs_controls)
        self.zero_row.set_visible(not hide_wcs_controls)

        layer_has_wcs = (
            self.doc and self.doc.active_layer and self.doc.active_layer.wcs
        )
        self.wcs_row.set_sensitive(not layer_has_wcs)
        if layer_has_wcs:
            self.wcs_row.set_tooltip_text(
                _(
                    "Overridden by the current layer. "
                    "Change it in the layer settings."
                )
            )
        else:
            self.wcs_row.set_tooltip_text("")

        current_wcs = self.machine.active_wcs
        if current_wcs in self.wcs_list:
            idx = self.wcs_list.index(current_wcs)
            if self.wcs_row.get_selected() != idx:
                self._updating_wcs_ui = True
                self.wcs_row.set_selected(idx)
                self._updating_wcs_ui = False

        wcs_label = self.machine.get_wcs_label(current_wcs)
        if wcs_label:
            title = f"{current_wcs} ({wcs_label})"
        else:
            title = current_wcs
        self.wcs_row.set_title(title)

        off_x, off_y, off_z = self.machine.get_active_wcs_offset()
        self.wcs_row.set_subtitle(
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
                offset = self.machine.get_wcs_offset(selected_wcs_ui)
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
