"""Touch-friendly machine control surface for shop mode.

When shop mode is enabled these two panels replace the WCS
preferences group and the jog widget in the bottom panel's Controls
tab: large per-axis position readouts with zero buttons, an offsets
card, big speed/step steppers, and a grid of oversized jog and job
buttons meant for use while standing at the machine.
"""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk
from raygeo.ops.axis import Axis

from ...context import get_context
from ...machine.cmd import MachineCmd
from ...machine.driver.driver import DeviceStatus
from ...machine.models.machine import JogDirection, Machine
from ...shared.tasker import task_mgr
from ...shared.units.formatter import get_preferred_unit
from ..icons import get_icon
from .wcs_dialog import WcsDialog

# Jog speed change per stepper tap, in mm/min.
_SPEED_STEP = 100
_SPEED_RANGE = (1, 60000)
# Jog distance bounds in mm; taps step one display unit.
_DISTANCE_RANGE = (0.1, 1000)

_AXES = (Axis.X, Axis.Y, Axis.Z)
_AXIS_NAMES = {Axis.X: "X", Axis.Y: "Y", Axis.Z: "Z"}
_AXIS_CSS = {
    Axis.X: "shop-mode-axis-x",
    Axis.Y: "shop-mode-axis-y",
    Axis.Z: "shop-mode-axis-z",
}


def _clamp(value, bounds):
    return max(bounds[0], min(bounds[1], value))


class ShopControlPanel(Gtk.Box):
    """Left column of the shop-mode surface: DRO, offsets and the
    jog speed/step steppers."""

    #: Emitted with ``speed`` (mm/min) when a stepper tap changes it.
    speed_changed = Signal()
    #: Emitted with ``distance`` (mm) when a stepper tap changes it.
    distance_changed = Signal()

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.set_spacing(10)
        self.machine: Machine | None = None
        self.machine_cmd: MachineCmd | None = None
        self._jog_speed = 1000
        self._jog_distance = 10.0
        self._syncing_alignment = False
        self._edit_dialog = None

        self._axis_values: dict[Axis, Gtk.Label] = {}
        self._axis_zero_btns: dict[Axis, Gtk.Button] = {}
        self._axis_rows: dict[Axis, Gtk.Widget] = {}
        self._offset_labels: dict[Axis, Gtk.Label] = {}

        dro_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        dro_card.set_spacing(6)
        dro_card.add_css_class("shop-mode-card")
        for axis in _AXES:
            dro_card.append(self._build_axis_row(axis))
        self.append(dro_card)

        self.append(self._build_offsets_card())
        self.append(self._build_steppers())

        self._config_handler_id = get_context().config.changed.connect(
            self._on_config_changed
        )
        self.connect("destroy", self._on_destroy)

    # ----- construction --------------------------------------------------

    def _build_axis_row(self, axis: Axis) -> Gtk.Widget:
        name = _AXIS_NAMES[axis]
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        tile = Gtk.Box()
        tile.add_css_class("shop-mode-axis-tile")
        letter = Gtk.Label(label=name)
        letter.add_css_class("shop-mode-axis")
        letter.add_css_class(_AXIS_CSS[axis])
        tile.append(letter)
        row.append(tile)

        value = Gtk.Label(label="---")
        value.add_css_class("shop-mode-value")
        value.set_hexpand(True)
        value.set_xalign(0.0)
        row.append(value)
        self._axis_values[axis] = value

        zero_btn = Gtk.Button(child=get_icon("zero-here-symbolic"))
        zero_btn.add_css_class("flat")
        zero_btn.add_css_class("shop-mode-zero")
        zero_btn.set_tooltip_text(
            _("Set current {axis} position as 0 for active WCS").format(
                axis=name
            )
        )
        zero_btn.connect("clicked", self._on_zero_clicked, axis)
        row.append(zero_btn)
        self._axis_zero_btns[axis] = zero_btn

        self._axis_rows[axis] = row
        return row

    def _build_offsets_card(self) -> Gtk.Widget:
        card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        card.add_css_class("shop-mode-card")

        info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info.set_hexpand(True)
        title = Gtk.Label(label=_("Offsets"), xalign=0.0)
        title.add_css_class("shop-mode-caption")
        info.append(title)
        for axis in _AXES:
            label = Gtk.Label(xalign=0.0)
            label.add_css_class("shop-mode-offset")
            label.add_css_class(_AXIS_CSS[axis])
            info.append(label)
            self._offset_labels[axis] = label
        card.append(info)

        self.edit_offsets_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        self.edit_offsets_btn.add_css_class("flat")
        self.edit_offsets_btn.add_css_class("shop-mode-zero")
        self.edit_offsets_btn.set_tooltip_text(_("Edit Offsets Manually"))
        self.edit_offsets_btn.connect("clicked", self._on_edit_offsets_clicked)
        card.append(self.edit_offsets_btn)

        self.alignment_btn = Gtk.ToggleButton(
            child=get_icon("pointer-alignment-symbolic")
        )
        self.alignment_btn.add_css_class("flat")
        self.alignment_btn.add_css_class("shop-mode-zero")
        self.alignment_btn.set_tooltip_text(
            _(
                "Pointer Alignment: aim moves and zeroing at the pointer "
                "dot instead of the cutting beam"
            )
        )
        self._alignment_handler_id = self.alignment_btn.connect(
            "toggled", self._on_alignment_toggled
        )
        card.append(self.alignment_btn)
        return card

    def _build_steppers(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        caption_group = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
        value_group = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)

        self._speed_unit_label = Gtk.Label()
        self._speed_unit_label.add_css_class("dim-label")
        self._speed_unit_label.add_css_class("shop-mode-unit")
        self._speed_value = Gtk.Label()
        self._speed_value.add_css_class("shop-mode-step-value")
        box.append(
            self._build_stepper_row(
                _("SPEED"),
                self._speed_unit_label,
                self._speed_value,
                self._on_speed_minus,
                self._on_speed_plus,
                caption_group,
                value_group,
            )
        )

        self._distance_unit_label = Gtk.Label()
        self._distance_unit_label.add_css_class("dim-label")
        self._distance_unit_label.add_css_class("shop-mode-unit")
        self._distance_value = Gtk.Label()
        self._distance_value.add_css_class("shop-mode-step-value")
        box.append(
            self._build_stepper_row(
                _("STEP"),
                self._distance_unit_label,
                self._distance_value,
                self._on_distance_minus,
                self._on_distance_plus,
                caption_group,
                value_group,
            )
        )
        return box

    def _build_stepper_row(
        self,
        title,
        unit_label,
        value_label,
        on_minus,
        on_plus,
        caption_group,
        value_group,
    ) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        caption = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        caption.set_valign(Gtk.Align.CENTER)
        title_label = Gtk.Label(label=title)
        title_label.add_css_class("shop-mode-caption")
        caption.append(title_label)
        caption.append(unit_label)
        caption_group.add_widget(caption)
        row.append(caption)

        minus = Gtk.Button(label="−")
        minus.add_css_class("shop-mode-stepper")
        minus.connect("clicked", on_minus)
        row.append(minus)

        value_label.set_valign(Gtk.Align.CENTER)
        value_group.add_widget(value_label)
        row.append(value_label)

        plus = Gtk.Button(label="+")
        plus.add_css_class("shop-mode-stepper")
        plus.connect("clicked", on_plus)
        row.append(plus)
        return row

    # ----- machine wiring ------------------------------------------------

    def set_machine(
        self, machine: Machine | None, machine_cmd: MachineCmd | None
    ):
        if self.machine:
            self.machine.state_changed.disconnect(self._on_state_changed)
            self.machine.wcs_updated.disconnect(self._on_wcs_updated)
            self.machine.changed.disconnect(self._on_wcs_updated)
            self.machine.connection_status_changed.disconnect(
                self._on_state_changed
            )
            self.machine.pointer_alignment_changed.disconnect(
                self._on_alignment_changed
            )
        self.machine = machine
        self.machine_cmd = machine_cmd
        if self.machine:
            self.machine.state_changed.connect(self._on_state_changed)
            self.machine.wcs_updated.connect(self._on_wcs_updated)
            self.machine.changed.connect(self._on_wcs_updated)
            self.machine.connection_status_changed.connect(
                self._on_state_changed
            )
            self.machine.pointer_alignment_changed.connect(
                self._on_alignment_changed
            )
        self.refresh()

    def _on_destroy(self, *args):
        get_context().config.changed.disconnect(self._config_handler_id)
        self.set_machine(None, None)

    def _on_config_changed(self, sender, **kwargs):
        self._refresh_stepper_labels()

    def _on_state_changed(self, sender, **kwargs):
        self.refresh()

    def _on_wcs_updated(self, sender, **kwargs):
        self.refresh()

    def _on_alignment_changed(self, sender, **kwargs):
        self._sync_alignment_btn()

    # ----- state ---------------------------------------------------------

    @property
    def jog_speed(self) -> int:
        return self._jog_speed

    @jog_speed.setter
    def jog_speed(self, speed: int):
        self._jog_speed = int(speed)
        self._refresh_stepper_labels()

    @property
    def jog_distance(self) -> float:
        return self._jog_distance

    @jog_distance.setter
    def jog_distance(self, distance: float):
        self._jog_distance = float(distance)
        self._refresh_stepper_labels()

    def refresh(self):
        """Re-reads positions, offsets and button sensitivity."""
        machine = self.machine
        if machine is None:
            for label in self._axis_values.values():
                label.set_label("---")
            for btn in self._axis_zero_btns.values():
                btn.set_sensitive(False)
            self.edit_offsets_btn.set_sensitive(False)
            self._sync_alignment_btn()
            return

        has_z = machine.has_z_axis
        self._axis_rows[Axis.Z].set_visible(has_z)
        self._offset_labels[Axis.Z].set_visible(has_z)

        is_active = machine.is_connected() or machine.is_driverless
        is_mcs = machine.active_wcs == machine.machine_space_wcs
        can_zero = is_active and not is_mcs

        m_pos = machine.device_state.machine_pos
        offset = machine.get_active_wcs_offset()
        for i, axis in enumerate(_AXES):
            value = self._axis_values[axis]
            raw = m_pos[i] if m_pos is not None else None
            if not is_active or raw is None:
                value.set_label("---")
            else:
                pos = raw if is_mcs else raw - offset[i]
                value.set_label(f"{pos:.3f}")
            self._axis_zero_btns[axis].set_sensitive(can_zero)
            self._offset_labels[axis].set_label(
                f"{_AXIS_NAMES[axis]}: {offset[i]:.3f}"
            )

        self.edit_offsets_btn.set_sensitive(not is_mcs)
        self._sync_alignment_btn()
        self._refresh_stepper_labels()

    def _sync_alignment_btn(self):
        btn = self.alignment_btn
        btn.handler_block(self._alignment_handler_id)
        if self.machine and self.machine.has_pointer_offset():
            btn.set_sensitive(True)
            btn.set_active(self.machine.pointer_alignment_enabled)
        else:
            btn.set_sensitive(False)
            btn.set_active(False)
        btn.handler_unblock(self._alignment_handler_id)

    def _refresh_stepper_labels(self):
        speed_unit = get_preferred_unit("speed")
        length_unit = get_preferred_unit("length")
        if speed_unit:
            self._speed_unit_label.set_label(speed_unit.label)
            self._speed_value.set_label(
                f"{speed_unit.from_base(self._jog_speed):.0f}"
            )
        if length_unit:
            self._distance_unit_label.set_label(length_unit.label)
            self._distance_value.set_label(
                f"{length_unit.from_base(self._jog_distance):.1f}"
            )

    # ----- actions -------------------------------------------------------

    def _on_zero_clicked(self, button, axis):
        machine = self.machine
        if machine:
            task_mgr.add_coroutine(
                lambda ctx: machine.set_work_origin_here(axis)
            )

    def _on_alignment_toggled(self, button):
        if self._syncing_alignment or not self.machine:
            return
        self.machine.set_pointer_alignment(button.get_active())

    def _on_edit_offsets_clicked(self, button):
        if not self.machine:
            return
        root = self.get_root()
        self._edit_dialog = WcsDialog(
            machine=self.machine,
            transient_for=root if isinstance(root, Gtk.Window) else None,
        )
        self._edit_dialog.connect("destroy", self._on_edit_dialog_destroy)
        self._edit_dialog.present()

    def _on_edit_dialog_destroy(self, *args):
        self._edit_dialog = None

    def _on_speed_minus(self, button):
        self._step_speed(-_SPEED_STEP)

    def _on_speed_plus(self, button):
        self._step_speed(_SPEED_STEP)

    def _step_speed(self, delta):
        speed = _clamp(self._jog_speed + delta, _SPEED_RANGE)
        if speed == self._jog_speed:
            return
        self.jog_speed = speed
        self.speed_changed.send(self, speed=speed)

    def _on_distance_minus(self, button):
        self._step_distance(-1.0)

    def _on_distance_plus(self, button):
        self._step_distance(1.0)

    def _step_distance(self, display_delta):
        unit = get_preferred_unit("length")
        if not unit:
            return
        display = unit.from_base(self._jog_distance) + display_delta
        distance = _clamp(unit.to_base(display), _DISTANCE_RANGE)
        if distance == self._jog_distance:
            return
        self.jog_distance = distance
        self.distance_changed.send(self, distance=distance)


class ShopJogPanel(Gtk.Box):
    """Right side of the shop-mode surface: oversized jog, quick-move
    and job buttons plus the WCS selector."""

    #: Emitted with ``position`` ("ll", "center", "ur") for quick moves.
    move_requested = Signal()
    #: Emitted when the move-to-WCS-zero button is clicked.
    move_to_zero_requested = Signal()

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.machine: Machine | None = None
        self.machine_cmd: MachineCmd | None = None
        self._jog_speed = 1000
        self._jog_distance = 10.0
        self._syncing_wcs = False
        self._wcs_list: list[str] = []

        grid = Gtk.Grid()
        grid.set_row_spacing(6)
        grid.set_column_spacing(6)
        grid.set_row_homogeneous(True)
        grid.set_column_homogeneous(True)
        grid.set_hexpand(True)
        grid.set_vexpand(True)
        self.append(grid)

        self._direction_btns: dict[tuple, Gtk.Button] = {}
        self._direction_widgets: dict[tuple, Gtk.Widget] = {}

        ur_btn = self._icon_button(
            "top-right-symbolic",
            _("Move to Upper-Right of Selection or Workarea"),
        )
        ur_btn.connect(
            "clicked", lambda b: self.move_requested.send(self, position="ur")
        )
        grid.attach(ur_btn, 0, 0, 1, 1)
        self._attach_jog(grid, 1, 0, (JogDirection.WEST, JogDirection.NORTH))
        self._attach_jog(grid, 2, 0, (JogDirection.NORTH,), "y")
        self._attach_jog(grid, 3, 0, (JogDirection.EAST, JogDirection.NORTH))

        self.hold_btn = Gtk.ToggleButton(child=get_icon("pause-symbolic"))
        self.hold_btn.add_css_class("shop-mode-button")
        self.hold_btn.add_css_class("shop-mode-hold")
        self.hold_btn.set_tooltip_text(_("Pause machine"))
        self.hold_btn.set_action_name("win.machine-hold")
        grid.attach(self.hold_btn, 4, 0, 1, 1)

        center_btn = self._icon_button(
            "center-symbolic",
            _("Move to Center of Selection or Workarea"),
        )
        center_btn.connect(
            "clicked",
            lambda b: self.move_requested.send(self, position="center"),
        )
        grid.attach(center_btn, 0, 1, 1, 1)
        self._attach_jog(grid, 1, 1, (JogDirection.WEST,), "x")
        self.home_all_btn = self._icon_button("home-symbolic", _("Home All"))
        self.home_all_btn.connect("clicked", self._on_home_all_clicked)
        grid.attach(self.home_all_btn, 2, 1, 1, 1)
        self._attach_jog(grid, 3, 1, (JogDirection.EAST,), "x")
        self.z_minus_btn = self._attach_jog(
            grid, 4, 1, (JogDirection.DOWN,), "z", "arrow-z-down-symbolic"
        )

        origin_btn = self._icon_button(
            "goto-origin-symbolic", _("Move to Origin of Active WCS")
        )
        origin_btn.connect(
            "clicked", lambda b: self.move_to_zero_requested.send(self)
        )
        grid.attach(origin_btn, 0, 2, 1, 1)
        self._attach_jog(grid, 1, 2, (JogDirection.WEST, JogDirection.SOUTH))
        self._attach_jog(grid, 2, 2, (JogDirection.SOUTH,), "y")
        self._attach_jog(grid, 3, 2, (JogDirection.EAST, JogDirection.SOUTH))
        self.z_plus_btn = self._attach_jog(
            grid, 4, 2, (JogDirection.UP,), "z", "arrow-z-up-symbolic"
        )

        self.wcs_dropdown = Gtk.DropDown()
        self.wcs_dropdown.add_css_class("shop-mode-wcs")
        self.wcs_dropdown.set_hexpand(True)
        self.wcs_dropdown.connect("notify::selected", self._on_wcs_selected)
        grid.attach(self.wcs_dropdown, 0, 3, 2, 1)

        self.cancel_btn = self._icon_button(
            "stop-symbolic", _("Cancel running job")
        )
        self.cancel_btn.add_css_class("destructive-action")
        self.cancel_btn.set_action_name("win.machine-cancel")
        grid.attach(self.cancel_btn, 2, 3, 1, 1)

        self.frame_btn = self._icon_button(
            "frame-symbolic", _("Frame job outline")
        )
        self.frame_btn.set_action_name("win.machine-frame")
        grid.attach(self.frame_btn, 3, 3, 1, 1)

        self.send_btn = self._icon_button(
            "play-arrow-symbolic", _("Send to machine")
        )
        self.send_btn.add_css_class("suggested-action")
        self.send_btn.set_action_name("win.machine-send")
        grid.attach(self.send_btn, 4, 3, 1, 1)

        self.connect("destroy", self._on_destroy)

    # ----- construction --------------------------------------------------

    def _icon_button(self, icon_name, tooltip) -> Gtk.Button:
        button = Gtk.Button()
        button.add_css_class("shop-mode-button")
        button.set_tooltip_text(tooltip)
        icon = get_icon(icon_name)
        icon.set_pixel_size(28)
        button.set_child(icon)
        button.set_hexpand(True)
        button.set_vexpand(True)
        return button

    def _attach_jog(
        self, grid, col, row, directions, axis_sub=None, icon_name=None
    ) -> Gtk.Button:
        if icon_name is None:
            icon_name = _JOG_ICONS[directions]
        button = self._icon_button(icon_name, _JOG_TOOLTIPS[directions])
        if axis_sub:
            overlay = Gtk.Overlay()
            overlay.set_child(button)
            sub = Gtk.Label(label=axis_sub)
            sub.add_css_class("shop-mode-axis-sub")
            sub.add_css_class(_AXIS_CSS[_SUB_AXIS[axis_sub]])
            sub.set_halign(Gtk.Align.END)
            sub.set_valign(Gtk.Align.END)
            sub.set_margin_end(6)
            sub.set_margin_bottom(4)
            overlay.add_overlay(sub)
            overlay.set_hexpand(True)
            overlay.set_vexpand(True)
            grid.attach(overlay, col, row, 1, 1)
            attached = overlay
        else:
            grid.attach(button, col, row, 1, 1)
            attached = button
        button.connect("clicked", self._on_jog_clicked, directions)
        self._direction_btns[directions] = button
        self._direction_widgets[directions] = attached
        return button

    # ----- machine wiring ------------------------------------------------

    def set_machine(
        self, machine: Machine | None, machine_cmd: MachineCmd | None
    ):
        if self.machine:
            self.machine.state_changed.disconnect(self._on_state_changed)
            self.machine.connection_status_changed.disconnect(
                self._on_state_changed
            )
            self.machine.changed.disconnect(self._on_machine_changed)
            self.machine.wcs_updated.disconnect(self._on_wcs_updated)
        self.machine = machine
        self.machine_cmd = machine_cmd
        if self.machine:
            self.machine.state_changed.connect(self._on_state_changed)
            self.machine.connection_status_changed.connect(
                self._on_state_changed
            )
            self.machine.changed.connect(self._on_machine_changed)
            self.machine.wcs_updated.connect(self._on_wcs_updated)
        self._sync_wcs_model()
        self._update_sensitivity()

    def _on_destroy(self, *args):
        self.set_machine(None, None)

    def _on_state_changed(self, sender, **kwargs):
        self._update_sensitivity()
        self._sync_hold_icon()

    def _on_machine_changed(self, sender, **kwargs):
        self._update_sensitivity()

    def _on_wcs_updated(self, sender, **kwargs):
        self._sync_wcs_model()

    @property
    def jog_speed(self) -> int:
        return self._jog_speed

    @jog_speed.setter
    def jog_speed(self, speed: int):
        self._jog_speed = int(speed)

    @property
    def jog_distance(self) -> float:
        return self._jog_distance

    @jog_distance.setter
    def jog_distance(self, distance: float):
        self._jog_distance = float(distance)

    # ----- state ---------------------------------------------------------

    def _sync_wcs_model(self):
        if not self.machine:
            return
        wcs_list = list(self.machine.supported_wcs)
        if wcs_list != self._wcs_list:
            self._wcs_list = wcs_list
            self.wcs_dropdown.set_model(Gtk.StringList.new(wcs_list))
        current = self.machine.active_wcs
        if current in self._wcs_list:
            idx = self._wcs_list.index(current)
            if self.wcs_dropdown.get_selected() != idx:
                self._syncing_wcs = True
                self.wcs_dropdown.set_selected(idx)
                self._syncing_wcs = False

    def _on_wcs_selected(self, dropdown, _pspec):
        if self._syncing_wcs or not self.machine:
            return
        machine = self.machine
        idx = dropdown.get_selected()
        if 0 <= idx < len(self._wcs_list):
            wcs = self._wcs_list[idx]
            if wcs != machine.active_wcs:
                task_mgr.add_coroutine(
                    lambda ctx, w=wcs: machine.switch_active_wcs(w),
                    key=(machine.id, "select-wcs"),
                )

    def _sync_hold_icon(self):
        holding = (
            self.machine is not None
            and self.machine.device_state.status == DeviceStatus.HOLD
        )
        icon = get_icon("play-arrow-symbolic" if holding else "pause-symbolic")
        icon.set_pixel_size(28)
        self.hold_btn.set_child(icon)

    def _jog_deltas(self, directions) -> dict[Axis, float]:
        if not self.machine:
            return {}
        deltas: dict[Axis, float] = {}
        for direction in directions:
            for axis, delta in self.machine.panel.calculate_jog(
                direction, self._jog_distance
            ).items():
                deltas[axis] = deltas.get(axis, 0.0) + delta
        return deltas

    def _update_sensitivity(self):
        machine = self.machine
        connected = machine is not None and machine.is_connected()

        for directions, button in self._direction_btns.items():
            ok = False
            if machine is not None and connected:
                ok = all(
                    machine.can_jog(axis)
                    for axis in self._jog_deltas(directions)
                )
            button.set_sensitive(ok)
            button.remove_css_class("warning")
            if ok and machine is not None and machine.soft_limits_enabled:
                exceeds = any(
                    machine.would_jog_exceed_limits(axis, delta)
                    for axis, delta in self._jog_deltas(directions).items()
                )
                if exceeds:
                    button.add_css_class("warning")

        self.home_all_btn.set_sensitive(connected)
        has_z = machine is not None and machine.has_z_axis
        self._direction_widgets[(JogDirection.UP,)].set_visible(has_z)
        self._direction_widgets[(JogDirection.DOWN,)].set_visible(has_z)
        self.hold_btn.set_sensitive(connected)
        self.cancel_btn.set_sensitive(connected)
        self._sync_hold_icon()

    # ----- actions -------------------------------------------------------

    def _on_jog_clicked(self, button, directions):
        if not self.machine or not self.machine_cmd:
            return
        deltas = self._jog_deltas(directions)
        if deltas:
            self.machine_cmd.jog(self.machine, deltas, self._jog_speed)

    def _on_home_all_clicked(self, button):
        if self.machine and self.machine_cmd:
            self.machine_cmd.home(self.machine)


_JOG_ICONS = {
    (JogDirection.WEST, JogDirection.NORTH): "arrow-north-west-symbolic",
    (JogDirection.NORTH,): "arrow-north-symbolic",
    (JogDirection.EAST, JogDirection.NORTH): "arrow-north-east-symbolic",
    (JogDirection.WEST,): "arrow-west-symbolic",
    (JogDirection.EAST,): "arrow-east-symbolic",
    (JogDirection.WEST, JogDirection.SOUTH): "arrow-south-west-symbolic",
    (JogDirection.SOUTH,): "arrow-south-symbolic",
    (JogDirection.EAST, JogDirection.SOUTH): "arrow-south-east-symbolic",
    (JogDirection.UP,): "arrow-z-up-symbolic",
    (JogDirection.DOWN,): "arrow-z-down-symbolic",
}

_JOG_TOOLTIPS = {
    (JogDirection.WEST, JogDirection.NORTH): _("Move North-West"),
    (JogDirection.NORTH,): _("Move North"),
    (JogDirection.EAST, JogDirection.NORTH): _("Move North-East"),
    (JogDirection.WEST,): _("Move West (Left)"),
    (JogDirection.EAST,): _("Move East (Right)"),
    (JogDirection.WEST, JogDirection.SOUTH): _("Move South-West"),
    (JogDirection.SOUTH,): _("Move South"),
    (JogDirection.EAST, JogDirection.SOUTH): _("Move South-East"),
    (JogDirection.UP,): _("Move Z Up"),
    (JogDirection.DOWN,): _("Move Z Down"),
}

_SUB_AXIS = {"x": Axis.X, "y": Axis.Y, "z": Axis.Z}

__all__ = ["ShopControlPanel", "ShopJogPanel"]
