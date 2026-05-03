from gi.repository import GLib, Gtk, Adw
from typing import Optional
from gettext import gettext as _
from ...machine.models.machine import Machine
from ...machine.models.laser import Laser
from ...machine.cmd import MachineCmd
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.slider import create_slider

_POWER_CSS = """
entry.power-value {
    min-width: 4em;
    max-width: 4em;
}
"""
apply_css(_POWER_CSS)


class LaserControlWidget(Gtk.Box):
    """Widget for manual laser on/off control with power and duration."""

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self.machine: Optional[Machine] = None
        self.machine_cmd: Optional[MachineCmd] = None
        self._is_on = False
        self._timer_source_id: Optional[int] = None
        self._remaining_ms: int = 0

        self._group = Adw.PreferencesGroup()
        self._group.add_css_class("compact")

        self._head_row = Adw.ComboRow(title=_("Laser Head"))
        self._head_row.connect(
            "notify::selected", self._on_head_selection_changed
        )
        self._toggle_btn = Gtk.ToggleButton()
        self._toggle_btn.set_child(get_icon("laser-off-symbolic"))
        self._toggle_btn.add_css_class("flat")
        self._toggle_btn.set_valign(Gtk.Align.CENTER)
        self._toggle_btn.set_tooltip_text(_("Toggle laser on/off"))
        self._toggle_btn.connect("clicked", self._on_toggle_clicked)
        self._head_row.add_suffix(self._toggle_btn)
        self._group.add(self._head_row)

        self._power_adj = Gtk.Adjustment(
            value=1.0,
            lower=0,
            upper=100,
            step_increment=0.5,
            page_increment=10,
        )
        self._power_scale = create_slider(
            adjustment=self._power_adj,
            digits=1,
            draw_value=False,
        )
        self._power_entry = Gtk.Entry()
        self._power_entry.set_width_chars(5)
        self._power_entry.set_max_width_chars(5)
        self._power_entry.set_hexpand(False)
        self._power_entry.set_halign(Gtk.Align.END)
        self._power_entry.set_alignment(1.0)
        self._power_entry.set_has_frame(False)
        self._power_entry.get_style_context().add_class("power-value")

        def update_power_entry(s):
            self._power_entry.set_text(f"{s.get_value():.1f} %")

        self._power_scale.connect("value-changed", update_power_entry)
        update_power_entry(self._power_scale)

        def commit_power_entry(e):
            text = e.get_text().rstrip(" %")
            try:
                self._power_adj.set_value(float(text))
            except ValueError:
                update_power_entry(self._power_scale)

        self._power_entry.connect("activate", commit_power_entry)
        focus_ctrl = Gtk.EventControllerFocus()
        focus_ctrl.connect(
            "leave", lambda c: commit_power_entry(self._power_entry)
        )
        self._power_entry.add_controller(focus_ctrl)

        self._power_row = Adw.ActionRow(title=_("Power"))
        self._power_row.set_subtitle(_("Laser power in percent"))
        suffix_box = Gtk.Box(spacing=6)
        suffix_box.set_hexpand(False)
        suffix_box.append(self._power_entry)
        suffix_box.append(self._power_scale)
        self._power_row.add_suffix(suffix_box)
        self._group.add(self._power_row)

        self._frequency_adj = Gtk.Adjustment(
            lower=1, upper=100000, step_increment=100, page_increment=1000
        )
        self._frequency_row = Adw.SpinRow(
            title=_("Frequency"),
            subtitle=_("PWM frequency in Hz"),
            adjustment=self._frequency_adj,
        )
        self._group.add(self._frequency_row)

        self._pulse_width_adj = Gtk.Adjustment(
            lower=1, upper=100000, step_increment=1, page_increment=10
        )
        self._pulse_width_row = Adw.SpinRow(
            title=_("Pulse Width"),
            subtitle=_("Pulse width in µs"),
            adjustment=self._pulse_width_adj,
        )
        self._group.add(self._pulse_width_row)

        duration_adjustment = Gtk.Adjustment(
            value=0, lower=0, upper=3600, step_increment=0.5
        )
        self._duration_row = Adw.SpinRow(
            title=_("Duration"),
            subtitle=_("Seconds (0 = continuous)"),
            adjustment=duration_adjustment,
            digits=1,
        )
        self._group.add(self._duration_row)

        self.append(self._group)

        self._countdown_label = Gtk.Label()
        self._countdown_label.add_css_class("dim-label")
        self._countdown_label.set_visible(False)
        self.append(self._countdown_label)

        self.connect("destroy", self._on_destroy)
        self._update_sensitivity()
        self._update_pwm_visibility()

    def set_machine(
        self, machine: Optional[Machine], machine_cmd: Optional[MachineCmd]
    ):
        if self.machine:
            self.machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            self.machine.changed.disconnect(self._on_machine_changed)
            self.machine.controller.laser_power_changed.disconnect(
                self._on_laser_power_changed
            )
        self.machine = machine
        self.machine_cmd = machine_cmd
        if self.machine:
            self.machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self.machine.changed.connect(self._on_machine_changed)
            self.machine.controller.laser_power_changed.connect(
                self._on_laser_power_changed
            )
            self._rebuild_head_model()
        self._update_sensitivity()

    def _rebuild_head_model(self):
        if not self.machine:
            return
        heads = self.machine.heads
        model = Gtk.StringList.new([h.name for h in heads])
        self._head_row.set_model(model)
        if heads:
            self._head_row.set_selected(0)
            self._sync_head_fields(heads[0])

    def _get_selected_head(self) -> Optional[Laser]:
        if not self.machine or not self.machine.heads:
            return None
        idx = self._head_row.get_selected()
        if 0 <= idx < len(self.machine.heads):
            return self.machine.heads[idx]
        return None

    def _sync_head_fields(self, head: Laser):
        self._head_row.set_subtitle(
            _("Tool {tool_number}, max power {max_power}").format(
                tool_number=head.tool_number, max_power=head.max_power
            )
        )
        self._power_adj.set_value(head.focus_power_percent * 100)
        self._frequency_adj.set_upper(head.max_pwm_frequency)
        self._frequency_row.set_value(head.pwm_frequency)
        self._pulse_width_adj.set_lower(head.min_pulse_width)
        self._pulse_width_adj.set_upper(head.max_pulse_width)
        self._pulse_width_row.set_value(head.pulse_width)
        self._update_pwm_visibility()

    def _update_pwm_visibility(self):
        head = self._get_selected_head()
        show = head is not None and head.laser_type.supports_pwm
        self._frequency_row.set_visible(show)
        self._pulse_width_row.set_visible(show)

    def _on_head_selection_changed(self, row, _pspec):
        head = self._get_selected_head()
        if head:
            self._sync_head_fields(head)

    def _on_machine_changed(self, sender):
        self._rebuild_head_model()

    def _on_laser_power_changed(self, sender, *, head, percent):
        is_on = percent > 0
        if is_on != self._is_on:
            self._cancel_timer()
            self._is_on = is_on
            self._update_toggle_ui()

    def _on_connection_status_changed(self, sender, **kwargs):
        if self.machine and not self.machine.is_connected() and self._is_on:
            self._cancel_timer()
            self._is_on = False
            self._update_toggle_ui()
        self._update_sensitivity()

    def _update_sensitivity(self):
        has_heads = self.machine is not None and len(self.machine.heads) > 0
        connected = self.machine is not None and self.machine.is_connected()
        self._head_row.set_sensitive(has_heads)
        self._power_row.set_sensitive(has_heads)
        self._frequency_row.set_sensitive(has_heads)
        self._pulse_width_row.set_sensitive(has_heads)
        self._duration_row.set_sensitive(has_heads)
        self._toggle_btn.set_sensitive(connected and has_heads)

    def _on_toggle_clicked(self, button):
        if self._is_on:
            self._turn_off()
        else:
            self._turn_on()

    def _turn_on(self):
        head = self._get_selected_head()
        if not head or not self.machine or not self.machine_cmd:
            return
        if not self.machine.is_connected():
            return

        percent = self._power_adj.get_value() / 100.0
        self.machine_cmd.set_focus_power(head, percent)

        self._is_on = True
        self._update_toggle_ui()
        self._update_sensitivity()

        duration_s = self._duration_row.get_value()
        if duration_s > 0:
            self._remaining_ms = int(duration_s * 1000)
            self._update_countdown_label()
            self._countdown_label.set_visible(True)
            self._timer_source_id = GLib.timeout_add(100, self._on_timer_tick)

    def _turn_off(self):
        self._cancel_timer()
        head = self._get_selected_head()
        if head and self.machine and self.machine.is_connected():
            if self.machine_cmd:
                self.machine_cmd.set_focus_power(head, 0)
        self._is_on = False
        self._update_toggle_ui()
        self._update_sensitivity()

    def _cancel_timer(self):
        if self._timer_source_id is not None:
            GLib.source_remove(self._timer_source_id)
            self._timer_source_id = None
        self._countdown_label.set_visible(False)
        self._remaining_ms = 0

    def _on_timer_tick(self) -> bool:
        self._remaining_ms -= 100
        if self._remaining_ms <= 0:
            self._turn_off()
            return GLib.SOURCE_REMOVE
        self._update_countdown_label()
        return GLib.SOURCE_CONTINUE

    def _update_countdown_label(self):
        secs = max(0, self._remaining_ms / 1000.0)
        self._countdown_label.set_label(
            _("{seconds:.1f} s remaining").format(seconds=secs)
        )

    def _update_toggle_ui(self):
        if self._is_on:
            self._toggle_btn.set_active(True)
            self._toggle_btn.set_child(get_icon("laser-off-symbolic"))
            self._toggle_btn.add_css_class("destructive-action")
        else:
            self._toggle_btn.set_active(False)
            self._toggle_btn.set_child(get_icon("laser-on-symbolic"))
            self._toggle_btn.remove_css_class("destructive-action")

    def _on_destroy(self, widget):
        if self._timer_source_id is not None:
            GLib.source_remove(self._timer_source_id)
            self._timer_source_id = None
