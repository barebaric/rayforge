import logging
from gi.repository import Gtk, Adw


logger = logging.getLogger(__name__)


class AdvancedPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Advanced"),
            icon_name="machine-settings-advanced-symbolic",
            **kwargs,
        )
        self.machine = machine

        path_group = Adw.PreferencesGroup(title=_("Path Processing"))
        path_group.set_description(
            _("Configure how paths are processed and optimized.")
        )
        self.add(path_group)

        self.arcs_row = Adw.SwitchRow(
            title=_("Support Arcs"),
            subtitle=_(
                "Generate arc commands for smoother paths. "
                "Disable if your machine does not support arcs"
            ),
        )
        self.arcs_row.set_active(self.machine.supports_arcs)
        self.arcs_row.connect("notify::active", self.on_arcs_changed)
        path_group.add(self.arcs_row)

        tolerance_adjustment = Gtk.Adjustment(
            lower=0.001, upper=10.0, step_increment=0.001, page_increment=0.01
        )
        self.arc_tolerance_row = Adw.SpinRow(
            title=_("Arc Tolerance"),
            subtitle=_(
                "Maximum deviation from original path when "
                "fitting arcs (in mm). Lower values "
                "drastically increase processing time and job size"
            ),
            adjustment=tolerance_adjustment,
        )
        self.arc_tolerance_row.set_digits(3)
        self.arc_tolerance_row.set_width_chars(5)
        tolerance_adjustment.set_value(self.machine.arc_tolerance)
        self.arc_tolerance_row.set_sensitive(self.machine.supports_arcs)
        self.arc_tolerance_row.connect(
            "changed", self.on_arc_tolerance_changed
        )
        path_group.add(self.arc_tolerance_row)

        homing_group = Adw.PreferencesGroup(title=_("Homing and Startup"))
        homing_group.set_description(
            _(
                "Configure homing behavior and startup settings, "
                "including automatic homing and alarm handling."
            )
        )
        self.add(homing_group)

        home_on_start_row = Adw.SwitchRow()
        home_on_start_row.set_title(_("Home On Start"))
        home_on_start_row.set_subtitle(
            _("Send a homing command when the application starts")
        )
        home_on_start_row.set_active(machine.home_on_start)
        home_on_start_row.connect(
            "notify::active", self.on_home_on_start_changed
        )
        homing_group.add(home_on_start_row)

        single_axis_homing_row = Adw.SwitchRow()
        single_axis_homing_row.set_title(_("Allow Single Axis Homing"))
        single_axis_homing_row.set_subtitle(
            _("Enable individual axis homing controls in the jog dialog")
        )
        single_axis_homing_row.set_active(machine.single_axis_homing_enabled)
        single_axis_homing_row.connect(
            "notify::active", self.on_single_axis_homing_changed
        )
        homing_group.add(single_axis_homing_row)

        clear_alarm_row = Adw.SwitchRow()
        clear_alarm_row.set_title(_("Clear Alarm On Connect"))
        clear_alarm_row.set_subtitle(
            _(
                "Automatically send an unlock command if "
                "connected in an ALARM state"
            )
        )
        clear_alarm_row.set_active(machine.clear_alarm_on_connect)
        clear_alarm_row.connect(
            "notify::active", self.on_clear_alarm_on_connect_changed
        )
        homing_group.add(clear_alarm_row)

    def on_arcs_changed(self, switch_row, _param):
        """Update the machine's arcs support when the value changes."""
        self.machine.supports_arcs = switch_row.get_active()
        self.arc_tolerance_row.set_sensitive(self.machine.supports_arcs)
        self.machine.changed.send(self.machine)

    def on_arc_tolerance_changed(self, spinrow):
        """Update to machine's arc tolerance when value changes."""
        self.machine.set_arc_tolerance(spinrow.get_value())

    def on_home_on_start_changed(self, row, _):
        self.machine.set_home_on_start(row.get_active())

    def on_single_axis_homing_changed(self, row, _):
        self.machine.set_single_axis_homing_enabled(row.get_active())

    def on_clear_alarm_on_connect_changed(self, row, _):
        self.machine.set_clear_alarm_on_connect(row.get_active())
