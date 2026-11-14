import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...machine.driver import get_driver_cls
from ...machine.driver.driver import Z_HOMING_MODE_LABELS
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.preferences_page import TrackedPreferencesPage
from .bidir_offset_table import BidirOffsetTableGroup

logger = logging.getLogger(__name__)


class AdvancedPreferencesPage(TrackedPreferencesPage):
    key = "advanced"
    path_prefix = "/machine-settings/"

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

        self.curves_row = Adw.SwitchRow(
            title=_("Support Bézier Curves"),
            subtitle=_(
                "Generate native cubic Bézier commands. "
                "Disable if your machine does not support them"
            ),
        )
        self.curves_row.set_active(self.machine.supports_curves)
        self.curves_row.connect("notify::active", self.on_curves_changed)
        path_group.add(self.curves_row)

        self.arc_tolerance_row = LengthSpinRow(
            _("Arc and Curve Tolerance"),
            _(
                "Maximum deviation from original path when "
                "fitting arcs and curves. Lower values "
                "drastically increase processing time and job size"
            ),
            lower=0.001,
            upper=10.0,
            step_increment=0.001,
            digits=3,
            value_in_base=self.machine.arc_tolerance,
        )
        self.arc_tolerance_row.set_width_chars(5)
        self.arc_tolerance_row.value_changed.connect(
            self.on_arc_tolerance_changed
        )
        path_group.add(self.arc_tolerance_row)

        self.bidir_offset_group = BidirOffsetTableGroup(self.machine)
        self.add(self.bidir_offset_group)

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

        park = machine.park_position_after_homing
        park_x, park_y = park if park is not None else (0.0, 0.0)

        self.park_after_homing_row = Adw.SwitchRow()
        self.park_after_homing_row.set_title(_("Park After Homing"))
        self.park_after_homing_row.set_subtitle(
            _(
                "After homing, move the head to the park position, "
                "e.g. to clear the homing pull-off"
            )
        )
        self.park_after_homing_row.set_active(park is not None)
        self.park_after_homing_row.connect(
            "notify::active", self.on_park_after_homing_changed
        )
        homing_group.add(self.park_after_homing_row)

        self.park_x_row = LengthSpinRow(
            _("Park Position X"),
            _("Machine-coordinate X of the park position"),
            lower=-10000.0,
            upper=10000.0,
            step_increment=1.0,
            digits=2,
            value_in_base=park_x,
        )
        self.park_x_row.value_changed.connect(self.on_park_position_changed)
        homing_group.add(self.park_x_row)

        self.park_y_row = LengthSpinRow(
            _("Park Position Y"),
            _("Machine-coordinate Y of the park position"),
            lower=-10000.0,
            upper=10000.0,
            step_increment=1.0,
            digits=2,
            value_in_base=park_y,
        )
        self.park_y_row.value_changed.connect(self.on_park_position_changed)
        homing_group.add(self.park_y_row)
        self._update_park_rows_sensitivity()

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

        self.z_homing_row = Adw.ComboRow(
            title=_("Z Homing"),
            subtitle=_(
                "How the Z axis finds its reference: a home switch, "
                "or the auto-focus probe"
            ),
        )
        self._z_homing_modes = ()
        self._z_homing_driver_name = None
        self._z_homing_handler = self.z_homing_row.connect(
            "notify::selected", self.on_z_homing_mode_changed
        )
        self._update_z_homing_row()
        homing_group.add(self.z_homing_row)

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

        self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)

    def on_arcs_changed(self, switch_row, _param):
        """Update the machine's arcs support when the value changes."""
        self.machine.set_supports_arcs(switch_row.get_active())

    def on_arc_tolerance_changed(self, spinrow):
        """Update to machine's arc tolerance when value changes."""
        self.machine.set_arc_tolerance(spinrow.get_value_in_base_units())

    def on_curves_changed(self, switch_row, _param):
        """Update the machine's curve support when the value changes."""
        self.machine.set_supports_curves(switch_row.get_active())

    def on_home_on_start_changed(self, row, _):
        self.machine.set_home_on_start(row.get_active())

    def on_park_after_homing_changed(self, row, _):
        if row.get_active():
            self.machine.set_park_position_after_homing(
                (
                    self.park_x_row.get_value_in_base_units(),
                    self.park_y_row.get_value_in_base_units(),
                )
            )
        else:
            self.machine.set_park_position_after_homing(None)
        self._update_park_rows_sensitivity()

    def on_park_position_changed(self, row):
        if not self.park_after_homing_row.get_active():
            return
        self.machine.set_park_position_after_homing(
            (
                self.park_x_row.get_value_in_base_units(),
                self.park_y_row.get_value_in_base_units(),
            )
        )

    def _update_park_rows_sensitivity(self):
        active = self.park_after_homing_row.get_active()
        self.park_x_row.set_sensitive(active)
        self.park_y_row.set_sensitive(active)

    def on_single_axis_homing_changed(self, row, _):
        self.machine.set_single_axis_homing_enabled(row.get_active())

    def on_clear_alarm_on_connect_changed(self, row, _):
        self.machine.set_clear_alarm_on_connect(row.get_active())

    def on_z_homing_mode_changed(self, row, _):
        index = row.get_selected()
        if 0 <= index < len(self._z_homing_modes):
            self.machine.set_z_homing_mode(self._z_homing_modes[index])

    def _on_machine_changed(self, machine):
        if machine.driver_name != self._z_homing_driver_name:
            self._update_z_homing_row()

    def _update_z_homing_row(self):
        """
        Lists the driver's Z homing modes, and only shows the row when
        there is a choice to make.
        """
        self._z_homing_driver_name = self.machine.driver_name
        driver_cls = get_driver_cls(self.machine.driver_name or "")
        self._z_homing_modes = driver_cls.z_homing_modes
        store = Gtk.StringList()
        for mode in self._z_homing_modes:
            store.append(Z_HOMING_MODE_LABELS[mode])
        mode = driver_cls.resolve_z_homing_mode(self.machine.z_homing_mode)
        self.z_homing_row.handler_block(self._z_homing_handler)
        self.z_homing_row.set_model(store)
        self.z_homing_row.set_selected(self._z_homing_modes.index(mode))
        self.z_homing_row.handler_unblock(self._z_homing_handler)
        self.z_homing_row.set_visible(len(self._z_homing_modes) > 1)

    def _on_destroy(self, *args):
        """Disconnects signals to prevent memory leaks."""
        self.machine.changed.disconnect(self._on_machine_changed)
