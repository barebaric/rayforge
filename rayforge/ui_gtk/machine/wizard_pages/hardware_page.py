"""Step 7 — Hardware configuration.

Surfaces the work-area X/Y extents, coordinate origin, soft-limit,
work-margins, axis-direction flags, max speeds and acceleration. The
reuse target is the existing ``HardwarePage`` widget set (see
``hardware_page.py``), but since that widget operates on a live
``Machine`` and the wizard holds an in-memory ``DeviceProfile``,
we rebuild a compact set of rows directly bound to *profile*. The
grouping (Axes / Work Area / Soft Limits) mirrors the device-settings
hardware page so the two look consistent.
"""

from gettext import gettext as _

from gi.repository import Adw, Gtk

from ....machine.device.profile import DeviceProfile
from ....machine.models.machine import Origin
from ....machine.models.machine_panel import PanelOrientation
from ...shared.pref_rows.acceleration_spin_row import AccelerationSpinRow
from ...shared.pref_rows.length_spin_row import LengthSpinRow
from ...shared.pref_rows.speed_spin_row import SpeedSpinRow
from . import WizardPage, _makePreferencesGroup

_ORIGIN_INDEX_TO_ENUM = {
    0: Origin.BOTTOM_LEFT,
    1: Origin.TOP_LEFT,
    2: Origin.TOP_RIGHT,
    3: Origin.BOTTOM_RIGHT,
}
_ORIGIN_ENUM_TO_INDEX = {v: k for k, v in _ORIGIN_INDEX_TO_ENUM.items()}
_PANEL_ORIENTATIONS = tuple(PanelOrientation)

# Sensible starting points surfaced when a profile carries no values;
# they mirror the Machine model defaults (machine.py).
_DEFAULT_TRAVEL_SPEED = 3000.0
_DEFAULT_CUT_SPEED = 1000.0
_DEFAULT_ACCELERATION = 1000.0


class HardwarePage(WizardPage):
    step_number = 7
    title = _("Hardware")
    subtitle = _("Work area, origin, speeds and acceleration.")

    def __init__(self, wizard, **kwargs):
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        # Grouping mirrors the device-settings HardwarePage: Axes,
        # Work Area (margins), Soft Limits, then wizard-only Speeds
        # and Behavior groups.
        axes_group = _makePreferencesGroup(
            title=_("Axes"),
            description=_("Configure the axis extents and coordinate system."),
        )
        self.content.append(axes_group)

        self.x_row = LengthSpinRow(
            _("X Extent"),
            _("Full X-axis travel range"),
            lower=10,
            upper=10000,
        )
        axes_group.add(self.x_row)

        self.y_row = LengthSpinRow(
            _("Y Extent"),
            _("Full Y-axis travel range"),
            lower=10,
            upper=10000,
        )
        axes_group.add(self.y_row)

        origin_store = Gtk.StringList()
        for label in (
            _("Bottom Left"),
            _("Top Left"),
            _("Top Right"),
            _("Bottom Right"),
        ):
            origin_store.append(label)
        self.origin_row = Adw.ComboRow(
            title=_("Coordinate Origin (0,0)"),
            subtitle=_(
                "Physical corner where coordinates are zero after homing"
            ),
            model=origin_store,
        )
        axes_group.add(self.origin_row)

        self.panel_orientation_row = Adw.ComboRow(
            title=_("Panel Orientation"),
            subtitle=_(
                "Rotate the flat workspace and remap output to the "
                "machine; rotary layers require Native"
            ),
            model=Gtk.StringList.new([o.label for o in _PANEL_ORIENTATIONS]),
        )
        axes_group.add(self.panel_orientation_row)

        # Direction reversals.
        self.reverse_x_row = Adw.SwitchRow(
            title=_("Reverse X-Axis Direction"),
            subtitle=_("Makes coordinate values negative"),
        )
        axes_group.add(self.reverse_x_row)
        self.reverse_y_row = Adw.SwitchRow(
            title=_("Reverse Y-Axis Direction"),
            subtitle=_("Makes coordinate values negative"),
        )
        axes_group.add(self.reverse_y_row)
        self.reverse_z_row = Adw.SwitchRow(
            title=_("Reverse Z-Axis Direction"),
            subtitle=_("Enable if +Z moves head down"),
        )
        axes_group.add(self.reverse_z_row)

        # Working margins.
        margins_group = _makePreferencesGroup(
            title=_("Work Area"),
            description=_(
                "Margins define the unusable space around the axis extents."
            ),
        )
        self.content.append(margins_group)

        # Work margins — four explicit rows so pyright follows the
        # attribute bindings (we read these back from apply_to_profile
        # and enter()).
        self.margin_left_row = self._build_margin_row(
            margins_group,
            _("Left Margin"),
            _("Unusable space from left edge"),
        )
        self.margin_top_row = self._build_margin_row(
            margins_group,
            _("Top Margin"),
            _("Unusable space from top edge"),
        )
        self.margin_right_row = self._build_margin_row(
            margins_group,
            _("Right Margin"),
            _("Unusable space from right edge"),
        )
        self.margin_bottom_row = self._build_margin_row(
            margins_group,
            _("Bottom Margin"),
            _("Unusable space from bottom edge"),
        )

        # Soft limits.
        self.soft_limits_group = _makePreferencesGroup(
            title=_("Soft Limits"),
            description=_(
                "Configurable safety bounds for jogging. "
                "Leave disabled to use work surface bounds."
            ),
        )
        self.content.append(self.soft_limits_group)

        self.soft_limits_enabled_row = Adw.SwitchRow(
            title=_("Enable Custom Soft Limits"),
            subtitle=_("Override work-surface bounds with custom limits"),
        )
        self.soft_limits_enabled_row.connect(
            "notify::active", self._on_soft_limits_toggle
        )
        self.soft_limits_group.add(self.soft_limits_enabled_row)

        self.soft_x_min_row = self._build_soft_limit_row(
            _("X Min"), _("Minimum X coordinate")
        )
        self.soft_y_min_row = self._build_soft_limit_row(
            _("Y Min"), _("Minimum Y coordinate")
        )
        self.soft_x_max_row = self._build_soft_limit_row(
            _("X Max"), _("Maximum X coordinate")
        )
        self.soft_y_max_row = self._build_soft_limit_row(
            _("Y Max"), _("Maximum Y coordinate")
        )

        # Speeds / accel.
        speed_group = _makePreferencesGroup(
            title=_("Speeds"),
            description=_("Limits in machine units per minute."),
        )
        self.content.append(speed_group)

        self.travel_speed_row = SpeedSpinRow(
            _("Max Travel Speed"),
            _("Maximum rapid movement speed"),
            upper=60000,
            step_increment=100,
            digits=0,
        )
        speed_group.add(self.travel_speed_row)

        self.cut_speed_row = SpeedSpinRow(
            _("Max Cut Speed"),
            _("Maximum cutting speed"),
            upper=60000,
            step_increment=100,
            digits=0,
        )
        speed_group.add(self.cut_speed_row)

        self.accel_row = AccelerationSpinRow(
            _("Acceleration"),
            _(
                "Used for time estimations and calculating the "
                "default overscan distance"
            ),
            upper=10000,
            digits=0,
        )
        speed_group.add(self.accel_row)

        # Behavior.
        behavior_group = _makePreferencesGroup(title=_("Behavior"))
        self.content.append(behavior_group)

        self.home_on_start_row = Adw.SwitchRow(
            title=_("Home on Start"),
            subtitle=_("Run homing cycle when machine connects"),
        )
        behavior_group.add(self.home_on_start_row)

        self.single_axis_homing_row = Adw.SwitchRow(
            title=_("Single-Axis Homing"),
            subtitle=_("Allow homing individual axes"),
        )
        behavior_group.add(self.single_axis_homing_row)

        # Whenever the user touches the soft-limits toggle or any of
        # the extents, we may need to clamp soft-limit adjustments.
        self.x_row.value_changed.connect(self._on_extents_changed)
        self.y_row.value_changed.connect(self._on_extents_changed)

        # The page is always consider-ready because the user can skip
        # fields they don't know yet (defaults are sensible). The
        # orchestrator will surface sanity-check warnings at Review.
        self.set_ready(True)

    # ----- row builders ---------------------------------------------------

    def _build_margin_row(
        self, group: Adw.PreferencesGroup, title: str, subtitle: str
    ) -> LengthSpinRow:
        row = LengthSpinRow(
            title=title,
            subtitle=subtitle,
            upper=10000,
        )
        group.add(row)
        return row

    def _build_soft_limit_row(
        self, title: str, subtitle: str
    ) -> LengthSpinRow:
        row = LengthSpinRow(
            title=title,
            subtitle=subtitle,
            upper=10000,
        )
        row.set_sensitive(False)
        self.soft_limits_group.add(row)
        return row

    def _on_extents_changed(self, row) -> None:
        x = self.x_row.get_value_in_base_units()
        y = self.y_row.get_value_in_base_units()
        self.soft_x_min_row.set_range(0.0, x)
        self.soft_x_max_row.set_range(0.0, x)
        self.soft_y_min_row.set_range(0.0, y)
        self.soft_y_max_row.set_range(0.0, y)

    def _on_soft_limits_toggle(self, row, _param) -> None:
        enabled = row.get_active()
        self.soft_x_min_row.set_sensitive(enabled)
        self.soft_y_min_row.set_sensitive(enabled)
        self.soft_x_max_row.set_sensitive(enabled)
        self.soft_y_max_row.set_sensitive(enabled)

    # ----- profile binding -----------------------------------------------

    def enter(self, profile: DeviceProfile) -> None:
        mc = profile.machine_config

        if mc.axis_extents:
            self.x_row.set_value_in_base_units(mc.axis_extents[0])
            self.y_row.set_value_in_base_units(mc.axis_extents[1])
        else:
            self.x_row.set_value_in_base_units(100.0)
            self.y_row.set_value_in_base_units(100.0)

        origin = mc.origin or Origin.BOTTOM_LEFT
        self.origin_row.set_selected(_ORIGIN_ENUM_TO_INDEX.get(origin, 0))

        orientation = mc.panel_orientation or PanelOrientation.NATIVE
        self.panel_orientation_row.set_selected(
            _PANEL_ORIENTATIONS.index(orientation)
        )

        # directional reversal flags aren't on MachineConfig; they live
        # on Machine directly. We treat them as ephemeral session state
        # via wizard.aux_state, defaulting to False.
        reverse = self.wizard.aux_state.setdefault("reverse", {})
        self.reverse_x_row.set_active(reverse.get("x", False))
        self.reverse_y_row.set_active(reverse.get("y", False))
        self.reverse_z_row.set_active(reverse.get("z", False))

        margins = mc.work_margins or (0.0, 0.0, 0.0, 0.0)
        self.margin_left_row.set_value_in_base_units(margins[0])
        self.margin_top_row.set_value_in_base_units(margins[1])
        self.margin_right_row.set_value_in_base_units(margins[2])
        self.margin_bottom_row.set_value_in_base_units(margins[3])

        soft = mc.soft_limits
        if soft:
            self.soft_limits_enabled_row.set_active(True)
            self.soft_x_min_row.set_value_in_base_units(soft[0])
            self.soft_y_min_row.set_value_in_base_units(soft[1])
            self.soft_x_max_row.set_value_in_base_units(soft[2])
            self.soft_y_max_row.set_value_in_base_units(soft[3])
        else:
            self.soft_limits_enabled_row.set_active(False)
            self.soft_x_min_row.set_value_in_base_units(0.0)
            self.soft_y_min_row.set_value_in_base_units(0.0)
            self.soft_x_max_row.set_value_in_base_units(
                self.x_row.get_value_in_base_units()
            )
            self.soft_y_max_row.set_value_in_base_units(
                self.y_row.get_value_in_base_units()
            )
        self._on_soft_limits_toggle(self.soft_limits_enabled_row, None)

        if mc.max_travel_speed is not None:
            self.travel_speed_row.set_value_in_base_units(mc.max_travel_speed)
        else:
            self.travel_speed_row.set_value_in_base_units(
                _DEFAULT_TRAVEL_SPEED
            )
        if mc.max_cut_speed is not None:
            self.cut_speed_row.set_value_in_base_units(mc.max_cut_speed)
        else:
            self.cut_speed_row.set_value_in_base_units(_DEFAULT_CUT_SPEED)
        if mc.acceleration is not None:
            self.accel_row.set_value_in_base_units(mc.acceleration)
        else:
            self.accel_row.set_value_in_base_units(_DEFAULT_ACCELERATION)

        self.home_on_start_row.set_active(bool(mc.home_on_start))
        self.single_axis_homing_row.set_active(
            bool(mc.single_axis_homing_enabled)
        )

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        mc = profile.machine_config

        x = self.x_row.get_value_in_base_units()
        y = self.y_row.get_value_in_base_units()
        if x > 0 and y > 0:
            mc.axis_extents = (float(x), float(y))

        mc.origin = _ORIGIN_INDEX_TO_ENUM.get(
            self.origin_row.get_selected(), Origin.BOTTOM_LEFT
        )
        orientation_index = self.panel_orientation_row.get_selected()
        if orientation_index < len(_PANEL_ORIENTATIONS):
            mc.panel_orientation = _PANEL_ORIENTATIONS[orientation_index]

        # stash reversals to aux_state (defers to Machine during
        # create_machine); the orchestrator applies them post-creation.
        reverse = self.wizard.aux_state.setdefault("reverse", {})
        reverse["x"] = self.reverse_x_row.get_active()
        reverse["y"] = self.reverse_y_row.get_active()
        reverse["z"] = self.reverse_z_row.get_active()

        margins = (
            self.margin_left_row.get_value_in_base_units(),
            self.margin_top_row.get_value_in_base_units(),
            self.margin_right_row.get_value_in_base_units(),
            self.margin_bottom_row.get_value_in_base_units(),
        )
        if any(m > 0 for m in margins):
            mc.work_margins = margins
        else:
            mc.work_margins = None

        if self.soft_limits_enabled_row.get_active():
            mc.soft_limits = (
                self.soft_x_min_row.get_value_in_base_units(),
                self.soft_y_min_row.get_value_in_base_units(),
                self.soft_x_max_row.get_value_in_base_units(),
                self.soft_y_max_row.get_value_in_base_units(),
            )
        else:
            mc.soft_limits = None

        travel = self.travel_speed_row.get_value_in_base_units()
        cut = self.cut_speed_row.get_value_in_base_units()
        accel = self.accel_row.get_value_in_base_units()
        mc.max_travel_speed = int(travel) if travel > 0 else None
        mc.max_cut_speed = int(cut) if cut > 0 else None
        mc.acceleration = int(accel) if accel > 0 else None

        mc.home_on_start = self.home_on_start_row.get_active() or None
        mc.single_axis_homing_enabled = (
            self.single_axis_homing_row.get_active() or None
        )
        return True


__all__ = ["HardwarePage"]
