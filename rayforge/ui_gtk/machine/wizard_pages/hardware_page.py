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
from ...shared.adwfix import get_spinrow_float
from ...shared.unit_spin_row import UnitSpinRowHelper
from . import WizardPage, _makePreferencesGroup

_ORIGIN_INDEX_TO_ENUM = {
    0: Origin.BOTTOM_LEFT,
    1: Origin.TOP_LEFT,
    2: Origin.TOP_RIGHT,
    3: Origin.BOTTOM_RIGHT,
}
_ORIGIN_ENUM_TO_INDEX = {v: k for k, v in _ORIGIN_INDEX_TO_ENUM.items()}

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

        x_adj = Gtk.Adjustment(
            lower=10, upper=10000, step_increment=1, page_increment=10
        )
        self.x_row = Adw.SpinRow(
            title=_("X Extent"),
            subtitle=_("Full X-axis travel range"),
            adjustment=x_adj,
            digits=2,
        )
        axes_group.add(self.x_row)
        self.x_helper = UnitSpinRowHelper(
            spin_row=self.x_row,
            quantity="length",
            max_value_in_base=10000.0,
            min_digits=2,
        )

        y_adj = Gtk.Adjustment(
            lower=10, upper=10000, step_increment=1, page_increment=10
        )
        self.y_row = Adw.SpinRow(
            title=_("Y Extent"),
            subtitle=_("Full Y-axis travel range"),
            adjustment=y_adj,
            digits=2,
        )
        axes_group.add(self.y_row)
        self.y_helper = UnitSpinRowHelper(
            spin_row=self.y_row,
            quantity="length",
            max_value_in_base=10000.0,
            min_digits=2,
        )

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
            margins_group, _("Left Margin"), _("Unusable space from left edge")
        )
        self.margin_top_row = self._build_margin_row(
            margins_group, _("Top Margin"), _("Unusable space from top edge")
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

        travel_adj = Gtk.Adjustment(
            lower=0, upper=60000, step_increment=100, page_increment=1000
        )
        self.travel_speed_row = Adw.SpinRow(
            title=_("Max Travel Speed"), adjustment=travel_adj
        )
        speed_group.add(self.travel_speed_row)
        self.travel_speed_helper = UnitSpinRowHelper(
            spin_row=self.travel_speed_row,
            quantity="speed",
            max_value_in_base=60000.0,
        )

        cut_adj = Gtk.Adjustment(
            lower=0, upper=60000, step_increment=100, page_increment=1000
        )
        self.cut_speed_row = Adw.SpinRow(
            title=_("Max Cut Speed"), adjustment=cut_adj
        )
        speed_group.add(self.cut_speed_row)
        self.cut_speed_helper = UnitSpinRowHelper(
            spin_row=self.cut_speed_row,
            quantity="speed",
            max_value_in_base=60000.0,
        )

        accel_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=10, page_increment=100
        )
        self.accel_row = Adw.SpinRow(
            title=_("Acceleration"), adjustment=accel_adj
        )
        speed_group.add(self.accel_row)
        self.accel_helper = UnitSpinRowHelper(
            spin_row=self.accel_row,
            quantity="acceleration",
            max_value_in_base=10000.0,
        )

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
        self.x_row.connect("notify::value", self._on_extents_changed)
        self.y_row.connect("notify::value", self._on_extents_changed)

        # The page is always consider-ready because the user can skip
        # fields they don't know yet (defaults are sensible). The
        # orchestrator will surface sanity-check warnings at Review.
        self.set_ready(True)

    # ----- row builders ---------------------------------------------------

    def _build_margin_row(
        self, group: Adw.PreferencesGroup, title: str, subtitle: str
    ) -> Adw.SpinRow:
        adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        row = Adw.SpinRow(
            title=title, subtitle=subtitle, adjustment=adj, digits=2
        )
        group.add(row)
        return row

    def _build_soft_limit_row(self, title: str, subtitle: str) -> Adw.SpinRow:
        adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        row = Adw.SpinRow(
            title=title,
            subtitle=subtitle,
            adjustment=adj,
            digits=2,
            sensitive=False,
        )
        self.soft_limits_group.add(row)
        return row

    def _on_extents_changed(self, _row, _param) -> None:
        x = get_spinrow_float(self.x_row)
        y = get_spinrow_float(self.y_row)
        self.soft_x_min_row.get_adjustment().set_upper(x)
        self.soft_x_max_row.get_adjustment().set_upper(x)
        self.soft_y_min_row.get_adjustment().set_upper(y)
        self.soft_y_max_row.get_adjustment().set_upper(y)

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
            self.x_helper.set_value_in_base_units(mc.axis_extents[0])
            self.y_helper.set_value_in_base_units(mc.axis_extents[1])
        else:
            self.x_helper.set_value_in_base_units(100.0)
            self.y_helper.set_value_in_base_units(100.0)

        origin = mc.origin or Origin.BOTTOM_LEFT
        self.origin_row.set_selected(_ORIGIN_ENUM_TO_INDEX.get(origin, 0))

        # directional reversal flags aren't on MachineConfig; they live
        # on Machine directly. We treat them as ephemeral session state
        # via wizard.aux_state, defaulting to False.
        reverse = self.wizard.aux_state.setdefault("reverse", {})
        self.reverse_x_row.set_active(reverse.get("x", False))
        self.reverse_y_row.set_active(reverse.get("y", False))
        self.reverse_z_row.set_active(reverse.get("z", False))

        margins = mc.work_margins or (0.0, 0.0, 0.0, 0.0)
        self.margin_left_row.set_value(margins[0])
        self.margin_top_row.set_value(margins[1])
        self.margin_right_row.set_value(margins[2])
        self.margin_bottom_row.set_value(margins[3])

        soft = mc.soft_limits
        if soft:
            self.soft_limits_enabled_row.set_active(True)
            self.soft_x_min_row.set_value(soft[0])
            self.soft_y_min_row.set_value(soft[1])
            self.soft_x_max_row.set_value(soft[2])
            self.soft_y_max_row.set_value(soft[3])
        else:
            self.soft_limits_enabled_row.set_active(False)
            self.soft_x_min_row.set_value(0.0)
            self.soft_y_min_row.set_value(0.0)
            self.soft_x_max_row.set_value(get_spinrow_float(self.x_row))
            self.soft_y_max_row.set_value(get_spinrow_float(self.y_row))
        self._on_soft_limits_toggle(self.soft_limits_enabled_row, None)

        if mc.max_travel_speed is not None:
            self.travel_speed_helper.set_value_in_base_units(
                mc.max_travel_speed
            )
        else:
            self.travel_speed_helper.set_value_in_base_units(
                _DEFAULT_TRAVEL_SPEED
            )
        if mc.max_cut_speed is not None:
            self.cut_speed_helper.set_value_in_base_units(mc.max_cut_speed)
        else:
            self.cut_speed_helper.set_value_in_base_units(_DEFAULT_CUT_SPEED)
        if mc.acceleration is not None:
            self.accel_helper.set_value_in_base_units(mc.acceleration)
        else:
            self.accel_helper.set_value_in_base_units(_DEFAULT_ACCELERATION)

        self.home_on_start_row.set_active(bool(mc.home_on_start))
        self.single_axis_homing_row.set_active(
            bool(mc.single_axis_homing_enabled)
        )

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        mc = profile.machine_config

        x = self.x_helper.get_value_in_base_units()
        y = self.y_helper.get_value_in_base_units()
        if x > 0 and y > 0:
            mc.axis_extents = (float(x), float(y))

        mc.origin = _ORIGIN_INDEX_TO_ENUM.get(
            self.origin_row.get_selected(), Origin.BOTTOM_LEFT
        )

        # stash reversals to aux_state (defers to Machine during
        # create_machine); the orchestrator applies them post-creation.
        reverse = self.wizard.aux_state.setdefault("reverse", {})
        reverse["x"] = self.reverse_x_row.get_active()
        reverse["y"] = self.reverse_y_row.get_active()
        reverse["z"] = self.reverse_z_row.get_active()

        margins = (
            get_spinrow_float(self.margin_left_row),
            get_spinrow_float(self.margin_top_row),
            get_spinrow_float(self.margin_right_row),
            get_spinrow_float(self.margin_bottom_row),
        )
        if any(m > 0 for m in margins):
            mc.work_margins = margins
        else:
            mc.work_margins = None

        if self.soft_limits_enabled_row.get_active():
            mc.soft_limits = (
                get_spinrow_float(self.soft_x_min_row),
                get_spinrow_float(self.soft_y_min_row),
                get_spinrow_float(self.soft_x_max_row),
                get_spinrow_float(self.soft_y_max_row),
            )
        else:
            mc.soft_limits = None

        travel = self.travel_speed_helper.get_value_in_base_units()
        cut = self.cut_speed_helper.get_value_in_base_units()
        accel = self.accel_helper.get_value_in_base_units()
        mc.max_travel_speed = int(travel) if travel > 0 else None
        mc.max_cut_speed = int(cut) if cut > 0 else None
        mc.acceleration = int(accel) if accel > 0 else None

        mc.home_on_start = self.home_on_start_row.get_active() or None
        mc.single_axis_homing_enabled = (
            self.single_axis_homing_row.get_active() or None
        )
        return True


__all__ = ["HardwarePage"]
