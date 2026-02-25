from gi.repository import Adw, Gtk

from ..shared.adwfix import get_spinrow_float
from ...machine.driver.driver import Axis
from ...machine.models.machine import Machine, Origin
from ..shared.preferences_page import TrackedPreferencesPage


class HardwarePage(TrackedPreferencesPage):
    key = "hardware"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Hardware"),
            icon_name="hardware-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._is_initializing = True

        axes_group = Adw.PreferencesGroup(title=_("Axes"))
        axes_group.set_description(
            _("Configure the axis extents and coordinate system.")
        )
        self.add(axes_group)

        x_extent_adjustment = Gtk.Adjustment(
            lower=50, upper=10000, step_increment=1, page_increment=10
        )
        self.x_extent_row = Adw.SpinRow(
            title=_("X Extent"),
            subtitle=_("Full X-axis travel range"),
            adjustment=x_extent_adjustment,
            digits=2,
        )
        x_extent_adjustment.set_value(self.machine.axis_extents[0])
        self.x_extent_row.connect("changed", self.on_x_extent_changed)
        axes_group.add(self.x_extent_row)

        y_extent_adjustment = Gtk.Adjustment(
            lower=50, upper=10000, step_increment=1, page_increment=10
        )
        self.y_extent_row = Adw.SpinRow(
            title=_("Y Extent"),
            subtitle=_("Full Y-axis travel range"),
            adjustment=y_extent_adjustment,
            digits=2,
        )
        y_extent_adjustment.set_value(self.machine.axis_extents[1])
        self.y_extent_row.connect("changed", self.on_y_extent_changed)
        axes_group.add(self.y_extent_row)

        origin_store = Gtk.StringList()
        origin_store.append(_("Bottom Left"))
        origin_store.append(_("Top Left"))
        origin_store.append(_("Top Right"))
        origin_store.append(_("Bottom Right"))
        origin_combo_row = Adw.ComboRow(
            title=_("Coordinate Origin (0,0)"),
            subtitle=_(
                "The physical corner where coordinates are zero after homing"
            ),
            model=origin_store,
        )
        origin_combo_row.set_selected(
            {
                Origin.BOTTOM_LEFT: 0,
                Origin.TOP_LEFT: 1,
                Origin.TOP_RIGHT: 2,
                Origin.BOTTOM_RIGHT: 3,
            }.get(self.machine.origin, 0)
        )
        origin_combo_row.connect("notify::selected", self.on_origin_changed)
        self.origin_combo_row = origin_combo_row
        axes_group.add(origin_combo_row)

        self.reverse_x_axis_row = Adw.SwitchRow()
        self.reverse_x_axis_row.set_title(_("Reverse X-Axis Direction"))
        self.reverse_x_axis_row.set_subtitle(
            _("Makes coordinate values negative")
        )
        self.reverse_x_axis_row.set_active(machine.reverse_x_axis)
        self.reverse_x_axis_row.connect(
            "notify::active", self.on_reverse_x_changed
        )
        axes_group.add(self.reverse_x_axis_row)

        self.reverse_y_axis_row = Adw.SwitchRow()
        self.reverse_y_axis_row.set_title(_("Reverse Y-Axis Direction"))
        self.reverse_y_axis_row.set_subtitle(
            _("Makes coordinate values negative")
        )
        self.reverse_y_axis_row.set_active(machine.reverse_y_axis)
        self.reverse_y_axis_row.connect(
            "notify::active", self.on_reverse_y_changed
        )
        axes_group.add(self.reverse_y_axis_row)

        self.reverse_z_axis_row = Adw.SwitchRow()
        self.reverse_z_axis_row.set_title(_("Reverse Z-Axis Direction"))
        self.reverse_z_axis_row.set_subtitle(
            _(
                "Enable if a positive Z command (e.g., G0 Z10) moves the head "
                "down"
            )
        )
        self.reverse_z_axis_row.set_active(machine.reverse_z_axis)
        self.reverse_z_axis_row.connect(
            "notify::active", self.on_reverse_z_changed
        )
        axes_group.add(self.reverse_z_axis_row)

        work_area_group = Adw.PreferencesGroup(title=_("Work Area"))
        work_area_group.set_description(
            _("Margins define the unusable space around the axis extents.")
        )
        self.add(work_area_group)

        ml, mt, mr, mb = self.machine.work_margins

        margin_left_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.margin_left_row = Adw.SpinRow(
            title=_("Left Margin"),
            subtitle=_("Unusable space from left edge"),
            adjustment=margin_left_adjustment,
            digits=2,
        )
        margin_left_adjustment.set_value(ml)
        self.margin_left_row.connect("changed", self.on_margins_changed)
        work_area_group.add(self.margin_left_row)

        margin_top_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.margin_top_row = Adw.SpinRow(
            title=_("Top Margin"),
            subtitle=_("Unusable space from top edge"),
            adjustment=margin_top_adjustment,
            digits=2,
        )
        margin_top_adjustment.set_value(mt)
        self.margin_top_row.connect("changed", self.on_margins_changed)
        work_area_group.add(self.margin_top_row)

        margin_right_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.margin_right_row = Adw.SpinRow(
            title=_("Right Margin"),
            subtitle=_("Unusable space from right edge"),
            adjustment=margin_right_adjustment,
            digits=2,
        )
        margin_right_adjustment.set_value(mr)
        self.margin_right_row.connect("changed", self.on_margins_changed)
        work_area_group.add(self.margin_right_row)

        margin_bottom_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.margin_bottom_row = Adw.SpinRow(
            title=_("Bottom Margin"),
            subtitle=_("Unusable space from bottom edge"),
            adjustment=margin_bottom_adjustment,
            digits=2,
        )
        margin_bottom_adjustment.set_value(mb)
        self.margin_bottom_row.connect("changed", self.on_margins_changed)
        work_area_group.add(self.margin_bottom_row)

        soft_limits_group = Adw.PreferencesGroup(title=_("Soft Limits"))
        soft_limits_group.set_description(
            _(
                "Configurable safety bounds for jogging. "
                "Leave disabled to use work surface bounds."
            )
        )
        self.add(soft_limits_group)

        self.soft_limits_enabled_row = Adw.SwitchRow()
        self.soft_limits_enabled_row.set_title(_("Enable Custom Soft Limits"))
        self.soft_limits_enabled_row.set_subtitle(
            _("Override work surface bounds with custom limits")
        )
        has_custom_limits = self.machine.soft_limits is not None
        self.soft_limits_enabled_row.set_active(has_custom_limits)
        self.soft_limits_enabled_row.connect(
            "notify::active", self.on_soft_limits_enabled_changed
        )
        soft_limits_group.add(self.soft_limits_enabled_row)

        soft_x_min_adjustment = Gtk.Adjustment(
            lower=0,
            upper=self.machine.axis_extents[0],
            step_increment=1,
            page_increment=10,
        )
        self.soft_x_min_adjustment = soft_x_min_adjustment
        self.soft_x_min_row = Adw.SpinRow(
            title=_("X Min"),
            subtitle=_("Minimum X coordinate"),
            adjustment=soft_x_min_adjustment,
            digits=2,
        )
        limits = self.machine.soft_limits or (0, 0, *self.machine.axis_extents)
        soft_x_min_adjustment.set_value(limits[0])
        self.soft_x_min_row.connect("changed", self.on_soft_limits_changed)
        self.soft_x_min_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_x_min_row)

        soft_y_min_adjustment = Gtk.Adjustment(
            lower=0,
            upper=self.machine.axis_extents[1],
            step_increment=1,
            page_increment=10,
        )
        self.soft_y_min_adjustment = soft_y_min_adjustment
        self.soft_y_min_row = Adw.SpinRow(
            title=_("Y Min"),
            subtitle=_("Minimum Y coordinate"),
            adjustment=soft_y_min_adjustment,
            digits=2,
        )
        soft_y_min_adjustment.set_value(limits[1])
        self.soft_y_min_row.connect("changed", self.on_soft_limits_changed)
        self.soft_y_min_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_y_min_row)

        soft_x_max_adjustment = Gtk.Adjustment(
            lower=0,
            upper=self.machine.axis_extents[0],
            step_increment=1,
            page_increment=10,
        )
        self.soft_x_max_adjustment = soft_x_max_adjustment
        self.soft_x_max_row = Adw.SpinRow(
            title=_("X Max"),
            subtitle=_("Maximum X coordinate"),
            adjustment=soft_x_max_adjustment,
            digits=2,
        )
        soft_x_max_adjustment.set_value(limits[2])
        self.soft_x_max_row.connect("changed", self.on_soft_limits_changed)
        self.soft_x_max_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_x_max_row)

        soft_y_max_adjustment = Gtk.Adjustment(
            lower=0,
            upper=self.machine.axis_extents[1],
            step_increment=1,
            page_increment=10,
        )
        self.soft_y_max_adjustment = soft_y_max_adjustment
        self.soft_y_max_row = Adw.SpinRow(
            title=_("Y Max"),
            subtitle=_("Maximum Y coordinate"),
            adjustment=soft_y_max_adjustment,
            digits=2,
        )
        soft_y_max_adjustment.set_value(limits[3])
        self.soft_y_max_row.connect("changed", self.on_soft_limits_changed)
        self.soft_y_max_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_y_max_row)

        self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)

        self._is_initializing = False
        self._update_z_axis_state()

    def _on_machine_changed(self, sender, **kwargs):
        if self._is_initializing:
            return
        self._update_z_axis_state()
        self._update_axis_extents_ui()
        self._update_soft_limits_ui()

    def _update_axis_extents_ui(self):
        self.x_extent_row.set_value(self.machine.axis_extents[0])
        self.y_extent_row.set_value(self.machine.axis_extents[1])

    def _update_soft_limits_ui(self):
        w, h = self.machine.axis_extents
        self.soft_x_min_adjustment.set_upper(w)
        self.soft_x_max_adjustment.set_upper(w)
        self.soft_y_min_adjustment.set_upper(h)
        self.soft_y_max_adjustment.set_upper(h)
        limits = self.machine.soft_limits or (0, 0, w, h)
        self.soft_x_min_row.set_value(limits[0])
        self.soft_y_min_row.set_value(limits[1])
        self.soft_x_max_row.set_value(limits[2])
        self.soft_y_max_row.set_value(limits[3])

    def _on_destroy(self, *args):
        self.machine.changed.disconnect(self._on_machine_changed)

    def on_origin_changed(self, row, _):
        selected_index = row.get_selected()
        origin_map = {
            0: Origin.BOTTOM_LEFT,
            1: Origin.TOP_LEFT,
            2: Origin.TOP_RIGHT,
            3: Origin.BOTTOM_RIGHT,
        }
        origin = origin_map.get(selected_index, Origin.BOTTOM_LEFT)
        self.machine.set_origin(origin)

    def on_reverse_x_changed(self, row, _):
        self.machine.set_reverse_x_axis(row.get_active())

    def on_reverse_y_changed(self, row, _):
        self.machine.set_reverse_y_axis(row.get_active())

    def on_reverse_z_changed(self, row, _):
        self.machine.set_reverse_z_axis(row.get_active())

    def on_x_extent_changed(self, spinrow):
        x = get_spinrow_float(spinrow)
        y = self.machine.axis_extents[1]
        self.machine.set_axis_extents(x, y)

    def on_y_extent_changed(self, spinrow):
        x = self.machine.axis_extents[0]
        y = get_spinrow_float(spinrow)
        self.machine.set_axis_extents(x, y)

    def on_margins_changed(self, _spinrow):
        ml = get_spinrow_float(self.margin_left_row)
        mt = get_spinrow_float(self.margin_top_row)
        mr = get_spinrow_float(self.margin_right_row)
        mb = get_spinrow_float(self.margin_bottom_row)

        extent_w, extent_h = self.machine.axis_extents
        ml = max(0, min(ml, extent_w - 1))
        mr = max(0, min(mr, extent_w - ml - 1))
        mt = max(0, min(mt, extent_h - 1))
        mb = max(0, min(mb, extent_h - mt - 1))

        self.machine.set_work_margins(ml, mt, mr, mb)

    def on_soft_limits_enabled_changed(self, row, _):
        enabled = row.get_active()
        self.soft_x_min_row.set_sensitive(enabled)
        self.soft_y_min_row.set_sensitive(enabled)
        self.soft_x_max_row.set_sensitive(enabled)
        self.soft_y_max_row.set_sensitive(enabled)

        if enabled:
            x_min = get_spinrow_float(self.soft_x_min_row)
            y_min = get_spinrow_float(self.soft_y_min_row)
            x_max = get_spinrow_float(self.soft_x_max_row)
            y_max = get_spinrow_float(self.soft_y_max_row)
            self.machine.set_soft_limits(x_min, y_min, x_max, y_max)
        else:
            self.machine._soft_limits = None
            self.machine.changed.send(self.machine)

    def on_soft_limits_changed(self, _spinrow):
        if not self.soft_limits_enabled_row.get_active():
            return
        x_min = get_spinrow_float(self.soft_x_min_row)
        y_min = get_spinrow_float(self.soft_y_min_row)
        x_max = get_spinrow_float(self.soft_x_max_row)
        y_max = get_spinrow_float(self.soft_y_max_row)
        self.machine.set_soft_limits(x_min, y_min, x_max, y_max)

    def _update_z_axis_state(self):
        if self._is_initializing:
            return

        has_z = self.machine.can_jog(Axis.Z)
        self.reverse_z_axis_row.set_visible(has_z)
