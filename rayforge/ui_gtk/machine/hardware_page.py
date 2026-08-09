from gettext import gettext as _
from typing import cast

from gi.repository import Adw, Gtk
from raygeo.ops.axis import Axis

from ...machine.models.machine import Machine, Origin
from ...pipeline.coordspace import WorkspaceOrientation
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.preferences_page import TrackedPreferencesPage

_WORKSPACE_ORIENTATIONS = tuple(WorkspaceOrientation)


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

        self.x_extent_row = LengthSpinRow(
            _("X Extent"),
            _("Full X-axis travel range"),
            lower=50,
            upper=10000,
            value_in_base=self.machine.axis_extents[0],
        )
        self.x_extent_row.value_changed.connect(self.on_x_extent_changed)
        axes_group.add(self.x_extent_row)

        self.y_extent_row = LengthSpinRow(
            _("Y Extent"),
            _("Full Y-axis travel range"),
            lower=50,
            upper=10000,
            value_in_base=self.machine.axis_extents[1],
        )
        self.y_extent_row.value_changed.connect(self.on_y_extent_changed)
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

        # In languages with long text, the combo row doesn't allocate enough
        # width for the dropdown, so we have to manually set the list box
        # width. This is a bit hacky but it works.
        combo_child_box = origin_combo_row.get_last_child()
        if combo_child_box:
            suffix_box = combo_child_box.get_last_child()
            if suffix_box:
                list_box = cast(Gtk.ListBox, suffix_box.get_first_child())
                if list_box:
                    list_box.set_size_request(100, -1)
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

        orientation_labels = [
            _("Native"),
            _("Rotate Left"),
            _("Rotate Right"),
        ]
        self.workspace_orientation_row = Adw.ComboRow(
            title=_("Workspace Orientation"),
            subtitle=_(
                "Rotate the flat workspace and remap output to the "
                "machine; rotary layers require Native"
            ),
            model=Gtk.StringList.new(orientation_labels),
        )
        self.workspace_orientation_row.set_selected(
            _WORKSPACE_ORIENTATIONS.index(self.machine.workspace_orientation)
        )
        self.workspace_orientation_row.connect(
            "notify::selected", self.on_workspace_orientation_changed
        )
        axes_group.add(self.workspace_orientation_row)

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

        self.margin_left_row = LengthSpinRow(
            _("Left Margin"),
            _("Unusable space from left edge"),
            upper=10000,
            value_in_base=ml,
        )
        self.margin_left_row.value_changed.connect(self.on_margins_changed)
        work_area_group.add(self.margin_left_row)

        self.margin_top_row = LengthSpinRow(
            _("Top Margin"),
            _("Unusable space from top edge"),
            upper=10000,
            value_in_base=mt,
        )
        self.margin_top_row.value_changed.connect(self.on_margins_changed)
        work_area_group.add(self.margin_top_row)

        self.margin_right_row = LengthSpinRow(
            _("Right Margin"),
            _("Unusable space from right edge"),
            upper=10000,
            value_in_base=mr,
        )
        self.margin_right_row.value_changed.connect(self.on_margins_changed)
        work_area_group.add(self.margin_right_row)

        self.margin_bottom_row = LengthSpinRow(
            _("Bottom Margin"),
            _("Unusable space from bottom edge"),
            upper=10000,
            value_in_base=mb,
        )
        self.margin_bottom_row.value_changed.connect(self.on_margins_changed)
        work_area_group.add(self.margin_bottom_row)

        self.wcs_origin_row = Adw.SwitchRow()
        self.wcs_origin_row.set_title(_("Workarea Origin Is Coordinate Zero"))
        self.wcs_origin_row.set_subtitle(
            _(
                "Treat workarea origin as coordinate zero. "
                "Hides WCS controls and uses workarea margins as offsets."
            )
        )
        self.wcs_origin_row.set_active(machine.wcs_origin_is_workarea_origin)
        self.wcs_origin_row.connect(
            "notify::active", self.on_wcs_origin_is_workarea_origin_changed
        )
        work_area_group.add(self.wcs_origin_row)

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
        limits = self.machine.soft_limits or (0, 0, *self.machine.axis_extents)
        self.soft_limits_enabled_row.set_active(has_custom_limits)
        self.soft_limits_enabled_row.connect(
            "notify::active", self.on_soft_limits_enabled_changed
        )
        soft_limits_group.add(self.soft_limits_enabled_row)

        self.soft_x_min_row = LengthSpinRow(
            _("X Min"),
            _("Minimum X coordinate"),
            upper=self.machine.axis_extents[0],
            value_in_base=limits[0],
        )
        self.soft_x_min_row.value_changed.connect(self.on_soft_limits_changed)
        self.soft_x_min_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_x_min_row)

        self.soft_y_min_row = LengthSpinRow(
            _("Y Min"),
            _("Minimum Y coordinate"),
            upper=self.machine.axis_extents[1],
            value_in_base=limits[1],
        )
        self.soft_y_min_row.value_changed.connect(self.on_soft_limits_changed)
        self.soft_y_min_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_y_min_row)

        self.soft_x_max_row = LengthSpinRow(
            _("X Max"),
            _("Maximum X coordinate"),
            upper=self.machine.axis_extents[0],
            value_in_base=limits[2],
        )
        self.soft_x_max_row.value_changed.connect(self.on_soft_limits_changed)
        self.soft_x_max_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_x_max_row)

        self.soft_y_max_row = LengthSpinRow(
            _("Y Max"),
            _("Maximum Y coordinate"),
            upper=self.machine.axis_extents[1],
            value_in_base=limits[3],
        )
        self.soft_y_max_row.value_changed.connect(self.on_soft_limits_changed)
        self.soft_y_max_row.set_sensitive(has_custom_limits)
        soft_limits_group.add(self.soft_y_max_row)

        self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)

        self._is_initializing = False
        self._update_soft_limits_ui()
        self._update_z_axis_state()

    def _on_machine_changed(self, sender, **kwargs):
        if self._is_initializing:
            return
        self._update_z_axis_state()
        self._update_axis_extents_ui()
        self._update_soft_limits_ui()
        self._update_workspace_orientation_ui()

    def _update_workspace_orientation_ui(self):
        selected = _WORKSPACE_ORIENTATIONS.index(
            self.machine.workspace_orientation
        )
        self.workspace_orientation_row.set_selected(selected)

    def _update_axis_extents_ui(self):
        self.x_extent_row.set_value_in_base_units(self.machine.axis_extents[0])
        self.y_extent_row.set_value_in_base_units(self.machine.axis_extents[1])

    def _update_soft_limits_ui(self):
        w, h = self.machine.axis_extents
        self.soft_x_min_row.set_range(0.0, w)
        self.soft_x_max_row.set_range(0.0, w)
        self.soft_y_min_row.set_range(0.0, h)
        self.soft_y_max_row.set_range(0.0, h)
        limits = self.machine.soft_limits or (0, 0, w, h)
        self.soft_x_min_row.set_value_in_base_units(limits[0])
        self.soft_y_min_row.set_value_in_base_units(limits[1])
        self.soft_x_max_row.set_value_in_base_units(limits[2])
        self.soft_y_max_row.set_value_in_base_units(limits[3])

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

    def on_workspace_orientation_changed(self, row, _):
        selected = row.get_selected()
        if selected >= len(_WORKSPACE_ORIENTATIONS):
            return
        orientation = _WORKSPACE_ORIENTATIONS[selected]
        self.machine.set_workspace_orientation(orientation)

    def on_reverse_x_changed(self, row, _):
        self.machine.set_reverse_x_axis(row.get_active())

    def on_reverse_y_changed(self, row, _):
        self.machine.set_reverse_y_axis(row.get_active())

    def on_reverse_z_changed(self, row, _):
        self.machine.set_reverse_z_axis(row.get_active())

    def on_x_extent_changed(self, row):
        x = self.x_extent_row.get_value_in_base_units()
        y = self.machine.axis_extents[1]
        self.machine.set_axis_extents(x, y)

    def on_y_extent_changed(self, row):
        x = self.machine.axis_extents[0]
        y = self.y_extent_row.get_value_in_base_units()
        self.machine.set_axis_extents(x, y)

    def on_margins_changed(self, row):
        ml = self.margin_left_row.get_value_in_base_units()
        mt = self.margin_top_row.get_value_in_base_units()
        mr = self.margin_right_row.get_value_in_base_units()
        mb = self.margin_bottom_row.get_value_in_base_units()

        extent_w, extent_h = self.machine.axis_extents
        ml = max(0, min(ml, extent_w - 1))
        mr = max(0, min(mr, extent_w - ml - 1))
        mt = max(0, min(mt, extent_h - 1))
        mb = max(0, min(mb, extent_h - mt - 1))

        self.machine.set_work_margins(ml, mt, mr, mb)

    def on_wcs_origin_is_workarea_origin_changed(self, row, _):
        self.machine.set_wcs_origin_is_workarea_origin(row.get_active())

    def on_soft_limits_enabled_changed(self, row, _):
        enabled = row.get_active()
        self.soft_x_min_row.set_sensitive(enabled)
        self.soft_y_min_row.set_sensitive(enabled)
        self.soft_x_max_row.set_sensitive(enabled)
        self.soft_y_max_row.set_sensitive(enabled)

        if enabled:
            x_min = self.soft_x_min_row.get_value_in_base_units()
            y_min = self.soft_y_min_row.get_value_in_base_units()
            x_max = self.soft_x_max_row.get_value_in_base_units()
            y_max = self.soft_y_max_row.get_value_in_base_units()
            self.machine.set_soft_limits(x_min, y_min, x_max, y_max)
        else:
            self.machine.clear_soft_limits()

    def on_soft_limits_changed(self, row):
        if not self.soft_limits_enabled_row.get_active():
            return
        x_min = self.soft_x_min_row.get_value_in_base_units()
        y_min = self.soft_y_min_row.get_value_in_base_units()
        x_max = self.soft_x_max_row.get_value_in_base_units()
        y_max = self.soft_y_max_row.get_value_in_base_units()
        self.machine.set_soft_limits(x_min, y_min, x_max, y_max)

    def _update_z_axis_state(self):
        if self._is_initializing:
            return

        has_z = self.machine.can_jog(Axis.Z)
        self.reverse_z_axis_row.set_visible(has_z)
