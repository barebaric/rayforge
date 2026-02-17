from gi.repository import Adw, Gtk

from ..shared.adwfix import get_spinrow_int
from ...machine.driver.driver import Axis
from ...machine.models.machine import Machine, Origin


class HardwarePage(Adw.PreferencesPage):
    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Hardware"),
            icon_name="hardware-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._is_initializing = True

        dimensions_group = Adw.PreferencesGroup(title=_("Dimensions"))
        dimensions_group.set_description(
            _("Configure the physical work area size.")
        )
        self.add(dimensions_group)

        width_adjustment = Gtk.Adjustment(
            lower=50, upper=10000, step_increment=1, page_increment=10
        )
        self.width_row = Adw.SpinRow(
            title=_("Width"),
            subtitle=_("Width of the machine work area in mm"),
            adjustment=width_adjustment,
        )
        width_adjustment.set_value(self.machine.dimensions[0])
        self.width_row.connect("changed", self.on_width_changed)
        dimensions_group.add(self.width_row)

        height_adjustment = Gtk.Adjustment(
            lower=50, upper=10000, step_increment=1, page_increment=10
        )
        self.height_row = Adw.SpinRow(
            title=_("Height"),
            subtitle=_("Height of the machine work area in mm"),
            adjustment=height_adjustment,
        )
        height_adjustment.set_value(self.machine.dimensions[1])
        self.height_row.connect("changed", self.on_height_changed)
        dimensions_group.add(self.height_row)

        axes_group = Adw.PreferencesGroup(title=_("Axes"))
        axes_group.set_description(
            _("Configure coordinate system and axis orientation.")
        )
        self.add(axes_group)

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

        x_offset_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.x_offset_row = Adw.SpinRow(
            title=_("X Offset"),
            subtitle=_("Offset to add to each gcode command on X axis"),
            adjustment=x_offset_adjustment,
        )
        x_offset_adjustment.set_value(self.machine.offsets[0])
        self.x_offset_row.connect("changed", self.on_x_offset_changed)
        axes_group.add(self.x_offset_row)

        y_offset_adjustment = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.y_offset_row = Adw.SpinRow(
            title=_("Y Offset"),
            subtitle=_("Offset to add to each gcode command on Y axis"),
            adjustment=y_offset_adjustment,
        )
        y_offset_adjustment.set_value(self.machine.offsets[1])
        self.y_offset_row.connect("changed", self.on_y_offset_changed)
        axes_group.add(self.y_offset_row)

        self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)

        self._is_initializing = False
        self._update_z_axis_state()

    def _on_machine_changed(self, sender, **kwargs):
        self._update_z_axis_state()

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

    def on_width_changed(self, spinrow):
        width = get_spinrow_int(spinrow)
        height = self.machine.dimensions[1]
        self.machine.set_dimensions(width, height)

    def on_height_changed(self, spinrow):
        width = self.machine.dimensions[0]
        height = get_spinrow_int(spinrow)
        self.machine.set_dimensions(width, height)

    def on_x_offset_changed(self, spinrow):
        y_offset = self.machine.offsets[1]
        x_offset = get_spinrow_int(spinrow)
        self.machine.set_offsets(x_offset, y_offset)

    def on_y_offset_changed(self, spinrow):
        x_offset = self.machine.offsets[0]
        y_offset = get_spinrow_int(spinrow)
        self.machine.set_offsets(x_offset, y_offset)

    def _update_z_axis_state(self):
        if self._is_initializing:
            return

        has_z = self.machine.can_jog(Axis.Z)
        self.reverse_z_axis_row.set_visible(has_z)
