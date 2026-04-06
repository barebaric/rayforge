from typing import Optional, cast
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...machine.models.machine import Machine
from ...machine.models.zone import Zone, ZoneShape
from ..shared.adwfix import get_spinrow_float
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage


class ZoneRow(Gtk.Box):
    def __init__(self, machine: Machine, zone: Zone):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.machine = machine
        self.zone = zone
        self._setup_ui()

    def _setup_ui(self):
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        info_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(info_box)

        self.title_label = Gtk.Label(
            label=self.zone.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        info_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(
            label=self._get_subtitle_text(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.add_css_class("caption")
        info_box.append(self.subtitle_label)

        self.enabled_switch = Gtk.Switch(
            active=self.zone.enabled,
            valign=Gtk.Align.CENTER,
        )
        self.enabled_switch.connect("notify::active", self._on_enabled)
        self.append(self.enabled_switch)

        remove_button = Gtk.Button(
            icon_name="edit-delete-symbolic",
            valign=Gtk.Align.CENTER,
            css_classes=["flat", "destructive-action"],
        )
        remove_button.connect("clicked", self._on_remove)
        self.append(remove_button)

    def _get_subtitle_text(self) -> str:
        shape_labels = {
            ZoneShape.RECT: _("Rectangle"),
            ZoneShape.BOX: _("Box"),
            ZoneShape.CYLINDER: _("Cylinder"),
        }
        return shape_labels.get(self.zone.shape, str(self.zone.shape))

    def _on_enabled(self, switch, _param):
        self.zone.set_enabled(switch.get_active())

    def _on_remove(self, button):
        self.machine.remove_nogo_zone(self.zone)


class ZoneListEditor(PreferencesGroupWithButton):
    def __init__(self, machine: Machine, **kwargs):
        super().__init__(button_label=_("Add Zone"), **kwargs)
        self.machine = machine
        self._row_widgets: list[ZoneRow] = []
        self._known_uids: set[str] = set()
        self._setup_ui()
        self.machine.changed.connect(self._on_machine_changed)
        self._rebuild()

    def _setup_ui(self):
        placeholder = Gtk.Label(
            label=_("No no-go zones configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.set_show_separators(True)

    def _on_machine_changed(self, sender, **kwargs):
        current_uids = set(self.machine.nogo_zones.keys())
        if current_uids != self._known_uids:
            self._rebuild()
            return

        i = 0
        while True:
            row = self.list_box.get_row_at_index(i)
            if not row:
                break
            zone_row = cast(ZoneRow, row.get_child())
            zone = zone_row.zone
            zone_row.title_label.set_label(zone.name)
            zone_row.subtitle_label.set_label(zone_row._get_subtitle_text())
            zone_row.enabled_switch.set_active(zone.enabled)
            i += 1

    def _rebuild(self):
        sorted_zones = sorted(
            self.machine.nogo_zones.values(), key=lambda z: z.name
        )
        self._known_uids = set(self.machine.nogo_zones.keys())

        selected_zone = None
        selected_row = self.list_box.get_selected_row()
        if selected_row:
            widget = cast(ZoneRow, selected_row.get_child())
            selected_zone = widget.zone

        while True:
            row = self.list_box.get_row_at_index(0)
            if not row:
                break
            self.list_box.remove(row)
        self._row_widgets.clear()

        new_selection_index = -1
        for i, zone in enumerate(sorted_zones):
            if zone is selected_zone:
                new_selection_index = i
            list_box_row = Gtk.ListBoxRow()
            list_box_row.set_child(self.create_row_widget(zone))
            self.list_box.append(list_box_row)

        if new_selection_index >= 0:
            row = self.list_box.get_row_at_index(new_selection_index)
            self.list_box.select_row(row)
        elif sorted_zones:
            row = self.list_box.get_row_at_index(0)
            self.list_box.select_row(row)
        else:
            if self.list_box.get_selected_row():
                self.list_box.unselect_all()
            else:
                self.list_box.emit("row-selected", None)

    def create_row_widget(self, item: Zone) -> Gtk.Widget:
        row = ZoneRow(self.machine, item)
        self._row_widgets.append(row)
        return row

    def _on_add_clicked(self, button: Gtk.Button):
        new_zone = Zone()
        new_zone.name = _("New Zone")
        self.machine.add_nogo_zone(new_zone)
        self._rebuild()

        sorted_zones = sorted(
            self.machine.nogo_zones.values(), key=lambda z: z.name
        )
        idx = next(
            i for i, z in enumerate(sorted_zones) if z.uid == new_zone.uid
        )
        row = self.list_box.get_row_at_index(idx)
        if row:
            self.list_box.select_row(row)


class NogoZonesPage(TrackedPreferencesPage):
    key = "nogo-zones"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("No-Go Zones"),
            icon_name="action-unavailable-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._is_updating = True

        zones_group = Adw.PreferencesGroup(
            title=_("No-Go Zones"),
            description=_(
                "Define restricted areas on the work surface. A warning "
                "will be shown before running or exporting a job whose "
                "toolpath enters any enabled no-go zone."
            ),
        )
        self.add(zones_group)

        self.zone_list_editor = ZoneListEditor(machine=self.machine)
        zones_group.add(self.zone_list_editor)

        self.config_group = Adw.PreferencesGroup(
            title=_("Zone Properties"),
            description=_("Configure the selected zone."),
        )
        self.add(self.config_group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.connect("changed", self._on_name_changed)
        self.name_row.connect("activate", self._on_name_applied)
        name_focus_ctrl = Gtk.EventControllerFocus()
        name_focus_ctrl.connect("leave", self._on_name_focus_left)
        self.name_row.add_controller(name_focus_ctrl)
        self.config_group.add(self.name_row)

        shape_store = Gtk.StringList()
        shape_store.append(_("Rectangle"))
        shape_store.append(_("Box"))
        shape_store.append(_("Cylinder"))
        self.shape_row = Adw.ComboRow(
            title=_("Shape"),
            subtitle=_("Zone geometry shape"),
            model=shape_store,
        )
        self.shape_row.connect("notify::selected", self._on_shape_changed)
        self.config_group.add(self.shape_row)

        x_adj = Gtk.Adjustment(
            lower=-10000, upper=10000, step_increment=1, page_increment=10
        )
        self.x_row = Adw.SpinRow(
            title=_("X"),
            subtitle=_("X position in {wcs}").format(
                wcs=self.machine.machine_space_wcs_display_name
            ),
            adjustment=x_adj,
            digits=2,
        )
        self.x_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.x_row)

        y_adj = Gtk.Adjustment(
            lower=-10000, upper=10000, step_increment=1, page_increment=10
        )
        self.y_row = Adw.SpinRow(
            title=_("Y"),
            subtitle=_("Y position in {wcs}").format(
                wcs=self.machine.machine_space_wcs_display_name
            ),
            adjustment=y_adj,
            digits=2,
        )
        self.y_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.y_row)

        z_adj = Gtk.Adjustment(
            lower=-10000, upper=10000, step_increment=1, page_increment=10
        )
        self.z_row = Adw.SpinRow(
            title=_("Z"),
            subtitle=_("Z position in {wcs}").format(
                wcs=self.machine.machine_space_wcs_display_name
            ),
            adjustment=z_adj,
            digits=2,
        )
        self.z_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.z_row)

        w_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.w_row = Adw.SpinRow(
            title=_("Width"),
            subtitle=_("Width in mm"),
            adjustment=w_adj,
            digits=2,
        )
        self.w_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.w_row)

        h_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.h_row = Adw.SpinRow(
            title=_("Height"),
            subtitle=_("Height in mm"),
            adjustment=h_adj,
            digits=2,
        )
        self.h_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.h_row)

        d_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.d_row = Adw.SpinRow(
            title=_("Depth"),
            subtitle=_("Depth (Z extent) in mm"),
            adjustment=d_adj,
            digits=2,
        )
        self.d_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.d_row)

        r_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.radius_row = Adw.SpinRow(
            title=_("Radius"),
            subtitle=_("Cylinder radius in mm"),
            adjustment=r_adj,
            digits=2,
        )
        self.radius_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.radius_row)

        cyl_h_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.cyl_height_row = Adw.SpinRow(
            title=_("Cylinder Height"),
            subtitle=_("Cylinder height in mm"),
            adjustment=cyl_h_adj,
            digits=2,
        )
        self.cyl_height_row.connect("notify::value", self._on_param_changed)
        self.config_group.add(self.cyl_height_row)

        self.zone_list_editor.list_box.connect(
            "row-selected", self._on_zone_selected
        )

        self.machine.changed.connect(self._on_machine_changed)
        self.connect("destroy", self._on_destroy)

        self._is_updating = False
        initial_row = self.zone_list_editor.list_box.get_selected_row()
        self._on_zone_selected(self.zone_list_editor.list_box, initial_row)

    def _get_selected_zone(self) -> Optional[Zone]:
        selected_row = self.zone_list_editor.list_box.get_selected_row()
        if not selected_row:
            return None
        zone_row = cast(ZoneRow, selected_row.get_child())
        return zone_row.zone

    def _on_zone_selected(self, listbox, row):
        has_selection = row is not None
        self.config_group.set_visible(has_selection)
        if not has_selection:
            return

        zone = self._get_selected_zone()
        if not zone:
            return

        self._is_updating = True

        self.name_row.set_text(zone.name)
        shape_map = {
            ZoneShape.RECT: 0,
            ZoneShape.BOX: 1,
            ZoneShape.CYLINDER: 2,
        }
        self.shape_row.set_selected(shape_map.get(zone.shape, 0))
        self.x_row.set_value(zone.params.get("x", 0.0))
        self.y_row.set_value(zone.params.get("y", 0.0))
        self.z_row.set_value(zone.params.get("z", 0.0))
        self.w_row.set_value(zone.params.get("w", 10.0))
        self.h_row.set_value(zone.params.get("h", 10.0))
        self.d_row.set_value(zone.params.get("d", 10.0))
        self.radius_row.set_value(zone.params.get("radius", 5.0))
        self.cyl_height_row.set_value(zone.params.get("height", 10.0))
        self._update_field_visibility(zone)

        self._is_updating = False

    def _update_field_visibility(self, zone: Zone):
        is_3d = zone.shape in (ZoneShape.BOX, ZoneShape.CYLINDER)
        self.z_row.set_visible(is_3d)
        self.d_row.set_visible(zone.shape == ZoneShape.BOX)
        self.radius_row.set_visible(zone.shape == ZoneShape.CYLINDER)
        self.cyl_height_row.set_visible(zone.shape == ZoneShape.CYLINDER)

    def _on_name_changed(self, entry_row):
        if self._is_updating:
            return
        zone = self._get_selected_zone()
        if zone:
            zone.set_name(entry_row.get_text())

    def _on_name_applied(self, entry_row):
        self.zone_list_editor._rebuild()

    def _on_name_focus_left(self, controller):
        if not self._is_updating:
            self.zone_list_editor._rebuild()

    def _on_shape_changed(self, row, _param):
        if self._is_updating:
            return
        zone = self._get_selected_zone()
        if not zone:
            return
        shape_map = {
            0: ZoneShape.RECT,
            1: ZoneShape.BOX,
            2: ZoneShape.CYLINDER,
        }
        new_shape = shape_map.get(row.get_selected(), ZoneShape.RECT)
        if new_shape == zone.shape:
            return
        self._is_updating = True
        zone.set_shape(new_shape)
        self.z_row.set_value(zone.params.get("z", 0.0))
        self.w_row.set_value(zone.params.get("w", 10.0))
        self.h_row.set_value(zone.params.get("h", 10.0))
        self.d_row.set_value(zone.params.get("d", 10.0))
        self.radius_row.set_value(zone.params.get("radius", 5.0))
        self.cyl_height_row.set_value(zone.params.get("height", 10.0))
        self._update_field_visibility(zone)
        self._is_updating = False

    def _on_param_changed(self, _spinrow, _param):
        if self._is_updating:
            return
        zone = self._get_selected_zone()
        if not zone:
            return

        self._is_updating = True
        zone.set_param("x", get_spinrow_float(self.x_row))
        zone.set_param("y", get_spinrow_float(self.y_row))
        if zone.shape in (ZoneShape.BOX, ZoneShape.CYLINDER):
            zone.set_param("z", get_spinrow_float(self.z_row))
        zone.set_param("w", get_spinrow_float(self.w_row))
        zone.set_param("h", get_spinrow_float(self.h_row))
        if zone.shape == ZoneShape.BOX:
            zone.set_param("d", get_spinrow_float(self.d_row))
        elif zone.shape == ZoneShape.CYLINDER:
            zone.set_param("radius", get_spinrow_float(self.radius_row))
            zone.set_param("height", get_spinrow_float(self.cyl_height_row))
        self._is_updating = False

    def _on_machine_changed(self, sender, **kwargs):
        pass

    def _on_destroy(self, *args):
        self.machine.changed.disconnect(self._on_machine_changed)
        self.machine.changed.disconnect(
            self.zone_list_editor._on_machine_changed
        )
