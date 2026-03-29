from typing import cast, Optional
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...machine.driver.driver import Axis
from ...machine.models.machine import Machine
from ...machine.models.rotary_module import RotaryModule
from ..icons import get_icon
from ..shared.adwfix import get_spinrow_float
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage


class RotaryModuleRow(Gtk.Box):
    """A widget representing a single RotaryModule in a ListBox."""

    def __init__(self, machine: Machine, module: RotaryModule):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.machine = machine
        self.module = module
        self._toggle_handler_id = None
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
            label=self.module.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        info_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(
            label=self._get_subtitle_text(),
            halign=Gtk.Align.START,
            xalign=0,
            wrap=True,
        )
        self.subtitle_label.add_css_class("dim-label")
        info_box.append(self.subtitle_label)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.add_css_class("flat")
        self.delete_button.connect("clicked", self._on_remove_clicked)
        suffix_box.append(self.delete_button)

        self.select_button = Gtk.ToggleButton()
        self.select_button.add_css_class("flat")
        self.select_button.set_child(get_icon("object-select-symbolic"))
        self.select_button.set_tooltip_text(_("Set as default"))
        self._toggle_handler_id = self.select_button.connect(
            "toggled", self._on_select_toggled
        )
        self.select_button.set_valign(Gtk.Align.CENTER)
        suffix_box.append(self.select_button)

        self._update_selection_state()

    def _get_subtitle_text(self) -> str:
        return _("Axis {axis}, {length}mm, chuck Ø{chuck:.0f}mm").format(
            axis=self.module.axis.name,
            length=self.module.length,
            chuck=self.module.chuck_diameter,
        )

    def _update_selection_state(self):
        is_default = (
            self.machine.default_rotary_module_uid is not None
            and self.machine.default_rotary_module_uid == self.module.uid
        )
        if self._toggle_handler_id is not None:
            self.select_button.handler_block(self._toggle_handler_id)
        self.select_button.set_active(is_default)
        if self._toggle_handler_id is not None:
            self.select_button.handler_unblock(self._toggle_handler_id)

    def _on_select_toggled(self, button: Gtk.ToggleButton):
        if not button.get_active():
            button.set_active(True)
            return

        self.machine.set_default_rotary_module_uid(self.module.uid)

    def _on_remove_clicked(self, button: Gtk.Button):
        self.machine.remove_rotary_module(self.module)


class RotaryModuleListEditor(PreferencesGroupWithButton):
    """An Adwaita widget for managing a list of rotary modules."""

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(button_label=_("Add Rotary Module"), **kwargs)
        self.machine = machine
        self._row_widgets: list[RotaryModuleRow] = []
        self._known_uids: set[str] = set()
        self._setup_ui()
        self.machine.changed.connect(self._on_machine_changed)
        self._rebuild()

    def _setup_ui(self):
        placeholder = Gtk.Label(
            label=_("No rotary modules configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.set_show_separators(True)

    def _on_machine_changed(self, sender, **kwargs):
        current_uids = set(self.machine.rotary_modules.keys())
        if current_uids != self._known_uids:
            self._rebuild()
            return

        i = 0
        while True:
            row = self.list_box.get_row_at_index(i)
            if not row:
                break
            module_row = cast(RotaryModuleRow, row.get_child())
            module = module_row.module
            module_row.title_label.set_label(module.name)
            module_row.subtitle_label.set_label(
                module_row._get_subtitle_text()
            )
            module_row._update_selection_state()
            i += 1

    def _rebuild(self):
        sorted_modules = sorted(
            self.machine.rotary_modules.values(), key=lambda m: m.name
        )
        self._known_uids = set(self.machine.rotary_modules.keys())

        selected_module = None
        selected_row = self.list_box.get_selected_row()
        if selected_row:
            widget = cast(RotaryModuleRow, selected_row.get_child())
            selected_module = widget.module

        while True:
            row = self.list_box.get_row_at_index(0)
            if not row:
                break
            self.list_box.remove(row)
        self._row_widgets.clear()

        new_selection_index = -1
        for i, module in enumerate(sorted_modules):
            if module is selected_module:
                new_selection_index = i
            list_box_row = Gtk.ListBoxRow()
            list_box_row.set_child(self.create_row_widget(module))
            self.list_box.append(list_box_row)

        if new_selection_index >= 0:
            row = self.list_box.get_row_at_index(new_selection_index)
            self.list_box.select_row(row)
        elif sorted_modules:
            row = self.list_box.get_row_at_index(0)
            self.list_box.select_row(row)
        else:
            if self.list_box.get_selected_row():
                self.list_box.unselect_all()
            else:
                self.list_box.emit("row-selected", None)

    def create_row_widget(self, item: RotaryModule) -> Gtk.Widget:
        row = RotaryModuleRow(self.machine, item)
        self._row_widgets.append(row)
        return row

    def _on_add_clicked(self, button: Gtk.Button):
        new_module = RotaryModule()
        new_module.name = _("New Rotary Module")
        self.machine.add_rotary_module(new_module)

        if self.machine.default_rotary_module_uid is None:
            self.machine.set_default_rotary_module_uid(new_module.uid)

        self._rebuild()

        sorted_modules = sorted(
            self.machine.rotary_modules.values(), key=lambda m: m.name
        )
        idx = next(
            i for i, m in enumerate(sorted_modules) if m.uid == new_module.uid
        )
        row = self.list_box.get_row_at_index(idx)
        if row:
            self.list_box.select_row(row)


class RotaryModulePage(TrackedPreferencesPage):
    key = "rotary-module"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Rotary Module"),
            icon_name="object-rotate-left-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._is_updating = True

        defaults_group = Adw.PreferencesGroup(
            title=_("Rotary Defaults"),
            description=_("Default settings applied to new layers."),
        )
        self.add(defaults_group)

        self.rotary_enabled_default_row = Adw.SwitchRow(
            title=_("Enable Rotary by Default"),
            subtitle=_("New layers will default to rotary mode"),
        )
        self.rotary_enabled_default_row.set_active(
            machine.rotary_enabled_default
        )
        self.rotary_enabled_default_row.connect(
            "notify::active", self._on_rotary_enabled_default_changed
        )
        defaults_group.add(self.rotary_enabled_default_row)

        modules_group = Adw.PreferencesGroup(
            title=_("Modules"),
            description=_(
                "Define the physical rotary modules attached to your "
                "machine. Select one as the default."
            ),
        )
        self.add(modules_group)

        self.module_list_editor = RotaryModuleListEditor(
            machine=self.machine,
        )
        modules_group.add(self.module_list_editor)

        self.config_group = Adw.PreferencesGroup(
            title=_("Module Properties"),
            description=_("Configure the selected module."),
        )
        self.add(self.config_group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.connect("changed", self._on_name_changed)
        self.name_row.connect("activate", self._on_name_applied)
        name_focus_ctrl = Gtk.EventControllerFocus()
        name_focus_ctrl.connect("leave", self._on_name_focus_left)
        self.name_row.add_controller(name_focus_ctrl)
        self.config_group.add(self.name_row)

        excluded = (Axis.X, Axis.Y)
        valid_axes = sorted(
            [a for a in Axis if a not in excluded],
            key=lambda a: str(a.name or ""),
        )
        self._valid_axes = valid_axes

        module_axis_store = Gtk.StringList()
        for a in valid_axes:
            module_axis_store.append(a.name or "")
        self.module_axis_row = Adw.ComboRow(
            title=_("Axis"),
            subtitle=_("Axis letter for this module"),
            model=module_axis_store,
        )
        self.module_axis_row.connect(
            "notify::selected", self._on_module_axis_changed
        )
        self.config_group.add(self.module_axis_row)

        default_diam_adj = Gtk.Adjustment(
            lower=1, upper=10000, step_increment=1, page_increment=10
        )
        self.default_diameter_row = Adw.SpinRow(
            title=_("Default Workpiece Diameter"),
            subtitle=_("Default diameter for new layers using this module"),
            adjustment=default_diam_adj,
            digits=1,
        )
        self.default_diameter_row.connect(
            "notify::value", self._on_default_diameter_changed
        )
        self.config_group.add(self.default_diameter_row)

        x_adj = Gtk.Adjustment(
            lower=-10000, upper=10000, step_increment=1, page_increment=10
        )
        self.x_row = Adw.SpinRow(
            title=_("X Position"),
            subtitle=_("X coordinate in machine space"),
            adjustment=x_adj,
            digits=2,
        )
        self.x_row.connect("notify::value", self._on_position_changed)
        self.config_group.add(self.x_row)

        y_adj = Gtk.Adjustment(
            lower=-10000, upper=10000, step_increment=1, page_increment=10
        )
        self.y_row = Adw.SpinRow(
            title=_("Y Position"),
            subtitle=_("Y coordinate in machine space"),
            adjustment=y_adj,
            digits=2,
        )
        self.y_row.connect("notify::value", self._on_position_changed)
        self.config_group.add(self.y_row)

        z_adj = Gtk.Adjustment(
            lower=-10000, upper=10000, step_increment=1, page_increment=10
        )
        self.z_row = Adw.SpinRow(
            title=_("Z Position"),
            subtitle=_("Z coordinate in machine space"),
            adjustment=z_adj,
            digits=2,
        )
        self.z_row.connect("notify::value", self._on_position_changed)
        self.config_group.add(self.z_row)

        length_adj = Gtk.Adjustment(
            lower=1, upper=10000, step_increment=1, page_increment=10
        )
        self.length_row = Adw.SpinRow(
            title=_("Length"),
            subtitle=_("Length of the rotary module along the axis"),
            adjustment=length_adj,
            digits=1,
        )
        self.length_row.connect("notify::value", self._on_length_changed)
        self.config_group.add(self.length_row)

        chuck_adj = Gtk.Adjustment(
            lower=1, upper=10000, step_increment=1, page_increment=10
        )
        self.chuck_diameter_row = Adw.SpinRow(
            title=_("Chuck Diameter"),
            subtitle=_("Diameter of the chuck end"),
            adjustment=chuck_adj,
            digits=1,
        )
        self.chuck_diameter_row.connect(
            "notify::value", self._on_chuck_diameter_changed
        )
        self.config_group.add(self.chuck_diameter_row)

        tailstock_adj = Gtk.Adjustment(
            lower=1, upper=10000, step_increment=1, page_increment=10
        )
        self.tailstock_diameter_row = Adw.SpinRow(
            title=_("Tailstock Diameter"),
            subtitle=_("Diameter of the tailstock end"),
            adjustment=tailstock_adj,
            digits=1,
        )
        self.tailstock_diameter_row.connect(
            "notify::value", self._on_tailstock_diameter_changed
        )
        self.config_group.add(self.tailstock_diameter_row)

        height_adj = Gtk.Adjustment(
            lower=1, upper=10000, step_increment=1, page_increment=10
        )
        self.max_height_row = Adw.SpinRow(
            title=_("Max Height"),
            subtitle=_("Maximum height above the mounting surface"),
            adjustment=height_adj,
            digits=1,
        )
        self.max_height_row.connect(
            "notify::value", self._on_max_height_changed
        )
        self.config_group.add(self.max_height_row)

        self.module_list_editor.list_box.connect(
            "row-selected", self._on_module_selected
        )

        self.machine.changed.connect(self._on_machine_changed)
        self.connect("map", self._on_page_mapped)
        self.connect("destroy", self._on_destroy)

        self._is_updating = False
        initial_row = self.module_list_editor.list_box.get_selected_row()
        self._on_module_selected(self.module_list_editor.list_box, initial_row)

    def _get_selected_module(self) -> Optional[RotaryModule]:
        selected_row = self.module_list_editor.list_box.get_selected_row()
        if not selected_row:
            return None
        module_row = cast(RotaryModuleRow, selected_row.get_child())
        return module_row.module

    def _on_module_selected(self, listbox, row):
        has_selection = row is not None
        self.config_group.set_visible(has_selection)
        if not has_selection:
            return

        module = self._get_selected_module()
        if not module:
            return

        self._is_updating = True

        self.name_row.set_text(module.name)
        try:
            selected = self._valid_axes.index(module.axis)
        except ValueError:
            selected = 0
        self.module_axis_row.set_selected(selected)
        self.default_diameter_row.set_value(module.default_diameter)
        self.x_row.set_value(module.x)
        self.y_row.set_value(module.y)
        self.z_row.set_value(module.z)
        self.length_row.set_value(module.length)
        self.chuck_diameter_row.set_value(module.chuck_diameter)
        self.tailstock_diameter_row.set_value(module.tailstock_diameter)
        self.max_height_row.set_value(module.max_height)

        self._is_updating = False

    def _on_rotary_enabled_default_changed(self, row, _param):
        if self._is_updating:
            return
        self.machine.set_rotary_enabled_default(row.get_active())

    def _on_name_changed(self, entry_row):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_name(entry_row.get_text())

    def _on_name_applied(self, entry_row):
        self.module_list_editor._rebuild()

    def _on_name_focus_left(self, controller):
        if not self._is_updating:
            self.module_list_editor._rebuild()

    def _on_module_axis_changed(self, row, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if not module:
            return
        selected = row.get_selected()
        if selected < len(self._valid_axes):
            module.set_axis(self._valid_axes[selected])

    def _on_default_diameter_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_default_diameter(get_spinrow_float(spinrow))

    def _on_position_changed(self, _spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if not module:
            return
        x = get_spinrow_float(self.x_row)
        y = get_spinrow_float(self.y_row)
        z = get_spinrow_float(self.z_row)
        module.set_position(x, y, z)

    def _on_length_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_length(get_spinrow_float(spinrow))

    def _on_chuck_diameter_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_chuck_diameter(get_spinrow_float(spinrow))

    def _on_tailstock_diameter_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_tailstock_diameter(get_spinrow_float(spinrow))

    def _on_max_height_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_max_height(get_spinrow_float(spinrow))

    def _on_machine_changed(self, sender, **kwargs):
        if self._is_updating:
            return
        self.rotary_enabled_default_row.set_active(
            self.machine.rotary_enabled_default
        )

    def _on_page_mapped(self, widget):
        if not self._is_updating:
            self.module_list_editor._rebuild()

    def _on_destroy(self, *args):
        self.machine.changed.disconnect(self._on_machine_changed)
        self.machine.changed.disconnect(
            self.module_list_editor._on_machine_changed
        )
