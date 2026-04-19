from pathlib import Path
from typing import List, cast, Optional
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...context import get_context
from ...core.model import Model
from ...core.ops.axis import Axis
from ...machine.models.machine import Machine
from ...machine.models.rotary_module import (
    RotaryModule,
    RotaryMode,
    RotaryType,
)
from ..icons import get_icon
from ..shared.adwfix import get_spinrow_float
from ..shared.model_selection_dialog import ModelSelectionDialog
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage
from ..sim3d.canvas3d.model_renderer import get_model_extent


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
        self.select_button.set_child(get_icon("check-symbolic"))
        self.select_button.set_tooltip_text(_("Set as default"))
        self._toggle_handler_id = self.select_button.connect(
            "toggled", self._on_select_toggled
        )
        self.select_button.set_valign(Gtk.Align.CENTER)
        suffix_box.append(self.select_button)

        self._update_selection_state()

    def _get_subtitle_text(self) -> str:
        if self.module.mode == RotaryMode.AXIS_REPLACEMENT:
            mode_label = _("Axis Replacement")
        else:
            mode_label = _("True 4th Axis")
        axis = self.module.axis.name
        return _("{mode}, Axis {axis}").format(mode=mode_label, axis=axis)

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
            icon_name="rotary-symbolic",
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

        self.general_group = Adw.PreferencesGroup(
            title=_("General"),
        )
        self.add(self.general_group)

        self.model_group = Adw.PreferencesGroup(
            title=_("Model"),
        )
        self.add(self.model_group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.connect("changed", self._on_name_changed)
        self.name_row.connect("activate", self._on_name_applied)
        name_focus_ctrl = Gtk.EventControllerFocus()
        name_focus_ctrl.connect("leave", self._on_name_focus_left)
        self.name_row.add_controller(name_focus_ctrl)
        self.general_group.add(self.name_row)

        mode_store = Gtk.StringList()
        mode_store.append(_("True 4th Axis"))
        mode_store.append(_("Axis Replacement"))
        self._mode_values = [
            RotaryMode.TRUE_4TH_AXIS,
            RotaryMode.AXIS_REPLACEMENT,
        ]
        self.mode_row = Adw.ComboRow(
            title=_("Connection Mode"),
            subtitle=_(
                "How the rotary is connected to the machine controller"
            ),
            model=mode_store,
        )
        self.mode_row.connect("notify::selected", self._on_mode_changed)
        self.general_group.add(self.mode_row)

        self._valid_axes: list[Axis] = []

        module_axis_store = Gtk.StringList()
        self.module_axis_row = Adw.ComboRow(
            title=_("Axis"),
            subtitle=_("Axis letter for this module"),
            model=module_axis_store,
        )
        self.module_axis_row.connect(
            "notify::selected", self._on_module_axis_changed
        )
        self.general_group.add(self.module_axis_row)

        self.reverse_axis_row = Adw.SwitchRow(
            title=_("Reversed Axis"),
            subtitle=_("Reverse the rotation direction of the rotary axis"),
        )
        self.reverse_axis_row.connect(
            "notify::active", self._on_reverse_axis_changed
        )
        self.general_group.add(self.reverse_axis_row)

        self.axis_position_x_row = Adw.SpinRow(
            title=_("Axis Offset X"),
            subtitle=_("Offset from module position to rotation axis (X)"),
            adjustment=Gtk.Adjustment(
                lower=-10000, upper=10000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.axis_position_x_row.connect(
            "notify::value", self._on_axis_position_changed
        )
        self.general_group.add(self.axis_position_x_row)

        self.axis_position_y_row = Adw.SpinRow(
            title=_("Axis Offset Y"),
            subtitle=_("Offset from module position to rotation axis (Y)"),
            adjustment=Gtk.Adjustment(
                lower=-10000, upper=10000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.axis_position_y_row.connect(
            "notify::value", self._on_axis_position_changed
        )
        self.general_group.add(self.axis_position_y_row)

        self.axis_position_z_row = Adw.SpinRow(
            title=_("Axis Offset Z"),
            subtitle=_("Offset from module position to rotation axis (Z)"),
            adjustment=Gtk.Adjustment(
                lower=-10000, upper=10000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.axis_position_z_row.connect(
            "notify::value", self._on_axis_position_changed
        )
        self.general_group.add(self.axis_position_z_row)

        rotary_type_store = Gtk.StringList()
        rotary_type_store.append(_("Jaws / Chuck"))
        rotary_type_store.append(_("Rollers"))
        self._rotary_type_values = [
            RotaryType.JAWS,
            RotaryType.ROLLERS,
        ]
        self.rotary_type_row = Adw.ComboRow(
            title=_("Drive Type"),
            subtitle=_("How the rotary module drives the workpiece rotation"),
            model=rotary_type_store,
        )
        self.rotary_type_row.connect(
            "notify::selected", self._on_rotary_type_changed
        )
        self.general_group.add(self.rotary_type_row)

        roller_diam_adj = Gtk.Adjustment(
            lower=0, upper=10000, step_increment=1, page_increment=10
        )
        self.roller_diameter_row = Adw.SpinRow(
            title=_("Roller Diameter"),
            subtitle=_("Diameter of the drive roller"),
            adjustment=roller_diam_adj,
            digits=1,
        )
        self.roller_diameter_row.connect(
            "notify::value", self._on_roller_diameter_changed
        )
        self.general_group.add(self.roller_diameter_row)

        mm_per_rot_adj = Gtk.Adjustment(
            lower=0, upper=100000, step_increment=1, page_increment=10
        )
        self.mu_per_rotation_row = Adw.SpinRow(
            title=_("Travel per Rotation"),
            subtitle=_(
                "Firmware distance for one full 360° rotation. "
                "0 = raw circumferential output."
            ),
            adjustment=mm_per_rot_adj,
            digits=2,
        )
        self.mu_per_rotation_row.connect(
            "notify::value", self._on_mm_per_rotation_changed
        )
        self.general_group.add(self.mu_per_rotation_row)

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
        self.general_group.add(self.default_diameter_row)

        max_len_adj = Gtk.Adjustment(
            lower=1, upper=10000, step_increment=10, page_increment=50
        )
        self.max_workpiece_length_row = Adw.SpinRow(
            title=_("Max Workpiece Length"),
            subtitle=_("Maximum workpiece length this module can accommodate"),
            adjustment=max_len_adj,
            digits=1,
        )
        self.max_workpiece_length_row.connect(
            "notify::value", self._on_max_workpiece_length_changed
        )
        self.general_group.add(self.max_workpiece_length_row)

        self.model_row = Adw.ActionRow(
            title=_("Model"),
            activatable=True,
        )
        self.model_row.connect("activated", self._on_model_activated)
        self.model_row.add_suffix(get_icon("go-next-symbolic"))
        self.model_group.add(self.model_row)

        self.scale_row = Adw.SpinRow(
            title=_("Scale"),
            subtitle=_("Uniform scale factor for the model"),
            adjustment=Gtk.Adjustment(
                lower=0.01, upper=1000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.scale_row.connect("notify::value", self._on_scale_changed)
        self.model_group.add(self.scale_row)

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
        self.model_group.add(self.x_row)

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
        self.model_group.add(self.y_row)

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
        self.model_group.add(self.z_row)

        self.rx_row = Adw.SpinRow(
            title=_("X Rotation"),
            subtitle=_("Degrees around the X axis"),
            adjustment=Gtk.Adjustment(
                lower=-360, upper=360, step_increment=1, page_increment=15
            ),
            digits=1,
        )
        self.rx_row.connect("notify::value", self._on_rotation_changed)
        self.model_group.add(self.rx_row)

        self.ry_row = Adw.SpinRow(
            title=_("Y Rotation"),
            subtitle=_("Degrees around the Y axis"),
            adjustment=Gtk.Adjustment(
                lower=-360, upper=360, step_increment=1, page_increment=15
            ),
            digits=1,
        )
        self.ry_row.connect("notify::value", self._on_rotation_changed)
        self.model_group.add(self.ry_row)

        self.rz_row = Adw.SpinRow(
            title=_("Z Rotation"),
            subtitle=_("Degrees around the Z axis"),
            adjustment=Gtk.Adjustment(
                lower=-360, upper=360, step_increment=1, page_increment=15
            ),
            digits=1,
        )
        self.rz_row.connect("notify::value", self._on_rotation_changed)
        self.model_group.add(self.rz_row)

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
        for g in (self.general_group, self.model_group):
            g.set_visible(has_selection)
        if not has_selection:
            return

        module = self._get_selected_module()
        if not module:
            return

        self._is_updating = True

        self.name_row.set_text(module.name)

        try:
            mode_idx = self._mode_values.index(module.mode)
        except ValueError:
            mode_idx = 0
        self.mode_row.set_selected(mode_idx)

        self.mu_per_rotation_row.set_value(module.mu_per_rotation)
        self._update_mode_dependent_rows(module)

        self.default_diameter_row.set_value(module.default_diameter)
        self.max_workpiece_length_row.set_value(module.max_workpiece_length)

        try:
            type_idx = self._rotary_type_values.index(module.rotary_type)
        except ValueError:
            type_idx = 0
        self.rotary_type_row.set_selected(type_idx)
        self._update_type_dependent_rows(module)

        self.roller_diameter_row.set_value(module.roller_diameter)
        self.reverse_axis_row.set_active(module.reverse_axis)
        self.axis_position_x_row.set_value(float(module.axis_position[0]))
        self.axis_position_y_row.set_value(float(module.axis_position[1]))
        self.axis_position_z_row.set_value(float(module.axis_position[2]))
        self._update_model_subtitle(module)
        t = module.transform
        self.x_row.set_value(float(t[0, 3]))
        self.y_row.set_value(float(t[1, 3]))
        self.z_row.set_value(float(t[2, 3]))
        rx, ry, rz = module.get_rotation()
        self.rx_row.set_value(rx)
        self.ry_row.set_value(ry)
        self.rz_row.set_value(rz)
        self.scale_row.set_value(module.get_scale())

        self._is_updating = False

    def _update_model_subtitle(self, module: RotaryModule):
        if module.model_id:
            model_mgr = get_context().model_mgr
            model = Model(name="", path=Path(module.model_id))
            resolved = model_mgr.resolve(model)
            if resolved:
                self.model_row.set_subtitle(resolved.stem)
                return
        self.model_row.set_subtitle(_("None"))

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

    def _get_valid_axes_for_mode(self, mode: RotaryMode) -> List[Axis]:
        if mode == RotaryMode.TRUE_4TH_AXIS:
            return [Axis.A, Axis.B, Axis.C, Axis.U]
        return [Axis.Y, Axis.Z]

    def _update_axis_dropdown(self, module: RotaryModule):
        axes = self._get_valid_axes_for_mode(module.mode)
        self._valid_axes = axes
        store = Gtk.StringList()
        for a in axes:
            store.append(a.name or "")
        self.module_axis_row.set_model(store)
        try:
            selected = axes.index(module.axis)
        except ValueError:
            selected = 0
            if axes:
                module.set_axis(axes[0])
        self.module_axis_row.set_selected(selected)

    def _update_mode_dependent_rows(self, module: RotaryModule):
        is_replacement = module.mode == RotaryMode.AXIS_REPLACEMENT
        self.mu_per_rotation_row.set_visible(is_replacement)
        self._update_axis_dropdown(module)

    def _update_type_dependent_rows(self, module: RotaryModule):
        is_roller = module.rotary_type == RotaryType.ROLLERS
        self.roller_diameter_row.set_visible(is_roller)

    def _on_mode_changed(self, row, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if not module:
            return
        selected = row.get_selected()
        if selected < len(self._mode_values):
            module.set_mode(self._mode_values[selected])
            self._update_mode_dependent_rows(module)

    def _on_rotary_type_changed(self, row, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if not module:
            return
        selected = row.get_selected()
        if selected < len(self._rotary_type_values):
            module.set_rotary_type(self._rotary_type_values[selected])
            self._update_type_dependent_rows(module)

    def _on_mm_per_rotation_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_mm_per_rotation(get_spinrow_float(spinrow))

    def _on_default_diameter_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_default_diameter(get_spinrow_float(spinrow))

    def _on_max_workpiece_length_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_max_workpiece_length(get_spinrow_float(spinrow))

    def _on_roller_diameter_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_roller_diameter(get_spinrow_float(spinrow))

    def _on_reverse_axis_changed(self, switchrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_reverse_axis(switchrow.get_active())

    def _on_axis_position_changed(self, spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_axis_position(
                self.axis_position_x_row.get_value(),
                self.axis_position_y_row.get_value(),
                self.axis_position_z_row.get_value(),
            )

    def _on_model_activated(self, row):
        module = self._get_selected_module()
        if not module:
            return

        model_mgr = get_context().model_mgr
        categories = model_mgr.get_categories()
        category = None
        for cat in categories:
            if cat.id == "rotary":
                category = cat
                break
        if category is None and categories:
            category = categories[0]

        root = self.get_root()
        dialog = ModelSelectionDialog(
            category=category,
            current_model_id=module.model_id,
            transient_for=cast(Gtk.Window, root) if root else None,
        )

        def on_response(d, response_id):
            if response_id != "select":
                d.destroy()
                return
            selected_id = d.get_selected_model_id()
            if selected_id != module.model_id:
                module.set_model_id(selected_id)
                if selected_id is not None:
                    self._apply_model_scale(module, selected_id)
            self._update_model_subtitle(module)
            self.module_list_editor._rebuild()
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _apply_model_scale(self, module, model_id):
        resolved = get_context().model_mgr.resolve(
            Model(name="", path=Path(model_id))
        )
        if resolved is None:
            return
        extent = get_model_extent(resolved)
        if extent and extent > 1e-6:
            module.set_scale(module.default_diameter / extent)

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

    def _on_rotation_changed(self, _spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if not module:
            return
        rx = get_spinrow_float(self.rx_row)
        ry = get_spinrow_float(self.ry_row)
        rz = get_spinrow_float(self.rz_row)
        module.set_rotation(rx, ry, rz)

    def _on_scale_changed(self, _spinrow, _param):
        if self._is_updating:
            return
        module = self._get_selected_module()
        if module:
            module.set_scale(get_spinrow_float(self.scale_row))

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
