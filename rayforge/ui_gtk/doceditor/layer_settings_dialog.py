from gettext import gettext as _
from typing import Optional

from gi.repository import Adw, Gdk, Gtk

from ...context import get_context
from ...core.layer import Layer
from ..icons import get_icon
from ..machine.wcs_dialog import WcsDialog
from ..shared.adwfix import get_spinrow_float
from ..shared.patched_dialog_window import PatchedDialogWindow


class LayerSettingsDialog(PatchedDialogWindow):
    """Dialog for configuring layer-level settings including rotary."""

    def __init__(self, layer: Layer, transient_for: Gtk.Window, **kwargs):
        super().__init__(transient_for=transient_for, **kwargs)
        self.layer = layer
        self._is_initializing = True

        self.set_title(_("{name} - Settings").format(name=layer.name))
        self.set_default_size(600, -1)
        self.set_modal(False)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header = Adw.HeaderBar()
        main_box.append(header)

        close_button = Gtk.Button(label=_("Close"))
        close_button.add_css_class("suggested-action")
        close_button.connect("clicked", lambda w: self.close())
        header.pack_end(close_button)

        content = Adw.PreferencesPage()
        main_box.append(content)

        general_group = Adw.PreferencesGroup(
            title=_("General"),
            description=_(
                "Basic layer settings such as appearance and "
                "coordinate system."
            ),
        )
        content.add(general_group)

        color_dialog = Gtk.ColorDialog()
        color_dialog.set_with_alpha(False)
        self.color_button = Gtk.ColorDialogButton(dialog=color_dialog)
        rgba = Gdk.RGBA()
        rgba.parse(layer.color)
        self.color_button.set_rgba(rgba)
        self.color_button.connect("notify::rgba", self._on_color_changed)

        color_row = Adw.ActionRow(
            title=_("Layer Color"),
            subtitle=_("Color used for operations in this layer"),
        )
        color_row.add_suffix(self.color_button)
        general_group.add(color_row)

        self._populate_wcs_store()
        self.wcs_row = Adw.ComboRow(
            title=_("Coordinate System"),
            subtitle=_(
                "The work coordinate system origin to use for this layer. "
                "By default, use the WCS selected in the main "
                "window"
            ),
            model=self._wcs_store,
        )
        self.edit_offsets_btn = Gtk.Button(child=get_icon("edit-symbolic"))
        self.edit_offsets_btn.set_tooltip_text(_("Edit Offsets Manually"))
        self.edit_offsets_btn.add_css_class("flat")
        self.edit_offsets_btn.set_valign(Gtk.Align.CENTER)
        self.edit_offsets_btn.connect("clicked", self._on_edit_offsets_clicked)
        self.wcs_row.add_suffix(self.edit_offsets_btn)

        self._select_current_wcs()
        self._update_edit_button_sensitivity()
        self.wcs_row.connect("notify::selected", self._on_wcs_changed)
        general_group.add(self.wcs_row)

        rotary_group = Adw.PreferencesGroup(
            title=_("Rotary Attachment"),
            description=_(
                "Configure rotary attachment for cylindrical objects. "
                "When enabled, Y-axis movements are converted to "
                "rotational movements in degrees."
            ),
        )
        content.add(rotary_group)

        self.rotary_enabled_row = Adw.SwitchRow()
        self.rotary_enabled_row.set_title(_("Enable Rotary Mode"))
        self.rotary_enabled_row.set_subtitle(
            _("Convert Y-axis to rotary axis")
        )
        self.rotary_enabled_row.set_active(layer.rotary_enabled)
        self.rotary_enabled_row.connect(
            "notify::active", self._on_rotary_enabled_changed
        )
        rotary_group.add(self.rotary_enabled_row)

        self._populate_module_store()
        self.module_row = Adw.ComboRow(
            title=_("Rotary Module"),
            subtitle=_("Select the rotary module for this layer"),
            model=self._module_store,
        )
        self._select_current_module()
        self.module_row.connect("notify::selected", self._on_module_changed)
        self.module_row.set_sensitive(layer.rotary_enabled)
        rotary_group.add(self.module_row)

        rotary_diameter_adjustment = Gtk.Adjustment(
            lower=1, upper=1000, step_increment=1, page_increment=10
        )
        self.rotary_diameter_row = Adw.SpinRow(
            title=_("Object Diameter"),
            subtitle=_("Diameter of the cylindrical object in machine units"),
            adjustment=rotary_diameter_adjustment,
            digits=2,
        )
        rotary_diameter_adjustment.set_value(layer.rotary_diameter)
        self.rotary_diameter_row.connect(
            "notify::value", self._on_rotary_diameter_changed
        )
        self.rotary_diameter_row.set_sensitive(layer.rotary_enabled)
        rotary_group.add(self.rotary_diameter_row)

        self._is_initializing = False

        if not self.layer.rotary_module_uid and self._module_uids:
            self.layer.set_rotary_module_uid(self._module_uids[0])

    def _populate_wcs_store(self):
        self._wcs_store = Gtk.StringList()
        self._wcs_values: list[Optional[str]] = [None]
        self._wcs_store.append(_("Default"))
        machine = get_context().machine
        if machine:
            for wcs in machine.supported_wcs:
                self._wcs_store.append(wcs)
                self._wcs_values.append(wcs)

    def _select_current_wcs(self):
        wcs = self.layer.wcs
        if wcs and wcs in self._wcs_values:
            self.wcs_row.set_selected(self._wcs_values.index(wcs))
        else:
            self.wcs_row.set_selected(0)

    def _get_selected_wcs(self) -> Optional[str]:
        idx = self.wcs_row.get_selected()
        if idx < len(self._wcs_values):
            return self._wcs_values[idx]
        return None

    def _on_wcs_changed(self, row, _param):
        if self._is_initializing:
            return
        wcs = self._get_selected_wcs()
        self.layer.set_wcs(wcs)
        self._update_edit_button_sensitivity()

    def _update_edit_button_sensitivity(self):
        wcs = self._get_selected_wcs()
        self.edit_offsets_btn.set_sensitive(wcs is not None)

    def _on_edit_offsets_clicked(self, button):
        machine = get_context().machine
        if not machine:
            return

        root = self.get_root()
        self._edit_dialog = WcsDialog(
            machine=machine,
            transient_for=root if isinstance(root, Gtk.Window) else None,
        )
        self._edit_dialog.connect("destroy", self._on_edit_dialog_destroy)
        self._edit_dialog.present()

    def _on_edit_dialog_destroy(self, *_):
        self._edit_dialog = None

    def _populate_module_store(self):
        self._module_store = Gtk.StringList()
        self._module_uids: list[str] = []
        machine = get_context().machine
        if machine:
            for module in sorted(
                machine.rotary_modules.values(), key=lambda m: m.name
            ):
                self._module_store.append(module.name)
                self._module_uids.append(module.uid)

    def _select_current_module(self):
        uid = self.layer.rotary_module_uid
        if uid and uid in self._module_uids:
            self.module_row.set_selected(self._module_uids.index(uid))
        elif self._module_uids:
            self.module_row.set_selected(0)
            self.layer.set_rotary_module_uid(self._module_uids[0])

    def _on_rotary_enabled_changed(self, row, _):
        if self._is_initializing:
            return
        enabled = row.get_active()
        self.module_row.set_sensitive(enabled)
        self.rotary_diameter_row.set_sensitive(enabled)
        self.layer.set_rotary_enabled(enabled)
        if enabled and self.layer.rotary_module_uid is None:
            machine = get_context().machine
            if machine:
                default_rm = machine.get_default_rotary_module()
                if default_rm:
                    self.layer.set_rotary_module_uid(default_rm.uid)
                    self.layer.set_rotary_diameter(default_rm.default_diameter)
                    self.rotary_diameter_row.set_value(
                        default_rm.default_diameter
                    )

    def _on_module_changed(self, row, _param):
        if self._is_initializing:
            return
        idx = row.get_selected()
        if idx < len(self._module_uids):
            uid = self._module_uids[idx]
            self.layer.set_rotary_module_uid(uid)
            machine = get_context().machine
            if machine:
                rm = machine.get_rotary_module_by_uid(uid)
                if rm:
                    self.layer.set_rotary_diameter(rm.default_diameter)
                    self.rotary_diameter_row.set_value(rm.default_diameter)

    def _on_rotary_diameter_changed(self, spinrow, _param):
        if self._is_initializing:
            return
        diameter = get_spinrow_float(spinrow)
        self.layer.set_rotary_diameter(diameter)

    def _on_color_changed(self, button, _param):
        if self._is_initializing:
            return
        rgba = button.get_rgba()
        r = int(round(rgba.red * 255))
        g = int(round(rgba.green * 255))
        b = int(round(rgba.blue * 255))
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.layer.set_color(hex_color)
