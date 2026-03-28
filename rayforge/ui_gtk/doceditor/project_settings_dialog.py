from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...core.doc import Doc
from ..shared.adwfix import get_spinrow_float
from ..shared.patched_dialog_window import PatchedDialogWindow


class ProjectSettingsDialog(PatchedDialogWindow):
    """Dialog for configuring project-level settings including rotary."""

    def __init__(self, doc: Doc, transient_for: Gtk.Window, **kwargs):
        super().__init__(transient_for=transient_for, **kwargs)
        self.doc = doc
        self._is_initializing = True

        self.set_title(_("Project Settings"))
        self.set_default_size(600, -1)
        self.set_modal(True)

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
        self.rotary_enabled_row.set_active(doc.rotary_enabled)
        self.rotary_enabled_row.connect(
            "notify::active", self._on_rotary_enabled_changed
        )
        rotary_group.add(self.rotary_enabled_row)

        rotary_diameter_adjustment = Gtk.Adjustment(
            lower=1, upper=1000, step_increment=1, page_increment=10
        )
        self.rotary_diameter_row = Adw.SpinRow(
            title=_("Object Diameter"),
            subtitle=_("Diameter of the cylindrical object in machine units"),
            adjustment=rotary_diameter_adjustment,
            digits=2,
        )
        rotary_diameter_adjustment.set_value(doc.rotary_diameter)
        self.rotary_diameter_row.connect(
            "notify::value", self._on_rotary_diameter_changed
        )
        self.rotary_diameter_row.set_sensitive(doc.rotary_enabled)
        rotary_group.add(self.rotary_diameter_row)

        self._is_initializing = False

    def _on_rotary_enabled_changed(self, row, _):
        if self._is_initializing:
            return
        enabled = row.get_active()
        self.rotary_diameter_row.set_sensitive(enabled)
        self.doc.set_rotary_enabled(enabled)

    def _on_rotary_diameter_changed(self, spinrow, _param):
        if self._is_initializing:
            return
        diameter = get_spinrow_float(spinrow)
        self.doc.set_rotary_diameter(diameter)
