"""A dialog for adding a new material."""

import logging
from gettext import gettext as _
from pathlib import Path
from typing import Any

from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from ...core.material import SUPPORTED_TEXTURE_SUFFIXES, Material
from ..icons import get_icon
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.slider import create_slider_row

logger = logging.getLogger(__name__)


def _make_pbr_slider_row(
    title: str, subtitle: str, value: float
) -> tuple[Adw.ActionRow, Gtk.Scale]:
    """Create an ActionRow with a 0..1 slider for a PBR parameter."""
    adjustment = Gtk.Adjustment(
        value=value,
        lower=0.0,
        upper=1.0,
        step_increment=0.05,
        page_increment=0.1,
    )
    return create_slider_row(title, adjustment, subtitle=subtitle, digits=2)


class AddMaterialDialog(Adw.MessageDialog):
    """A dialog for creating a new material."""

    def __init__(self, material: Material | None = None, **kwargs):
        super().__init__(**kwargs)

        self.material = material
        self.is_edit_mode = material is not None

        if self.is_edit_mode:
            self.set_heading(_("Edit Material"))
            self.set_body(_("Update the material details:"))
            self.add_response("cancel", _("Cancel"))
            self.add_response("save", _("Save"))
            self.set_response_appearance(
                "save", Adw.ResponseAppearance.SUGGESTED
            )
            self.set_default_response("save")
        else:
            self.set_heading(_("Add New Material"))
            self.set_body(_("Enter the details for the new material:"))
            self.add_response("cancel", _("Cancel"))
            self.add_response("add", _("Add"))
            self.set_response_appearance(
                "add", Adw.ResponseAppearance.SUGGESTED
            )
            self.set_default_response("add")

        self.name_entry = Adw.EntryRow(title=_("Name"))
        self.category_entry = Adw.EntryRow(title=_("Category"))

        self._texture_path: Path | None = None
        self.texture_row = Adw.ActionRow(title=_("Texture"))
        self.texture_row.set_subtitle(_("None"))
        self._choose_texture_button = Gtk.Button(label=_("Choose..."))
        self._choose_texture_button.set_valign(Gtk.Align.CENTER)
        self._choose_texture_button.connect("clicked", self._on_choose_texture)
        self.texture_row.add_suffix(self._choose_texture_button)
        self._clear_texture_button = Gtk.Button(
            child=get_icon("clear-symbolic")
        )
        self._clear_texture_button.add_css_class("flat")
        self._clear_texture_button.set_valign(Gtk.Align.CENTER)
        self._clear_texture_button.connect("clicked", self._on_clear_texture)
        self._clear_texture_button.set_visible(False)
        self.texture_row.add_suffix(self._clear_texture_button)

        color_dialog = Gtk.ColorDialog()
        color_dialog.set_with_alpha(False)
        self.color_button = Gtk.ColorDialogButton(dialog=color_dialog)
        self.color_button.set_size_request(32, 32)
        self.color_button.connect("notify::rgba", self._on_color_set)
        self._color: str | None = "#f0f0f0"
        self._clear_color_button = Gtk.Button(child=get_icon("clear-symbolic"))
        self._clear_color_button.add_css_class("flat")
        self._clear_color_button.set_valign(Gtk.Align.CENTER)
        self._clear_color_button.set_tooltip_text(
            _("Unset the tint color (texture is shown as-is)")
        )
        self._clear_color_button.connect("clicked", self._on_clear_color)
        self._clear_color_button.set_visible(False)
        self.color_row = Adw.ActionRow(
            title=_("Color"), activatable_widget=self.color_button
        )
        self.color_row.add_suffix(self.color_button)
        self.color_row.add_suffix(self._clear_color_button)

        self.tintable_row = Adw.SwitchRow(
            title=_("Tintable"),
            subtitle=_("Allow tinting the texture with a color"),
        )
        self.tintable_row.connect("notify::active", self._on_tintable_changed)

        self.texture_scale_row = LengthSpinRow(
            _("Texture Scale"),
            _("Size one texture tile covers on the material"),
            lower=1.0,
            upper=2000.0,
            value_in_base=300.0,
        )
        self.texture_scale_row.set_sensitive(False)

        self.roughness_row, self.roughness_scale = _make_pbr_slider_row(
            _("Roughness"),
            _("How rough or polished the surface appears"),
            0.8,
        )
        self.metallic_row, self.metallic_scale = _make_pbr_slider_row(
            _("Metallic"),
            _("Whether the surface reflects light like a metal"),
            0.0,
        )

        # Use a preferences group for a clean layout
        group = Adw.PreferencesGroup()
        group.add(self.name_entry)
        group.add(self.category_entry)
        group.add(self.texture_row)
        group.add(self.texture_scale_row)
        group.add(self.color_row)
        group.add(self.tintable_row)
        group.add(self.roughness_row)
        group.add(self.metallic_row)

        self.set_extra_child(group)
        group.set_margin_start(24)
        group.set_margin_end(24)
        group.set_margin_bottom(24)

        # If editing, populate the fields with existing data
        if self.is_edit_mode:
            self._populate_fields()

        # Set initial focus on the name entry
        self.name_entry.grab_focus()

        # Widen the dialog: libadwaita dialogs are presented at their
        # minimum preferred width, so add the extra width to that.
        self.set_default_size(600, -1)

        # Connect Enter key handler to entries
        # Adw.EntryRow has an internal entry widget we need to access
        self.name_entry.connect("entry-activated", self._on_enter_key)
        self.category_entry.connect("entry-activated", self._on_enter_key)

    def _on_enter_key(self, widget):
        """Handle Enter key pressed in entry fields."""
        # Get the default response and emit the response signal
        default_response = self.get_default_response()
        if default_response:
            self.response(default_response)

    def get_name(self) -> str:
        """Get the text from the name entry."""
        return self.name_entry.get_text()

    def get_category(self) -> str:
        """Get the text from the category entry."""
        return self.category_entry.get_text()

    def _color_button_hex(self) -> str:
        """Read the current color button value as a hex string."""
        rgba = self.color_button.get_rgba()
        r = int(rgba.red * 255)
        g = int(rgba.green * 255)
        b = int(rgba.blue * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def get_color_hex(self) -> str | None:
        """Get the color as a hex string, or None when unset (not tinted)."""
        return self._color

    def _on_color_set(self, button: Gtk.ColorDialogButton, pspec=None):
        """Track color changes from the color picker."""
        self._color = self._color_button_hex()
        self._update_color_row()

    def _on_clear_color(self, button: Gtk.Button):
        """Unset the color so the texture is shown without a tint."""
        self._color = None
        self._update_color_row()

    def _update_color_row(self):
        """Refresh the color row subtitle and clear button visibility."""
        if self._color is None:
            self.color_row.set_subtitle(_("Not tinted"))
            self._clear_color_button.set_visible(False)
        else:
            self.color_row.set_subtitle(self._color)
            self._clear_color_button.set_visible(True)

    def _on_tintable_changed(self, row: Adw.SwitchRow, pspec):
        """Keep row sensitivity in sync when the tintable switch flips."""
        self._update_sensitivity()

    def _update_sensitivity(self):
        """
        The color row stays active while a texture is selected only when
        the material is tintable; the texture scale is active whenever a
        texture is chosen.
        """
        has_texture = self._texture_path is not None
        color_active = (not has_texture) or self.tintable_row.get_active()
        self.color_row.set_sensitive(color_active)
        self.texture_scale_row.set_sensitive(has_texture)

    def _on_choose_texture(self, button: Gtk.Button):
        """Open a file chooser for a texture image (WebP or PNG)."""
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Choose Texture Image"))
        file_filter = Gtk.FileFilter()
        file_filter.set_name(_("Texture images"))
        file_filter.add_suffix("webp")
        file_filter.add_suffix("png")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(file_filter)
        dialog.set_filters(filters)
        dialog.open(self, None, self._on_file_dialog_finished)

    def _on_file_dialog_finished(self, dialog: Gtk.FileDialog, result):
        """Handle the file chooser result."""
        try:
            file = dialog.open_finish(result)
        except GLib.Error:
            return
        if file is None:
            return

        raw_path = file.get_path()
        if raw_path is None:
            return
        path = Path(raw_path)
        if path.suffix.lower() not in SUPPORTED_TEXTURE_SUFFIXES:
            self._show_error(
                _("Only WebP and PNG texture images are supported.")
            )
            return

        self._texture_path = path
        self.texture_row.set_subtitle(path.name)
        self._clear_texture_button.set_visible(True)
        self._update_sensitivity()

    def _on_clear_texture(self, button: Gtk.Button):
        """Clear the chosen texture."""
        self._texture_path = None
        self.texture_row.set_subtitle(_("None"))
        self._clear_texture_button.set_visible(False)
        self._update_sensitivity()

    def _show_error(self, message: str):
        """Show an error dialog."""
        err_dialog = Adw.MessageDialog(
            transient_for=self,
            heading=_("Error"),
            body=message,
        )
        err_dialog.add_response("ok", _("OK"))
        err_dialog.present()

    def _populate_fields(self):
        """Populate the dialog fields with existing material data."""
        if not self.material:
            return

        self.name_entry.set_text(self.material.name)
        self.category_entry.set_text(self.material.category)

        # Set the color (may be None when the material is tintable and
        # the tint was unset).
        self.tintable_row.set_active(bool(self.material.appearance.tintable))
        self._color = self.material.appearance.color
        if self._color is not None and self._color.startswith("#"):
            # Try using GTK's built-in color parsing
            rgba = Gdk.RGBA()
            if rgba.parse(self._color):
                self.color_button.set_rgba(rgba)
        self._update_color_row()

        self.roughness_scale.set_value(self.material.appearance.roughness)
        self.metallic_scale.set_value(self.material.appearance.metallic)

        texture_path = self.material.get_texture_path()
        if texture_path is not None:
            self._texture_path = texture_path
            self.texture_row.set_subtitle(texture_path.name)
            self._clear_texture_button.set_visible(True)

        self.texture_scale_row.set_value_in_base_units(
            self.material.appearance.texture_size_mm
        )
        self._update_sensitivity()

    def get_material_data(self) -> dict[str, Any]:
        """Returns a dictionary with the entered material data."""
        return {
            "name": self.get_name().strip(),
            "category": self.get_category().strip() or _("Custom"),
            "color": self.get_color_hex(),
            "tintable": self.tintable_row.get_active(),
            "roughness": self.roughness_scale.get_value(),
            "metallic": self.metallic_scale.get_value(),
            "texture_size_mm": (
                self.texture_scale_row.get_value_in_base_units()
            ),
            "texture": self._texture_path,
        }
