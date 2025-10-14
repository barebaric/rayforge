"""A dialog for adding a new material."""

import logging
from typing import Dict, Any, Optional
from gi.repository import Gtk, Adw, Gdk
from ...core.material import Material

logger = logging.getLogger(__name__)


class AddMaterialDialog(Adw.MessageDialog):
    """A dialog for creating a new material."""

    def __init__(self, material: Optional[Material] = None, **kwargs):
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

        self.color_button = Gtk.ColorButton(margin_bottom=0)
        self.color_button.set_size_request(32, 32)
        self.color_row = Adw.ActionRow(
            title=_("Color"), activatable_widget=self.color_button
        )
        self.color_row.add_suffix(self.color_button)

        # Use a preferences group for a clean layout
        group = Adw.PreferencesGroup()
        group.add(self.name_entry)
        group.add(self.category_entry)
        group.add(self.color_row)

        self.set_extra_child(group)

        # If editing, populate the fields with existing data
        if self.is_edit_mode:
            self._populate_fields()

        # Set initial focus on the name entry
        self.name_entry.grab_focus()

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

    def get_color_hex(self) -> str:
        """Get the color as a hex string."""
        rgba = self.color_button.get_rgba()
        r = int(rgba.red * 255)
        g = int(rgba.green * 255)
        b = int(rgba.blue * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _populate_fields(self):
        """Populate the dialog fields with existing material data."""
        if not self.material:
            return

        self.name_entry.set_text(self.material.name)
        self.category_entry.set_text(self.material.category)

        # Set the color button to the material's color
        color_hex = self.material.appearance.color
        if color_hex.startswith("#"):
            # Try using GTK's built-in color parsing
            rgba = Gdk.RGBA()
            if rgba.parse(color_hex):
                self.color_button.set_rgba(rgba)

    def get_material_data(self) -> Dict[str, Any]:
        """Returns a dictionary with the entered material data."""
        return {
            "name": self.get_name().strip(),
            "category": self.get_category().strip() or _("Custom"),
            "color": self.get_color_hex(),
        }
