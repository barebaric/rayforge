"""Material list UI components for Rayforge."""

import logging
import shutil
import uuid
from gettext import gettext as _
from pathlib import Path
from typing import cast

from blinker import Signal
from gi.repository import Adw, Gtk

from ...context import get_context
from ...core.material import (
    SUPPORTED_TEXTURE_SUFFIXES,
    Material,
    MaterialAppearance,
)
from ...core.material_library import MaterialLibrary
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.texture_loader import create_material_swatch
from .add_material_dialog import AddMaterialDialog

logger = logging.getLogger(__name__)


class MaterialRow(Gtk.Box):
    """A widget representing a single Material in a ListBox."""

    def __init__(
        self,
        material: Material,
        library: MaterialLibrary,
        on_delete_callback,
        on_edit_callback,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.material = material
        self.library = library
        self.on_delete_callback = on_delete_callback
        self.on_edit_callback = on_edit_callback
        self._setup_ui()

    def _setup_ui(self):
        """Builds the user interface for the row."""
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        self.prepend(create_material_swatch(self.material))

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        title_label = Gtk.Label(
            label=self.material.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        labels_box.append(title_label)

        subtitle_label = Gtk.Label(
            label=self.material.category,
            halign=Gtk.Align.START,
            xalign=0,
        )
        subtitle_label.add_css_class("dim-label")
        labels_box.append(subtitle_label)

        if not self.library.read_only:
            # Suffix area for buttons
            suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
            self.append(suffix_box)

            edit_button = Gtk.Button(child=get_icon("edit-symbolic"))
            edit_button.add_css_class("flat")
            edit_button.connect("clicked", self._on_edit_clicked)
            suffix_box.append(edit_button)

            delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
            delete_button.add_css_class("flat")
            delete_button.connect("clicked", self._on_delete_clicked)
            suffix_box.append(delete_button)

    def _on_delete_clicked(self, button: Gtk.Button):
        """Handle the delete button being clicked."""
        self.on_delete_callback(self.material)

    def _on_edit_clicked(self, button: Gtk.Button):
        """Handle the edit button being clicked."""
        self.on_edit_callback(self.material)


class MaterialListWidget(PreferencesGroupWithButton):
    """
    An Adwaita widget for displaying materials from a selected library.
    """

    def __init__(self, **kwargs):
        # This list correctly uses the default SelectionMode.NONE
        super().__init__(
            button_label=_("Add New Material"),
            empty_placeholder=_("No materials in selected library."),
            **kwargs,
        )
        self.material_added = Signal()
        self.material_deleted = Signal()
        self._setup_ui()
        self._current_library: MaterialLibrary | None = None

    def _setup_ui(self):
        """Configures the widget's list box."""
        self.list_box.set_show_separators(True)

    def set_library(self, library: MaterialLibrary | None):
        """Set the current library and update the materials list."""
        logger.debug(
            f"MaterialListEditor: Setting library to "
            f"'{library.library_id if library is not None else 'None'}'"
        )
        self._current_library = library
        self.add_button.set_sensitive(
            library is not None and not library.read_only
        )
        self._populate_materials()

    def _populate_materials(self):
        """Populate the list with materials from the current library."""
        if self._current_library is None:
            self.set_items([])
            return

        materials = sorted(
            self._current_library.get_all_materials(), key=lambda m: m.name
        )
        self.set_items(materials)

    def create_row_widget(self, item: Material) -> Gtk.Widget:
        """Creates a MaterialRow for the given material."""
        assert self._current_library is not None
        return MaterialRow(
            item,
            self._current_library,
            self._on_delete_material,
            self._on_edit_material,
        )

    def _on_delete_material(self, material: Material):
        """Handle material deletion with confirmation."""
        if self._current_library is None:
            return

        # Reject deletion if the material is still in use
        root = self.get_root()
        recipe_mgr = get_context().recipe_mgr
        if recipe_mgr.is_material_in_use(material.uid):
            err_dialog = Adw.MessageDialog(
                transient_for=cast(Gtk.Window, root) if root else None,
                heading=_("Cannot Delete Material"),
                body=_(
                    "This material is currently used by one or more recipes. "
                    "Please remove the recipes that use this material before "
                    "deleting it."
                ),
            )
            err_dialog.add_response("ok", _("OK"))
            err_dialog.present()
            return  # Stop the deletion process

        # Ask for confirmation
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Delete '{name}'?").format(name=material.name),
            body=_(
                "The material will be permanently removed from the library. "
                "This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")

        def on_response(d, response_id):
            if response_id == "delete":
                if (
                    self._current_library is not None
                    and self._current_library.remove_material(material.uid)
                ):
                    self._populate_materials()
                    self.material_deleted.send(
                        self, library=self._current_library
                    )
                else:
                    logger.error(f"Failed to remove material '{material.uid}'")
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_edit_material(self, material: Material):
        """Handle material editing."""
        if self._current_library is None:
            return

        root = self.get_root()
        dialog = AddMaterialDialog(
            material=material,
            transient_for=cast(Gtk.Window, root) if root else None,
        )

        def on_response(d, response_id):
            if response_id in ("add", "save"):
                data = d.get_material_data()
                if data["name"] and self._current_library is not None:
                    self._update_material(
                        data, material, self._current_library
                    )
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _update_material(
        self, data: dict, material: Material, library: MaterialLibrary
    ):
        """Update an existing material in the library."""
        # Update material properties
        material.name = data["name"]
        material.category = data["category"]
        material.appearance.color = data["color"]
        material.appearance.tintable = bool(data.get("tintable", False))
        material.appearance.roughness = float(data.get("roughness", 0.8))
        material.appearance.metallic = float(data.get("metallic", 0.0))
        material.appearance.texture_size_mm = float(
            data.get("texture_size_mm", 300.0)
        )

        texture_source = data.get("texture")
        if texture_source is not None:
            self._install_material_texture(material, texture_source)
        else:
            material.appearance.texture = None

        # Save the updated material
        if material.file_path:
            try:
                material.save_to_file(material.file_path)
                self._populate_materials()
                logger.info(
                    f"Updated material '{data['name']}' in library "
                    f"'{library.library_id}'"
                )
                self.material_added.send(self, library=library)
            except (OSError, ValueError) as e:
                logger.error(f"Failed to update material: {e}")
                root = self.get_root()
                err_dialog = Adw.MessageDialog(
                    transient_for=cast(Gtk.Window, root) if root else None,
                    heading=_("Error"),
                    body=_("Failed to update material."),
                )
                err_dialog.add_response("ok", _("OK"))
                err_dialog.present()

    def _on_add_clicked(self, button: Gtk.Button):
        """Handle add material button click."""
        logger.debug("MaterialListEditor: Add material button clicked")
        if self._current_library is None:
            logger.error(
                "MaterialListEditor: _on_add_clicked failed because "
                "_current_library is None. The dialog will not be shown."
            )
            return

        root = self.get_root()
        dialog = AddMaterialDialog(
            transient_for=cast(Gtk.Window, root) if root else None
        )

        def on_response(d, response_id):
            if response_id == "add":
                data = d.get_material_data()
                if data["name"] and self._current_library is not None:
                    self._add_material(data, self._current_library)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _add_material(self, data: dict, library: MaterialLibrary):
        """Add a new material to the current library."""
        material = Material(
            uid=str(uuid.uuid4()),
            name=data["name"],
            description="",
            category=data["category"],
            appearance=MaterialAppearance(
                color=data["color"],
                tintable=bool(data.get("tintable", False)),
                roughness=float(data.get("roughness", 0.8)),
                metallic=float(data.get("metallic", 0.0)),
                texture_size_mm=float(data.get("texture_size_mm", 300.0)),
            ),
        )

        if library.add_material(material):
            texture_source = data.get("texture")
            if texture_source is not None:
                self._install_material_texture(material, texture_source)
            self._populate_materials()
            logger.info(
                f"Added material '{data['name']}' to library "
                f"'{library.library_id}'"
            )
            self.material_added.send(self, library=library)
        else:
            root = self.get_root()
            err_dialog = Adw.MessageDialog(
                transient_for=cast(Gtk.Window, root) if root else None,
                heading=_("Error"),
                body=_("Failed to add material to library."),
            )
            err_dialog.add_response("ok", _("OK"))
            err_dialog.present()

    def _install_material_texture(
        self, material: Material, texture_source: Path
    ):
        """
        Copy a chosen texture file into the material's library.

        The texture is stored next to the material YAML as
        "<uid>.<suffix>" (WebP or PNG) and referenced by that relative
        name, so the material stays valid if the library is moved.
        """
        if material.file_path is None:
            logger.error(
                f"Cannot install texture for '{material.uid}': "
                "material has no file path"
            )
            return
        if texture_source.suffix.lower() not in SUPPORTED_TEXTURE_SUFFIXES:
            logger.error(
                f"Ignoring texture '{texture_source}' for "
                f"'{material.uid}': only WebP and PNG are supported"
            )
            return

        dest = (
            material.file_path.parent
            / f"{material.uid}{texture_source.suffix.lower()}"
        )
        try:
            shutil.copy2(texture_source, dest)
        except OSError as e:
            logger.error(f"Failed to copy texture '{texture_source}': {e}")
            return
        material.appearance.texture = dest.name
        try:
            material.save_to_file(material.file_path)
        except OSError as e:
            logger.error(f"Failed to save material after texture copy: {e}")
