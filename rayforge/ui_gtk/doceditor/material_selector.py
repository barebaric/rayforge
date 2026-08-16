"""A dialog for selecting a material from available libraries."""

import logging
from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw, Gdk, GLib, Gtk

from ...context import get_context
from ...core.material import Material
from ...core.material_library import MaterialLibrary
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.texture_loader import (
    create_material_swatch,
    create_material_swatch_texture,
)

logger = logging.getLogger(__name__)


css = """
.material-selector-list {
    background: none;
}

/* Square footprint shared with the colour picker button: 45x45
   (the colour button's natural width) with the swatch inset 5px.
   Both buttons centre vertically so the row cannot stretch them. */
.material-row-button {
    min-width: 45px;
    min-height: 45px;
    padding: 0;
}
"""


def _draw_swatch_fill(
    area: Gtk.DrawingArea,
    cr,
    width: int,
    height: int,
    texture: Gdk.Texture,
):
    """Paint *texture* stretched over the whole draw area."""
    pixbuf = Gdk.pixbuf_get_from_texture(texture)
    if pixbuf is None or width <= 0 or height <= 0:
        return
    cr.save()
    cr.scale(width / pixbuf.get_width(), height / pixbuf.get_height())
    Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
    cr.paint()
    cr.restore()


class MaterialSelectorRow(Adw.ActionRow):
    """A widget representing a single Material in the selector ListBox."""

    def __init__(self, material: Material):
        super().__init__(title=material.name, activatable=True)
        self.material = material

        self.swatch = create_material_swatch(material)
        self.add_prefix(self.swatch)


class MaterialRow(Adw.ActionRow):
    """A preferences row selecting a material via a swatch button.

    Mirrors the colour selector rows: the suffix holds a clickable
    preview that opens :class:`MaterialSelectorDialog`.  The subtitle
    tracks the selection — the library-qualified material name, or
    *empty_subtitle* when nothing is selected — so no prefix widget
    is needed and the row stays aligned either way.
    """

    def __init__(
        self,
        title: str = "",
        empty_subtitle: str | None = None,
        on_select: Callable[[str], None] | None = None,
    ):
        super().__init__(title=title)
        self._empty_subtitle = empty_subtitle
        self._on_select = on_select
        self.material: Material | None = None

        apply_css(css)
        self.button = Gtk.Button(valign=Gtk.Align.CENTER)
        self.button.add_css_class("material-row-button")
        self.button.set_tooltip_text(_("Select material"))
        self.button.connect("clicked", self._on_button_clicked)
        self.add_suffix(self.button)

        self.set_material(None)

    def set_material(self, material: Material | None):
        """Show *material*'s swatch and name (None = nothing selected)."""
        self.material = material
        if material is None:
            icon = get_icon("image-x-generic-symbolic")
            icon.set_pixel_size(16)
            self.button.set_child(icon)
            self.set_subtitle(self._empty_subtitle or _("None"))
            return
        # Zero-natural-size draw area so the button's natural size
        # comes from its CSS minimum alone (like ColorDialogButton);
        # the swatch fills the chrome inset by the button margins,
        # mirroring the internal GtkColorSwatch.
        area = Gtk.DrawingArea()
        area.set_margin_top(5)
        area.set_margin_bottom(5)
        area.set_margin_start(5)
        area.set_margin_end(5)
        texture = create_material_swatch_texture(material, size=32)
        area.set_draw_func(_draw_swatch_fill, texture)
        self.button.set_child(area)
        self.set_subtitle(self._material_label(material))

    @staticmethod
    def _material_label(material: Material) -> str:
        """The library-qualified display name of *material*."""
        material_mgr = get_context().material_mgr
        for library in material_mgr.get_libraries():
            if library.get_material(material.uid):
                return f"{library.display_name}: {material.name}"
        return material.name

    def _on_button_clicked(self, button: Gtk.Button):
        root = self.get_root()
        if not isinstance(root, Gtk.Window):
            return
        dialog = MaterialSelectorDialog(
            parent=root, on_select_callback=self._on_material_selected
        )
        dialog.present()

    def _on_material_selected(self, material_uid: str | None):
        if material_uid is None:
            return
        if self._on_select is not None:
            self._on_select(material_uid)
        # The consumer may not refresh the row itself (e.g. the layer
        # dialog has no update wiring), so resolve and display the
        # selection here.
        material = get_context().material_mgr.get_material_or_none(
            material_uid
        )
        if material is not None:
            self.set_material(material)


class MaterialSelectorDialog(Adw.MessageDialog):
    """A dialog for selecting a material."""

    def __init__(self, parent: Gtk.Window, on_select_callback):
        super().__init__(transient_for=parent)
        self.on_select_callback = on_select_callback
        self._current_library: MaterialLibrary | None = None
        self._all_materials: list[Material] = []
        self.libraries: list[MaterialLibrary] = []

        self.set_heading(_("Select Material"))
        self.set_body(_("Choose a material from the available libraries."))

        # This is the proper, targeted CSS for the ListBox.
        # It manually creates the grouped, rounded-corner appearance.
        apply_css(css)

        # Main content area
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_top(12)
        self.set_extra_child(content_box)

        # Library dropdown
        self.library_dropdown = Gtk.DropDown()
        self.library_dropdown.connect(
            "notify::selected-item", self._on_library_changed
        )
        content_box.append(self.library_dropdown)

        # Search entry
        self.search_entry = Gtk.SearchEntry()
        self.search_entry.connect("search-changed", self._on_search_changed)
        content_box.append(self.search_entry)

        # Scrolled window for the list
        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=300,
            vexpand=True,
        )
        scrolled_window.add_css_class("card")
        content_box.append(scrolled_window)

        # Material list
        self.material_list = Gtk.ListBox()
        self.material_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.material_list.add_css_class("material-selector-list")
        self.material_list.connect(
            "row-activated", self._on_material_activated
        )
        scrolled_window.set_child(self.material_list)

        # Add response button
        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")

        self.connect("map", self._on_dialog_mapped)
        self._populate_libraries()

    def _on_dialog_mapped(self, _dialog):
        """Focuses the search entry when the dialog is shown."""
        GLib.idle_add(self.search_entry.grab_focus)

    def _populate_libraries(self):
        """Populates the library dropdown."""
        material_mgr = get_context().material_mgr
        model = Gtk.StringList()
        self.libraries = sorted(
            material_mgr.get_libraries(), key=lambda lib: lib.display_name
        )
        for lib in self.libraries:
            model.append(lib.display_name)

        self.library_dropdown.set_model(model)
        if self.libraries:
            self.library_dropdown.set_selected(0)

    def _on_library_changed(self, dropdown, _):
        """Handles library selection change."""
        selected_index = dropdown.get_selected()
        if selected_index < 0 or selected_index >= len(self.libraries):
            self._current_library = None
        else:
            self._current_library = self.libraries[selected_index]

        if self._current_library:
            self._all_materials = self._current_library.get_all_materials()
        else:
            self._all_materials = []
        self._filter_and_populate_materials()

    def _on_search_changed(self, entry: Gtk.SearchEntry):
        """Handles search text changes."""
        self._filter_and_populate_materials()

    def _filter_and_populate_materials(self):
        """Filters and populates the material list based on search."""
        search_text = self.search_entry.get_text().lower()

        while child := self.material_list.get_row_at_index(0):
            self.material_list.remove(child)

        for material in self._all_materials:
            if search_text in material.name.lower():
                row = MaterialSelectorRow(material)
                self.material_list.append(row)

    def _on_material_activated(
        self, listbox: Gtk.ListBox, row: MaterialSelectorRow
    ):
        """Handles when a material is selected."""
        if isinstance(row, MaterialSelectorRow):
            selected_material = row.material
            self.on_select_callback(selected_material.uid)
            self.close()
