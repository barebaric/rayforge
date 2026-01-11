"""A dialog for selecting a material from available libraries."""

import logging
from typing import Optional, List
from gi.repository import Gtk, Adw
from ...context import get_context
from ...core.material import Material
from ...core.material_library import MaterialLibrary
from ..shared.gtk import apply_css

logger = logging.getLogger(__name__)


css = """
.material-selector-list {
    background: none;
}
"""


class MaterialSelectorRow(Adw.ActionRow):
    """A widget representing a single Material in the selector ListBox."""

    def __init__(self, material: Material):
        super().__init__(title=material.name, activatable=True)
        self.material = material

        # Color indicator
        color_box = Gtk.Box()
        color_box.set_valign(Gtk.Align.CENTER)
        color_box.set_size_request(24, 24)
        color_box.add_css_class("material-color-selector")
        color_provider = Gtk.CssProvider()
        color_data = (
            ".material-color-selector {{ background-color: {}; }}"
        ).format(self.material.get_display_color())
        color_provider.load_from_string(color_data)
        color_box.get_style_context().add_provider(
            color_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.add_prefix(color_box)


class MaterialSelectorDialog(Adw.MessageDialog):
    """A dialog for selecting a material."""

    def __init__(self, parent: Gtk.Window, on_select_callback):
        super().__init__(transient_for=parent)
        self.on_select_callback = on_select_callback
        self._current_library: Optional[MaterialLibrary] = None
        self._all_materials: List[Material] = []
        self.libraries: List[MaterialLibrary] = []

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

        self._populate_libraries()

    def _populate_libraries(self):
        """Populates the library dropdown."""
        material_mgr = get_context().material_mgr
        model = Gtk.StringList()
        self.libraries = sorted(
            material_mgr.get_libraries(), key=lambda lib: lib.display_name
        )
        for lib in self.libraries:
            display_name = (
                _("Core Materials")
                if lib.source == "core"
                else lib.display_name
            )
            model.append(display_name)

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
