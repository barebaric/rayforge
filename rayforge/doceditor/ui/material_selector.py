"""A dialog for selecting a material from available libraries."""

import logging
from typing import Optional, List
from gi.repository import Gtk, Adw, GObject
from ...config import material_mgr
from ...core.material import Material
from ...core.material_library import MaterialLibrary

logger = logging.getLogger(__name__)


class MaterialSelectorRow(Gtk.Box):
    """A widget representing a single Material in the selector ListBox."""

    def __init__(self, material: Material):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.material = material
        self._setup_ui()

    def _setup_ui(self):
        """Builds the user interface for the row."""
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        # Color indicator
        color_box = Gtk.Box()
        color_box.set_valign(Gtk.Align.CENTER)
        color_box.set_size_request(24, 24)
        color_box.add_css_class("material-color-selector")
        color_provider = Gtk.CssProvider()
        color_data = (
            ".material-color-selector {{ background-color: {}; }}"
        ).format(self.material.get_display_color())
        color_provider.load_from_data(color_data.encode())
        color_box.get_style_context().add_provider(
            color_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.append(color_box)

        # Labels
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


class MaterialListBoxRow(Gtk.ListBoxRow):
    """Custom ListBoxRow to hold a reference to a Material."""

    material: Material

    def __init__(self, material: Material):
        super().__init__()
        self.material = material
        self.set_child(MaterialSelectorRow(material))


# GObject wrapper for MaterialLibrary for use in Gio.ListStore
class LibraryListItem(GObject.Object):
    __gtype_name__ = "LibraryListItem"

    def __init__(self, library: MaterialLibrary):
        super().__init__()
        self.library = library


class MaterialSelectorDialog(Adw.Window):
    """A dialog for selecting a material."""

    def __init__(self, parent: Gtk.Window, on_select_callback):
        super().__init__(transient_for=parent)
        self.on_select_callback = on_select_callback
        self._current_library: Optional[MaterialLibrary] = None
        self._all_materials: List[Material] = []
        self.libraries: List[MaterialLibrary] = []

        self._setup_ui()
        self._populate_libraries()

    def _setup_ui(self):
        self.set_title(_("Select Material"))
        self.set_default_size(350, 500)
        self.set_modal(True)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header = Adw.HeaderBar()
        main_box.append(header)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(12)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        main_box.append(content_box)

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
        content_box.append(scrolled_window)

        # Material list
        self.material_list = Gtk.ListBox()
        self.material_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.material_list.connect(
            "row-activated", self._on_material_activated
        )
        scrolled_window.set_child(self.material_list)

    def _populate_libraries(self):
        """Populates the library dropdown."""
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
                row = MaterialListBoxRow(material)
                self.material_list.append(row)

    def _on_material_activated(self, listbox, row):
        """Handles when a material is selected."""
        if isinstance(row, MaterialListBoxRow):
            selected_material = row.material
            self.on_select_callback(selected_material.uid)
            self.close()
