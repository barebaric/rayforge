"""Material library list UI components for Rayforge."""

import logging
from typing import Optional, cast, List
from gi.repository import Gtk, Adw
from blinker import Signal
from ...context import get_context
from ...core.material_library import MaterialLibrary
from ..settings.preferences_group import PreferencesGroupWithButton
from ..icons import get_icon

logger = logging.getLogger(__name__)


class LibraryRow(Gtk.Box):
    """A widget representing a single Material Library in a ListBox."""

    def __init__(
        self, library: MaterialLibrary, on_delete_callback, on_edit_callback
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        logger.debug(
            f"LibraryRow.__init__: Creating instance for library "
            f"'{library.library_id if library is not None else 'None'}'"
        )
        self.library = library
        self.on_delete_callback = on_delete_callback
        self.on_edit_callback = on_edit_callback
        self.delete_button: Gtk.Button
        self.edit_button: Gtk.Button
        self.title_label: Gtk.Label
        self.subtitle_label: Gtk.Label
        self._setup_ui()

    def _setup_ui(self):
        """Builds the user interface for the row."""
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        label_text = (
            _("Core Materials")
            if self.library.source == "core"
            else self.library.display_name
        )
        self.title_label = Gtk.Label(
            label=label_text,
            halign=Gtk.Align.START,
            xalign=0,
        )
        labels_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(
            label=self._get_subtitle_text(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        self.subtitle_label.add_css_class("dim-label")
        labels_box.append(self.subtitle_label)

        # Add edit and delete buttons for user libraries only
        if not self.library.read_only:
            # Suffix area for buttons
            suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
            self.append(suffix_box)

            self.edit_button = Gtk.Button(
                child=get_icon("document-edit-symbolic")
            )
            self.edit_button.add_css_class("flat")
            self.edit_button.connect("clicked", self._on_edit_clicked)
            suffix_box.append(self.edit_button)

            self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
            self.delete_button.add_css_class("flat")
            self.delete_button.connect("clicked", self._on_delete_clicked)
            suffix_box.append(self.delete_button)

    def _get_subtitle_text(self) -> str:
        """Generates the subtitle text from library properties."""
        if self.library.source == "core":
            return _("Read-only core library")
        elif self.library.source == "user":
            material_count = len(self.library)
            if material_count == 1:
                return _("1 material")
            return _("{count} materials").format(count=material_count)
        else:
            return _("Plugin library ({source})").format(
                source=self.library.source
            )

    def _on_delete_clicked(self, button: Gtk.Button):
        """Handle delete button click."""
        self.on_delete_callback(self.library)

    def _on_edit_clicked(self, button: Gtk.Button):
        """Handle edit button click."""
        self.on_edit_callback(self.library)


class LibraryListWidget(PreferencesGroupWithButton):
    """
    An Adwaita widget for displaying and managing a list of material libraries.
    """

    def __init__(self, **kwargs):
        # Pass the correct selection mode to the parent constructor.
        # This is the single source of truth for selection behavior.
        super().__init__(
            button_label=_("Add New Library"),
            selection_mode=Gtk.SelectionMode.SINGLE,
            **kwargs,
        )
        self.library_selected = Signal()
        # Keep a persistent reference to the Python widget objects
        self._row_widgets: List[LibraryRow] = []
        self._setup_ui()

    def _setup_ui(self):
        """Configures the widget's list box and placeholder."""
        placeholder = Gtk.Label(
            label=_("No user libraries found."),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_show_separators(True)
        # With the widget correctly initialized, 'row-selected' is reliable.
        self.list_box.connect("row-selected", self._on_library_selected)

    def populate_and_select(self, select_name: Optional[str] = None):
        """
        Populates the list with libraries and selects a specific one.

        Args:
            select_name: The name (ID) of the library to select. If None,
                         selects the first library in the list.
        """
        material_mgr = get_context().material_mgr
        material_mgr.reload_libraries()
        libraries = sorted(
            material_mgr.get_libraries(),
            key=lambda lib: (lib.source != "core", lib.display_name),
        )
        # Clear the old references before creating new ones
        self._row_widgets.clear()
        # This now calls the corrected base class method.
        self.set_items(libraries)

        row_to_select = None
        if libraries:
            if select_name:
                i = 0
                while row := self.list_box.get_row_at_index(i):
                    child = row.get_child()
                    if (
                        isinstance(child, LibraryRow)
                        and child.library.library_id == select_name
                    ):
                        row_to_select = row
                        break
                    i += 1
            else:
                row_to_select = self.list_box.get_row_at_index(0)

        if row_to_select:
            self.list_box.select_row(row_to_select)
        elif not libraries:
            # Ensure selection is cleared if no libraries exist
            self._on_library_selected(self.list_box, None)

    def create_row_widget(self, item: MaterialLibrary) -> Gtk.Widget:
        """Creates a LibraryRow for the given library."""
        logger.debug(
            f"LibraryListEditor: Creating LibraryRow for library "
            f"'{item.library_id}' (display: '{item.display_name}')"
        )
        row_widget = LibraryRow(
            item, self._on_delete_library, self._on_edit_library
        )

        # Store a reference to prevent garbage collection
        self._row_widgets.append(row_widget)
        return row_widget

    def _on_delete_library(self, library: MaterialLibrary):
        """Handle library deletion with confirmation dialog."""
        material_mgr = get_context().material_mgr
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Delete '{name}'?").format(name=library.display_name),
            body=_(
                "The library folder and all its materials will be "
                "permanently removed. This action cannot be undone."
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
                if not material_mgr.remove_user_library(library.library_id):
                    logger.error(
                        f"Failed to remove library '{library.library_id}'"
                    )
                self.populate_and_select()
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_edit_library(self, library: MaterialLibrary):
        """Handle library editing."""
        material_mgr = get_context().material_mgr
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Edit Library"),
            body=_("Enter a new name for the library:"),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("save", _("Save"))
        dialog.set_response_appearance(
            "save", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.set_default_response("cancel")

        entry = Gtk.Entry(placeholder_text=_("Library name"))
        entry.set_text(library.display_name)
        dialog.set_extra_child(entry)

        # Connect Enter key handler
        entry.connect("activate", lambda widget: dialog.response("save"))

        def on_response(d, response_id):
            if response_id == "save":
                new_display_name = entry.get_text().strip()
                if (
                    not new_display_name
                    or new_display_name == library.display_name
                ):
                    d.destroy()
                    return

                # Update the library display name directly
                library.set_display_name(new_display_name)

                # Save the changes to disk
                if material_mgr.update_library(library.library_id):
                    self.populate_and_select(select_name=library.library_id)
                else:
                    err_dialog = Adw.MessageDialog(
                        transient_for=cast(Gtk.Window, root) if root else None,
                        heading=_("Error"),
                        body=_("Failed to rename library."),
                    )
                    err_dialog.add_response("ok", _("OK"))
                    err_dialog.present()
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()
        entry.grab_focus()

    def _on_add_clicked(self, button: Gtk.Button):
        """Handle add library button click."""
        material_mgr = get_context().material_mgr
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Add New Library"),
            body=_("Enter a name for the new library folder:"),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("add", _("Add"))
        dialog.set_response_appearance("add", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("cancel")

        entry = Gtk.Entry(placeholder_text=_("Library name"))
        dialog.set_extra_child(entry)

        # Connect Enter key handler
        entry.connect("activate", lambda widget: dialog.response("add"))

        def on_response(d, response_id):
            if response_id == "add":
                display_name = entry.get_text().strip()
                if not display_name:
                    d.destroy()
                    return

                new_lib_id = material_mgr.create_user_library(display_name)
                if new_lib_id:
                    self.populate_and_select(select_name=new_lib_id)
                else:
                    err_dialog = Adw.MessageDialog(
                        transient_for=cast(Gtk.Window, root) if root else None,
                        heading=_("Error"),
                        body=_(
                            "Failed to create library. A folder with that "
                            "name may already exist."
                        ),
                    )
                    err_dialog.add_response("ok", _("OK"))
                    err_dialog.present()
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()
        entry.grab_focus()

    def _on_library_selected(
        self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]
    ):
        """Handle library selection."""
        logger.debug("LibraryListEditor: Handling library selection")
        library = None
        selected_row = listbox.get_selected_row()

        if selected_row:
            child = selected_row.get_child()
            if isinstance(child, LibraryRow):
                library = child.library
        self.library_selected.send(self, library=library)
