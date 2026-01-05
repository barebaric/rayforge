"""Material manager UI component for Rayforge."""

import logging
from typing import Optional
from gi.repository import Adw
from ...core.material_library import MaterialLibrary
from .material_library_list import LibraryListWidget
from .material_list import MaterialListWidget

logger = logging.getLogger(__name__)


class MaterialManager(Adw.PreferencesPage):
    """
    Widget for managing materials and libraries.
    """

    library_list_editor: LibraryListWidget
    material_list_editor: MaterialListWidget

    def __init__(self):
        """Initialize the material manager."""
        super().__init__(
            title=_("Materials"),
            icon_name="material-symbolic",
        )

        self.library_list_editor = LibraryListWidget(
            title=_("Material Libraries"),
            description=_(
                "Manage your material libraries. Select a library to "
                "view its materials."
            ),
        )
        self.add(self.library_list_editor)

        self.material_list_editor = MaterialListWidget(
            title=_("Materials"),
            description=_("Materials in the selected library."),
        )
        self.add(self.material_list_editor)

        self.library_list_editor.library_selected.connect(
            self._on_library_selected
        )
        self.material_list_editor.material_added.connect(
            self._on_material_event
        )
        self.material_list_editor.material_deleted.connect(
            self._on_material_event
        )

        self.library_list_editor.populate_and_select()

    def _on_library_selected(
        self, sender, library: Optional[MaterialLibrary] = None
    ):
        """Handle library selection change."""
        logger.debug(
            f"MaterialManager: Library selected: "
            f"'{library.library_id if library is not None else 'None'}'"
        )
        self.material_list_editor.set_library(library)

    def _on_material_event(self, sender, library: MaterialLibrary):
        """
        Handle a material being added to or removed from a library.

        This re-populates and re-selects the library list to ensure the
        material count in the subtitle is updated.
        """
        self.library_list_editor.populate_and_select(library.library_id)
