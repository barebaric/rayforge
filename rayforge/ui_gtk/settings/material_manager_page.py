"""Material manager UI component for Rayforge."""

import logging
from gettext import gettext as _

from gi.repository import Adw

from ...context import get_context
from ...core.material_library import MaterialLibrary
from ..doceditor.material_library_list import LibraryListWidget
from ..doceditor.material_list import MaterialListWidget
from ..doceditor.material_selector import MaterialRow
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.preferences_page import TrackedPreferencesPage

logger = logging.getLogger(__name__)


class MaterialManagerPage(TrackedPreferencesPage):
    """
    Widget for managing materials and libraries.
    """

    key = "materials"

    library_list_editor: LibraryListWidget
    material_list_editor: MaterialListWidget

    def __init__(self):
        """Initialize the material manager."""
        super().__init__(
            title=_("Materials"),
            icon_name="material-symbolic",
        )

        defaults_group = Adw.PreferencesGroup(
            title=_("New Stock Defaults"),
            description=_(
                "Material and thickness applied when a new stock or "
                "rotary layer is created."
            ),
        )
        self.default_material_row = MaterialRow(
            _("Default Stock Material"),
            _("Material used when a new stock or rotary layer is created"),
            on_select=self._on_default_material_selected,
        )
        config = get_context().config
        current_uid = config.default_stock_material_uid
        if current_uid:
            current = get_context().material_mgr.get_material_or_none(
                current_uid
            )
            self.default_material_row.set_material(current)
        defaults_group.add(self.default_material_row)

        self.default_thickness_row = LengthSpinRow(
            _("Default Stock Thickness"),
            _("Thickness applied to new stock assets"),
            upper=999,
            value_in_base=config.default_stock_thickness_mm,
        )
        self.default_thickness_row.value_changed.connect(
            self._on_default_thickness_changed
        )
        defaults_group.add(self.default_thickness_row)
        self.add(defaults_group)

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

        get_context().material_mgr.libraries_changed.connect(
            self._on_libraries_changed
        )

        self.library_list_editor.populate_and_select()

    def _on_default_material_selected(self, material_uid: str | None):
        """Persist the user's default stock material choice."""
        if material_uid is None:
            return
        get_context().config.set_default_stock_material(material_uid)
        material = get_context().material_mgr.get_material_or_none(
            material_uid
        )
        if material is not None:
            self.default_material_row.set_material(material)

    def _on_default_thickness_changed(self, row: LengthSpinRow):
        """Persist the user's default stock thickness choice."""
        get_context().config.set_default_stock_thickness(
            row.get_value_in_base_units()
        )

    def _on_library_selected(
        self, sender, library: MaterialLibrary | None = None
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

    def _on_libraries_changed(self, sender):
        """Handle libraries being added or removed."""
        self.library_list_editor.populate_and_select()
