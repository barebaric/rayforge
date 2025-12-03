import logging
from gi.repository import Gtk
from typing import Optional, List, TYPE_CHECKING

from ...core.group import Group
from ...core.item import DocItem
from ...core.stock import StockItem
from ...core.workpiece import WorkPiece
from ...shared.ui.expander import Expander
from .property_providers import (
    PropertyProvider,
    TransformPropertyProvider,
    WorkpieceInfoProvider,
    TabsPropertyProvider,
)

if TYPE_CHECKING:
    from ..editor import DocEditor


logger = logging.getLogger(__name__)


class DocItemPropertiesWidget(Expander):
    """
    An orchestrator widget that displays properties for selected document
    items.

    It composes its UI from a set of registered "Property Provider" components,
    each responsible for a specific aspect of an item (e.g., transformation,
    source file, tabs).
    """

    def __init__(
        self,
        editor: "DocEditor",
        items: Optional[List[DocItem]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.editor = editor
        self.items: List[DocItem] = []
        self._rows_container = Gtk.ListBox()
        self._rows_container.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(self._rows_container)

        self.set_title(_("Item Properties"))
        self.set_expanded(True)

        # Register the providers that will build the UI.
        self.providers: List[PropertyProvider] = [
            TransformPropertyProvider(),
            WorkpieceInfoProvider(),
            TabsPropertyProvider(),
        ]

        self.set_items(items)

    def set_items(self, items: Optional[List[DocItem]]):
        """Sets the currently selected items and rebuilds the UI."""
        for item in self.items:
            item.updated.disconnect(self._on_item_data_changed)
            item.transform_changed.disconnect(self._on_item_data_changed)

        self.items = items or []

        count = len(self.items)
        if count == 1:
            self.set_subtitle(_("1 item selected"))
        elif count > 1:
            self.set_subtitle(_(f"{count} items selected"))
        else:
            self.set_subtitle("")

        for item in self.items:
            item.updated.connect(self._on_item_data_changed)
            item.transform_changed.connect(self._on_item_data_changed)

        self._rebuild_ui()

    def _on_item_data_changed(self, item):
        """
        Handles data changes from the DocItem model by rebuilding the UI to
        reflect the new state.
        """
        logger.debug(
            f"Item data changed for {item.name}, rebuilding properties UI."
        )
        self._rebuild_ui()

    def _rebuild_ui(self):
        """
        Clears the current UI and rebuilds it by querying all registered
        property providers based on the current item selection.
        """
        # Clear existing rows
        child = self._rows_container.get_first_child()
        while child:
            self._rows_container.remove(child)
            child = self._rows_container.get_first_child()

        if not self.items:
            self.set_sensitive(False)
            self.set_title(_("Item Properties"))
            return

        self.set_sensitive(True)
        self._update_title(self.items[0])

        for provider in self.providers:
            if provider.can_handle(self.items):
                rows = provider.create_rows(self.editor, self.items)
                for row in rows:
                    self._rows_container.append(row)

    def _update_title(self, item: DocItem):
        """Sets the main title of the expander based on selection."""
        if len(self.items) > 1:
            self.set_title(_("Multiple Items"))
        elif isinstance(item, StockItem):
            self.set_title(_("Stock Properties"))
        elif isinstance(item, WorkPiece):
            self.set_title(_("Workpiece Properties"))
        elif isinstance(item, Group):
            self.set_title(_("Group Properties"))
        else:
            self.set_title(_("Item Properties"))
