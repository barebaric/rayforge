import logging
from gi.repository import Gtk
from typing import Optional, List, TYPE_CHECKING, Tuple
from ...core.group import Group
from ...core.item import DocItem
from ...core.stock import StockItem
from ...core.workpiece import WorkPiece
from ..shared.expander import Expander
from .property_providers import (
    PropertyProvider,
    TransformPropertyProvider,
    WorkpieceInfoProvider,
    TabsPropertyProvider,
)

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


logger = logging.getLogger(__name__)


class DocItemPropertiesWidget(Expander):
    """
    An orchestrator widget that displays properties for selected document
    items.

    It composes its UI from a set of registered "Property Provider" components,
    each responsible for a specific aspect of an item (e.g., transformation,
    source file, tabs). It manages persistent widgets to avoid interrupting
    user edits.
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

        # Register and instantiate the providers that will build the UI.
        self.providers: List[PropertyProvider] = [
            TransformPropertyProvider(),
            WorkpieceInfoProvider(),
            TabsPropertyProvider(),
        ]
        # This will hold tuples of (provider, [list_of_widgets])
        self._provider_widget_map: List[
            Tuple[PropertyProvider, List[Gtk.Widget]]
        ] = []
        self._initialize_providers_ui()

        self.set_items(items)

    def _initialize_providers_ui(self):
        """
        Creates all widgets for all providers one time and adds them to the
        container in a hidden state.
        """
        for provider in self.providers:
            widgets = provider.create_widgets()
            self._provider_widget_map.append((provider, widgets))
            for widget in widgets:
                widget.set_visible(False)
                self._rows_container.append(widget)

    def set_items(self, items: Optional[List[DocItem]]):
        """Sets the currently selected items and updates the UI."""
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

        self._update_ui()

    def _on_item_data_changed(self, item):
        """
        Handles data changes from the DocItem model by updating the UI to
        reflect the new state.
        """
        logger.debug(
            f"Item data changed for {item.name}, updating properties UI."
        )
        self._update_ui()

    def _update_ui(self):
        """
        Updates the UI by querying all registered property providers and
        managing the visibility and content of their persistent widgets.
        """
        if not self.items:
            self.set_sensitive(False)
            self.set_title(_("Item Properties"))
            # Hide all provider widgets when nothing is selected
            for provider, widgets in self._provider_widget_map:
                for widget in widgets:
                    widget.set_visible(False)
            return

        self.set_sensitive(True)
        self._update_title(self.items[0])

        for provider, widgets in self._provider_widget_map:
            can_handle = provider.can_handle(self.items)
            for widget in widgets:
                widget.set_visible(can_handle)

            if can_handle:
                provider.update_widgets(self.editor, self.items)

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
