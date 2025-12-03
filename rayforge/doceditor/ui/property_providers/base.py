# rayforge/doceditor/ui/property_providers/base.py

from abc import ABC, abstractmethod
from gi.repository import Gtk
from typing import List, TYPE_CHECKING
from ....core.item import DocItem

if TYPE_CHECKING:
    from ...editor import DocEditor


class PropertyProvider(ABC):
    """
    Defines the contract for a component that can provide UI for a specific
    aspect of one or more DocItems.
    """

    def __init__(self):
        self.editor: "DocEditor"
        self.items: List[DocItem] = []
        self._in_update: bool = False
        # A list of all widgets created by this provider to manage them
        self._rows: List[Gtk.Widget] = []

    @abstractmethod
    def can_handle(self, items: List[DocItem]) -> bool:
        """
        Returns True if this provider is applicable to the given selection
        of items.
        """
        ...

    @abstractmethod
    def create_rows(
        self, editor: "DocEditor", items: List[DocItem]
    ) -> List[Gtk.Widget]:
        """
        Generates the list of Adw.PreferencesRow widgets for the given
        selection.
        """
        ...
