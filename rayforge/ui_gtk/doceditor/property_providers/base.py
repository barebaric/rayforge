import logging
from abc import ABC, abstractmethod
from gi.repository import Gtk
from typing import List, TYPE_CHECKING
from ....core.item import DocItem

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


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
        logger.debug(
            f"PropertyProvider '{self.__class__.__name__}' initialized."
        )

    @abstractmethod
    def can_handle(self, items: List[DocItem]) -> bool:
        """
        Returns True if this provider is applicable to the given selection
        of items.
        """
        ...

    @abstractmethod
    def create_widgets(self) -> List[Gtk.Widget]:
        """
        Creates the necessary Gtk.Widget instances for this provider.
        This method is called only once. The created widgets should be
        stored as instance members for later access by `update_widgets`.
        """
        ...

    @abstractmethod
    def update_widgets(self, editor: "DocEditor", items: List[DocItem]):
        """
        Updates the state of the widgets created by `create_widgets` to
        reflect the properties of the currently selected items.
        """
        ...
