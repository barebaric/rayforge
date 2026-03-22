import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Type, TYPE_CHECKING

from gi.repository import Gtk

from ....core.item import DocItem

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class PropertyProviderRegistry:
    """
    Registry for property provider classes.

    Allows registration of providers that can create UI widgets for
    specific types of document items. Providers are sorted by priority
    (lower priority appears first in UI).
    """

    def __init__(self):
        self._providers: List[Type["PropertyProvider"]] = []
        self._addon_map: Dict[Type["PropertyProvider"], str] = {}

    def register(
        self, provider_cls: Type["PropertyProvider"], addon_name: str
    ) -> None:
        """
        Register a property provider class.

        Providers are sorted by priority when instances are created.

        Args:
            provider_cls: The provider class to register
            addon_name: Optional addon name for cleanup during unload
        """
        if provider_cls not in self._providers:
            self._providers.append(provider_cls)
            if addon_name:
                self._addon_map[provider_cls] = addon_name
            logger.debug(
                f"Registered property provider: {provider_cls.__name__}"
            )

    def unregister(self, provider_cls: Type["PropertyProvider"]) -> bool:
        """
        Unregister a property provider class.

        Returns True if the provider was found and removed.
        """
        if provider_cls in self._providers:
            self._providers.remove(provider_cls)
            self._addon_map.pop(provider_cls, None)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all providers registered by a specific addon.

        Args:
            addon_name: The name of the addon to clean up

        Returns:
            The number of providers unregistered
        """
        to_remove = [
            cls for cls, name in self._addon_map.items() if name == addon_name
        ]
        for cls in to_remove:
            self._providers.remove(cls)
            del self._addon_map[cls]
        if to_remove:
            logger.debug(
                f"Unregistered {len(to_remove)} property providers "
                f"from addon '{addon_name}'"
            )
        return len(to_remove)

    def create_instances(self) -> List["PropertyProvider"]:
        """
        Create instances of all registered providers, sorted by priority.

        Lower priority values appear first in the UI.

        Returns a list of provider instances ready for use.
        """
        sorted_providers = sorted(
            self._providers,
            key=lambda cls: getattr(cls, "priority", 100),
        )
        return [cls() for cls in sorted_providers]

    def all(self) -> List[Type["PropertyProvider"]]:
        """Return all registered provider classes."""
        return self._providers.copy()


property_provider_registry = PropertyProviderRegistry()


class PropertyProvider(ABC):
    """
    Defines the contract for a component that can provide UI for a specific
    aspect of one or more DocItems.
    """

    priority: int = 100
    """Lower priority values appear first in the UI."""

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
