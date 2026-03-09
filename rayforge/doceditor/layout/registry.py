from typing import Dict, List, Optional, Set, Type, TYPE_CHECKING
from blinker import Signal

if TYPE_CHECKING:
    from .base import LayoutStrategy


class LayoutStrategyRegistry:
    """
    Registry for layout strategy classes.

    Allows addons to register custom layout strategies. UI metadata
    (labels, shortcuts, menu/toolbar placement) should be registered
    via the ActionRegistry.
    """

    def __init__(self):
        self._strategies: Dict[str, Type["LayoutStrategy"]] = {}
        self._addon_items: Dict[str, Set[str]] = {}
        self.changed = Signal()

    def register(
        self,
        strategy_class: Type["LayoutStrategy"],
        name: str,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register a layout strategy class.

        Args:
            strategy_class: The LayoutStrategy subclass to register.
            name: Unique name for this strategy.
            addon_name: Optional name of the addon registering this strategy.
        """
        if name in self._strategies:
            if addon_name:
                old_info = self._strategies.get(name)
                if old_info and name in self._addon_items:
                    for addon in list(self._addon_items.keys()):
                        if name in self._addon_items.get(addon, set()):
                            self._addon_items[addon].discard(name)

        self._strategies[name] = strategy_class

        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

        self.changed.send(self)

    def unregister(self, name: str) -> bool:
        """
        Unregister a layout strategy by name.

        Args:
            name: The name of the strategy to unregister.

        Returns:
            True if the strategy was unregistered, False if not found.
        """
        if name not in self._strategies:
            return False

        for addon_name in list(self._addon_items.keys()):
            self._addon_items[addon_name].discard(name)

        del self._strategies[name]
        self.changed.send(self)
        return True

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all strategies registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of strategies unregistered.
        """
        if addon_name not in self._addon_items:
            return 0

        names = self._addon_items.pop(addon_name)
        count = 0
        for name in names:
            if name in self._strategies:
                del self._strategies[name]
                count += 1
        if count > 0:
            self.changed.send(self)
        return count

    def get(self, name: str) -> Optional[Type["LayoutStrategy"]]:
        """
        Look up a strategy class by name.

        Args:
            name: The name of the strategy.

        Returns:
            The strategy class, or None if not found.
        """
        return self._strategies.get(name)

    def list_all(self) -> List[Type["LayoutStrategy"]]:
        """
        Return a list of all registered strategy classes.

        Returns:
            List of LayoutStrategy subclasses.
        """
        return list(self._strategies.values())

    def list_names(self) -> List[str]:
        """
        Return a list of all registered strategy names.

        Returns:
            List of strategy names.
        """
        return list(self._strategies.keys())


layout_registry = LayoutStrategyRegistry()


def register_builtin_layout_strategies():
    """
    Register built-in layout strategies.

    This function should be called during application initialization
    before addons register their own strategies.
    """
    from .auto import PixelPerfectLayoutStrategy

    layout_registry.register(
        PixelPerfectLayoutStrategy,
        name="pixel-perfect",
        addon_name="core",
    )
