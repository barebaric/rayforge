from typing import Dict, List, Optional, Set, Type, TYPE_CHECKING
from gettext import gettext as _

if TYPE_CHECKING:
    from .base import LayoutStrategy


class LayoutStrategyInfo:
    """Information about a registered layout strategy."""

    def __init__(
        self,
        strategy_class: Type["LayoutStrategy"],
        name: str,
        action_id: Optional[str] = None,
        label: Optional[str] = None,
        shortcut: Optional[str] = None,
        addon_name: Optional[str] = None,
    ):
        self.strategy_class = strategy_class
        self.name = name
        self.action_id = action_id
        self.label = label
        self.shortcut = shortcut
        self.addon_name = addon_name

    def __repr__(self):
        return f"LayoutStrategyInfo({self.name}, {self.addon_name})"


class LayoutStrategyRegistry:
    """
    Registry for layout strategies.

    Allows addons to register custom layout strategies with metadata
    for UI integration.
    """

    def __init__(self):
        self._strategies: Dict[str, LayoutStrategyInfo] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self,
        strategy_class: Type["LayoutStrategy"],
        name: str,
        action_id: Optional[str] = None,
        label: Optional[str] = None,
        shortcut: Optional[str] = None,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register a layout strategy.

        Args:
            strategy_class: The LayoutStrategy subclass to register.
            name: Unique name for this strategy.
            action_id: Optional action ID for UI integration.
            label: Optional display label for UI.
            shortcut: Optional keyboard shortcut.
            addon_name: Optional name of the addon registering this strategy.
        """
        if name in self._strategies:
            if addon_name:
                old_addon = self._strategies[name].addon_name
                if old_addon and old_addon in self._addon_items:
                    self._addon_items[old_addon].discard(name)

        info = LayoutStrategyInfo(
            strategy_class=strategy_class,
            name=name,
            action_id=action_id,
            label=label,
            shortcut=shortcut,
            addon_name=addon_name,
        )
        self._strategies[name] = info

        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

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

        info = self._strategies[name]
        if info.addon_name and info.addon_name in self._addon_items:
            self._addon_items[info.addon_name].discard(name)

        del self._strategies[name]
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
        return count

    def get(self, name: str) -> Optional[Type["LayoutStrategy"]]:
        """
        Look up a strategy class by name.

        Args:
            name: The name of the strategy.

        Returns:
            The strategy class, or None if not found.
        """
        info = self._strategies.get(name)
        return info.strategy_class if info else None

    def get_info(self, name: str) -> Optional[LayoutStrategyInfo]:
        """
        Look up strategy info by name.

        Args:
            name: The name of the strategy.

        Returns:
            The LayoutStrategyInfo object, or None if not found.
        """
        return self._strategies.get(name)

    def list_all(self) -> List[LayoutStrategyInfo]:
        """
        Return a list of all registered strategy info objects.

        Returns:
            List of LayoutStrategyInfo objects.
        """
        return list(self._strategies.values())


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
        action_id="layout-pixel-perfect",
        label=_("Auto Layout (Simple)"),
        shortcut="<Ctrl><Alt>p",
        addon_name="core",
    )
