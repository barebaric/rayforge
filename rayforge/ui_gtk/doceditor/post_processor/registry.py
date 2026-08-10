"""Registry mapping transformer classes to their settings widget classes."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rayforge.pipeline.transformer.base import OpsTransformer

logger = logging.getLogger(__name__)


class TransformerWidgetRegistry:
    """
    Registry for post-processor transformer settings widget classes.

    Maps an :class:`OpsTransformer` subclass to the
    :class:`TransformerSettingsGroup` subclass that renders its
    settings UI. Addons register their widgets at module-import time;
    pages look up widget classes directly from the singleton — no
    pluggy hook call needed.

    Satisfies the :class:`~rayforge.addon_mgr.addon_manager.AddonRegistry`
    protocol for automatic cleanup on addon unload.
    """

    def __init__(self):
        self._widgets: dict[type, type] = {}
        self._addon_items: dict[str, set[type]] = {}

    def register(
        self,
        transformer_cls: "type[OpsTransformer]",
        widget_cls: type,
        addon_name: str | None = None,
    ) -> None:
        """
        Register a widget class for a transformer type.

        Args:
            transformer_cls: The OpsTransformer subclass.
            widget_cls: The TransformerSettingsGroup subclass that
                renders its settings.
            addon_name: Optional name of the addon registering this
                widget. Used for cleanup when the addon is unloaded.
        """
        self._widgets[transformer_cls] = widget_cls
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(transformer_cls)
        logger.debug(
            "Registered widget %s for transformer %s",
            widget_cls.__name__,
            transformer_cls.__name__,
        )

    def get(self, transformer_cls: type) -> type | None:
        """
        Look up the widget class for a transformer type.

        Returns:
            The widget class, or None if no widget is registered.
        """
        return self._widgets.get(transformer_cls)

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all widgets registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of widgets unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for cls in items:
            if cls in self._widgets:
                del self._widgets[cls]
                count += 1
        return count


transformer_widget_registry = TransformerWidgetRegistry()
