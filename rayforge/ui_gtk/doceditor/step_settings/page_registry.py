"""Registry mapping step assembler names to their settings page classes."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rayforge.ui_gtk.doceditor.step_settings.pages.base import (
        StepSettingsPage,
    )

logger = logging.getLogger(__name__)


class StepSettingsPageRegistry:
    """
    Registry for step settings page classes.

    Maps a step's assembler name (``step.ASSEMBLER_NAME``) to the
    :class:`StepSettingsPage` subclass that renders its settings.
    Addons register their pages ahead of time via the
    ``register_step_settings_pages`` hook; the step settings dialog
    looks up the page class directly from the singleton.

    Satisfies the :class:`~rayforge.addon_mgr.addon_manager.AddonRegistry`
    protocol for automatic cleanup on addon unload.
    """

    def __init__(self):
        self._pages: dict[str, type] = {}
        self._addon_items: dict[str, set[str]] = {}

    def register(
        self,
        assembler_name: str,
        page_cls: "type[StepSettingsPage]",
        addon_name: str | None = None,
    ) -> None:
        """
        Register a settings page class for an assembler name.

        Args:
            assembler_name: The step assembler name (``step.ASSEMBLER_NAME``).
            page_cls: The StepSettingsPage subclass.
            addon_name: Optional name of the addon registering this
                page. Used for cleanup when the addon is unloaded.
        """
        self._pages[assembler_name] = page_cls
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(assembler_name)
        logger.debug(
            "Registered settings page %s for assembler %s",
            page_cls.__name__,
            assembler_name,
        )

    def get(self, assembler_name: str) -> type | None:
        """
        Look up the settings page class for an assembler name.

        Returns:
            The page class, or None if not registered.
        """
        return self._pages.get(assembler_name)

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all pages registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of pages unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._pages:
                del self._pages[name]
                count += 1
        return count


step_settings_page_registry = StepSettingsPageRegistry()
