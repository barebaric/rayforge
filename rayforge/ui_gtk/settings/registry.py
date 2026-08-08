"""Registry for addon-contributed Settings dialog pages."""

import logging
from collections.abc import Callable

from blinker import Signal

logger = logging.getLogger(__name__)


class SettingsPageRegistry:
    """
    Collects settings page classes contributed by addons.

    A page class is a no-argument widget constructor that exposes
    ``get_title()`` and ``get_icon_name()`` (e.g. a
    :class:`~rayforge.ui_gtk.shared.preferences_page.TrackedPreferencesPage`
    subclass). The
    :class:`~rayforge.ui_gtk.settings.settings_dialog.SettingsWindow`
    instantiates each registered class when it is built.

    Implements the :class:`~rayforge.addon_mgr.addon_manager.AddonRegistry`
    protocol so that pages are removed automatically when their addon is
    unloaded.

    Emits the :attr:`changed` signal whenever pages are added or removed
    so that open settings windows can rebuild themselves live.
    """

    def __init__(self) -> None:
        self._pages: list[tuple[Callable[[], object], str]] = []
        self.changed = Signal()

    def register(
        self,
        page_class: Callable[[], object],
        addon_name: str = "",
    ) -> None:
        """
        Register a settings page class.

        Re-registering the same class is a no-op (so an addon being
        reloaded does not produce duplicate pages).

        Args:
            page_class: A no-arg widget constructor.
            addon_name: The canonical name of the contributing addon,
                used for cleanup on unload.
        """
        if any(cls is page_class for cls, _ in self._pages):
            return
        self._pages.append((page_class, addon_name))
        logger.debug(
            f"Registered settings page {page_class!r} for '{addon_name}'"
        )
        self.changed.send(self)

    def get_pages(self) -> list[Callable[[], object]]:
        """Return all registered page classes in insertion order."""
        return [cls for cls, _ in self._pages]

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Remove all pages registered by the named addon.

        Returns:
            The number of pages removed.
        """
        before = len(self._pages)
        self._pages = [
            (cls, name) for cls, name in self._pages if name != addon_name
        ]
        removed = before - len(self._pages)
        if removed:
            logger.info(
                f"Removed {removed} settings pages from '{addon_name}'"
            )
            self.changed.send(self)
        return removed


settings_page_registry = SettingsPageRegistry()
