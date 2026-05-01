from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Callable, List, Dict
from gettext import gettext as _
from gi.repository import Gtk, Gio

if TYPE_CHECKING:
    from ...core.item import DocItem
    from .surface import WorkSurface

logger = logging.getLogger(__name__)

ContextMenuHandler = Callable[
    ["WorkSurface", Optional["DocItem"], Gtk.Gesture, Gio.Menu], None
]


class ContextMenuExtensionRegistry:
    """
    Registry for context menu extension handlers.

    Handlers are called when a context menu is about to be shown,
    allowing addons to add custom menu items.
    """

    def __init__(self):
        self._handlers: List[ContextMenuHandler] = []
        self._addon_map: Dict[str, str] = {}

    def register(self, handler: ContextMenuHandler, addon_name: str):
        """Register a context menu extension handler."""
        self._handlers.append(handler)
        if addon_name:
            self._addon_map[handler.__name__] = addon_name
        logger.debug(f"Registered context menu handler: {handler.__name__}")

    def unregister(self, handler: ContextMenuHandler) -> bool:
        """Unregister a context menu extension handler."""
        try:
            self._handlers.remove(handler)
            self._addon_map.pop(handler.__name__, None)
            return True
        except ValueError:
            return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all handlers registered by a specific addon.

        Args:
            addon_name: The name of the addon to clean up

        Returns:
            The number of handlers unregistered
        """
        to_remove = [
            h
            for h in self._handlers
            if self._addon_map.get(h.__name__) == addon_name
        ]
        for h in to_remove:
            self._handlers.remove(h)
            self._addon_map.pop(h.__name__, None)
        if to_remove:
            logger.debug(
                f"Unregistered {len(to_remove)} context menu handlers "
                f"from addon '{addon_name}'"
            )
        return len(to_remove)

    def invoke_all(
        self,
        surface: "WorkSurface",
        item: Optional["DocItem"],
        gesture: Gtk.Gesture,
        menu: Gio.Menu,
    ):
        """Invoke all registered handlers."""
        for handler in self._handlers:
            try:
                handler(surface, item, gesture, menu)
            except Exception as e:
                logger.error(
                    f"Error in context menu handler {handler.__name__}: {e}",
                    exc_info=True,
                )


context_menu_extension_registry = ContextMenuExtensionRegistry()


def _populate_standard_items(menu: Gio.Menu):
    """
    Helper to append standard items to a menu using flat structure with
    separators.
    """
    menu.append_item(
        Gio.MenuItem.new(_("Move Up a Layer"), "win.layer-move-up")
    )
    menu.append_item(
        Gio.MenuItem.new(_("Move Down a Layer"), "win.layer-move-down")
    )

    # Separator
    menu.append_section(None, Gio.Menu.new())

    menu.append_item(Gio.MenuItem.new(_("Group"), "win.group"))
    menu.append_item(Gio.MenuItem.new(_("Ungroup"), "win.ungroup"))

    # Separator
    menu.append_section(None, Gio.Menu.new())

    menu.append_item(
        Gio.MenuItem.new(_("Convert to Stock"), "win.convert-to-stock")
    )

    # Separator
    menu.append_section(None, Gio.Menu.new())

    menu.append_item(Gio.MenuItem.new(_("Remove"), "win.remove"))


def _create_item_context_menu() -> Gio.Menu:
    """Builds the standard context menu for DocItems."""
    menu = Gio.Menu.new()
    _populate_standard_items(menu)
    return menu


def _create_geometry_context_menu() -> Gio.Menu:
    """Builds the context menu for interacting with a workpiece's path."""
    menu = Gio.Menu.new()
    menu.append_item(Gio.MenuItem.new(_("Add Tab Here"), "win.tab-add"))
    return menu


def _create_tab_context_menu() -> Gio.Menu:
    """Builds the context menu for an existing tab handle."""
    menu = Gio.Menu.new()
    menu.append_item(Gio.MenuItem.new(_("Remove Tab"), "win.tab-remove"))
    return menu


# Pre-build and cache the menu models once when the module is loaded.
_MENU_MODELS = {
    "item": _create_item_context_menu(),
    "geometry": _create_geometry_context_menu(),
    "tab": _create_tab_context_menu(),
}


def _show_popover(
    surface: "WorkSurface", gesture: Gtk.Gesture, menu_model: Gio.Menu
):
    """Helper to create and show a popover menu from a model."""
    popover = Gtk.PopoverMenu.new_from_model(menu_model)
    popover.set_parent(surface)
    popover.set_has_arrow(False)

    # Position usually defaults to bottom/right, rely on set_pointing_to for
    # exact placement.
    popover.set_position(Gtk.PositionType.RIGHT)

    ok, rect = gesture.get_bounding_box()
    if ok:
        popover.set_pointing_to(rect)

    popover.popup()


def show_item_context_menu(
    surface: "WorkSurface",
    gesture: Gtk.Gesture,
    item: Optional["DocItem"] = None,
):
    """
    Displays the context menu for general items like WorkPieces or Groups.

    Emits the context_menu_requested signal and invokes registered extension
    handlers to allow addons to add custom menu items. The item parameter is
    passed to handlers so they can determine if the menu should be extended.
    """
    menu = Gio.Menu.new()
    _populate_standard_items(menu)

    # Invoke registered extension handlers
    context_menu_extension_registry.invoke_all(surface, item, gesture, menu)

    # Also emit signal for direct connections
    surface.context_menu_requested.send(
        surface, item=item, gesture=gesture, menu=menu
    )

    _show_popover(surface, gesture, menu)


def show_geometry_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """Displays the context menu for adding a tab to a geometry path."""
    _show_popover(surface, gesture, _MENU_MODELS["geometry"])


def show_tab_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """Displays the context menu for an existing tab."""
    _show_popover(surface, gesture, _MENU_MODELS["tab"])


def show_background_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """Displays the context menu for empty canvas space."""
    menu = Gio.Menu.new()
    menu.append_item(Gio.MenuItem.new(_("New Sketch"), "win.new_sketch"))
    menu.append_item(Gio.MenuItem.new(_("New Stock"), "win.add-stock"))
    menu.append_section(None, Gio.Menu.new())
    menu.append_item(Gio.MenuItem.new(_("Import File\u2026"), "win.import"))
    menu.append_section(None, Gio.Menu.new())
    menu.append_item(Gio.MenuItem.new(_("Paste"), "win.paste"))
    _show_popover(surface, gesture, menu)
