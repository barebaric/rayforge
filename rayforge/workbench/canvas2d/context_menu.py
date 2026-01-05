from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from gi.repository import Gtk, Gio

if TYPE_CHECKING:
    from .surface import WorkSurface

logger = logging.getLogger(__name__)


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

    menu.append_item(Gio.MenuItem.new(_("Remove"), "win.remove"))


def _create_item_context_menu() -> Gio.Menu:
    """Builds the standard context menu for DocItems."""
    menu = Gio.Menu.new()
    _populate_standard_items(menu)
    return menu


def _create_sketch_item_context_menu() -> Gio.Menu:
    """Builds the context menu for Sketch WorkPieces."""
    menu = Gio.Menu.new()
    menu.append_item(Gio.MenuItem.new(_("Edit Sketch"), "win.edit_sketch"))
    menu.append_item(
        Gio.MenuItem.new(_("Export Sketch..."), "win.export_sketch")
    )

    # Separator before standard items
    menu.append_section(None, Gio.Menu.new())

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
    "sketch-item": _create_sketch_item_context_menu(),
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


def show_item_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """
    Displays the context menu for general items like WorkPieces or Groups.
    """
    _show_popover(surface, gesture, _MENU_MODELS["item"])


def show_sketch_item_context_menu(
    surface: "WorkSurface", gesture: Gtk.Gesture
):
    """
    Displays the context menu for Sketch items.
    """
    _show_popover(surface, gesture, _MENU_MODELS["sketch-item"])


def show_geometry_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """Displays the context menu for adding a tab to a geometry path."""
    _show_popover(surface, gesture, _MENU_MODELS["geometry"])


def show_tab_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """Displays the context menu for an existing tab."""
    _show_popover(surface, gesture, _MENU_MODELS["tab"])
