from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from gi.repository import Gtk, Gio

if TYPE_CHECKING:
    from .surface import WorkSurface

logger = logging.getLogger(__name__)


def show_context_menu(surface: "WorkSurface", gesture: Gtk.Gesture):
    """
    Builds and displays the context menu on the WorkSurface.

    The menu items displayed are dynamically constructed, and their sensitivity
    is automatically handled by GTK based on the state of their
    associated Gio.Actions.
    """
    menu = Gio.Menu.new()

    # Section 1: Layer Management
    menu.append_item(
        Gio.MenuItem.new(_("Move Up a Layer"), "win.layer-move-up")
    )
    menu.append_item(
        Gio.MenuItem.new(_("Move Down a Layer"), "win.layer-move-down")
    )

    # Add a separator before the next section
    menu.append_section(None, Gio.Menu.new())

    # Section 2: Grouping
    menu.append_item(Gio.MenuItem.new(_("Group"), "win.group"))
    menu.append_item(Gio.MenuItem.new(_("Ungroup"), "win.ungroup"))

    # Add another separator
    menu.append_section(None, Gio.Menu.new())

    # Section 3: Deletion
    # Use a more descriptive label if multiple items are selected.
    num_selected = len(surface.get_selected_elements())
    remove_label = _("Remove Item") if num_selected <= 1 else _("Remove Items")
    menu.append_item(Gio.MenuItem.new(remove_label, "win.remove"))

    # Create the popover menu from the model
    popover = Gtk.PopoverMenu.new_from_model(menu)
    popover.set_parent(surface)
    popover.set_has_arrow(False)
    popover.set_position(Gtk.PositionType.RIGHT)

    # Set the pointing_to rectangle to the click location to position the menu
    ok, rect = gesture.get_bounding_box()
    if ok:
        popover.set_pointing_to(rect)

    popover.popup()
