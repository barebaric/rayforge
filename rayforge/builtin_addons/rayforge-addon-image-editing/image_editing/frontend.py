"""
Frontend entry point for the image_editing addon.

Adds the image editing tools to the Tools menu. They act on the selected
workpieces and are undoable.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement

from .commands import toggle_image_edit
from .ops import INVERT

if TYPE_CHECKING:
    from rayforge.ui_gtk.actions import ActionManager

ADDON_NAME = "image_editing"
INVERT_ACTION = "invert-image"


def _has_selection(action_manager: "ActionManager") -> bool:
    return bool(action_manager.win.surface.get_selected_workpieces())


@hookimpl
def register_actions(action_registry):
    """Register actions with menu placement."""
    action = Gio.SimpleAction.new(INVERT_ACTION, None)

    def on_activate(action, param):
        window = action_registry.window
        toggle_image_edit(
            window.doc_editor.history_manager,
            window.surface.get_selected_workpieces(),
            INVERT,
            _("Invert Image"),
        )

    action.connect("activate", on_activate)
    action_registry.register(
        action_name=INVERT_ACTION,
        action=action,
        addon_name=ADDON_NAME,
        label=_("Invert Image"),
        menu=MenuPlacement(menu_id="tools", priority=60),
        enabled_when=_has_selection,
    )
