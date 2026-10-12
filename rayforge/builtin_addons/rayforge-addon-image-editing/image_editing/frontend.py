"""
Frontend entry point for the image_editing addon.

Adds the image editing tools to the Tools menu. They act on the selected
workpieces and are undoable.
"""

from gettext import gettext as _

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement, action_registry
from rayforge.ui_gtk.actions import action_extension_registry

from .commands import toggle_image_edit
from .ops import INVERT

ADDON_NAME = "image_editing"
INVERT_ACTION = "invert-image"


def _register_actions(action_manager):
    action = Gio.SimpleAction.new(INVERT_ACTION, None)

    def on_activate(action, param):
        window = action_manager.win
        toggle_image_edit(
            window.doc_editor.history_manager,
            window.surface.get_selected_workpieces(),
            INVERT,
            _("Invert Image"),
        )

    action.connect("activate", on_activate)
    action_manager.win.add_action(action)
    action_manager.actions[INVERT_ACTION] = action
    action_registry.register(
        action_name=INVERT_ACTION,
        action=action,
        addon_name=ADDON_NAME,
        label=_("Invert Image"),
        menu=MenuPlacement(menu_id="tools", priority=60),
    )


def _update_action_states(action_manager):
    action = action_manager.actions.get(INVERT_ACTION)
    if action:
        selected = action_manager.win.surface.get_selected_workpieces()
        action.set_enabled(bool(selected))


@hookimpl
def register_actions(action_registry):
    action_extension_registry.register_setup(_register_actions, ADDON_NAME)
    action_extension_registry.register_state_update(
        _update_action_states, ADDON_NAME
    )
