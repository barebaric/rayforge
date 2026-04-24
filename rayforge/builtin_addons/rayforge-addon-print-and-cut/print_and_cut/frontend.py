import gettext
import logging
from pathlib import Path

from gi.repository import Gio

from rayforge.core.group import Group
from rayforge.core.hooks import hookimpl
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.action_registry import (
    MenuPlacement,
    action_registry,
)
from rayforge.ui_gtk.actions import action_extension_registry

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation("print_and_cut", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "print_and_cut"
logger = logging.getLogger(__name__)


def _register_actions(action_manager):
    action = Gio.SimpleAction.new("align-workpiece", None)

    def on_activate(action, param):
        window = action_manager.win
        surface = window.surface
        selected = surface.get_selected_top_level_items()
        if len(selected) != 1:
            return
        item = selected[0]
        if not isinstance(item, (WorkPiece, Group)):
            return
        from rayforge.context import get_context
        from .wizard import PrintAndCutWizard

        ctx = get_context()
        machine = ctx.machine
        if not machine:
            return
        wizard = PrintAndCutWizard(
            parent=window,
            item=item,
            machine=machine,
            machine_cmd=window.machine_cmd,
            editor=window.doc_editor,
        )
        wizard.present()

    action.connect("activate", on_activate)
    action_manager.win.add_action(action)
    action_manager.actions["align-workpiece"] = action
    action_registry.register(
        action_name="align-workpiece",
        action=action,
        addon_name=ADDON_NAME,
        label=_("Align to Physical Position"),
        menu=MenuPlacement(menu_id="tools", priority=50),
    )


def _update_action_states(action_manager):
    selected = action_manager.win.surface.get_selected_top_level_items()
    action = action_manager.actions.get("align-workpiece")
    if action:
        enabled = len(selected) == 1 and isinstance(
            selected[0], (WorkPiece, Group)
        )
        action.set_enabled(enabled)


@hookimpl
def register_actions(action_registry):
    action_extension_registry.register_setup(_register_actions, ADDON_NAME)
    action_extension_registry.register_state_update(
        _update_action_states, ADDON_NAME
    )
