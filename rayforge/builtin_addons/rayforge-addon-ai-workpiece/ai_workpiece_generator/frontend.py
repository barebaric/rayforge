from gettext import gettext as _

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement
from .controller import controller
from .widgets import AIWorkpieceGeneratorDialog


ADDON_NAME = "ai_workpiece_generator"


@hookimpl
def register_actions(action_registry):
    """Register action for AI workpiece generation with menu placement."""
    action = Gio.SimpleAction.new("ai_generate_workpiece", None)

    def on_activate(action, param) -> None:
        window = action_registry.window
        editor = window.doc_editor
        dialog = AIWorkpieceGeneratorDialog(
            editor=editor,
            controller=controller,
            parent=window,
        )
        dialog.present()

    action.connect("activate", on_activate)
    action_registry.register(
        action_name="ai_generate_workpiece",
        action=action,
        addon_name=ADDON_NAME,
        label=_("Generate Workpiece with AI..."),
        menu=MenuPlacement(menu_id="tools", priority=50),
    )
