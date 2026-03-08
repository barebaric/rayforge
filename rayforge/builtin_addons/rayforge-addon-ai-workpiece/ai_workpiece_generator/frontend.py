from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from .controller import controller
from .widgets import AIWorkpieceGeneratorDialog

if TYPE_CHECKING:
    from rayforge.ui_gtk.mainwindow import MainWindow


ADDON_NAME = "ai_workpiece_generator"


@hookimpl
def register_menu_items(menu_registry):
    menu_registry.register(
        item_id="ai_workpiece_generator.generate",
        label=_("Generate Workpiece with AI..."),
        action="win.ai_generate_workpiece",
        menu="Tools",
        priority=50,
        addon_name=ADDON_NAME,
    )


@hookimpl
def register_actions(window: "MainWindow") -> None:
    action = Gio.SimpleAction.new("ai_generate_workpiece", None)

    def on_activate(action, param) -> None:
        editor = window.doc_editor
        dialog = AIWorkpieceGeneratorDialog(
            editor=editor,
            controller=controller,
            parent=window,
        )
        dialog.present()

    action.connect("activate", on_activate)
    window.action_registry.register(
        "ai_generate_workpiece", action, "ai_workpiece_generator"
    )


@hookimpl
def register_step_widgets(widget_registry):
    pass
