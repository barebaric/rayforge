from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from .controller import controller
from .widgets import AISvgGeneratorDialog

if TYPE_CHECKING:
    from rayforge.ui_gtk.mainwindow import MainWindow

ADDON_NAME = "ai_svg_generator"


@hookimpl
def register_menu_items(menu_registry):
    menu_registry.register(
        item_id="ai_svg_generator.generate",
        label=_("Generate Workpiece with AI..."),
        action="win.ai_generate_svg",
        menu="Tools",
        priority=50,
        addon_name=ADDON_NAME,
    )


@hookimpl
def register_actions(window: "MainWindow") -> None:
    action = Gio.SimpleAction.new("ai_generate_svg", None)

    def on_activate(action, param) -> None:
        editor = window.doc_editor
        dialog = AISvgGeneratorDialog(
            editor=editor,
            controller=controller,
            parent=window,
        )
        dialog.present()

    action.connect("activate", on_activate)
    window.add_action(action)


@hookimpl
def register_step_widgets(widget_registry) -> None:
    pass
