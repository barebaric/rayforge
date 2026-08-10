"""
Frontend entry point for laser-essentials addon.

Registers UI widgets and actions with the main application.
"""

from gettext import gettext as _
from pathlib import Path

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement
from rayforge.ui_gtk.icons import register_icon_path

from .commands import MaterialTestCmd
from .widgets import ASSEMBLER_WIDGETS

ADDON_NAME = "laser_essentials"
_ICONS_DIR = Path(__file__).parent / "resources" / "icons"

register_icon_path(_ICONS_DIR)


@hookimpl
def register_step_settings_pages(step_settings_page_registry):
    """Register step settings page classes based on assembler name."""
    for assembler_name, page_cls in ASSEMBLER_WIDGETS.items():
        step_settings_page_registry.register(
            assembler_name, page_cls, ADDON_NAME
        )


@hookimpl
def register_commands(command_registry):
    """Register editor command handlers."""
    command_registry.register("material_test", MaterialTestCmd, ADDON_NAME)


@hookimpl
def register_actions(action_registry):
    """Register actions with menu placement."""
    action = Gio.SimpleAction.new("material_test", None)

    def on_activate(action, param):
        window = action_registry.window
        editor = window.doc_editor
        editor.material_test.create_test_grid()

    action.connect("activate", on_activate)
    action_registry.register(
        action_name="material_test",
        action=action,
        addon_name=ADDON_NAME,
        label=_("Create Material Test Grid"),
        menu=MenuPlacement(menu_id="tools", priority=100),
    )
