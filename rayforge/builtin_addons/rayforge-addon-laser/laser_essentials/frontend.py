"""
Frontend entry point for laser-essentials addon.

Registers UI widgets and actions with the main application.
"""

import gettext
from pathlib import Path

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement
from rayforge.ui_gtk.icons import register_icon_path
from .commands import MaterialTestCmd
from .widgets import PRODUCER_WIDGETS

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation(
    "laser_essentials", localedir=_localedir, fallback=True
)
_ = _t.gettext

ADDON_NAME = "laser_essentials"
_ICONS_DIR = Path(__file__).parent / "resources" / "icons"

register_icon_path(_ICONS_DIR)


@hookimpl
def step_settings_loaded(dialog, step, producer):
    """Add step settings widgets based on producer type."""
    if producer is None:
        return

    widget_cls = PRODUCER_WIDGETS.get(type(producer))
    if widget_cls:
        dialog.add(
            widget_cls(
                dialog.editor,
                step.typelabel,
                producer,
                dialog,
                step,
            )
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
