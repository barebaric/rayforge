"""
Frontend entry point for laser-essentials addon.

Registers UI widgets and actions with the main application.
"""

from gettext import gettext as _
from pathlib import Path

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement
from rayforge.ui_gtk.doceditor.property_providers import (
    property_provider_registry,
)
from rayforge.ui_gtk.icons import register_icon_path

from .commands import CalibrationCmd, CalibrationSync
from .widgets import ASSEMBLER_WIDGETS
from .widgets.calibration_provider import (
    CalibrationPropertyProvider,
    show_new_test,
)

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
    """
    Register editor command handlers, and the Test Settings they are
    edited with. The main window creates its property panels after the
    commands are registered.
    """
    command_registry.register("calibration", CalibrationCmd, ADDON_NAME)
    property_provider_registry.register(
        CalibrationPropertyProvider, ADDON_NAME
    )
    command_registry.register("calibration_sync", CalibrationSync, ADDON_NAME)


def _create_test(action_registry, generator_id: str):
    window = action_registry.window
    editor = window.doc_editor
    cmd = editor.calibration
    try:
        record = cmd.create(generator_id)
    except ValueError as e:
        editor.notification_requested.send(editor, message=str(e))
        return
    show_new_test(window, cmd.workpieces(record))


@hookimpl
def register_actions(action_registry):
    """Register actions with menu placement."""
    tests = (
        (
            "material_test",
            "material_test_grid",
            _("Create Material Test Grid"),
        ),
        ("interval_test", "interval_test", _("Create Interval Test")),
        ("focus_test", "focus_test", _("Create Focus Test")),
    )
    for priority, (action_name, generator_id, label) in enumerate(
        tests, start=100
    ):
        action = Gio.SimpleAction.new(action_name, None)
        action.connect(
            "activate",
            lambda _action, _param, gid=generator_id: _create_test(
                action_registry, gid
            ),
        )
        action_registry.register(
            action_name=action_name,
            action=action,
            addon_name=ADDON_NAME,
            label=label,
            menu=MenuPlacement(menu_id="tools", priority=priority),
        )
