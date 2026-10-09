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

from .commands import CalibrationTestCmd, MaterialTestCmd
from .widgets import ASSEMBLER_WIDGETS
from .widgets.calibration_test_dialogs import (
    FocusTestDialog,
    IntervalTestDialog,
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
    """Register editor command handlers."""
    command_registry.register("material_test", MaterialTestCmd, ADDON_NAME)
    command_registry.register(
        "calibration_tests", CalibrationTestCmd, ADDON_NAME
    )


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

    def _max_speed(window) -> float:
        machine = window.doc_editor.context.machine
        return float(machine.max_cut_speed) if machine else 60000.0

    interval_action = Gio.SimpleAction.new("interval_test", None)

    def on_interval_test(action, param):
        window = action_registry.window
        editor = window.doc_editor
        IntervalTestDialog(
            window,
            editor.calibration_tests.create_interval_test,
            max_speed=_max_speed(window),
        ).present()

    interval_action.connect("activate", on_interval_test)
    action_registry.register(
        action_name="interval_test",
        action=interval_action,
        addon_name=ADDON_NAME,
        label=_("Create Interval Test"),
        menu=MenuPlacement(menu_id="tools", priority=101),
    )

    focus_action = Gio.SimpleAction.new("focus_test", None)

    def on_focus_test(action, param):
        window = action_registry.window
        editor = window.doc_editor
        cmd = editor.calibration_tests
        FocusTestDialog(
            window,
            cmd.create_focus_test,
            cmd.available_focus_modes(),
            max_speed=_max_speed(window),
        ).present()

    focus_action.connect("activate", on_focus_test)
    action_registry.register(
        action_name="focus_test",
        action=focus_action,
        addon_name=ADDON_NAME,
        label=_("Create Focus Test"),
        menu=MenuPlacement(menu_id="tools", priority=102),
    )
