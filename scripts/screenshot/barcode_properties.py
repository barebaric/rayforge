"""Screenshot: Barcode properties group and a Code 128 symbol.

Usage: pixi run screenshot addon:barcodes
"""

import importlib
import logging
import time

from raygeo.barcode import Symbology
from utils import (
    get_target,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    wait_for_settled,
)

from rayforge.context import get_context
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("addon:barcodes")


def _addon_module(name: str):
    """Import a module from the addon manager's runtime package."""
    return importlib.import_module(f"rayforge_addons.barcodes.barcodes.{name}")


def _add_sample_barcode(commands):
    editor = win.doc_editor
    workpiece = commands.add_barcode(editor, commands.default_position(editor))

    def _set(setting, value):
        commands.set_barcode_setting(
            editor, workpiece.get_geometry_provider(), setting, value
        )

    _set("symbology", Symbology.CODE128)
    _set("content", "SN-0042")
    _set("show_text", True)
    _set("module_mm", 0.5)
    _set("bar_height_mm", 18.0)
    commands.rename_barcode_asset(
        editor, workpiece.get_geometry_provider(), "Serial label"
    )

    win.surface.update_from_doc()
    win.surface.select_items([workpiece])
    return workpiece


def _prepare_view():
    """Collapse the transform card and scroll to the Barcode group."""
    win.item_props_widget._main_expander.set_expanded(False)
    adjustment = win._right_pane.get_vadjustment()
    adjustment.set_value(adjustment.get_upper() - adjustment.get_page_size())
    for toast in list(win._active_toasts):
        toast.dismiss()


@restore_config
def main():
    set_window_size(win, 1400, 1000)

    context = get_context()
    _ = context.addon_mgr
    if "barcodes" not in context.addon_mgr.loaded_addons:
        logger.error("The barcodes addon is not loaded")
        app.quit_idle()
        return

    commands = _addon_module("commands")
    run_on_main_thread(lambda: _add_sample_barcode(commands))

    if not wait_for_settled(win, timeout=15):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    run_on_main_thread(_prepare_view)
    time.sleep(1.0)
    logger.info("Taking screenshot: %s", TARGET)
    take_screenshot()
    app.quit_idle()


main()
