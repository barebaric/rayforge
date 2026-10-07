"""Screenshot: Macro editor dialog."""

import logging
import time
from gettext import gettext as _
from pathlib import Path

import utils
from gi.repository import GLib
from utils import (
    _save_webp_deterministic,
    capture_full_screen,
    get_output_name,
    get_target,
    get_toplevel_window_box,
    park_mouse_pointer,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
)

from rayforge.machine.models.macro import Macro
from rayforge.ui_gtk.machine.gcode_editor import GcodeEditorDialog
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET = get_target("machine-settings:hooks-macros:editor")


def _log_popover_size(popover):
    logger.info(
        "Variables popover size: %sx%s",
        popover.get_width(),
        popover.get_height(),
    )
    return GLib.SOURCE_REMOVE


def _open_dialog():
    machine = win.doc_editor.context.machine
    macros = list(machine.macros.values())
    if macros:
        macro = macros[0]
    else:
        macro = Macro(name=_("New Macro"), code=["G21", "G90", "M8"])

    dialog = GcodeEditorDialog(
        win,
        macro,
        allow_name_edit=True,
        existing_macros=macros,
    )
    dialog.present()
    if TARGET.endswith(":variables"):
        popover = dialog.code_editor.variables_popover
        popover.popup()
        GLib.timeout_add(600, _log_popover_size, popover)


def _take_popover_screenshot() -> bool:
    """Capture dialog and popover from the root window.

    The popover renders as its own surface outside the dialog, so a
    plain window capture would cut it. Crop the full screen to the
    dialog box expanded by a margin instead; no synthetic frame, as
    the crop deliberately includes area beyond the window.
    """
    park_mouse_pointer()
    time.sleep(0.5)
    img = capture_full_screen()
    box = get_toplevel_window_box()
    if img is None or box is None:
        logger.error("Failed to capture the popover screenshot")
        return False
    x, y, _w, _h = box
    # The popover opens below-left of the editor toolbar: crop around
    # it, keeping the toolbar row and the dialog's left edge visible.
    left = max(0, x - 240)
    top = max(0, y + 100)
    right = min(img.width, x + 380)
    bottom = min(img.height, y + 500)
    cropped = img.crop((left, top, right, bottom))
    output_path = utils.OUTPUT_DIR / get_output_name()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _save_webp_deterministic(cropped, Path(output_path))


@restore_config
def main():
    set_window_size(win, 2400, 1650)
    run_on_main_thread(_open_dialog)
    time.sleep(0.5)
    if TARGET.endswith(":variables"):
        _take_popover_screenshot()
    else:
        take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
