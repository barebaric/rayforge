"""Screenshot: Color Rule editor dialog."""

import logging
import time

from utils import (
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


@restore_config
def main():
    set_window_size(win, 1200, 900)

    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    from rayforge.core.color_preset import ColorPreset
    from rayforge.ui_gtk.settings.color_presets_page import ColorPresetDialog

    def add_presets_and_open():
        preset_mgr = win.doc_editor.context.color_preset_mgr
        preset_mgr.add_preset(
            ColorPreset(color="#ff0000", step_type="ContourStep", label="Cut")
        )

        dialog = ColorPresetDialog(parent=win)
        dialog.present()
        return dialog

    dialog = run_on_main_thread(add_presets_and_open)

    time.sleep(1.0)

    take_screenshot()

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
