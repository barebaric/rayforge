"""Screenshot: App settings - Color Rules page."""

import logging
import time

from utils import (
    open_app_settings,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    wait_for_settled,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "color_presets"


@restore_config
def main():
    set_window_size(win, 1200, 900)

    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    # Add a few sample color presets so the list shows content.
    from rayforge.core.color_preset import ColorPreset

    context = win.doc_editor.context
    preset_mgr = context.color_preset_mgr

    def add_presets():
        preset_mgr.add_preset(
            ColorPreset(color="#ff0000", step_type="ContourStep", label="Cut")
        )
        preset_mgr.add_preset(
            ColorPreset(
                color="#0000ff", step_type="EngraveStep", label="Engrave"
            )
        )
        preset_mgr.add_preset(
            ColorPreset(
                color="#00ff00", step_type="ContourStep", label="Score"
            )
        )

    run_on_main_thread(add_presets)

    time.sleep(0.3)

    logger.info("Opening Color Rules settings page...")
    open_app_settings(win, PAGE)

    time.sleep(0.5)

    take_screenshot()

    time.sleep(0.25)

    app.quit_idle()


main()
