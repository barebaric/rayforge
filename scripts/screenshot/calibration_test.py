"""
Screenshot: the Test Settings of a calibration test in the settings
dialog of its layer.

Usage: pixi run screenshot material-test
       pixi run screenshot interval-test
       pixi run screenshot focus-test
"""

import logging
import time

from utils import (
    clear_window_subtitle,
    create_calibration_test,
    get_target,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
)

from rayforge.ui_gtk.doceditor.layer_settings_dialog import LayerSettingsDialog
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

ACTIONS = {
    "material-test": "material_test",
    "interval-test": "interval_test",
    "focus-test": "focus_test",
}


@restore_config
def main():
    set_window_size(win, 2400, 1650)
    target = get_target("material-test")
    create_calibration_test(win, ACTIONS[target])
    time.sleep(1.5)
    clear_window_subtitle(win)

    layer = win.doc_editor.doc.active_layer
    run_on_main_thread(
        lambda: LayerSettingsDialog(
            layer, transient_for=win, editor=win.doc_editor
        ).present()
    )
    time.sleep(1.5)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
