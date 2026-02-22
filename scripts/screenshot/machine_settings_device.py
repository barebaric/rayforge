#!/usr/bin/env python3
"""Screenshot: Machine settings - Device page."""

import time
import logging
from rayforge.uiscript import app, win
from rayforge.machine.driver.grbl_util import get_grbl_setting_varsets
from utils import open_machine_settings, take_screenshot

logger = logging.getLogger(__name__)
PAGE = "device"


def inject_fake_device_settings(dialog):
    """Inject fake device settings into the device settings page."""
    from utils import _run_on_main_thread

    device_page = dialog.content_stack.get_child_by_name("device")

    def _inject():
        var_sets = get_grbl_setting_varsets()

        device_page._is_busy = False
        device_page._clear_error_state()
        device_page._not_connected_warning_dismissed = True

        device_page.machine.driver.supports_settings = True

        device_page._rebuild_settings_widgets(var_sets)
        device_page._update_ui_state()

    _run_on_main_thread(_inject)


def main():
    time.sleep(0.25)
    dialog = open_machine_settings(win, PAGE)

    inject_fake_device_settings(dialog)

    time.sleep(0.25)
    take_screenshot(f"machine-{PAGE}.png")
    time.sleep(0.25)
    app.quit_idle()


main()
