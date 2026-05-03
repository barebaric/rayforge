#!/usr/bin/env python3
"""Screenshot: Configuration wizard pages."""

import os
import time
import logging
from rayforge.uiscript import app, win
from rayforge.machine.device.profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)
from utils import (
    take_screenshot,
    run_on_main_thread,
    set_window_size,
)

logger = logging.getLogger(__name__)

FAKE_WARNINGS = [
    "Laser mode is not enabled ($32=0). Enable it for best results"
    " with laser cutters.",
]

FAKE_PROFILE = DeviceProfile(
    meta=DeviceMeta(
        name="Ortur Laser Master 2",
        description="Auto-configured via probe wizard",
    ),
    machine_config=MachineConfig(
        driver_config={
            "firmware_version": "1.1h",
            "rx_buffer_size": 128,
            "arc_tolerance": 0.002,
        },
        axis_extents=(400.0, 430.0),
        max_travel_speed=3000,
        max_cut_speed=1000,
        acceleration=500,
        home_on_start=True,
        single_axis_homing_enabled=True,
        heads=[{"max_power": 1000}],
    ),
    dialect_config={},
)


def main():
    target = os.environ.get("TARGET", "app-settings:machines:wizard:connect")

    set_window_size(win, 1400, 1000)
    time.sleep(0.25)

    from rayforge.ui_gtk.machine.config_wizard import ConfigWizard

    def open_wizard():
        wizard = ConfigWizard(transient_for=win)
        wizard.present()
        return wizard

    wizard = run_on_main_thread(open_wizard)
    time.sleep(0.5)

    if target == "app-settings:machines:wizard:connect":
        take_screenshot("app-settings-machines-wizard-connect.png")
    elif target == "app-settings:machines:wizard:review":
        run_on_main_thread(
            lambda: wizard._populate_review_page(
                FAKE_PROFILE, FAKE_WARNINGS
            )
        )
        run_on_main_thread(
            lambda: wizard._stack.set_visible_child_name("review")
        )
        time.sleep(0.5)
        take_screenshot("app-settings-machines-wizard-review.png")

    time.sleep(0.25)

    def close_wizard():
        wizard.close()

    run_on_main_thread(close_wizard)
    app.quit_idle()


main()
