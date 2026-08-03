#!/usr/bin/env python3
"""Screenshot: Unified wizard Step 3 (Connection) and Step 10 (Review).

The wizard's adaptive routing means many of the intermediate steps
either auto-skip or are only reachable via the live probe flow.
For screenshots we open the wizard, jump to a representative step,
and snapshot the rendered UI.

Targets:
    app-settings:machines:wizard:connect    — Step 3 (connection)
    app-settings:machines:wizard:review      — Step 10 (review)
"""

import logging
import os
import time

from utils import (
    run_on_main_thread,
    set_window_size,
    take_screenshot,
)

from rayforge.machine.device.profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)
from rayforge.machine.models.machine import Origin
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


FAKE_PROFILE = DeviceProfile(
    meta=DeviceMeta(
        name="Ortur Laser Master 2",
        vendor="Ortur",
        model="Aufero Laser Master 2",
        description="Auto-configured via unified wizard",
    ),
    machine_config=MachineConfig(
        driver="GrblSerialDriver",
        driver_args={"port": "/dev/ttyUSB0", "baud_rate": 115200},
        axis_extents=(400.0, 430.0),
        origin=Origin.BOTTOM_LEFT,
        max_travel_speed=3000,
        max_cut_speed=1000,
        acceleration=500,
        home_on_start=True,
        single_axis_homing_enabled=True,
        heads=[{"head_class": "LaserHead", "max_power": 1000}],
    ),
    dialect_config={},
)


def main():
    target = os.environ.get("TARGET", "app-settings:machines:wizard:connect")

    set_window_size(win, 1400, 1000)
    time.sleep(0.25)

    from rayforge.ui_gtk.machine.unified_wizard import UnifiedWizard

    def open_wizard():
        wizard = UnifiedWizard(transient_for=win)
        wizard.present()

        # Pre-load a known-profile state and route to the requested step.
        wizard.profile = FAKE_PROFILE
        return wizard

    wizard = run_on_main_thread(open_wizard)
    time.sleep(0.5)

    if target == "app-settings:machines:wizard:connect":
        run_on_main_thread(lambda: wizard._navigate_to("connection"))
        take_screenshot("app-settings-machines-wizard-connect.png")
    elif target == "app-settings:machines:wizard:review":
        run_on_main_thread(lambda: wizard._navigate_to("review"))
        time.sleep(0.5)
        take_screenshot("app-settings-machines-wizard-review.png")

    time.sleep(0.25)

    def close_wizard():
        wizard.close()

    run_on_main_thread(close_wizard)
    app.quit_idle()


main()
