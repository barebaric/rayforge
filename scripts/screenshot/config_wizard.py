#!/usr/bin/env python3
"""Screenshot: Unified machine configuration wizard pages.

The wizard's adaptive routing means many of the intermediate steps
either auto-skip or are only reachable via the live probe flow.
For screenshots we open the wizard, jump to the requested step,
and snapshot the rendered UI.

Targets (``app-settings:machines:wizard:<step>``):

* ``profile``      — Step 1 (pick source)
* ``controller``   — Step 2 (choose controller)
* ``connect``      — Step 3 (connection)
* ``probe``        — Step 4 (discover device)
* ``ai-provider``  — Step 5 (AI provider)
* ``ai-lookup``    — Step 6 (AI spec lookup)
* ``hardware``     — Step 7 (hardware)
* ``head``         — Step 8 (head)
* ``rotary``       — Step 9 (rotary module)
* ``camera``       — Step 10 (cameras)
* ``review``       — Step 11 (review & name)
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

# Target -> (wizard step name, output filename).
PAGES = {
    "app-settings:machines:wizard:profile": (
        "profile",
        "app-settings-machines-wizard-profile.png",
    ),
    "app-settings:machines:wizard:controller": (
        "controller",
        "app-settings-machines-wizard-controller.png",
    ),
    "app-settings:machines:wizard:connect": (
        "connection",
        "app-settings-machines-wizard-connect.png",
    ),
    "app-settings:machines:wizard:probe": (
        "probe",
        "app-settings-machines-wizard-probe.png",
    ),
    "app-settings:machines:wizard:ai-provider": (
        "ai_provider",
        "app-settings-machines-wizard-ai-provider.png",
    ),
    "app-settings:machines:wizard:ai-lookup": (
        "ai_lookup",
        "app-settings-machines-wizard-ai-lookup.png",
    ),
    "app-settings:machines:wizard:hardware": (
        "hardware",
        "app-settings-machines-wizard-hardware.png",
    ),
    "app-settings:machines:wizard:head": (
        "head",
        "app-settings-machines-wizard-head.png",
    ),
    "app-settings:machines:wizard:rotary": (
        "rotary",
        "app-settings-machines-wizard-rotary.png",
    ),
    "app-settings:machines:wizard:camera": (
        "camera",
        "app-settings-machines-wizard-camera.png",
    ),
    "app-settings:machines:wizard:review": (
        "review",
        "app-settings-machines-wizard-review.png",
    ),
}


def main():
    target = os.environ.get("TARGET", "app-settings:machines:wizard:connect")
    if target not in PAGES:
        logger.error("Unknown wizard screenshot target: %s", target)
        app.quit_idle()
        return
    step, output = PAGES[target]

    set_window_size(win, 1400, 1000)
    time.sleep(0.25)

    from rayforge.ui_gtk.machine.unified_wizard import UnifiedWizard

    def open_wizard():
        wizard = UnifiedWizard(transient_for=win)
        wizard.present()

        # Pre-load a known-profile state and route to the requested
        # step.
        wizard.profile = FAKE_PROFILE
        return wizard

    wizard = run_on_main_thread(open_wizard)
    time.sleep(0.5)

    if step == "probe":
        # The probe page auto-starts a live probe on entry. Build the
        # page first and suppress that so the screenshot shows the
        # idle "Probe Now" state rather than a connection failure.
        def suppress_auto_probe():
            page = wizard._get_page("probe")
            page._probed = True

        run_on_main_thread(suppress_auto_probe)

    run_on_main_thread(lambda: wizard._navigate_to(step))
    time.sleep(0.5)
    take_screenshot(output)

    time.sleep(0.25)

    def close_wizard():
        wizard.close()

    run_on_main_thread(close_wizard)
    app.quit_idle()


main()
