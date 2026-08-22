"""Screenshot: Unified machine configuration wizard pages.

The wizard's adaptive routing means many of the intermediate steps
either auto-skip or are only reachable via the live probe flow.
For screenshots we open the wizard, jump to the requested step,
and snapshot the rendered UI.

Targets (``config-wizard:<step>``):

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
import time

from utils import (
    get_target,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_screenshot,
    target_to_filename,
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


# Wizard step name is the target's leaf with ``-`` -> ``_``.
def wizard_step_for(target: str) -> str:
    return target.split(":")[-1].replace("-", "_")


@restore_config
def main():
    target = get_target("config-wizard:connect")
    if not target.startswith("config-wizard:"):
        logger.error("Unknown wizard screenshot target: %s", target)
        app.quit_idle()
        return
    step = wizard_step_for(target)

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
    take_screenshot(target_to_filename(target))

    time.sleep(0.25)

    def close_wizard():
        wizard.close()

    run_on_main_thread(close_wizard)
    app.quit_idle()


main()
