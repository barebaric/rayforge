"""Screenshot: remote machine (raydriver) setup.

Targets:

* ``config-wizard:remote``    — the wizard connection page for a
  remote machine (daemon endpoint + device picker).
* ``machine-settings:remote`` — machine settings (General) for a
  remote machine.
"""

import logging
import time

from utils import (
    get_target,
    open_machine_settings,
    restore_config,
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


REMOTE_PROFILE = DeviceProfile(
    meta=DeviceMeta(
        name="Sculpfun iCube",
        description="Remote machine hosted by a raydriver daemon",
    ),
    machine_config=MachineConfig(
        driver="RaydriverRemoteDriver",
        driver_args={
            "remote_host": "pi.local",
            "remote_port": 8765,
            "device": "/dev/ttyUSB0",
            "_raydriver_protocol": "grbl",
        },
        axis_extents=(400.0, 400.0),
        origin=Origin.BOTTOM_LEFT,
        max_travel_speed=3000,
        max_cut_speed=1000,
        acceleration=500,
        heads=[{"head_class": "LaserHead", "max_power": 1000}],
    ),
    dialect_config={},
)


@restore_config
def main():
    target = get_target("config-wizard:remote")
    set_window_size(win, 1400, 1000)
    time.sleep(0.25)

    if target == "machine-settings:remote":
        from rayforge.context import get_context

        def setup_machine():
            context = get_context()
            machine = REMOTE_PROFILE.create_machine(context)
            context.machine_mgr.add_machine(machine)
            context.machine_mgr.set_active_machine(machine)

        run_on_main_thread(setup_machine)
        time.sleep(0.5)
        open_machine_settings(win, "general")
        time.sleep(0.5)
        take_screenshot()
        time.sleep(0.25)
        app.quit_idle()
        return

    from rayforge.ui_gtk.machine.unified_wizard import UnifiedWizard

    def open_wizard():
        wizard = UnifiedWizard(transient_for=win)
        wizard.present()
        wizard.profile = REMOTE_PROFILE
        return wizard

    wizard = run_on_main_thread(open_wizard)
    time.sleep(0.5)
    run_on_main_thread(lambda: wizard._navigate_to("connect"))
    time.sleep(0.75)
    take_screenshot()
    time.sleep(0.25)
    run_on_main_thread(wizard.close)
    app.quit_idle()


main()
