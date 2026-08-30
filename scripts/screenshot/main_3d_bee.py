"""
Screenshot: Main window in 3D mode with the bee project.

Usage: pixi run screenshot main:3d-bee
"""

import logging
import time

from gi.repository import GLib
from utils import (
    clear_window_subtitle,
    hide_panel,
    load_project,
    restore_config,
    restore_panel_states,
    run_on_main_thread,
    save_panel_states,
    seek_3d_playback,
    set_window_size,
    show_bottom_tab,
    show_panel,
    take_cropped_screenshot,
    wait_for_3d_rendered,
    wait_for_settled,
    wcs,
)

from rayforge.context import get_context
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

PANELS = ["show_3d_view", "toggle_bottom_panel"]
MACHINE_ID = "a2c84b52-eaed-40d5-91ba-3b54370e5c3c"


def wait_for_active_machine(machine_id: str, timeout: float = 10.0) -> bool:
    """Wait until the machine with the given id is the active one."""
    start = time.time()
    while time.time() - start < timeout:
        machine = run_on_main_thread(lambda: get_context().config.machine)
        if machine and machine.id == machine_id:
            return True
        time.sleep(0.1)
    return False


def activate_machine(machine_id: str) -> None:
    """Make the machine with the given id the active machine."""

    def _switch() -> None:
        machine_mgr = get_context().machine_mgr
        machine = machine_mgr.get_machine_by_id(machine_id)
        if machine is None:
            raise ValueError(f"No machine found with ID {machine_id}")
        machine_mgr.set_active_machine(machine)

    run_on_main_thread(_switch)
    if not wait_for_active_machine(machine_id):
        raise TimeoutError(
            f"Machine {machine_id} did not become active in time"
        )
    logger.info(f"Active machine set to {machine_id}")


def hide_grid(win) -> None:
    """Turn off the grid via the win.show_grid action."""

    def _hide() -> None:
        action = win.action_manager.get_action("show_grid")
        action.change_state(GLib.Variant.new_boolean(False))

    run_on_main_thread(_hide)


def set_theme(theme: str) -> None:
    """Switch the application theme (e.g. 'light' or 'dark')."""

    def _apply() -> None:
        get_context().config.set_theme(theme)

    run_on_main_thread(_apply)
    time.sleep(0.5)


@restore_config
def main():
    set_window_size(win, 2000, 1375)

    logger.info("Activating machine...")
    try:
        activate_machine(MACHINE_ID)
    except (ValueError, TimeoutError) as e:
        logger.error(f"Failed to activate machine: {e}")
        app.quit_idle()
        return

    logger.info("Switching to light mode...")
    set_theme("light")

    load_project(win, "bee.ryp")
    logger.info("Waiting for document to settle...")

    if not wait_for_settled(win, timeout=20):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Setting up 3D mode")

    saved_states = save_panel_states(win, PANELS)

    with wcs(win, "G54"):
        show_panel(win, "show_3d_view", True)
        hide_panel(win, "toggle_bottom_panel")
        show_bottom_tab(win, "gcode")
        hide_grid(win)

        logger.info(
            "Waiting for pipeline to settle after 3D view activation..."
        )
        if not wait_for_settled(win, timeout=30):
            logger.error("Pipeline did not settle after 3D view activation")
            app.quit_idle()
            return

        logger.info("Waiting for 3D scene to render...")
        if not wait_for_3d_rendered(win, timeout=15):
            logger.error("3D scene did not render in time")
            app.quit_idle()
            return

        seek_3d_playback(win, 1.0)

        clear_window_subtitle(win)
        logger.info("Taking screenshot: main-3d-bee.webp")
        take_cropped_screenshot(
            from_top=350,
            from_bottom=400,
            from_left=450,
            from_right=450,
        )

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
