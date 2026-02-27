#!/usr/bin/env python3
"""
Screenshot: Main window in simulation mode.

Usage: pixi run screenshot main:simulation
"""

import time
import logging
from gi.repository import GLib
from rayforge.uiscript import app, win
from utils import (
    load_project,
    wait_for_settled,
    show_panel,
    hide_panel,
    save_panel_states,
    restore_panel_states,
    take_screenshot,
    clear_window_subtitle,
    set_window_size,
    run_on_main_thread,
)

logger = logging.getLogger(__name__)

PANELS = ["toggle_control_panel", "toggle_gcode_preview"]


def _wait_for_simulation_components(timeout: int = 10) -> bool:
    """Wait for simulation components to be initialized and ops loaded."""

    def get_step_count():
        overlay = win.simulator_cmd.simulation_overlay
        return overlay.timeline.get_step_count() if overlay else 0

    for i in range(timeout):
        overlay = run_on_main_thread(
            lambda: win.simulator_cmd.simulation_overlay
        )
        if overlay:
            count = run_on_main_thread(get_step_count)
            if count > 0:
                logger.info("Simulation components initialized with ops")
                time.sleep(0.5)
                return True
            logger.info(
                f"Simulation overlay ready, waiting for ops... ({i + 1}s)"
            )
        else:
            logger.info(f"Waiting for simulation components... ({i + 1}s)")
        time.sleep(1)

    logger.warning("Simulation components not initialized")
    return False


def activate_simulation_mode() -> bool:
    """
    Activate simulation mode.

    Returns:
        True if activation was successful.
    """

    def _activate() -> None:
        action = win.action_manager.get_action("simulate_mode")
        action.set_state(GLib.Variant.new_boolean(False))
        action.activate(GLib.Variant.new_boolean(True))

    run_on_main_thread(_activate)
    logger.info("Simulation mode activated")

    time.sleep(2)

    is_sim = run_on_main_thread(lambda: win.surface.is_simulation_mode())
    if is_sim:
        return _wait_for_simulation_components()

    if win.simulator_cmd:
        run_on_main_thread(lambda: win.simulator_cmd._enter_mode())
        return _wait_for_simulation_components()

    return False


def advance_simulation(fraction: float = 0.8) -> bool:
    """
    Advance the simulation to a given fraction of the job (0.0 to 1.0).

    Args:
        fraction: How far into the job to advance (default 80%).

    Returns:
        True if successful.
    """
    if not win.simulator_cmd or not win.simulator_cmd.preview_controls:
        logger.warning("Simulation controls not available")
        return False

    preview_controls = win.simulator_cmd.preview_controls

    def _advance() -> bool:
        num_lines = preview_controls.num_gcode_lines
        if num_lines <= 0:
            logger.warning("No G-code lines in simulation")
            return False
        target_line = int(num_lines * fraction)
        preview_controls.set_playback_position(target_line)
        logger.info(
            f"Advanced simulation to line {target_line}/{num_lines} "
            f"({fraction * 100:.0f}%)"
        )
        return True

    result = run_on_main_thread(_advance)
    time.sleep(0.25)
    return result


def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "contour.ryp")
    logger.info("Waiting for document to settle...")

    if not wait_for_settled(win, timeout=60):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    logger.info("Setting up simulation mode")

    if not activate_simulation_mode():
        logger.error("Failed to activate simulation mode")
        app.quit_idle()
        return

    advance_simulation(fraction=0.8)

    saved_states = save_panel_states(win, PANELS)
    hide_panel(win, "toggle_control_panel")
    show_panel(win, "toggle_gcode_preview", True)

    time.sleep(0.25)

    clear_window_subtitle(win)
    logger.info("Taking screenshot: main-simulation.png")
    take_screenshot("main-simulation.png")

    restore_panel_states(win, saved_states)

    time.sleep(0.25)
    app.quit_idle()


main()
