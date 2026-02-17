#!/usr/bin/env python3
"""
Script to start Rayforge and take screenshots.
Supports capturing the main window and machine dialog pages.
"""

import logging
import sys
import time
import threading
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
# __file__ is scripts/media/take_screenshot.py, so we need to go up two levels
# to get to the project root (rayforge directory)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def take_screenshot(output_name: str = "ss-main.png") -> bool:
    """
    Take a screenshot using gnome-screenshot or import.

    Args:
        output_name: Name of the output file.

    Returns:
        bool: True if screenshot was taken successfully, False otherwise.
    """
    import subprocess

    try:
        output_dir = project_root / "website" / "content" / "docs" / "images"
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Wait for the window to fully render
        time.sleep(2)

        # Try gnome-screenshot first
        result = subprocess.run(
            ["gnome-screenshot", "-w", "-f", str(output_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info(f"Screenshot saved to {output_path}")
            return True

        # If gnome-screenshot fails, try import (ImageMagick)
        result = subprocess.run(
            ["import", "-window", "root", str(output_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info(f"Screenshot saved to {output_path}")
            return True

        logger.error("Failed to take screenshot with available tools")
        return False

    except Exception as e:
        logger.error(f"Error taking screenshot: {e}")
        return False


def wait_for_operations(window, timeout: int = 30) -> bool:
    """
    Wait for operations to be generated.

    Args:
        window: The main window instance.
        timeout: Maximum time to wait in seconds.

    Returns:
        bool: True if operations were generated, False otherwise.
    """
    if not (
        hasattr(window, "doc_editor")
        and hasattr(window.doc_editor, "ops_generator")
    ):
        logger.error("Could not find ops_generator")
        return False

    generator = window.doc_editor.ops_generator
    wait_count = 0

    while generator.is_busy and wait_count < timeout:
        time.sleep(1)
        wait_count += 1
        logger.info(f"Waiting for operations generation... ({wait_count}s)")

    ops = window._aggregate_ops_for_3d_view()
    logger.info(f"Aggregated {len(ops)} operations for simulation")

    return len(ops) > 0


def activate_simulation_mode_via_action(window) -> bool:
    """
    Activate simulation mode using the action system.

    Args:
        window: The main window instance.

    Returns:
        bool: True if activation was successful, False otherwise.
    """
    from gi.repository import GLib

    if not hasattr(window, "action_manager"):
        logger.error("Window has no action_manager")
        return False

    action = window.action_manager.get_action("simulate_mode")
    if not action:
        logger.error("Could not find simulate_mode action")
        return False

    try:
        # First, set the state to False to ensure it's properly initialized
        action.set_state(GLib.Variant.new_boolean(False))
        # Now activate with True
        action.activate(GLib.Variant.new_boolean(True))
        logger.info("Simulation mode activated via action")
        return True
    except Exception as e:
        logger.warning(f"Failed to activate simulation mode via action: {e}")
        return False


def activate_simulation_mode_directly(window) -> bool:
    """
    Activate simulation mode by directly calling _enter_mode.

    Args:
        window: The main window instance.

    Returns:
        bool: True if activation was successful, False otherwise.
    """
    if not hasattr(window, "simulator_cmd") or not window.simulator_cmd:
        logger.error("Window has no simulator_cmd")
        return False

    try:
        window.simulator_cmd._enter_mode()
        logger.info("Simulation mode activated via direct call")
        return True
    except Exception as e:
        logger.error(f"Failed to activate simulation mode directly: {e}")
        return False


def wait_for_simulation_components(window, timeout: int = 10) -> bool:
    """
    Wait for simulation components to be fully initialized.

    Args:
        window: The main window instance.
        timeout: Maximum time to wait in seconds.

    Returns:
        bool: True if components are initialized, False otherwise.
    """
    if not hasattr(window, "simulator_cmd") or not window.simulator_cmd:
        return False

    wait_count = 0
    while (
        not window.simulator_cmd.simulation_overlay
        or not window.simulator_cmd.preview_controls
    ) and wait_count < timeout:
        time.sleep(1)
        wait_count += 1
        logger.info(f"Waiting for simulation components... ({wait_count}s)")

    if (
        window.simulator_cmd.simulation_overlay
        and window.simulator_cmd.preview_controls
    ):
        logger.info("Simulation components fully initialized")
        # Wait a bit more for visual stabilization
        time.sleep(2)
        return True

    logger.warning("Simulation components not fully initialized")
    return False


def is_simulation_mode_active(window) -> bool:
    """
    Check if simulation mode is active.

    Args:
        window: The main window instance.

    Returns:
        bool: True if simulation mode is active, False otherwise.
    """
    if not (
        hasattr(window, "surface")
        and hasattr(window.surface, "is_simulation_mode")
    ):
        return False
    return window.surface.is_simulation_mode()


def activate_simulation_mode(window) -> bool:
    """
    Activate simulation mode with fallback mechanisms.

    Args:
        window: The main window instance.

    Returns:
        bool: True if activation was successful, False otherwise.
    """
    # Try to activate via action first
    if activate_simulation_mode_via_action(window):
        time.sleep(3)
        if is_simulation_mode_active(window):
            return wait_for_simulation_components(window)

    # Fallback to direct activation
    logger.warning("Simulation mode may not be active, trying fallback")
    if activate_simulation_mode_directly(window):
        return wait_for_simulation_components(window)

    return False


def setup_application(include_test_file: bool = True):
    """
    Set up the application with necessary imports and configurations.

    Args:
        include_test_file: Whether to include the test file in sys.argv.
    """
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Adw", "1")

    # Override sys.argv to pass the file path (only for main screenshots)
    if include_test_file:
        test_file = str(project_root / "tests/image/png/color.png")
        sys.argv = ["rayforge", test_file]
    else:
        sys.argv = ["rayforge"]

    # Set environment variables for Linux
    import os

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Import the app module
    import rayforge.app

    return rayforge.app


def create_patched_app_class(
    screenshot_type: str = "main", machine_page: Optional[str] = None
):
    """
    Create a patched App class that activates simulation mode and takes a
    screenshot.

    Args:
        screenshot_type: Type of screenshot to take ("main" or "machine").
        machine_page: Machine dialog page to capture (for "machine" type).

    Returns:
        The patched App class.
    """
    from gi.repository import Adw
    from rayforge.ui_gtk.mainwindow import MainWindow

    class PatchedApp(Adw.Application):
        def __init__(self, args):
            super().__init__(application_id="org.rayforge.rayforge")
            self.set_accels_for_action("win.quit", ["<Ctrl>Q"])
            self.args = args
            self._screenshot_taken = False
            self._screenshot_type = screenshot_type
            self._machine_page = machine_page

        def _schedule_delayed_actions(self, win):
            """Schedule delayed actions for simulation activation and
            screenshot."""

            def delayed_actions():
                # Wait for the window to be fully shown and file to be
                # processed
                time.sleep(10)

                # Wait for operations to be generated
                if not wait_for_operations(win):
                    logger.error("Failed to generate operations")
                    self._quit_application()
                    return

                # Activate simulation mode
                if not activate_simulation_mode(win):
                    logger.error("Failed to activate simulation mode")
                    self._quit_application()
                    return

                # Take the screenshot
                if not self._screenshot_taken:
                    take_screenshot()
                    self._screenshot_taken = True

                # Quit the app after taking the screenshot
                self._quit_application()

            # Run in a thread to not block the UI
            thread = threading.Thread(target=delayed_actions, daemon=True)
            thread.start()

        def _schedule_machine_dialog_screenshot(self, win):
            """Schedule delayed actions for machine dialog screenshot."""
            from gi.repository import GLib

            def open_dialog():
                """Open the machine settings dialog."""
                if not self._open_machine_dialog(win):
                    logger.error("Failed to open machine dialog")
                    self._quit_application()
                    return False
                return False

            def take_screenshot_and_quit():
                """Take the screenshot and quit."""
                if not self._screenshot_taken:
                    output_name = f"machine-{self._machine_page}.png"
                    take_screenshot(output_name)
                    self._screenshot_taken = True
                self._quit_application()
                return False

            # Schedule actions with delays
            GLib.timeout_add_seconds(3, open_dialog)
            GLib.timeout_add_seconds(5, take_screenshot_and_quit)

        def do_activate(self):
            """Activate the application and set up the window."""
            from rayforge.core.vectorization_spec import TraceSpec
            import mimetypes
            from rayforge.context import get_context

            # Initialize the full context before creating the window
            get_context().initialize_full_context()

            win = MainWindow(application=self)
            win.set_default_size(1600, 1100)
            logger.info("Window size set to 1600x1100")

            # For machine screenshots, don't load a file
            if (
                self._screenshot_type != "machine"
                and self.args.filenames
                and self.args.filenames[0]
                not in ("--screenshot-type", "--machine-page")
            ):
                for filename in self.args.filenames:
                    if filename.startswith("--"):
                        continue
                    mime_type, _ = mimetypes.guess_type(filename)
                    vector_spec = (
                        None if self.args.direct_vector else TraceSpec()
                    )
                    win.doc_editor.file.load_file_from_path(
                        filename=Path(filename),
                        mime_type=mime_type,
                        vectorization_spec=vector_spec,
                    )
            win.present()

            # Schedule the screenshot and simulation activation
            if self._screenshot_type == "machine":
                self._schedule_machine_dialog_screenshot(win)
            else:
                self._schedule_delayed_actions(win)

        def _open_machine_dialog(self, win) -> bool:
            """Open the machine settings dialog on the specified page."""
            from rayforge.context import get_context

            try:
                config = get_context().config
                machine = config.machine
                if not machine:
                    logger.error("No default machine found")
                    return False

                from rayforge.ui_gtk.machine.settings_dialog import (
                    MachineSettingsDialog,
                )

                dialog = MachineSettingsDialog(
                    machine=machine,
                    transient_for=win,
                    initial_page=self._machine_page,
                )
                dialog.present()
                msg = f"Opened machine dialog on page: {self._machine_page}"
                logger.info(msg)
                return True

            except Exception as e:
                logger.error(f"Failed to open machine dialog: {e}")
                return False

        def _quit_application(self):
            """Quit the application."""
            logger.info("Quitting application...")
            self.quit()

    return PatchedApp


def initialize_logging_and_imports():
    """Initialize logging and necessary imports."""
    import logging
    import cairo

    # Set logging level
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Application starting with log level INFO")

    # Print PyCairo version
    logger.info(f"PyCairo version: {cairo.version}")

    # Register the standalone 'cairo' module
    import gi

    gi.require_foreign("cairo")
    gi.require_version("cairo", "1.0")
    gi.require_version("Gtk", "4.0")
    gi.require_version("GdkPixbuf", "2.0")

    # Initialize the 3D canvas module
    from rayforge.ui_gtk import canvas3d

    canvas3d.initialize()


def shutdown_application():
    """Perform graceful shutdown of the application."""
    import asyncio
    import rayforge.shared.tasker
    from rayforge.context import get_context

    logger.info("Application exiting.")
    context = get_context()

    async def shutdown_async():
        logger.info("Starting graceful async shutdown...")
        # The context now handles shutting down all its owned managers
        await context.shutdown()
        logger.info("Async shutdown complete.")

    loop = rayforge.shared.tasker.task_mgr.loop
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(shutdown_async(), loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    else:
        logger.warning(
            "Task manager loop not running, skipping async shutdown."
        )

    # Save configuration
    if context.config_mgr:
        context.config_mgr.save()
        logger.info("Saved config.")

    rayforge.shared.tasker.task_mgr.shutdown()
    logger.info("Task manager shut down.")


def main():
    """Main function to start the app and take a screenshot."""
    import argparse
    from rayforge import __version__

    parser = argparse.ArgumentParser(
        description="A GCode generator for laser cutters."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "filenames",
        help="Paths to one or more input SVG or image files.",
        nargs="*",
    )
    parser.add_argument(
        "--direct-vector",
        action="store_true",
        help="Import SVG files as direct vectors instead of tracing them.",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--screenshot-type",
        default="main",
        choices=["main", "machine"],
        help="Type of screenshot to take (default: main)",
    )
    parser.add_argument(
        "--machine-page",
        default="general",
        choices=[
            "general",
            "hardware",
            "advanced",
            "gcode",
            "hooks-macros",
            "device",
            "laser",
            "camera",
            "maintenance",
        ],
        help="Machine dialog page to capture (default: general)",
    )

    args = parser.parse_args()

    # Set logging level based on the command-line argument
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Application starting with log level {args.loglevel.upper()}")

    # Set up application with or without test file based on screenshot type
    include_test_file = args.screenshot_type != "machine"
    setup_application(include_test_file)

    # Initialize logging and imports
    initialize_logging_and_imports()

    # Create and run the patched application
    PatchedApp = create_patched_app_class(
        screenshot_type=args.screenshot_type,
        machine_page=args.machine_page,
    )
    app = PatchedApp(args)
    exit_code = app.run(None)

    # Shutdown sequence
    shutdown_application()

    return exit_code


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # This is expected when we quit the app
        pass
