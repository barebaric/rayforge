#!/usr/bin/env python3
"""
Script to start the RayForge application and take a screenshot of the main
window.
The screenshot is saved to docs/ss-main.png.
"""

import logging
import sys
import time
import threading
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def take_screenshot() -> bool:
    """
    Take a screenshot of the main window using gnome-screenshot or import.

    Returns:
        bool: True if screenshot was taken successfully, False otherwise.
    """
    import subprocess

    try:
        output_path = project_root / "docs" / "ss-main.png"
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


def setup_application():
    """Set up the application with necessary imports and configurations."""
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Adw", "1")

    # Override sys.argv to pass the file path
    test_file = str(project_root / "tests/image/png/color.png")
    sys.argv = ["rayforge", test_file]

    # Set environment variables for Linux
    import os

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Import the app module
    import rayforge.app

    return rayforge.app


def create_patched_app_class():
    """
    Create a patched App class that activates simulation mode and takes a
    screenshot.

    Returns:
        The patched App class.
    """
    from gi.repository import GLib, Adw
    from rayforge.mainwindow import MainWindow

    class PatchedApp(Adw.Application):
        def __init__(self, args):
            super().__init__(application_id="org.rayforge.rayforge")
            self.set_accels_for_action("win.quit", ["<Ctrl>Q"])
            self.args = args
            self._screenshot_taken = False

        def do_activate(self):
            """Activate the application and set up the window."""
            from rayforge.core.vectorization_config import TraceConfig
            import mimetypes

            win = MainWindow(application=self)
            win.set_default_size(1600, 1100)
            logger.info("Window size set to 1600x1100")

            # Load the test file
            if self.args.filenames:
                for filename in self.args.filenames:
                    mime_type, _ = mimetypes.guess_type(filename)
                    vector_config = (
                        None if self.args.direct_vector else TraceConfig()
                    )
                    win.doc_editor.file.load_file_from_path(
                        filename=Path(filename),
                        mime_type=mime_type,
                        vector_config=vector_config,
                    )
            win.present()

            # Schedule the screenshot and simulation activation
            self._schedule_delayed_actions(win)

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

        def _quit_application(self):
            """Quit the application."""
            GLib.idle_add(lambda: sys.exit(0))

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
    from rayforge.workbench import canvas3d

    canvas3d.initialize()


def initialize_configuration():
    """Initialize configuration managers."""
    import rayforge.shared.tasker
    import rayforge.config

    rayforge.config.initialize_managers()


def shutdown_application():
    """Perform graceful shutdown of the application."""
    import asyncio
    import rayforge.config
    import rayforge.shared.tasker

    logger.info("Application exiting.")

    async def shutdown_async():
        logger.info("Starting graceful async shutdown...")
        if rayforge.config.machine_mgr:
            await rayforge.config.machine_mgr.shutdown()
        logger.info("Async shutdown complete.")

    loop = rayforge.shared.tasker.task_mgr._loop
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

    if rayforge.config.config_mgr:
        rayforge.config.config_mgr.save()
    logger.info("Saved config.")

    rayforge.shared.tasker.task_mgr.shutdown()
    logger.info("Task manager shut down.")


def main():
    """Main function to start the app and take a screenshot."""
    # Store the original main function
    rayforge_app = setup_application()

    # Create a new main function that patches the App class
    def patched_main():
        """Patched main function that activates simulation mode."""
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

        args = parser.parse_args()

        # Set logging level based on the command-line argument
        log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        logger.info(
            f"Application starting with log level {args.loglevel.upper()}"
        )

        # Initialize logging and imports
        initialize_logging_and_imports()

        # Initialize configuration
        initialize_configuration()

        # Create and run the patched application
        PatchedApp = create_patched_app_class()
        app = PatchedApp(args)
        exit_code = app.run(None)

        # Shutdown sequence
        shutdown_application()

        return exit_code

    # Replace the main function
    rayforge_app.main = patched_main

    # Start the application
    try:
        patched_main()
    except SystemExit:
        # This is expected when we quit the app
        pass


if __name__ == "__main__":
    main()
