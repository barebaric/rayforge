#!/usr/bin/env python3
"""
Script to start Rayforge and take screenshots.
Supports capturing the main window, machine dialog pages, and step
settings dialog pages.
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
        # Save to website-new static images directory
        output_dir = project_root / "website" / "static" / "images"
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
    screenshot_type: str = "main",
    machine_page: Optional[str] = None,
    step_page: Optional[str] = None,
    step_index: Optional[int] = None,
    engrave_mode: Optional[str] = None,
    settings_page: Optional[str] = None,
    recipe_page: Optional[str] = None,
):
    """
    Create a patched App class that activates simulation mode and takes a
    screenshot.

    Args:
        screenshot_type: Type of screenshot to take ("main", "machine",
            "step-settings", or "settings").
        machine_page: Machine dialog page to capture (for "machine" type).
        step_page: Step settings dialog page to capture (for "step-settings"
            type).
        step_index: Index of the step to capture (for "step-settings" type).
        engrave_mode: Engrave mode to set for engrave steps (e.g.,
            "POWER_MODULATION", "CONSTANT_POWER", "DITHER", "MULTI_PASS").
        settings_page: Application settings page to capture (for "settings"
            type).
        recipe_page: Recipe dialog page to capture (for "recipe-editor" type).

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
            self._step_page = step_page
            self._step_index = step_index
            self._engrave_mode = engrave_mode
            self._settings_page = settings_page
            self._recipe_page = recipe_page
            self._settings_dialog = None
            self._recipe_dialog = None

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

        def _schedule_settings_dialog_screenshot(self, win):
            """Schedule delayed actions for settings dialog screenshot."""
            from gi.repository import GLib

            logger.info(
                f"Scheduling settings dialog screenshot for page: "
                f"{self._settings_page}"
            )

            def open_dialog():
                """Open the settings dialog."""
                logger.info("Opening settings dialog...")
                if not self._open_settings_dialog(win):
                    logger.error("Failed to open settings dialog")
                    self._quit_application()
                    return False
                return False

            def take_screenshot_and_quit():
                """Take the screenshot and quit."""
                logger.info("Taking screenshot and quitting...")
                if not self._screenshot_taken:
                    output_name = f"application-{self._settings_page}.png"
                    take_screenshot(output_name)
                    self._screenshot_taken = True
                self._quit_application()
                return False

            # Schedule actions with delays
            GLib.timeout_add_seconds(3, open_dialog)
            GLib.timeout_add_seconds(5, take_screenshot_and_quit)

        def _schedule_recipe_editor_screenshot(self, win):
            """Schedule delayed actions for recipe editor dialog screenshot."""
            from gi.repository import GLib

            self._settings_dialog = None
            self._recipe_dialog = None

            def open_settings_dialog():
                """Open the settings dialog on recipes page."""
                logger.info("Opening settings dialog on recipes page...")
                if not self._open_settings_dialog(win):
                    logger.error("Failed to open settings dialog")
                    self._quit_application()
                    return False
                return False

            def open_recipe_editor():
                """Click add recipe button to open the editor."""
                logger.info("Opening recipe editor dialog...")
                if not self._open_recipe_editor_from_settings():
                    logger.error("Failed to open recipe editor dialog")
                    self._quit_application()
                    return False
                return False

            def switch_page():
                """Switch to the specified recipe dialog page."""
                if not self._recipe_dialog:
                    logger.error("Recipe dialog not open")
                    self._quit_application()
                    return False
                page = self._recipe_page or "general"
                logger.info(f"Switching to recipe page: {page}")
                try:
                    from gi.repository import Adw, Gtk

                    # First switch the view stack
                    self._recipe_dialog.view_stack.set_visible_child_name(page)

                    # Find the header bar through the dialog's widget tree
                    def find_header_bar(widget, depth=0):
                        """Recursively find HeaderBar in widget tree."""
                        if depth > 10:
                            return None
                        if isinstance(widget, Adw.HeaderBar):
                            return widget
                        child = widget.get_first_child()
                        while child:
                            result = find_header_bar(child, depth + 1)
                            if result:
                                return result
                            child = child.get_next_sibling()
                        return None

                    header_bar = find_header_bar(self._recipe_dialog)
                    if header_bar:
                        switcher_box = header_bar.get_title_widget()
                        if switcher_box and isinstance(switcher_box, Gtk.Box):
                            button_index = {
                                "general": 0,
                                "applicability": 1,
                                "settings": 2,
                            }
                            idx = button_index.get(page, 0)
                            button = switcher_box.get_first_child()
                            for i in range(idx):
                                if button is None:
                                    break
                                button = button.get_next_sibling()
                            if button and isinstance(button, Gtk.ToggleButton):
                                button.set_active(True)
                                logger.info(f"Activated button for {page}")
                except Exception as e:
                    logger.error(f"Failed to switch page: {e}")
                return False

            def take_screenshot_and_quit():
                """Take the screenshot and quit."""
                logger.info("Taking screenshot and quitting...")
                if not self._screenshot_taken:
                    page = self._recipe_page or "general"
                    output_name = f"recipe-editor-{page}.png"
                    take_screenshot(output_name)
                    self._screenshot_taken = True
                self._quit_application()
                return False

            # Schedule actions with delays
            GLib.timeout_add_seconds(3, open_settings_dialog)
            GLib.timeout_add_seconds(5, open_recipe_editor)
            GLib.timeout_add_seconds(6, switch_page)
            GLib.timeout_add_seconds(8, take_screenshot_and_quit)

        def _schedule_step_dialog_screenshot(self, win):
            """Schedule delayed actions for step settings dialog screenshot."""
            from gi.repository import GLib

            self._step_dialog = None

            def open_dialog():
                """Open the step settings dialog."""
                self._step_dialog = self._open_step_dialog(win)
                if not self._step_dialog:
                    logger.error("Failed to open step dialog")
                    self._quit_application()
                    return False
                return False

            def switch_page_and_screenshot():
                """Switch to the specified page and take screenshot."""
                # Page is set via initial_page parameter when dialog is created
                # So we don't need to manually switch pages
                logger.info(f"Using page: {self._step_page}")
                if not self._switch_step_page(win):
                    logger.error("Failed to switch step page")
                    self._quit_application()
                    return False
                return False

            def take_screenshot_and_quit():
                """Take the screenshot and quit."""
                if not self._screenshot_taken:
                    step_type = self._get_step_type_name(win)
                    mode_suffix = (
                        f"-{self._engrave_mode.lower()}"
                        if self._engrave_mode and step_type == "engrave"
                        else ""
                    )
                    output_name = (
                        f"step-{step_type}{mode_suffix}-{self._step_page}.png"
                    )
                    take_screenshot(output_name)
                    self._screenshot_taken = True
                self._quit_application()
                return False

            # Schedule actions with delays
            GLib.timeout_add_seconds(3, open_dialog)
            GLib.timeout_add_seconds(4, switch_page_and_screenshot)
            GLib.timeout_add_seconds(6, take_screenshot_and_quit)

        def do_activate(self):
            """Activate the application and set up the window."""
            from rayforge.core.vectorization_spec import TraceSpec
            import mimetypes
            from rayforge.context import get_context

            # Initialize the full context before creating the window
            get_context().initialize_full_context()

            win = MainWindow(application=self)
            win.set_default_size(2400, 1650)
            logger.info("Window size set to 2400x1650")

            # For machine, settings and step-settings screenshots, don't load
            # a file. Use load_project_from_path for .ryp files,
            # load_file_from_path for others.
            skip_args = (
                "--screenshot-type",
                "--machine-page",
                "--settings-page",
            )
            if (
                self._screenshot_type
                not in ("machine", "settings", "recipe-editor")
                and self.args.filenames
                and self.args.filenames[0] not in skip_args
            ):
                for filename in self.args.filenames:
                    if filename.startswith("--"):
                        continue
                    file_path = Path(filename)
                    if file_path.suffix == ".ryp":
                        win.doc_editor.file.load_project_from_path(file_path)
                    else:
                        mime_type, _ = mimetypes.guess_type(filename)
                        vector_spec = (
                            None if self.args.direct_vector else TraceSpec()
                        )
                        win.doc_editor.file.load_file_from_path(
                            filename=file_path,
                            mime_type=mime_type,
                            vectorization_spec=vector_spec,
                        )
            win.present()

            # Schedule the screenshot and simulation activation
            logger.info(f"Scheduling screenshot type: {self._screenshot_type}")
            if self._screenshot_type == "machine":
                self._schedule_machine_dialog_screenshot(win)
            elif self._screenshot_type == "settings":
                self._schedule_settings_dialog_screenshot(win)
            elif self._screenshot_type == "step-settings":
                self._schedule_step_dialog_screenshot(win)
            elif self._screenshot_type == "recipe-editor":
                self._schedule_recipe_editor_screenshot(win)
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

        def _open_settings_dialog(self, win) -> bool:
            """Open the settings dialog on the specified page."""
            try:
                from rayforge.ui_gtk.settings.settings_dialog import (
                    SettingsWindow,
                )

                page = self._settings_page or "general"
                if self._screenshot_type == "recipe-editor":
                    page = "recipes"
                dialog = SettingsWindow(initial_page=page)
                dialog.set_transient_for(win)
                dialog.present()
                self._settings_dialog = dialog
                msg = f"Opened settings dialog on page: {page}"
                logger.info(msg)
                return True

            except Exception as e:
                logger.error(f"Failed to open settings dialog: {e}")
                return False

        def _open_recipe_editor_from_settings(self) -> bool:
            """Open the recipe editor from the settings dialog."""
            try:
                if not self._settings_dialog:
                    logger.error("Settings dialog not open")
                    return False

                from rayforge.core.recipe import Recipe
                from rayforge.ui_gtk.doceditor.edit_recipe_dialog import (
                    AddEditRecipeDialog,
                )

                recipe = Recipe(name="3mm Plywood Cut")
                recipe.description = (
                    "A recipe for cutting 3mm plywood with a diode laser"
                )

                self._recipe_dialog = AddEditRecipeDialog(
                    parent=self._settings_dialog,
                    recipe=recipe,
                )
                self._recipe_dialog.set_default_size(700, 800)
                self._recipe_dialog.present()
                logger.info("Opened recipe editor dialog")
                return True

            except Exception as e:
                logger.error(f"Failed to open recipe editor dialog: {e}")
                return False

        def _open_step_dialog(self, win):
            """Open the step settings dialog for the specified step."""
            try:
                step = self._get_step_by_index(win)
                if not step:
                    logger.error(
                        f"Could not find step at index {self._step_index}"
                    )
                    return None

                from rayforge.pipeline.producer.raster import DepthMode

                if self._engrave_mode and step.typelabel == _("Engrave"):
                    producer_dict = step.opsproducer_dict
                    params = producer_dict.setdefault("params", {})
                    try:
                        mode = DepthMode[self._engrave_mode]
                        params["depth_mode"] = mode.name
                        msg = f"Set engrave mode to: {self._engrave_mode}"
                        logger.info(msg)
                    except KeyError:
                        logger.warning(
                            f"Invalid engrave mode: {self._engrave_mode}"
                        )

                from rayforge.ui_gtk.doceditor.step_settings_dialog import (
                    StepSettingsDialog,
                )

                dialog = StepSettingsDialog(
                    editor=win.doc_editor,
                    step=step,
                    transient_for=win,
                    initial_page=self._step_page,
                )
                dialog.set_default_size(600, 900)
                dialog.present()
                msg = f"Opened step dialog for step: {step.name}"
                logger.info(msg)
                return dialog

            except Exception as e:
                logger.error(f"Failed to open step dialog: {e}")
                return None

        def _get_step_by_index(self, win):
            """Get a step by its index in the document."""
            try:
                step_index = self._step_index or 0
                for layer in win.doc_editor.doc.layers:
                    if layer.workflow and layer.workflow.steps:
                        if step_index < len(layer.workflow.steps):
                            return layer.workflow.steps[step_index]
                        step_index -= len(layer.workflow.steps)
                return None
            except Exception as e:
                logger.error(f"Error finding step: {e}")
                return None

        def _get_step_type_name(self, win) -> str:
            """Get the type name of the current step for the filename."""
            step = self._get_step_by_index(win)
            if step:
                step_type = step.typelabel.lower().replace(" ", "-")
                return step_type
            return "unknown"

        def _switch_step_page(self, win) -> bool:
            """Switch to the specified page in the step settings dialog."""
            # Page is set via initial_page parameter when dialog is created
            # So we don't need to manually switch pages
            logger.info(f"Using page: {self._step_page}")
            return True

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
    try:
        if context.config_mgr:
            context.config_mgr.save()
            logger.info("Saved config.")
    except RuntimeError:
        logger.debug("Config manager already shut down.")

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
        choices=[
            "main",
            "machine",
            "settings",
            "step-settings",
            "recipe-editor",
        ],
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
    parser.add_argument(
        "--settings-page",
        default="general",
        choices=[
            "general",
            "machines",
            "materials",
            "recipes",
            "packages",
        ],
        help="Settings dialog page to capture (default: general)",
    )
    parser.add_argument(
        "--step-page",
        default="step-settings",
        choices=["step-settings", "post-processing"],
        help="Step settings dialog page to capture (default: step-settings)",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Index of the step to capture (default: 0)",
    )
    parser.add_argument(
        "--engrave-mode",
        default=None,
        choices=[
            "POWER_MODULATION",
            "CONSTANT_POWER",
            "DITHER",
            "MULTI_PASS",
        ],
        help="Engrave mode to set for engrave steps",
    )
    parser.add_argument(
        "--recipe-page",
        default="general",
        choices=["general", "applicability", "settings"],
        help="Recipe dialog page to capture (default: general)",
    )

    args = parser.parse_args()

    # Set logging level based on the command-line argument
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Application starting with log level {args.loglevel.upper()}")
    logger.info(f"Parsed screenshot_type: {args.screenshot_type}")
    logger.info(f"Parsed settings_page: {args.settings_page}")

    # Set up application with or without test file based on screenshot type
    # For machine and step-settings screenshots, don't include
    # default test file
    include_test_file = args.screenshot_type == "main"
    setup_application(include_test_file)

    # Initialize logging and imports
    initialize_logging_and_imports()

    # Create and run the patched application
    PatchedApp = create_patched_app_class(
        screenshot_type=args.screenshot_type,
        machine_page=args.machine_page,
        step_page=args.step_page,
        step_index=args.step_index,
        engrave_mode=args.engrave_mode,
        settings_page=args.settings_page,
        recipe_page=args.recipe_page,
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
