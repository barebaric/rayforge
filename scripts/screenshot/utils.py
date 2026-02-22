#!/usr/bin/env python3
"""
Shared utilities for screenshot scripts.

These scripts are designed to be run via `rayforge --uiscript`.
Scripts run in a background thread, so UI operations use
GLib.idle_add for thread safety.
"""

import logging
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Callable, List, Optional, Tuple, TypeVar, TYPE_CHECKING

from gi.repository import GLib

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.ui_gtk.doceditor.edit_recipe_dialog import (
        AddEditRecipeDialog,
    )
    from rayforge.ui_gtk.doceditor.step_settings_dialog import (
        StepSettingsDialog,
    )
    from rayforge.ui_gtk.machine.settings_dialog import MachineSettingsDialog
    from rayforge.ui_gtk.mainwindow import MainWindow
    from rayforge.ui_gtk.settings.settings_dialog import SettingsWindow

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "website" / "static" / "screenshots"
TESTS_DIR = PROJECT_ROOT / "tests"

SCREENSHOT_TOOLS = [
    (["gnome-screenshot", "-w", "-f"], "gnome-screenshot"),
    (["import", "-window", "root"], "ImageMagick import"),
]

T = TypeVar("T")


def _run_on_main_thread(func: Callable[[], T], timeout: float = 10.0) -> T:
    """
    Run a function on the main GTK thread and wait for completion.
    """
    result: List[T] = []
    exception: List[Optional[Exception]] = [None]
    done = Event()

    def wrapper() -> None:
        try:
            result.append(func())
        except Exception as e:
            exception[0] = e
        finally:
            done.set()

    GLib.idle_add(wrapper)
    if done.wait(timeout=timeout):
        if exception[0]:
            raise exception[0]
        return result[0]
    raise TimeoutError(f"Function did not complete within {timeout}s")


def take_screenshot(output_name: str) -> bool:
    """
    Take a screenshot of the active window.

    Args:
        output_name: Filename (saved to website/static/images/).

    Returns:
        True if screenshot was saved successfully.
    """
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time.sleep(0.5)

    for cmd_args, tool_name in SCREENSHOT_TOOLS:
        result = subprocess.run(
            [*cmd_args, str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Screenshot saved to {output_path} using {tool_name}")
            return True

    logger.error("Failed to take screenshot with available tools")
    return False


def take_cropped_screenshot(
    output_name: str,
    *,
    from_bottom: Optional[int] = None,
    from_top: Optional[int] = None,
    from_left: Optional[int] = None,
    from_right: Optional[int] = None,
) -> bool:
    """
    Take a screenshot of the active window and crop it.

    Args:
        output_name: Filename (saved to OUTPUT_DIR).
        from_bottom: Crop this many pixels from the bottom.
        from_top: Crop this many pixels from the top.
        from_left: Crop this many pixels from the left.
        from_right: Crop this many pixels from the right.

    Returns:
        True if screenshot was saved successfully.
    """
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time.sleep(0.5)

    temp_path = output_path.with_suffix(".temp.png")

    success = False
    for cmd_args, tool_name in SCREENSHOT_TOOLS:
        result = subprocess.run(
            [*cmd_args, str(temp_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            success = True
            break

    if not success:
        logger.error("Failed to take screenshot with available tools")
        return False

    try:
        from PIL import Image

        img = Image.open(temp_path)
        width, height = img.size

        left = from_left or 0
        top = from_top or 0
        right = width - (from_right or 0)
        bottom = height - (from_bottom or 0)

        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path)
        temp_path.unlink()
        logger.info(f"Cropped screenshot saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to crop screenshot: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def wait_for_settled(win: "MainWindow", timeout: float = 30.0) -> bool:
    """
    Wait for the document to finish processing.

    Returns:
        True if settled within timeout.
    """
    return win.doc_editor.wait_until_settled_sync(timeout=timeout)


def load_project(win: "MainWindow", project_name: str) -> None:
    """Load a project file from the tests/assets directory."""
    project_path = TESTS_DIR / "assets" / project_name
    if not project_path.exists():
        raise FileNotFoundError(f"Project not found: {project_path}")

    def _load() -> None:
        win.doc_editor.file.load_project_from_path(project_path)

    _run_on_main_thread(_load)
    logger.info(f"Loaded project: {project_name}")


def show_panel(
    win: "MainWindow", panel_name: str, visible: bool = True
) -> None:
    """
    Show or hide a UI panel.

    Args:
        panel_name: Action name (e.g., "toggle_control_panel").
        visible: True to show, False to hide.
    """

    def _show() -> None:
        action = win.action_manager.get_action(panel_name)
        action.change_state(GLib.Variant.new_boolean(visible))

    _run_on_main_thread(_show)


def hide_panel(win: "MainWindow", panel_name: str) -> None:
    """Hide a UI panel."""
    show_panel(win, panel_name, visible=False)


def activate_simulation_mode(win: "MainWindow") -> bool:
    """
    Activate simulation mode.

    Returns:
        True if activation was successful.
    """

    def _activate() -> None:
        action = win.action_manager.get_action("simulate_mode")
        action.set_state(GLib.Variant.new_boolean(False))
        action.activate(GLib.Variant.new_boolean(True))

    _run_on_main_thread(_activate)
    logger.info("Simulation mode activated")

    time.sleep(2)

    is_sim = _run_on_main_thread(lambda: win.surface.is_simulation_mode())
    if is_sim:
        return _wait_for_simulation_components(win)

    if win.simulator_cmd:
        _run_on_main_thread(lambda: win.simulator_cmd._enter_mode())
        return _wait_for_simulation_components(win)

    return False


def _wait_for_simulation_components(
    win: "MainWindow", timeout: int = 10
) -> bool:
    """Wait for simulation components to be initialized."""
    for i in range(timeout):
        overlay = _run_on_main_thread(
            lambda: win.simulator_cmd.simulation_overlay
        )
        if overlay:
            logger.info("Simulation components initialized")
            time.sleep(1)
            return True
        logger.info(f"Waiting for simulation components... ({i + 1}s)")
        time.sleep(1)

    logger.warning("Simulation components not initialized")
    return False


def open_machine_settings(
    win: "MainWindow", page: str = "general"
) -> "MachineSettingsDialog":
    """Open machine settings dialog on the specified page."""
    from rayforge.context import get_context
    from rayforge.ui_gtk.machine.settings_dialog import MachineSettingsDialog

    def _open() -> "MachineSettingsDialog":
        config = get_context().config
        machine = config.machine
        if not machine:
            raise ValueError("No machine configured")
        dialog = MachineSettingsDialog(
            machine=machine,
            transient_for=win,
            initial_page=page,
        )
        dialog.present()
        return dialog

    dialog = _run_on_main_thread(_open)
    logger.info(f"Opened machine settings on page: {page}")
    return dialog


def open_app_settings(
    win: "MainWindow", page: str = "general"
) -> "SettingsWindow":
    """Open app settings dialog on the specified page."""
    from rayforge.ui_gtk.settings.settings_dialog import SettingsWindow

    def _open() -> "SettingsWindow":
        dialog = SettingsWindow(initial_page=page)
        dialog.set_transient_for(win)
        dialog.present()
        return dialog

    dialog = _run_on_main_thread(_open)
    logger.info(f"Opened app settings on page: {page}")
    return dialog


def open_step_settings(
    win: "MainWindow", step_index: int = 0, page: str = "step-settings"
) -> "StepSettingsDialog":
    """Open step settings dialog for the step at the given index."""
    from rayforge.ui_gtk.doceditor.step_settings_dialog import (
        StepSettingsDialog,
    )

    step = get_step_by_index(win, step_index)
    if not step:
        raise ValueError(f"Step at index {step_index} not found")

    def _open() -> "StepSettingsDialog":
        dialog = StepSettingsDialog(
            editor=win.doc_editor,
            step=step,
            transient_for=win,
        )
        dialog.set_default_size(600, 900)
        dialog.present()
        dialog.set_initial_page(page)
        return dialog

    dialog = _run_on_main_thread(_open)
    logger.info(f"Opened step settings for: {step.name} on page: {page}")
    return dialog


def get_step_by_index(win: "MainWindow", index: int) -> Optional["Step"]:
    """Get a step by its index across all layers."""

    def _get() -> Optional["Step"]:
        step_index = index
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                if step_index < len(layer.workflow.steps):
                    return layer.workflow.steps[step_index]
                step_index -= len(layer.workflow.steps)
        return None

    return _run_on_main_thread(_get)


def get_all_steps(win: "MainWindow") -> List["Step"]:
    """Get all steps across all layers."""

    def _get() -> List["Step"]:
        steps: List["Step"] = []
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                steps.extend(layer.workflow.steps)
        return steps

    return _run_on_main_thread(_get)


def get_step_types(win: "MainWindow") -> List[str]:
    """Get all unique step types (typelabels) in the document."""

    def _get() -> List[str]:
        types: set = set()
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                for step in layer.workflow.steps:
                    types.add(step.typelabel.lower().replace(" ", "-"))
        return sorted(types)

    return _run_on_main_thread(_get)


def find_step_by_type(
    win: "MainWindow", step_type: str
) -> Tuple[Optional["Step"], int]:
    """Find first step matching the given type."""

    def _find() -> Tuple[Optional["Step"], int]:
        normalized = step_type.lower().replace(" ", "-")
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                for i, step in enumerate(layer.workflow.steps):
                    if step.typelabel.lower().replace(" ", "-") == normalized:
                        return step, i
        return None, -1

    return _run_on_main_thread(_find)


def open_recipe_editor(
    win: "MainWindow", page: str = "general"
) -> "AddEditRecipeDialog":
    """Open recipe editor dialog from app settings."""
    from rayforge.core.recipe import Recipe
    from rayforge.ui_gtk.doceditor.edit_recipe_dialog import (
        AddEditRecipeDialog,
    )

    settings_dialog = open_app_settings(win, "recipes")
    time.sleep(0.5)

    recipe = Recipe(name="3mm Plywood Cut")
    recipe.description = "A recipe for cutting 3mm plywood with a diode laser"

    def _open() -> "AddEditRecipeDialog":
        dialog = AddEditRecipeDialog(
            parent=settings_dialog,
            recipe=recipe,
        )
        dialog.set_default_size(700, 800)
        dialog.present()

        button_map = {
            "general": dialog.btn_general,
            "applicability": dialog.btn_applicability,
            "settings": dialog.btn_settings,
        }
        if page in button_map:
            button_map[page].set_active(True)
        return dialog

    dialog = _run_on_main_thread(_open)
    logger.info(f"Opened recipe editor on page: {page}")
    return dialog


def open_material_test(win: "MainWindow") -> "StepSettingsDialog":
    """Open material test grid dialog."""
    from rayforge.pipeline.steps import create_material_test_step
    from rayforge.ui_gtk.doceditor.step_settings_dialog import (
        StepSettingsDialog,
    )

    def _open() -> "StepSettingsDialog":
        step = create_material_test_step(win.doc_editor.context)
        step.name = "Material Test Grid"
        dialog = StepSettingsDialog(
            editor=win.doc_editor,
            step=step,
            transient_for=win,
            initial_page="step-settings",
        )
        dialog.set_default_size(600, 900)
        dialog.present()
        return dialog

    dialog = _run_on_main_thread(_open)
    logger.info("Opened material test grid dialog")
    return dialog
