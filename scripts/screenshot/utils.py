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
from gi.repository import Adw, GLib
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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


def _save_png_deterministic(img: Image.Image, output_path: Path) -> bool:
    """
    Save a PNG image deterministically, only updating if content changed.

    Strips metadata and uses consistent compression to ensure identical
    screenshots produce identical files.
    """
    img = img.copy()
    img.info.clear()

    if output_path.exists():
        try:
            existing = Image.open(output_path)
            if existing.size == img.size and existing.mode == img.mode:
                if _images_visually_equal(existing, img):
                    logger.info(f"Screenshot unchanged: {output_path}")
                    return True
        except Exception as e:
            logger.debug(f"Comparison failed: {e}")

    pnginfo = PngInfo()
    img.save(output_path, format="PNG", compress_level=9, pnginfo=pnginfo)
    logger.info(f"Screenshot saved to {output_path}")
    return True


def _images_visually_equal(
    img1: Image.Image,
    img2: Image.Image,
    threshold: int = 5,
    max_different: float = 0.001,
) -> bool:
    """
    Compare two images using a perceptual heuristic.

    Args:
        img1: First image to compare.
        img2: Second image to compare.
        threshold: Minimum per-channel difference to count as changed (0-255).
        max_different: Maximum fraction of pixels that can differ (0.0-1.0).

    Returns:
        True if images are visually equal within tolerance.
    """
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    significant_diff = np.any(diff > threshold, axis=-1)
    different_pixels = np.sum(significant_diff)
    total_pixels = arr1.shape[0] * arr1.shape[1]

    return different_pixels / total_pixels <= max_different


def run_on_main_thread(func: Callable[[], T], timeout: float = 10.0) -> T:
    """
    Run a function on the main GTK thread and wait for completion.
    """
    result: List[T] = []
    exception: List[Optional[Exception]] = [None]
    done = Event()

    def wrapper() -> bool:
        try:
            result.append(func())
        except Exception as e:
            exception[0] = e
        finally:
            done.set()
        return GLib.SOURCE_REMOVE

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

    temp_path = output_path.with_suffix(".temp.png")

    for cmd_args, tool_name in SCREENSHOT_TOOLS:
        result = subprocess.run(
            [*cmd_args, str(temp_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            try:
                img = Image.open(temp_path)
                _save_png_deterministic(img, output_path)
                temp_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to process screenshot: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                return False

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
        _save_png_deterministic(cropped, output_path)
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

    run_on_main_thread(_load)
    logger.info(f"Loaded project: {project_name}")


def set_window_size(
    win: "MainWindow", width: int, height: int, timeout: float = 1.0
) -> bool:
    """
    Force the window to a specific size, handling maximized state.

    Args:
        win: The main window.
        width: Desired width in pixels.
        height: Desired height in pixels.
        timeout: Time to wait for size to be applied.

    Returns:
        True if size was successfully applied.
    """

    def _set_size() -> None:
        if win.is_maximized():
            win.unmaximize()
        win.set_default_size(width, height)
        win.set_size_request(width, height)

    run_on_main_thread(_set_size)

    actual_width = 0
    actual_height = 0
    start = time.time()
    while time.time() - start < timeout:
        actual_width = run_on_main_thread(lambda: win.get_width())
        actual_height = run_on_main_thread(lambda: win.get_height())
        if actual_width == width and actual_height == height:
            logger.info(f"Window size set to {width}x{height}")
            return True
        time.sleep(0.1)

    logger.warning(
        f"Window size not applied (expected {width}x{height}, "
        f"got {actual_width}x{actual_height})"
    )
    return False


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

    run_on_main_thread(_show)


def hide_panel(win: "MainWindow", panel_name: str) -> None:
    """Hide a UI panel."""
    show_panel(win, panel_name, visible=False)


def get_panel_state(win: "MainWindow", panel_name: str) -> bool:
    """Get the current visibility state of a panel."""

    def get_state() -> bool:
        action = win.action_manager.get_action(panel_name)
        state = action.get_state()
        if state is None:
            return False
        return state.get_boolean()

    return run_on_main_thread(get_state)


def save_panel_states(
    win: "MainWindow", panel_names: list[str]
) -> dict[str, bool]:
    """Save the current state of multiple panels."""
    return {name: get_panel_state(win, name) for name in panel_names}


def restore_panel_states(win: "MainWindow", states: dict[str, bool]) -> None:
    """Restore panel states from a saved dictionary."""
    for name, visible in states.items():
        show_panel(win, name, visible)


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

    dialog = run_on_main_thread(_open)
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

    dialog = run_on_main_thread(_open)
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

    dialog = run_on_main_thread(_open)
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

    return run_on_main_thread(_get)


def get_all_steps(win: "MainWindow") -> List["Step"]:
    """Get all steps across all layers."""

    def _get() -> List["Step"]:
        steps: List["Step"] = []
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                steps.extend(layer.workflow.steps)
        return steps

    return run_on_main_thread(_get)


def get_step_types(win: "MainWindow") -> List[str]:
    """Get all unique step types (typelabels) in the document."""

    def _get() -> List[str]:
        types: set = set()
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                for step in layer.workflow.steps:
                    types.add(step.typelabel.lower().replace(" ", "-"))
        return sorted(types)

    return run_on_main_thread(_get)


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

    return run_on_main_thread(_find)


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

    dialog = run_on_main_thread(_open)
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
        )
        dialog.set_initial_page("step-settings")
        dialog.set_default_size(600, 900)
        dialog.present()
        return dialog

    dialog = run_on_main_thread(_open)
    logger.info("Opened material test grid dialog")
    return dialog


def clear_window_subtitle(win: "MainWindow") -> None:
    """
    Clear the version subtitle from the main window for deterministic
    screenshots.
    """

    def _clear() -> None:
        title_widget = win.header_bar.get_title_widget()
        if isinstance(title_widget, Adw.WindowTitle):
            title_widget.set_subtitle("")

    run_on_main_thread(_clear)
