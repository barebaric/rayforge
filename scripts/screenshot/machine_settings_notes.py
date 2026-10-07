"""Screenshot: Machine settings - Notes page and editor dialog."""

import logging
import subprocess
import time

from utils import (
    get_target,
    open_machine_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
)

from rayforge.context import get_context
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

PAGE = "notes"
TARGET = get_target(f"machine-settings:{PAGE}")

SAMPLE_NOTES = (
    "## Maintenance log\n"
    "\n"
    "- Lens cleaned and realigned (2026-09-12)\n"
    "- X belt re-tensioned after 40 hours\n"
    "- Air assist: always on for acrylic\n"
)


def stage_notes() -> None:
    """Put sample notes on the machine so the page shows content."""

    def _stage() -> None:
        machine = get_context().config.machine
        assert machine is not None
        machine.user_notes = SAMPLE_NOTES

    run_on_main_thread(_stage)


def open_editor(dialog):
    """Open the Markdown editor dialog from the notes page."""

    def _open():
        page = dialog.content_stack.get_child_by_name(PAGE)
        page._on_edit_clicked(None)
        return page._editor_dialog

    return run_on_main_thread(_open)


def activate_window(title: str) -> None:
    """Raise the dialog window so it is the topmost for the capture."""
    try:
        subprocess.run(
            ["xdotool", "search", "--name", title, "windowactivate"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        time.sleep(0.25)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("xdotool not available, relying on window focus")


@restore_config
def main():
    time.sleep(0.25)
    stage_notes()
    dialog = open_machine_settings(win, PAGE)
    time.sleep(0.25)

    if TARGET.endswith(":editor"):
        editor = open_editor(dialog)
        time.sleep(0.5)
        activate_window("Edit My Notes")
        take_screenshot()

        def _close():
            editor.close()

        run_on_main_thread(_close)
        time.sleep(0.25)
    else:
        take_screenshot()

    time.sleep(0.25)
    app.quit_idle()


main()
