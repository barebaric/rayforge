"""
Screenshot: Machine settings - Advanced page and the bidirectional
scan offset table.

Usage:
  pixi run screenshot machine-settings:advanced
  pixi run screenshot machine-settings:advanced:bidir-offset-table
"""

import logging
import time

from PIL import Image
from utils import (
    DECOR_MARGIN_PX,
    DECOR_OFFSET_Y_PX,
    OUTPUT_DIR,
    _add_window_decorations,
    _save_webp_deterministic,
    get_target,
    open_machine_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
    target_to_filename,
)

from rayforge.context import get_context
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "advanced"

TABLE_ROWS = [(1000, 0.10), (3000, 0.25), (6000, 0.45), (10000, 0.80)]


def stage_offset_table() -> None:
    """Stage a representative offset table on the active machine."""

    def _stage():
        machine = get_context().config.machine
        machine.set_bidir_offset_table(TABLE_ROWS)

    run_on_main_thread(_stage)
    time.sleep(0.25)


def _expand_first_row(dialog):
    """Show the speed and offset fields of the table's first row."""

    def _expand():
        page = dialog.content_stack.get_child_by_name(PAGE)
        group = page.bidir_offset_group
        if group.entry_rows:
            group.entry_rows[0].set_expanded(True)

    return _expand


def enlarge_dialog(dialog) -> None:
    """Grow the dialog so the advanced page fits without scrolling."""

    def _resize():
        dialog.set_default_size(800, 1240)

    run_on_main_thread(_resize)
    time.sleep(0.5)


def capture_offset_table(dialog) -> None:
    """Crop the bidirectional scan offset group from the dialog shot."""
    output_path = OUTPUT_DIR / target_to_filename(get_target(PAGE))
    if not output_path.exists():
        logger.error("Dialog screenshot is missing for cropping")
        return

    def _group_box():
        page = dialog.content_stack.get_child_by_name(PAGE)
        group = page.bidir_offset_group
        pos = group.translate_coordinates(dialog, 0, 0)
        return pos, group.get_width(), group.get_height()

    box = run_on_main_thread(_group_box)
    if box is None:
        logger.error("Could not locate the offset table group")
        return
    (x, y), w, h = box
    pad = 14
    ox = DECOR_MARGIN_PX
    oy = DECOR_MARGIN_PX + DECOR_OFFSET_Y_PX
    img = Image.open(output_path)
    cropped = _add_window_decorations(
        img.crop(
            (
                max(x - pad + ox, 0),
                max(y - pad + oy, 0),
                min(x + w + pad + ox, img.width),
                min(y + h + pad + oy, img.height),
            )
        )
    )
    _save_webp_deterministic(cropped, output_path)


@restore_config
def main():
    target = get_target(f"machine-settings:{PAGE}")
    time.sleep(0.25)
    stage_offset_table()
    dialog = open_machine_settings(win, PAGE)
    time.sleep(0.25)
    enlarge_dialog(dialog)
    if target.endswith(":bidir-offset-table"):
        run_on_main_thread(_expand_first_row(dialog))
        time.sleep(0.25)
        take_screenshot()
        time.sleep(0.25)
        capture_offset_table(dialog)
    else:
        take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
