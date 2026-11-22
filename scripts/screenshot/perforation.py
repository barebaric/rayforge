"""
Screenshot: Perforation post-processor applied to a contour.

Usage: pixi run screenshot operations:perforation
"""

import logging
import time

from utils import (
    clear_window_subtitle,
    load_project,
    restore_config,
    run_on_main_thread,
    set_window_size,
    take_region_screenshot,
    wait_for_settled,
)

from rayforge.core.workpiece import WorkPiece
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)

TARGET_PX_PER_MM = 26.0
CROP_SIZE = (1200, 800)


def _content_boxes() -> list[tuple[float, float, float, float]]:
    items = [
        item
        for layer in win.doc_editor.doc.layers
        for item in layer.children
        if isinstance(item, WorkPiece)
    ]
    items.extend(win.doc_editor.doc.stock_items)
    return [item.bbox for item in items if item.size != (0.0, 0.0)]


def frame_closeup() -> tuple[float, float] | None:
    """Zoom in on the workpiece so the dash pattern is clearly visible.

    Centers the view on a point inside the workpiece and returns the
    focused world position, or None when there is nothing to frame.
    """

    def _frame() -> tuple[float, float] | None:
        boxes = _content_boxes()
        if not boxes:
            logger.warning("No content to frame")
            return None
        min_x = min(b[0] for b in boxes)
        min_y = min(b[1] for b in boxes)
        max_x = max(b[0] + b[2] for b in boxes)
        max_y = max(b[1] + b[3] for b in boxes)

        work_boxes = [
            b
            for layer in win.doc_editor.doc.layers
            for b in [
                item.bbox
                for item in layer.children
                if isinstance(item, WorkPiece)
            ]
        ]
        if work_boxes:
            box = work_boxes[0]
        else:
            box = (min_x, min_y, max_x - min_x, max_y - min_y)

        focus_x = box[0] + 0.79 * box[2]
        focus_y = box[1] + 0.5 * box[3]

        scale_x, scale_y = win.surface.get_view_scale()
        px_per_mm = max((scale_x + scale_y) / 2.0, 1e-9)
        zoom = win.surface.zoom_level * TARGET_PX_PER_MM / px_per_mm
        win.surface.set_zoom(max(zoom, 0.01))

        for _pass in range(2):
            pos = win.surface.view_transform.transform_point(focus_x, focus_y)
            dx = win.surface.get_width() / 2.0 - pos[0]
            dy = win.surface.get_height() / 2.0 - pos[1]
            scale_x, scale_y = win.surface.get_view_scale()
            win.surface.set_pan(
                win.surface.pan_x_mm - dx / scale_x,
                win.surface.pan_y_mm + dy / scale_y,
            )
        return focus_x, focus_y

    return run_on_main_thread(_frame)


def closeup_region(focus: tuple[float, float]) -> tuple[int, int, int, int]:
    """Return the main-window region around the focused point."""

    def _region() -> tuple[int, int, int, int]:
        pos = win.surface.view_transform.transform_point(*focus)
        ox, oy = win.surface.translate_coordinates(win, 0, 0)
        cx, cy = ox + pos[0], oy + pos[1]
        w, h = CROP_SIZE
        return (
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2),
        )

    return run_on_main_thread(_region)


@restore_config
def main():
    set_window_size(win, 2400, 1650)

    load_project(win, "perforation.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=30):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return
    logger.info("Document settled")

    focus = frame_closeup()
    if focus is None:
        app.quit_idle()
        return
    time.sleep(0.5)

    clear_window_subtitle(win)
    time.sleep(0.25)

    logger.info("Taking close-up screenshot: operations-perforation.webp")
    take_region_screenshot(win, closeup_region(focus))

    time.sleep(0.25)
    app.quit_idle()


main()
