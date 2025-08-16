from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..doceditor.layout import (
    BboxAlignLeftStrategy,
    BboxAlignCenterStrategy,
    BboxAlignRightStrategy,
    BboxAlignTopStrategy,
    BboxAlignMiddleStrategy,
    BboxAlignBottomStrategy,
    LayoutStrategy,
    SpreadHorizontallyStrategy,
    SpreadVerticallyStrategy,
    PixelPerfectLayoutStrategy,
)
from ..shared.tasker import task_mgr
from ..undo import ChangePropertyCommand

if TYPE_CHECKING:
    from ..mainwindow import MainWindow
    from ..shared.tasker.task import Task


logger = logging.getLogger(__name__)


def _execute_layout_task(
    win: "MainWindow", strategy: LayoutStrategy, transaction_name: str
):
    """
    A synchronous helper that configures and launches a background layout task.

    The actual model mutation happens in the `when_done` callback, which is
    guaranteed to run on the main GTK thread.
    """

    def when_done(task: "Task"):
        """
        This callback runs on the main thread after the task finishes.
        It safely applies the calculated changes to the document.
        """
        if task.get_status() != "completed":
            logger.error(
                f"Layout task '{transaction_name}' did not complete "
                f"successfully. Status: {task.get_status()}"
            )
            # You could add a toast notification here if desired.
            return

        deltas = task.result()
        if not deltas:
            return  # No changes to apply

        with win.doc.history_manager.transaction(transaction_name) as t:
            for wp, delta_matrix in deltas.items():
                old_matrix = wp.matrix.copy()
                new_matrix = delta_matrix @ old_matrix
                cmd = ChangePropertyCommand(
                    target=wp,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    # This simple coroutine just runs the calculation in the background
    # and returns the result.
    async def layout_coro(context):
        return strategy.calculate_deltas(context)

    # Launch the coroutine and attach the main-thread callback.
    task_mgr.add_coroutine(
        layout_coro,
        when_done=when_done,
        key=f"layout-{transaction_name}",  # key to prevent concurrent runs
    )


def center_horizontally(win: "MainWindow"):
    """Action handler for centering workpieces horizontally."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    surface_w, _ignore = win.surface.get_size()
    strategy = BboxAlignCenterStrategy(
        selected_wps, surface_width_mm=surface_w
    )
    _execute_layout_task(win, strategy, _("Center Horizontally"))


def center_vertically(win: "MainWindow"):
    """Action handler for centering workpieces vertically."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    _ignore, surface_h = win.surface.get_size()
    strategy = BboxAlignMiddleStrategy(
        selected_wps, surface_height_mm=surface_h
    )
    _execute_layout_task(win, strategy, _("Center Vertically"))


def align_left(win: "MainWindow"):
    """Action handler for aligning workpieces to the left."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    strategy = BboxAlignLeftStrategy(selected_wps)
    _execute_layout_task(win, strategy, _("Align Left"))


def align_right(win: "MainWindow"):
    """Action handler for aligning workpieces to the right."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    surface_w, _ignore = win.surface.get_size()
    strategy = BboxAlignRightStrategy(selected_wps, surface_width_mm=surface_w)
    _execute_layout_task(win, strategy, _("Align Right"))


def align_top(win: "MainWindow"):
    """Action handler for aligning workpieces to the top."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    _ignore, surface_h = win.surface.get_size()
    strategy = BboxAlignTopStrategy(selected_wps, surface_height_mm=surface_h)
    _execute_layout_task(win, strategy, _("Align Top"))


def align_bottom(win: "MainWindow"):
    """Action handler for aligning workpieces to the bottom."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    strategy = BboxAlignBottomStrategy(selected_wps)
    _execute_layout_task(win, strategy, _("Align Bottom"))


def spread_horizontally(win: "MainWindow"):
    """Action handler for spreading workpieces horizontally."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    strategy = SpreadHorizontallyStrategy(selected_wps)
    _execute_layout_task(win, strategy, _("Spread Horizontally"))


def spread_vertically(win: "MainWindow"):
    """Action handler for spreading workpieces vertically."""
    selected_wps = win.surface.get_selected_workpieces()
    if not selected_wps:
        return

    strategy = SpreadVerticallyStrategy(selected_wps)
    _execute_layout_task(win, strategy, _("Spread Vertically"))


def layout_pixel_perfect(win: "MainWindow"):
    """Action handler for the pixel-perfect packing layout."""
    workpieces_to_layout = win.surface.get_selected_workpieces()

    # If nothing is selected, apply to all workpieces in the document.
    if not workpieces_to_layout:
        workpieces_to_layout = win.doc.all_workpieces

    if not workpieces_to_layout:
        return

    strategy = PixelPerfectLayoutStrategy(
        workpieces=workpieces_to_layout,
        margin_mm=0.5,
        resolution_px_per_mm=8.0,
        allow_rotation=True,
    )
    _execute_layout_task(win, strategy, _("Auto Layout"))
