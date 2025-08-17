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
            for item, delta_matrix in deltas.items():
                old_matrix = item.matrix.copy()
                new_matrix = delta_matrix @ old_matrix
                cmd = ChangePropertyCommand(
                    target=item,
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
    """Action handler for centering selected items horizontally."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    surface_w, _ignore = win.surface.get_size_mm()
    strategy = BboxAlignCenterStrategy(
        selected_items, surface_width_mm=surface_w
    )
    _execute_layout_task(win, strategy, _("Center Horizontally"))


def center_vertically(win: "MainWindow"):
    """Action handler for centering selected items vertically."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    _ignore, surface_h = win.surface.get_size_mm()
    strategy = BboxAlignMiddleStrategy(
        selected_items, surface_height_mm=surface_h
    )
    _execute_layout_task(win, strategy, _("Center Vertically"))


def align_left(win: "MainWindow"):
    """Action handler for aligning selected items to the left."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    strategy = BboxAlignLeftStrategy(selected_items)
    _execute_layout_task(win, strategy, _("Align Left"))


def align_right(win: "MainWindow"):
    """Action handler for aligning selected items to the right."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    surface_w, _ignore = win.surface.get_size_mm()
    strategy = BboxAlignRightStrategy(
        selected_items, surface_width_mm=surface_w
    )
    _execute_layout_task(win, strategy, _("Align Right"))


def align_top(win: "MainWindow"):
    """Action handler for aligning selected items to the top."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    _ignore, surface_h = win.surface.get_size_mm()
    strategy = BboxAlignTopStrategy(
        selected_items, surface_height_mm=surface_h
    )
    _execute_layout_task(win, strategy, _("Align Top"))


def align_bottom(win: "MainWindow"):
    """Action handler for aligning selected items to the bottom."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    strategy = BboxAlignBottomStrategy(selected_items)
    _execute_layout_task(win, strategy, _("Align Bottom"))


def spread_horizontally(win: "MainWindow"):
    """Action handler for spreading selected items horizontally."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    strategy = SpreadHorizontallyStrategy(selected_items)
    _execute_layout_task(win, strategy, _("Spread Horizontally"))


def spread_vertically(win: "MainWindow"):
    """Action handler for spreading selected items vertically."""
    selected_items = win.surface.get_selected_items()
    if not selected_items:
        return

    strategy = SpreadVerticallyStrategy(selected_items)
    _execute_layout_task(win, strategy, _("Spread Vertically"))


def layout_pixel_perfect(win: "MainWindow"):
    """Action handler for the pixel-perfect packing layout."""
    selected_items = win.surface.get_selected_items()

    # Determine the actual items to be laid out based on selection context.
    if not selected_items:
        # If nothing is selected, apply to all workpieces in the document.
        items_to_layout = win.doc.all_workpieces
    else:
        # For any selection, only pack the top-level selected items.
        # E.g., if a group and its child are both selected, only pack the
        # group.
        items_to_layout = []
        selected_set = set(selected_items)
        for item in selected_items:
            has_selected_ancestor = False
            p = item.parent
            while p:
                if p in selected_set:
                    has_selected_ancestor = True
                    break
                p = p.parent
            if not has_selected_ancestor:
                items_to_layout.append(item)

    if not items_to_layout:
        return

    strategy = PixelPerfectLayoutStrategy(
        items=items_to_layout,
        margin_mm=0.5,
        resolution_px_per_mm=8.0,
        allow_rotation=True,
    )
    _execute_layout_task(win, strategy, _("Auto Layout"))
