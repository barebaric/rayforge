from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple
from ..shared.util.glib import idle_add
from ..core.item import DocItem
from ..core.undo import ChangePropertyCommand
from .layout import (
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
    PositionAtStrategy,
)

if TYPE_CHECKING:
    from ..shared.tasker.manager import TaskManager
    from ..shared.tasker.task import Task
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class LayoutCmd:
    """Handles alignment, distribution, and automatic layout of items."""

    def __init__(self, editor: "DocEditor", task_manager: "TaskManager"):
        self._editor = editor
        self._task_manager = task_manager

    def _execute_layout_task(
        self, strategy: LayoutStrategy, transaction_name: str
    ):
        """
        A synchronous helper that configures and launches a background layout
        task.

        The actual model mutation happens in the `when_done` callback, which is
        guaranteed to run on the main GTK thread.
        """

        # Define the handler that will receive error signals from the strategy.
        def on_error_reported(sender, message: str):
            """
            Receives an error message from the strategy (from a background
            thread) and safely schedules a UI notification on the main thread.
            """
            # Wrap the call in a lambda to ensure the keyword argument is
            # passed correctly by GLib.idle_add.
            idle_add(
                self._editor.notification_requested.send, self, message=message
            )

        # Connect the handler before running the task.
        strategy.error_reported.connect(on_error_reported)

        def when_done(task: "Task"):
            """
            This callback runs on the main thread after the task finishes.
            It disconnects the signal handler and safely applies the
            calculated changes to the document.
            """
            # Disconnect the handler to prevent potential memory leaks.
            strategy.error_reported.disconnect(on_error_reported)

            if task.get_status() != "completed":
                logger.error(
                    "Layout task '%s' did not complete successfully. "
                    "Status: %s",
                    transaction_name,
                    task.get_status(),
                )
                return

            # The result of the task is the dictionary of transformation
            # deltas.
            deltas = task.result()

            if not deltas:
                return  # No changes to apply

            with self._editor.history_manager.transaction(
                transaction_name
            ) as t:
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
        self._task_manager.add_coroutine(
            layout_coro,
            when_done=when_done,
            key=f"layout-{transaction_name}",  # key to prevent concurrent runs
        )

    def center_horizontally(
        self, selected_items: List[DocItem], surface_width_mm: float
    ):
        """Action handler for centering selected items horizontally."""
        if not selected_items:
            return

        strategy = BboxAlignCenterStrategy(
            selected_items, surface_width_mm=surface_width_mm
        )
        self._execute_layout_task(strategy, _("Center Horizontally"))

    def center_vertically(
        self, selected_items: List[DocItem], surface_height_mm: float
    ):
        """Action handler for centering selected items vertically."""
        if not selected_items:
            return

        strategy = BboxAlignMiddleStrategy(
            selected_items, surface_height_mm=surface_height_mm
        )
        self._execute_layout_task(strategy, _("Center Vertically"))

    def align_left(self, selected_items: List[DocItem]):
        """Action handler for aligning selected items to the left."""
        if not selected_items:
            return

        strategy = BboxAlignLeftStrategy(selected_items)
        self._execute_layout_task(strategy, _("Align Left"))

    def align_right(
        self, selected_items: List[DocItem], surface_width_mm: float
    ):
        """Action handler for aligning selected items to the right."""
        if not selected_items:
            return

        strategy = BboxAlignRightStrategy(
            selected_items, surface_width_mm=surface_width_mm
        )
        self._execute_layout_task(strategy, _("Align Right"))

    def align_top(
        self, selected_items: List[DocItem], surface_height_mm: float
    ):
        """Action handler for aligning selected items to the top."""
        if not selected_items:
            return

        strategy = BboxAlignTopStrategy(
            selected_items, surface_height_mm=surface_height_mm
        )
        self._execute_layout_task(strategy, _("Align Top"))

    def align_bottom(self, selected_items: List[DocItem]):
        """Action handler for aligning selected items to the bottom."""
        if not selected_items:
            return

        strategy = BboxAlignBottomStrategy(selected_items)
        self._execute_layout_task(strategy, _("Align Bottom"))

    def spread_horizontally(self, selected_items: List[DocItem]):
        """Action handler for spreading selected items horizontally."""
        if not selected_items:
            return

        strategy = SpreadHorizontallyStrategy(selected_items)
        self._execute_layout_task(strategy, _("Spread Horizontally"))

    def spread_vertically(self, selected_items: List[DocItem]):
        """Action handler for spreading selected items vertically."""
        if not selected_items:
            return

        strategy = SpreadVerticallyStrategy(selected_items)
        self._execute_layout_task(strategy, _("Spread Vertically"))

    def position_at(
        self, selected_items: List[DocItem], position_mm: Tuple[float, float]
    ):
        """Action handler for positioning the selection's center at a point."""
        if not selected_items:
            return

        strategy = PositionAtStrategy(
            items=selected_items, position_mm=position_mm
        )
        self._execute_layout_task(strategy, _("Position at Point"))

    def layout_pixel_perfect(self, selected_items: List[DocItem]):
        """Action handler for the pixel-perfect packing layout."""
        # Determine the actual items to be laid out based on selection context.
        if not selected_items:
            # If nothing is selected, get all top-level content items from
            # the current active layer only.
            items_to_layout = []
            active_layer = self._editor.doc.active_layer
            if active_layer:
                items_to_layout.extend(active_layer.get_content_items())
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
        self._execute_layout_task(strategy, _("Auto Layout"))
