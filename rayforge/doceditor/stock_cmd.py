from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..core.stock import StockItem
from ..core.geo import Geometry
from ..undo import ListItemCommand

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class StockCmd:
    """Handles commands related to stock material."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def add_stock_item(self):
        """
        Sets or resets the StockItem to a default size based on the
        machine dimensions. If a stock item already exists, it is replaced.
        This is a single undoable operation.
        """
        doc = self._editor.doc
        stock_layer = doc.stock_layer
        if stock_layer is None:
            logger.error(
                "Cannot add StockItem: No StockLayer found in the document."
            )
            return

        # Find any existing stock item. There should be at most one.
        existing_stock_item = next(
            (c for c in stock_layer.children if isinstance(c, StockItem)), None
        )

        # Get machine dimensions, with a fallback
        machine = self._editor._config_manager.config.machine
        machine_w, machine_h = (200.0, 200.0)  # A sensible fallback
        if machine:
            machine_w, machine_h = machine.dimensions

        # Calculate a proportional size (e.g., 80%) and centered position
        stock_w = machine_w * 0.8
        stock_h = machine_h * 0.8
        stock_x = (machine_w - stock_w) / 2
        stock_y = (machine_h - stock_h) / 2

        # Create geometry for a rectangle of the calculated size at the origin
        default_geometry = Geometry()
        default_geometry.move_to(0, 0)
        default_geometry.line_to(stock_w, 0)
        default_geometry.line_to(stock_w, stock_h)
        default_geometry.line_to(0, stock_h)
        default_geometry.close_path()
        new_stock_item = StockItem(
            geometry=default_geometry, name=_("New Stock")
        )

        # The StockItem constructor sets its matrix to scale to the geometry
        # size. Now, we set its world position, which updates the matrix's
        # translation part.
        new_stock_item.pos = (stock_x, stock_y)

        with self._editor.history_manager.transaction(
            _("Set Stock Item")
        ) as t:
            # If an old item exists, create and execute a command to remove it.
            if existing_stock_item:
                remove_cmd = ListItemCommand(
                    owner_obj=stock_layer,
                    item=existing_stock_item,
                    undo_command="add_child",
                    redo_command="remove_child",
                    name=_("Remove old stock"),
                )
                t.execute(remove_cmd)

            # Create and execute the command to add the new item.
            add_cmd = ListItemCommand(
                owner_obj=stock_layer,
                item=new_stock_item,
                undo_command="remove_child",
                redo_command="add_child",
                name=_("Add new stock"),
            )
            t.execute(add_cmd)

        # Automatically make the stock layer active
        doc.active_layer = stock_layer
