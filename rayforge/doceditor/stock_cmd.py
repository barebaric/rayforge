from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..core.stock import StockItem
from ..core.geo import Geometry
from ..undo import ChangePropertyCommand
from ..undo.models.list_cmd import ListItemCommand, ReorderListCommand

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class StockCmd:
    """Handles commands related to stock material."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def add_stock_item(self):
        """
        Adds a new StockItem with a default size based on machine dimensions.
        This is a single undoable operation.
        """
        doc = self._editor.doc

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

        # Generate auto-numbered name
        stock_count = len(doc.stock_items) + 1
        stock_name = _("Stock {count}").format(count=stock_count)
        new_stock_item = StockItem(geometry=default_geometry, name=stock_name)

        # The StockItem constructor sets its matrix to scale to the geometry
        # size. Now, we set its world position, which updates the matrix's
        # translation part.
        new_stock_item.pos = (stock_x, stock_y)

        # Create and execute the command through the history manager
        command = ListItemCommand(
            owner_obj=doc,
            item=new_stock_item,
            undo_command="remove_stock_item",
            redo_command="add_stock_item",
            name=_("Add Stock Item"),
        )
        doc.history_manager.execute(command)

    def delete_stock_item(self, stock_item: StockItem):
        """
        Deletes a StockItem with an undoable command.

        Args:
            stock_item: The StockItem to delete
        """
        doc = self._editor.doc

        command = ListItemCommand(
            owner_obj=doc,
            item=stock_item,
            undo_command="add_stock_item",
            redo_command="remove_stock_item",
            name=_("Remove Stock Item"),
        )
        doc.history_manager.execute(command)

    def toggle_stock_visibility(self, stock_item: StockItem):
        """
        Toggles the visibility of a StockItem with an undoable command.

        Args:
            stock_item: The StockItem to toggle visibility for
        """
        from ..undo.models.property_cmd import ChangePropertyCommand

        new_visibility = not stock_item.visible

        command = ChangePropertyCommand(
            target=stock_item,
            property_name="visible",
            new_value=new_visibility,
            setter_method_name="set_visible",
            name=_("Toggle stock visibility"),
        )
        self._editor.doc.history_manager.execute(command)

    def reorder_stock_items(self, new_order: list[StockItem]):
        """
        Reorders stock items with an undoable command.

        Args:
            new_order: The new list of StockItems in the desired order
        """
        doc = self._editor.doc

        command = ReorderListCommand(
            target_obj=doc,
            list_property_name="stock_items",
            new_list=new_order,
            name=_("Reorder Stock Items"),
        )
        doc.history_manager.execute(command)

    def rename_stock_item(self, stock_item: StockItem, new_name: str):
        """
        Renames a StockItem with an undoable command.

        Args:
            stock_item: The StockItem to rename.
            new_name: The new name for the StockItem.
        """
        from ..undo.models.property_cmd import ChangePropertyCommand

        if new_name == stock_item.name:
            return

        command = ChangePropertyCommand(
            target=stock_item,
            property_name="name",
            new_value=new_name,
            setter_method_name="set_name",
            name=_("Rename stock item"),
        )
        self._editor.doc.history_manager.execute(command)

    def set_stock_thickness(self, stock_item: StockItem, new_thickness: float):
        """
        Sets the thickness of a StockItem with an undoable command.

        Args:
            stock_item: The StockItem to modify.
            new_thickness: The new thickness for the StockItem.
        """
        if new_thickness == stock_item.thickness:
            return

        command = ChangePropertyCommand(
            target=stock_item,
            property_name="thickness",
            new_value=new_thickness,
            setter_method_name="set_thickness",
            name=_("Change stock thickness"),
        )
        self._editor.doc.history_manager.execute(command)
