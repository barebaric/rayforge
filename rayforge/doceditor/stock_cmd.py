from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..core.stock import StockItem
from ..core.stock_asset import StockAsset
from ..core.geo import Geometry
from ..core.matrix import Matrix
from ..undo import ChangePropertyCommand, Command

if TYPE_CHECKING:
    from .editor import DocEditor
    from ..core.doc import Doc

logger = logging.getLogger(__name__)


class _AddStockCommand(Command):
    """
    A private command to handle the creation of a StockAsset and StockItem.
    """

    def __init__(
        self,
        doc: "Doc",
        name: str,
        geometry: Geometry,
        pos: tuple[float, float],
    ):
        super().__init__(name=_("Add Stock"))
        self.doc = doc
        self.asset = StockAsset(name=name, geometry=geometry)
        self.item = StockItem(stock_asset_uid=self.asset.uid, name=name)
        w, h = self.asset.get_natural_size()
        self.item.matrix = Matrix.scale(w, h)
        self.item.pos = pos
        self.asset_uid = self.asset.uid

    def execute(self):
        self.do()

    def do(self):
        self.doc.add_asset(self.asset, silent=True)
        self.doc.add_child(self.item)

    def undo(self):
        self.doc.remove_child(self.item)
        self.doc.remove_asset_by_uid(self.asset_uid)


class StockCmd:
    """Handles commands related to stock material."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def add_stock(self):
        """
        Adds a new StockAsset and a linking StockItem to the document.
        This is a single undoable operation.
        """
        doc = self._editor.doc
        machine = self._editor.context.config.machine
        machine_w, machine_h = (200.0, 200.0)
        if machine:
            machine_w, machine_h = machine.dimensions

        stock_w = machine_w * 0.8
        stock_h = machine_h * 0.8
        stock_x = (machine_w - stock_w) / 2
        stock_y = (machine_h - stock_h) / 2

        default_geometry = Geometry()
        default_geometry.move_to(0, 0)
        default_geometry.line_to(stock_w, 0)
        default_geometry.line_to(stock_w, stock_h)
        default_geometry.line_to(0, stock_h)
        default_geometry.close_path()

        stock_count = len(doc.stock_assets) + 1
        stock_name = _("Stock {count}").format(count=stock_count)

        command = _AddStockCommand(
            doc, stock_name, default_geometry, (stock_x, stock_y)
        )
        doc.history_manager.execute(command)

    def toggle_stock_visibility(self, stock_item: StockItem):
        """
        Toggles the visibility of a StockItem with an undoable command.
        """
        new_visibility = not stock_item.visible
        command = ChangePropertyCommand(
            target=stock_item,
            property_name="visible",
            new_value=new_visibility,
            setter_method_name="set_visible",
            name=_("Toggle stock visibility"),
        )
        self._editor.doc.history_manager.execute(command)

    def rename_stock_asset(self, stock_asset: StockAsset, new_name: str):
        """
        Renames a StockAsset with an undoable command. It also finds and
        renames all StockItem instances that use this asset.
        """
        if new_name == stock_asset.name:
            return

        with self._editor.doc.history_manager.transaction(
            _("Rename Stock Asset")
        ) as t:
            # Command to rename the asset definition
            t.execute(
                ChangePropertyCommand(
                    target=stock_asset,
                    property_name="name",
                    new_value=new_name,
                    setter_method_name="set_name",
                )
            )
            # Find and rename all instances
            for item in self._editor.doc.stock_items:
                if item.stock_asset_uid == stock_asset.uid:
                    t.execute(
                        ChangePropertyCommand(
                            target=item,
                            property_name="name",
                            new_value=new_name,
                            setter_method_name="set_name",
                        )
                    )

    def set_stock_thickness(self, stock_item: StockItem, new_thickness: float):
        """
        Sets the thickness of a StockAsset with an undoable command.
        """
        stock_asset = stock_item.stock_asset
        if not stock_asset or new_thickness == stock_asset.thickness:
            return

        command = ChangePropertyCommand(
            target=stock_asset,
            property_name="thickness",
            new_value=new_thickness,
            setter_method_name="set_thickness",
            name=_("Change stock thickness"),
        )
        self._editor.doc.history_manager.execute(command)
        stock_item.updated.send(stock_item)

    def set_stock_material(self, stock_item: StockItem, new_material_uid: str):
        """
        Sets the material of a StockAsset with an undoable command.
        """
        stock_asset = stock_item.stock_asset
        if not stock_asset or new_material_uid == stock_asset.material_uid:
            return

        command = ChangePropertyCommand(
            target=stock_asset,
            property_name="material_uid",
            new_value=new_material_uid,
            setter_method_name="set_material",
            name=_("Change stock material"),
        )
        self._editor.doc.history_manager.execute(command)
        stock_item.updated.send(stock_item)
