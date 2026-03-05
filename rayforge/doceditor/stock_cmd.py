from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from gettext import gettext as _
from ..core.geo import Geometry
from ..core.matrix import Matrix
from ..core.undo import ChangePropertyCommand, Command
from ..core.stock import StockItem
from ..core.stock_asset import StockAsset
from ..core.workpiece import WorkPiece

from ..usage import get_usage_tracker

if TYPE_CHECKING:
    from .editor import DocEditor
    from ..core.doc import Doc
    from ..core.item import DocItem

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


class RemoveStockAssetCommand(Command):
    """Command to remove a StockAsset from the document."""

    def __init__(self, doc: "Doc", asset_uid: str):
        super().__init__(name=_("Remove Stock Asset"))
        self.doc = doc
        self.asset_uid = asset_uid
        self._removed_asset = None

    def execute(self):
        self._removed_asset = self.doc.get_asset_by_uid(self.asset_uid)
        if self._removed_asset:
            self.doc.remove_asset_by_uid(self.asset_uid)

    def undo(self):
        if self._removed_asset:
            self.doc.add_asset(self._removed_asset, silent=True)


class ConvertToStockCommand(Command):
    """
    Command to convert a WorkPiece to a StockItem with its own StockAsset.
    """

    def __init__(self, doc: "Doc", workpiece: WorkPiece):
        super().__init__(name=_("Convert to Stock"))
        self.doc = doc
        self.workpiece = workpiece
        self.original_parent: "DocItem | None" = workpiece.parent
        self.original_index = 0

        geometry = workpiece.get_world_geometry()
        if geometry is None:
            geometry = Geometry()

        self.asset = StockAsset(name=workpiece.name, geometry=geometry)
        self.stock_item = StockItem(
            stock_asset_uid=self.asset.uid, name=workpiece.name
        )
        w, h = self.asset.get_natural_size()
        if w > 0 and h > 0:
            self.stock_item.matrix = Matrix.scale(w, h)
        self.stock_item.pos = workpiece.pos
        self.stock_item.angle = workpiece.angle
        self.asset_uid = self.asset.uid

    def execute(self):
        if self.original_parent:
            children = list(self.original_parent.children)
            self.original_index = children.index(self.workpiece)
            self.original_parent.remove_child(self.workpiece)

        self.doc.add_asset(self.asset, silent=True)
        self.doc.add_child(self.stock_item)

    def undo(self):
        self.doc.remove_child(self.stock_item)
        self.doc.remove_asset_by_uid(self.asset_uid)

        if self.original_parent:
            self.original_parent.add_child(
                self.workpiece, index=self.original_index
            )


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
        if machine:
            __, __, wa_w, wa_h = machine.work_area
            ref_x, ref_y = machine.get_reference_position_world()
            stock_x = ref_x
            stock_y = ref_y
            stock_w = wa_w * 0.8
            stock_h = wa_h * 0.8
            space = machine.get_coordinate_space()
            stock_x, stock_y = space.world_position_from_origin(
                ref_x, ref_y, (stock_w, stock_h)
            )
            logger.debug(
                "Calculated stock position (%.2f, %.2f)", stock_x, stock_y
            )
        else:
            stock_x, stock_y, stock_w, stock_h = 0, 0, 200.0, 200.0

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
        get_usage_tracker().track_page_view(
            "/doc/add-asset/stock", "Add Stock Asset"
        )

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

    def convert_to_stock(self, workpiece: WorkPiece) -> StockItem:
        """
        Converts a WorkPiece to a StockItem with its own StockAsset.
        This is a single undoable operation.
        """
        command = ConvertToStockCommand(self._editor.doc, workpiece)
        self._editor.doc.history_manager.execute(command)
        return command.stock_item
