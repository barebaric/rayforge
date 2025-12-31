from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..core.asset import IAsset
from ..core.undo import ListItemCommand, ChangePropertyCommand

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class AssetCmd:
    """Handles commands related to document assets."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self.doc = editor.doc

    def rename_asset(self, asset: IAsset, new_name: str):
        """
        Renames an asset and any dependent items in a single transaction.
        For example, renaming a StockAsset also renames its StockItems.
        """
        if not new_name.strip() or new_name == asset.name:
            return

        with self.doc.history_manager.transaction(_("Rename Asset")) as t:
            # 1. Rename the asset definition itself. The property setter will
            #    trigger the necessary signals.
            t.execute(
                ChangePropertyCommand(
                    target=asset,
                    property_name="name",
                    new_value=new_name,
                )
            )

            # 2. Find and rename dependent DocItems in an agnostic way.
            for item in self.doc.get_descendants():
                if item.depends_on_asset(asset):
                    t.execute(
                        ChangePropertyCommand(
                            target=item,
                            property_name="name",
                            new_value=new_name,
                        )
                    )

    def delete_asset(self, asset_to_delete: IAsset):
        """
        Deletes an asset and all document items that depend on it in a single
        undoable transaction.
        """
        history = self.doc.history_manager
        dependent_items = []

        # 1. Find all DocItems that depend on this asset, agnostically.
        for item in self.doc.get_descendants():
            if item.depends_on_asset(asset_to_delete):
                dependent_items.append(item)

        # 2. Create a single transaction to remove everything.
        tx_name = _("Delete Asset '{name}'").format(name=asset_to_delete.name)
        with history.transaction(tx_name) as t:
            # First, remove the dependent DocItems from the document tree.
            for item in dependent_items:
                if not item.parent:
                    continue
                t.execute(
                    ListItemCommand(
                        owner_obj=item.parent,
                        item=item,
                        undo_command="add_child",
                        redo_command="remove_child",
                        name=_("Remove dependent item"),
                    )
                )

            # Finally, remove the asset definition itself.
            t.execute(
                ListItemCommand(
                    owner_obj=self.doc,
                    item=asset_to_delete,
                    undo_command="add_asset",
                    redo_command="remove_asset",
                    name=_("Remove asset definition"),
                )
            )
