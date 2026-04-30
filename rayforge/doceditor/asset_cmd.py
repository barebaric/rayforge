from __future__ import annotations
import logging
from typing import Dict, Any, TYPE_CHECKING
from gettext import gettext as _

from ..core.asset import IAsset
from ..core.asset_registry import asset_type_registry
from ..core.stock_asset import StockAsset
from ..core.undo import ListItemCommand, ChangePropertyCommand, Command

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..core.geometry_provider import IGeometryProvider
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class UpdateAssetCommand(Command):
    """
    A command that updates an asset in the document's registry.

    For assets that are also IGeometryProvider, it recalculates geometry
    and resizes all WorkPiece instances that depend on this asset.
    """

    def __init__(
        self,
        doc: "Doc",
        asset_uid: str,
        new_data: Dict[str, Any],
        name: str = _("Update Asset"),
    ):
        super().__init__(name)
        self.doc = doc
        self.asset_uid = asset_uid

        # --- Store old state for undo ---
        old_asset = doc.get_asset_by_uid(asset_uid)
        if not old_asset:
            raise ValueError(f"Asset with UID {asset_uid} not found.")
        self.old_data = old_asset.to_dict()
        self.new_data = new_data
        self.asset_type_name = old_asset.asset_type_name

        # Store old matrices of all affected workpieces for a perfect undo
        # (only relevant for geometry providers)
        self.old_matrices = {
            wp.uid: wp.matrix.copy()
            for wp in doc.all_workpieces
            if wp.geometry_provider_uid == asset_uid
        }

    def _apply_state(self, data: Dict[str, Any]):
        """Helper to apply an asset dictionary to the document state."""
        # 1. Deserialize and update the asset in the document registry
        asset_class = asset_type_registry.get(self.asset_type_name)
        if not asset_class:
            raise TypeError(f"Unknown asset type '{self.asset_type_name}'")
        asset_instance = asset_class.from_dict(data)
        self.doc.assets[self.asset_uid] = asset_instance

        # 2. If this is a geometry provider, update dependent workpieces
        from ..core.geometry_provider import IGeometryProvider

        if isinstance(asset_instance, IGeometryProvider):
            self._update_dependent_workpieces(asset_instance)

        # 3. Send a general doc update signal for pipeline, etc.
        self.doc.updated.send(self.doc)

    def _update_dependent_workpieces(
        self, provider: "IGeometryProvider"
    ) -> None:
        """Update all workpieces that depend on this geometry provider."""
        for workpiece in self.doc.all_workpieces:
            if workpiece.geometry_provider_uid != self.asset_uid:
                continue

            params = workpiece.geometry_provider_params or {}
            geometry, _ = provider.get_geometry(params=params)

            if geometry.is_empty():
                new_width = 0.0
                new_height = 0.0
            else:
                min_x, min_y, max_x, max_y = geometry.rect()
                new_width = max(max_x - min_x, 1e-9)
                new_height = max(max_y - min_y, 1e-9)

            # Update the workpiece's own dimension attributes
            workpiece.natural_width_mm = new_width
            workpiece.natural_height_mm = new_height

            # This resizes the workpiece's matrix while preserving its
            # center
            workpiece.set_size(new_width, new_height)

            # This clears _boundaries_cache and _render_cache
            workpiece.clear_render_cache()

            # Signal for UI to redraw this specific workpiece
            workpiece.updated.send(workpiece)

    def execute(self):
        logger.debug(
            f"Executing UpdateAssetCommand for asset {self.asset_uid}"
        )
        self._apply_state(self.new_data)

    def undo(self):
        logger.debug(f"Undoing UpdateAssetCommand for asset {self.asset_uid}")
        # Re-apply the old asset data, which will call set_size
        self._apply_state(self.old_data)

        # `set_size` preserves center, which might not be what we want for
        # undo.
        # To guarantee a perfect undo, explicitly restore original matrices.
        for wp in self.doc.all_workpieces:
            if wp.uid in self.old_matrices:
                wp.matrix = self.old_matrices[wp.uid].copy()


class AssetCmd:
    """Handles commands related to document assets."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    @property
    def doc(self):
        return self._editor.doc

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
        logger.debug(
            "delete_asset called: name=%s, uid=%s",
            asset_to_delete.name,
            asset_to_delete.uid,
        )
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

    def toggle_asset_visibility(self, asset: IAsset):
        """
        Toggles the visibility of an asset and all its dependent items
        with an undoable command.
        """
        new_hidden = not asset.hidden

        with self.doc.history_manager.transaction(
            _("Toggle Asset Visibility")
        ) as t:
            # Toggle the asset itself
            t.execute(
                ChangePropertyCommand(
                    target=asset,
                    property_name="hidden",
                    new_value=new_hidden,
                    setter_method_name="set_hidden",
                )
            )
            # For StockAsset, also update all StockItem instances
            if isinstance(asset, StockAsset):
                for item in self.doc.stock_items:
                    if item.stock_asset_uid == asset.uid:
                        t.execute(
                            ChangePropertyCommand(
                                target=item,
                                property_name="visible",
                                new_value=not new_hidden,
                                setter_method_name="set_visible",
                            )
                        )
