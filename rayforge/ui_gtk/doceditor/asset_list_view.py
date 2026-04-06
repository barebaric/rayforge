import logging
from gi.repository import Gtk
from blinker import Signal
from typing import cast, TYPE_CHECKING, Dict, List
from gettext import gettext as _
from ...core.doc import Doc
from ...core.asset import IAsset
from ...core.asset_registry import asset_type_registry
from ...core.stock import StockItem
from ...core.undo import Command
from ..shared.draglist import DragListBox
from ..shared.expander import ExpanderWithButton
from ..shared.gtk import apply_css
from ..shared.popover_menu import PopoverMenu
from .asset_row_factory import create_asset_row_widget
from .asset_row_widget import IAssetRowWidget


if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

css = """
/* Generic styles for any asset row */
.asset-list-box > row.active { /* Gtk adds :active on activation */
    background-color: @accent_bg_color;
    color: @accent_fg_color;
    border-radius: 6px;
}

.asset-list-box > row.active .asset-row-view {
    background-color: transparent;
}

.asset-list-box > row.active .dim-label {
    opacity: 0.7;
}

/* Style for any asset row with an editable title */
.asset-row-view entry.asset-title-entry,
.asset-row-view entry.asset-title-entry:focus {
    border: none;
    outline: none;
    box-shadow: none;
    background: transparent;
    padding: 0;
    margin: 0;
    min-height: 0;
}

/* Style the caret color for any active asset row entry */
.asset-list-box > row.active .asset-row-view entry {
    caret-color: @accent_fg_color;
}
"""


class _ReorderAssetsCommand(Command):
    """A command to handle reordering assets in the document."""

    def __init__(self, doc: Doc, new_asset_order: List["IAsset"]):
        super().__init__(name=_("Reorder Assets"))
        self.doc = doc
        self.old_order = doc.asset_order[:]
        self.new_order = [asset.uid for asset in new_asset_order]

    def execute(self):
        self.doc.set_asset_order(self.new_order)

    def undo(self):
        self.doc.set_asset_order(self.old_order)


class AssetListView(ExpanderWithButton):
    """
    A widget that displays a collapsible, reorderable list of all
    document assets (Stock, Sketches, etc.).
    """

    add_asset_requested = Signal()
    asset_activated = Signal()

    def __init__(self, editor: "DocEditor", **kwargs):
        super().__init__(button_label=_("Add Asset"), **kwargs)
        apply_css(css)
        self.editor = editor
        self.doc = editor.doc

        self.set_title(_("Assets"))
        self.set_expanded(True)

        self.draglist = DragListBox()
        self.draglist.add_css_class("asset-list-box")
        self.draglist.connect("row-activated", self.on_row_activated)
        self.draglist.reordered.connect(self.on_assets_reordered)
        self.append_content(self.draglist)

        self.add_button.connect("clicked", self.on_add_button_clicked)

        self._connect_signals()
        self.on_doc_changed(self.doc)

    def set_doc(self, doc: Doc):
        """Updates the widget to track a new document instance."""
        if self.doc == doc:
            return

        self._disconnect_signals()
        self.doc = doc
        self._connect_signals()
        self.on_doc_changed(self.doc)

    def _connect_signals(self):
        self.doc.updated.connect(self.on_doc_changed)
        self.doc.descendant_added.connect(self.on_doc_changed)
        self.doc.descendant_removed.connect(self.on_doc_changed)

    def _disconnect_signals(self):
        self.doc.updated.disconnect(self.on_doc_changed)
        self.doc.descendant_added.disconnect(self.on_doc_changed)
        self.doc.descendant_removed.disconnect(self.on_doc_changed)

    def on_add_button_clicked(self, button: Gtk.Button):
        """Shows a popup to select and add a new asset type."""
        asset_types = []
        for type_name, asset_class in asset_type_registry.all_types().items():
            if asset_class.is_addable:
                display_name = f"Add {type_name.title()}"
                asset_types.append((_(display_name), type_name))

        popup = PopoverMenu(items=asset_types)
        popup.set_parent(button)
        popup.popup()
        popup.connect("closed", self.on_add_dialog_response)

    def on_add_dialog_response(self, popup: PopoverMenu):
        """Handles the creation of a new asset after the popup closes."""
        if popup.selected_item:
            self.add_asset_requested.send(self, type_name=popup.selected_item)

    def on_doc_changed(self, sender, **kwargs):
        """
        Handles document updates. Filters out irrelevant changes (like
        WorkPiece additions) to prevent unnecessary list rebuilds.
        """
        # Filter descendant events to prevent unnecessary rebuilds that can
        # conflict with drag-drop operations.
        child = kwargs.get("child")
        if child:
            # The UI for a StockAssetRowWidget depends on whether a
            # corresponding StockItem instance exists. Therefore, we must
            # refresh when a StockItem is added or removed. Any other
            # descendant change (e.g., adding a WorkPiece) is irrelevant
            # to this view and should be ignored.
            if not isinstance(child, StockItem):
                return

        visible_assets = [a for a in self.doc.get_all_assets() if not a.hidden]
        count = len(visible_assets)
        self.set_subtitle(
            _("{count} asset").format(count=count)
            if count == 1
            else _("{count} assets").format(count=count)
        )
        self.update_list()

    def update_list(self):
        new_assets = [a for a in self.doc.get_all_assets() if not a.hidden]
        uid_to_widget: Dict[str, IAssetRowWidget] = {}
        child = self.draglist.get_first_child()

        while child:
            row = cast(Gtk.ListBoxRow, child)
            # The child of the row is an hbox created by DragListBox
            hbox = row.get_child()
            if isinstance(hbox, Gtk.Box):
                # The actual asset widget is the last child of the hbox
                widget = hbox.get_last_child()
                if isinstance(widget, IAssetRowWidget):
                    # Detach the widget from its parent hbox so we can reuse it
                    hbox.remove(widget)
                    uid_to_widget[widget.asset.uid] = widget
            child = child.get_next_sibling()

        self.draglist.remove_all()

        for asset in new_assets:
            widget = uid_to_widget.get(asset.uid)

            if widget is None:
                widget = create_asset_row_widget(asset, self.editor)

            if widget is None:
                continue

            if isinstance(widget, IAssetRowWidget):
                widget.update_ui()
                widget.edit_clicked.connect(self.on_edit_clicked)
                widget.delete_clicked.connect(self.on_delete_clicked)
                widget.visibility_changed.connect(self.on_visibility_changed)

            new_row = Gtk.ListBoxRow()
            new_row.data = asset  # type: ignore
            new_row.set_activatable(True)
            new_row.set_child(cast(Gtk.Widget, widget))

            self.draglist.add_row(new_row)

    def on_assets_reordered(self, draglist: DragListBox):
        """
        Handles the reordering of assets in the list and applies the change
        to the document model via an undoable command.
        """
        new_asset_order = [row.data for row in draglist]  # type: ignore
        command = _ReorderAssetsCommand(
            doc=self.doc, new_asset_order=new_asset_order
        )
        self.editor.doc.history_manager.execute(command)

    def on_row_activated(self, listbox: Gtk.ListBox, row: Gtk.ListBoxRow):
        asset = row.data  # type: ignore
        self.asset_activated.send(self, asset=asset)

    def on_edit_clicked(self, row_widget: IAssetRowWidget):
        """Handles edit button click for any asset type."""
        self.asset_activated.send(self, asset=row_widget.asset)

    def on_delete_clicked(self, row_widget: IAssetRowWidget):
        """Handles delete button click for any asset type."""
        logger.debug(
            "on_delete_clicked: asset=%s, uid=%s",
            row_widget.asset,
            row_widget.asset.uid,
        )
        self.editor.asset.delete_asset(row_widget.asset)

    def on_visibility_changed(self, row_widget: IAssetRowWidget):
        """Handles visibility toggle for any asset type."""
        self.editor.asset.toggle_asset_visibility(row_widget.asset)
