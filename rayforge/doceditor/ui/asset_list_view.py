import logging
from gi.repository import Gtk, Adw
from blinker import Signal
from typing import cast, TYPE_CHECKING, Dict, List
from ...core.doc import Doc
from ...core.sketcher.sketch import Sketch
from ...core.stock_asset import StockAsset
from ...shared.ui.draglist import DragListBox
from ...shared.ui.expander import Expander
from ...shared.util.gtk import apply_css
from ...undo import ListItemCommand, Command
from ...icons import get_icon
from .asset_row_factory import create_asset_row_widget
from .asset_row_widget import StockAssetRowWidget, SketchAssetRowWidget
from ...shared.ui.popover_menu import PopoverMenu


if TYPE_CHECKING:
    from ..editor import DocEditor
    from ...core.asset import IAsset

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

/* Specific to stock rows that have an editable entry */
.stock-asset-row entry.asset-title-entry,
.stock-asset-row entry.asset-title-entry:focus {
    border: none;
    outline: none;
    box-shadow: none;
    background: transparent;
    padding: 0;
    margin: 0;
    min-height: 0;
}

.asset-list-box > row.active .stock-asset-row entry {
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


class AssetListView(Expander):
    """
    A widget that displays a collapsible, reorderable list of all
    document assets (Stock, Sketches, etc.).
    """

    add_stock_clicked = Signal()
    add_sketch_clicked = Signal()
    sketch_activated = Signal()
    stock_activated = Signal()

    def __init__(self, editor: "DocEditor", **kwargs):
        super().__init__(**kwargs)
        apply_css(css)
        self.editor = editor
        self.doc = editor.doc

        self.set_title(_("Assets"))
        self.set_expanded(True)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(content_box)

        self.draglist = DragListBox()
        self.draglist.add_css_class("asset-list-box")
        self.draglist.connect("row-activated", self.on_row_activated)
        self.draglist.reordered.connect(self.on_assets_reordered)
        content_box.append(self.draglist)

        # A Gtk.Button, styled as a card, serves as our "Add" button
        add_button = Gtk.Button()
        add_button.add_css_class("darkbutton")
        add_button.connect("clicked", self.on_add_button_clicked)
        content_box.append(add_button)

        # The button's content is a box with an icon and a label.
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_margin_top(10)
        button_box.set_margin_end(12)
        button_box.set_margin_bottom(10)
        button_box.set_margin_start(12)

        add_icon = get_icon("add-symbolic")
        button_box.append(add_icon)

        lbl = _("Add Asset")
        add_label = Gtk.Label()
        add_label.set_markup(f"<span weight='normal'>{lbl}</span>")
        add_label.set_xalign(0)
        button_box.append(add_label)
        add_button.set_child(button_box)

        self.doc.updated.connect(self.on_doc_changed)
        self.doc.descendant_added.connect(self.on_doc_changed)
        self.doc.descendant_removed.connect(self.on_doc_changed)
        self.on_doc_changed(self.doc)

    def on_add_button_clicked(self, button: Gtk.Button):
        """Shows a popup to select and add a new asset type."""
        asset_types = [
            (_("Add Stock Material"), "stock"),
            (_("Add Sketch"), "sketch"),
        ]
        popup = PopoverMenu(items=asset_types)
        popup.set_parent(button)
        popup.popup()
        popup.connect("closed", self.on_add_dialog_response)

    def on_add_dialog_response(self, popup: PopoverMenu):
        """Handles the creation of a new asset after the popup closes."""
        if popup.selected_item == "stock":
            self.add_stock_clicked.send(self)
        elif popup.selected_item == "sketch":
            self.add_sketch_clicked.send(self)

    def on_doc_changed(self, sender, **kwargs):
        count = len(self.doc.get_all_assets())
        self.set_subtitle(
            _("{count} asset").format(count=count)
            if count == 1
            else _("{count} assets").format(count=count)
        )
        self.update_list()

    def update_list(self):
        new_assets = self.doc.get_all_assets()
        uid_to_widget: Dict[str, Gtk.Widget] = {}
        child = self.draglist.get_first_child()

        while child:
            row = cast(Gtk.ListBoxRow, child)
            # The child of the row is an hbox created by DragListBox
            hbox = row.get_child()
            if isinstance(hbox, Gtk.Box):
                # The actual asset widget is the last child of the hbox
                widget = hbox.get_last_child()
                if isinstance(
                    widget, (StockAssetRowWidget, SketchAssetRowWidget)
                ):
                    # Detach the widget from its parent hbox so we can reuse it
                    hbox.remove(widget)
                    uid_to_widget[widget.asset.uid] = widget
            child = child.get_next_sibling()

        self.draglist.remove_all()

        for asset in new_assets:
            widget = uid_to_widget.get(asset.uid)

            if widget:
                if isinstance(
                    widget, (StockAssetRowWidget, SketchAssetRowWidget)
                ):
                    widget.update_ui()
            else:
                widget = create_asset_row_widget(asset, self.editor)

            if not widget:
                continue

            new_row = Gtk.ListBoxRow()
            new_row.data = asset  # type: ignore
            new_row.set_activatable(True)
            new_row.set_child(widget)

            if isinstance(widget, SketchAssetRowWidget):
                widget.edit_clicked.connect(self.on_edit_sketch_clicked)
                widget.delete_clicked.connect(self.on_delete_sketch_clicked)
            elif isinstance(widget, StockAssetRowWidget):
                widget.delete_clicked.connect(self.on_delete_stock_clicked)

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
        if asset.asset_type_name == "sketch":
            self.sketch_activated.send(self, sketch=cast(Sketch, asset))
        elif asset.asset_type_name == "stock":
            stock_asset = cast(StockAsset, asset)
            # Find the corresponding item on the canvas to activate
            stock_item = next(
                (
                    item
                    for item in self.doc.stock_items
                    if item.stock_asset_uid == stock_asset.uid
                ),
                None,
            )
            if stock_item:
                self.stock_activated.send(self, stock_item=stock_item)

    def on_edit_sketch_clicked(self, sketch_row_widget: SketchAssetRowWidget):
        sketch = sketch_row_widget.asset
        self.sketch_activated.send(self, sketch=sketch)

    def on_delete_sketch_clicked(
        self, sketch_row_widget: SketchAssetRowWidget
    ):
        sketch_to_delete = sketch_row_widget.asset
        workpieces_using_sketch = [
            wp
            for wp in self.doc.all_workpieces
            if wp.sketch_uid == sketch_to_delete.uid
        ]

        if workpieces_using_sketch:
            root = self.get_root()
            parent_window = (
                cast(Gtk.Window, root)
                if isinstance(root, Gtk.Window)
                else None
            )
            dialog = Adw.MessageDialog(
                transient_for=parent_window,
                heading=_("Cannot Delete Sketch"),
                body=_(
                    "This sketch is still in use by {count} workpiece(s) on "
                    "the canvas. Please delete those workpieces first."
                ).format(count=len(workpieces_using_sketch)),
            )
            dialog.add_response("ok", _("OK"))
            dialog.set_default_response("ok")
            dialog.connect("response", lambda d, r: d.close())
            dialog.present()
            return

        command = ListItemCommand(
            owner_obj=self.doc,
            item=sketch_to_delete,
            undo_command="add_asset",
            redo_command="remove_asset",
            name=_("Delete Sketch Definition"),
        )
        self.editor.doc.history_manager.execute(command)

    def on_delete_stock_clicked(self, stock_row_widget: StockAssetRowWidget):
        stock_item_to_delete = stock_row_widget.stock_item
        if stock_item_to_delete:
            self.editor.stock.delete_stock_item(stock_item_to_delete)
