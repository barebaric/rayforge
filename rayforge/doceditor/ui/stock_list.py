import logging
from gi.repository import Gtk
from blinker import Signal
from typing import cast

from ...core.stock import StockItem
from ...shared.ui.draglist import DragListBox
from .stock_view import StockItemView
from ...shared.ui.expander import Expander
from ...icons import get_icon

logger = logging.getLogger(__name__)


class StockListView(Expander):
    """
    A widget that displays a collapsible, reorderable list of StockItems.
    """

    stock_activated = Signal()

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        self.doc = editor.doc

        self.set_title(_("Stock Material"))
        self.set_expanded(True)

        # A container for all content that will be revealed by the expander
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(content_box)

        # The reorderable list of StockItems goes inside the content box
        self.draglist = DragListBox()
        self.draglist.add_css_class("stock-list-box")
        self.draglist.reordered.connect(self.on_stock_reordered)
        self.draglist.connect("row-activated", self.on_row_activated)
        content_box.append(self.draglist)

        # An "Add" button
        add_button = Gtk.Button()
        add_button.add_css_class("darkbutton")
        add_button.connect("clicked", self.on_button_add_clicked)
        content_box.append(add_button)

        # The button's content is a box with an icon and a label.
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_margin_top(10)
        button_box.set_margin_end(12)
        button_box.set_margin_bottom(10)
        button_box.set_margin_start(12)

        add_icon = get_icon("add-symbolic")
        button_box.append(add_icon)

        lbl = _("Add Stock")
        add_label = Gtk.Label()
        add_label.set_markup(f"<span weight='normal'>{lbl}</span>")
        add_label.set_xalign(0)
        button_box.append(add_label)
        add_button.set_child(button_box)

        # Connect to document changes and perform initial population
        self.doc.updated.connect(self.on_doc_changed)
        self.doc.descendant_added.connect(self.on_doc_changed)
        self.doc.descendant_removed.connect(self.on_doc_changed)
        self.on_doc_changed(self.doc)

    def on_doc_changed(self, sender, **kwargs):
        """
        Updates the list and subtitle when the document structure changes.
        """
        count = len(self.doc.stock_items)
        self.set_subtitle(
            _("{count} item").format(count=count)
            if count == 1
            else _("{count} items").format(count=count)
        )
        self.update_list()

    def update_list(self):
        """
        Re-populates the draglist to match the state of the document's
        stock items and ensures the initial active state is correctly
        displayed.
        """
        self.draglist.remove_all()
        # You can only delete a stock item if there is more than one.
        can_delete_stock_item = len(self.doc.stock_items) > 1

        for stock_item in self.doc.stock_items:
            list_box_row = Gtk.ListBoxRow()
            list_box_row.data = stock_item  # type: ignore
            stock_item_view = StockItemView(self.doc, stock_item, self.editor)

            is_deletable = can_delete_stock_item
            # For now, all items are deletable if more than one
            is_deletable = is_deletable

            stock_item_view.delete_clicked.connect(
                self.on_delete_stock_item_clicked
            )
            list_box_row.set_child(stock_item_view)
            self.draglist.add_row(list_box_row)

            # The StockItemView now has a parent. Manually call
            # update_style() here to guarantee the initial CSS class is set
            # correctly based on the model's state at creation time.
            # Stock items don't have an active state like layers

    def on_row_activated(self, listbox, row):
        """Handles user clicks to change the active stock item."""
        if row and row.data:
            stock_item = cast(StockItem, row.data)

            # Send a signal for other parts of the UI (e.g., MainWindow)
            self.stock_activated.send(self, stock_item=stock_item)

    def on_button_add_clicked(self, button):
        """Handles creation of a new stock item with an undoable command."""
        # Delegate to the StockCmd which handles undo/redo properly
        self.editor.stock.add_stock_item()

    def on_delete_stock_item_clicked(self, stock_item_view):
        """Handles deletion of a stock item with an undoable command."""
        stock_item_to_delete = stock_item_view.stock_item

        # Delegate to the StockCmd which handles undo/redo properly
        self.editor.stock.delete_stock_item(stock_item_to_delete)

    def on_stock_reordered(self, sender):
        """Handles reordering of StockItems with an undoable command."""
        new_order = [row.data for row in self.draglist]  # type: ignore
        self.editor.stock.reorder_stock_items(new_order)
