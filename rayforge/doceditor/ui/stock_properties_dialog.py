import logging
from gi.repository import Gtk, Adw, GLib
from typing import TYPE_CHECKING, Tuple

from ...core.stock import StockItem
from ...shared.ui.unit_spin_row import UnitSpinRowHelper

if TYPE_CHECKING:
    from ..editor import DocEditor

logger = logging.getLogger(__name__)


class StockPropertiesDialog(Adw.Window):
    """
    A non-modal window for editing stock item properties.
    """

    def __init__(
        self, parent: Gtk.Window, stock_item: StockItem, editor: "DocEditor"
    ):
        super().__init__(transient_for=parent)
        self.stock_item = stock_item
        self.editor = editor
        self.doc = editor.doc

        # Used to delay updates from continuous-change widgets
        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args: Tuple = ()

        # Connect to stock item updates to refresh UI
        self.stock_item.updated.connect(self.on_stock_item_updated)

        # Make sure to disconnect when the dialog is destroyed
        self.connect("destroy", self._on_destroy)

        self.set_title(_("Stock Properties"))
        self.set_default_size(400, 300)
        self.set_modal(False)
        self.set_resizable(True)

        # Create a vertical box to hold the header bar and the content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Add a header bar for title and window controls (like close)
        header = Adw.HeaderBar()
        main_box.append(header)

        # Create the main content
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_top(24)
        content_box.set_margin_bottom(24)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        main_box.append(content_box)

        # Properties group
        properties_group = Adw.PreferencesGroup()
        properties_group.set_title(_("Stock Properties"))

        # Name field
        name_row = Adw.EntryRow()
        name_row.set_title(_("Name"))
        name_row.set_text(self.stock_item.name)
        # Connect to the "changed" signal for instant apply
        name_row.connect("changed", self.on_name_changed)
        properties_group.add(name_row)

        # Thickness field using SpinRow
        thickness_adjustment = Gtk.Adjustment(
            lower=0,
            upper=999,
            step_increment=1,
            page_increment=10,
        )
        thickness_row = Adw.SpinRow(
            title=_("Thickness"),
            subtitle=_("Material thickness"),
            adjustment=thickness_adjustment,
        )
        self.thickness_helper = UnitSpinRowHelper(
            spin_row=thickness_row,
            quantity="length",
            max_value_in_base=999,
        )
        if self.stock_item.thickness is not None:
            self.thickness_helper.set_value_in_base_units(
                self.stock_item.thickness
            )
        self.thickness_helper.changed.connect(self.on_thickness_changed)
        properties_group.add(thickness_row)

        content_box.append(properties_group)

    def _on_destroy(self, widget):
        """Clean up signal connections when dialog is destroyed."""
        if hasattr(self, "stock_item") and self.stock_item:
            self.stock_item.updated.disconnect(self.on_stock_item_updated)

    def _debounce(self, callback, *args, delay_ms=300):
        """
        Debounce a callback function to avoid excessive updates.
        """
        if self._debounce_timer:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0

        self._debounced_callback = callback
        self._debounced_args = args
        self._debounce_timer = GLib.timeout_add(
            delay_ms, self._on_debounce_timer
        )

    def _on_debounce_timer(self):
        """
        Called when the debounce timer expires.
        """
        self._debounce_timer = 0
        if self._debounced_callback:
            callback = self._debounced_callback
            args = self._debounced_args
            self._debounced_callback = None
            self._debounced_args = ()
            callback(*args)
        return False  # Don't repeat the timer

    def on_name_changed(self, entry):
        """Handle name entry changes with instant apply."""
        new_name = entry.get_text()
        if new_name and new_name != self.stock_item.name:
            self._debounce(self._apply_name_change, new_name)

    def on_thickness_changed(self, helper: UnitSpinRowHelper):
        """Handle thickness spin button changes with instant apply."""
        new_thickness = helper.get_value_in_base_units()
        if new_thickness != self.stock_item.thickness:
            self._debounce(self._apply_thickness_change, new_thickness)

    def _apply_name_change(self, new_name):
        """Apply the name change."""
        if new_name and new_name != self.stock_item.name:
            self.editor.stock.rename_stock_item(self.stock_item, new_name)

    def on_stock_item_updated(self, sender, **kwargs):
        """Update the UI when the stock item changes."""
        # Update the thickness field if it has changed
        if self.stock_item.thickness is not None:
            self.thickness_helper.set_value_in_base_units(
                self.stock_item.thickness
            )

    def _apply_thickness_change(self, new_thickness):
        """Apply the thickness change."""
        if new_thickness != self.stock_item.thickness:
            self.editor.stock.set_stock_thickness(
                self.stock_item, new_thickness
            )
