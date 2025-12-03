import logging
from gi.repository import Gtk, Adw, GLib
from typing import TYPE_CHECKING, Tuple, Optional
from ...context import get_context
from ...core.stock import StockItem
from ...shared.ui.unit_spin_row import UnitSelectorSpinRow
from .material_selector import MaterialSelectorDialog

if TYPE_CHECKING:
    from ..editor import DocEditor
    from ...core.material import Material

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
        self.set_default_size(500, 400)
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

        # Name field
        self.name_row = Adw.EntryRow()
        self.name_row.set_title(_("Name"))
        self.name_row.set_text(self.stock_item.name)
        self.name_row.connect("changed", self.on_name_changed)
        properties_group.add(self.name_row)

        # Thickness field using UnitSelectorSpinRow
        self.thickness_selector = UnitSelectorSpinRow(
            quantity="length",
            title=_("Thickness"),
            subtitle=_("Material thickness"),
            max_value_in_base=999,
        )
        if self.stock_item.thickness is not None:
            self.thickness_selector.set_value_in_base_units(
                self.stock_item.thickness
            )
        self.thickness_selector.changed.connect(self.on_thickness_changed)
        properties_group.add(self.thickness_selector.row)

        # Material display row
        self.material_row = Adw.ActionRow()
        self.material_row.set_title(_("Material"))

        # Add a button to open the material selector
        self.material_button = Gtk.Button(label=_("Select"))
        self.material_button.set_valign(Gtk.Align.CENTER)
        self.material_button.connect("clicked", self.on_select_material)
        self.material_row.add_suffix(self.material_button)

        properties_group.add(self.material_row)

        # Initialize material display
        self._update_material_display()

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

    def on_thickness_changed(self, selector: UnitSelectorSpinRow):
        """Handle thickness selector changes with instant apply."""
        new_thickness = selector.get_value_in_base_units()
        if new_thickness != self.stock_item.thickness:
            self._debounce(self._apply_thickness_change, new_thickness)

    def on_select_material(self, button: Gtk.Button):
        """Shows the material selector dialog."""
        dialog = MaterialSelectorDialog(
            parent=self, on_select_callback=self._on_material_selected
        )
        dialog.present()

    def _on_material_selected(self, material_uid: Optional[str]):
        """Callback for when a material is selected from the dialog."""
        if material_uid is not None:
            self.editor.stock.set_stock_material(self.stock_item, material_uid)

    def _apply_name_change(self, new_name):
        """Apply the name change."""
        stock_asset = self.stock_item.stock_asset
        if stock_asset and new_name and new_name != stock_asset.name:
            self.editor.asset.rename_asset(stock_asset, new_name)

    def on_stock_item_updated(self, sender, **kwargs):
        """Update the UI when the stock item changes."""
        # Update name if it has changed
        if self.name_row.get_text() != self.stock_item.name:
            self.name_row.set_text(self.stock_item.name)

        # Update the thickness field if it has changed
        if self.stock_item.thickness is not None:
            self.thickness_selector.set_value_in_base_units(
                self.stock_item.thickness
            )

        # Update the material display if it has changed
        self._update_material_display()

    def _apply_thickness_change(self, new_thickness):
        """Apply the thickness change."""
        if new_thickness != self.stock_item.thickness:
            self.editor.stock.set_stock_thickness(
                self.stock_item, new_thickness
            )

    def _update_material_display(self):
        """Update the material display label."""
        if not self.stock_item.material_uid:
            self.material_row.set_subtitle(_("None"))
            return

        material = self.stock_item.material
        if material:
            library_name = self._get_material_library_name(material)
            if library_name:
                self.material_row.set_subtitle(
                    f"{library_name}: {material.name}"
                )
            else:
                self.material_row.set_subtitle(material.name)
        else:
            self.material_row.set_subtitle(
                f"❓ {self.stock_item.material_uid}"
            )

    def _get_material_library_name(
        self, material: "Material"
    ) -> Optional[str]:
        """Get the display name of the library that contains this material."""
        material_mgr = get_context().material_mgr
        # Search through all libraries to find which one contains this material
        for library in material_mgr.get_libraries():
            if library.get_material(material.uid):
                return library.display_name

        return None
