import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, Gdk, GLib, Gtk

from ...core.stock import StockItem
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from .material_selector import MaterialRow

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class StockPropertiesDialog(PatchedDialogWindow):
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
        self._debounced_args: tuple = ()

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

        # Thickness field
        self.thickness_row = LengthSpinRow(
            _("Thickness"),
            _("Material thickness"),
            upper=999,
        )
        if self.stock_item.thickness is not None:
            self.thickness_row.set_value_in_base_units(
                self.stock_item.thickness
            )
        self.thickness_row.value_changed.connect(self.on_thickness_changed)
        properties_group.add(self.thickness_row)

        # Material display row
        self.material_row = MaterialRow(
            _("Material"),
            on_select=self._on_material_selected,
        )
        properties_group.add(self.material_row)

        # Per-instance color row (only usable for tintable materials; for
        # other materials the color comes from the material definition).
        self.color_row = Adw.ActionRow(title=_("Color"))
        color_dialog = Gtk.ColorDialog()
        color_dialog.set_with_alpha(False)
        self.color_button = Gtk.ColorDialogButton(dialog=color_dialog)
        self.color_button.set_size_request(45, 45)
        self.color_button.set_valign(Gtk.Align.CENTER)
        self.color_button.connect("notify::rgba", self._on_color_set)
        # Clear button reverts to the material's default color (inherit).
        self._clear_color_button = Gtk.Button(child=get_icon("clear-symbolic"))
        self._clear_color_button.add_css_class("flat")
        self._clear_color_button.set_valign(Gtk.Align.CENTER)
        self._clear_color_button.set_tooltip_text(
            _("Use the material's default color")
        )
        self._clear_color_button.connect("clicked", self._on_clear_color)
        self._clear_color_button.set_visible(False)
        self.color_row.add_suffix(self.color_button)
        self.color_row.add_suffix(self._clear_color_button)
        properties_group.add(self.color_row)
        self._updating_color = False

        # Initialize material display
        self.material_row.set_material(self.stock_item.material)
        self._update_color_display()

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

    def on_thickness_changed(self, row: LengthSpinRow):
        """Handle thickness changes with instant apply."""
        new_thickness = row.get_value_in_base_units()
        if new_thickness != self.stock_item.thickness:
            self._debounce(self._apply_thickness_change, new_thickness)

    def _on_material_selected(self, material_uid: str | None):
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
            self.thickness_row.set_value_in_base_units(
                self.stock_item.thickness
            )

        # Update the material display if it has changed
        self.material_row.set_material(self.stock_item.material)
        self._update_color_display()

    def _apply_thickness_change(self, new_thickness):
        """Apply the thickness change."""
        if new_thickness != self.stock_item.thickness:
            self.editor.stock.set_stock_thickness(
                self.stock_item, new_thickness
            )

    def _on_color_set(self, button: Gtk.ColorDialogButton, pspec=None):
        """Apply a per-instance color chosen in the color picker."""
        if self._updating_color:
            return
        rgba = button.get_rgba()
        color = (
            f"#{int(rgba.red * 255):02x}"
            f"{int(rgba.green * 255):02x}"
            f"{int(rgba.blue * 255):02x}"
        )
        self.editor.stock.set_stock_color(self.stock_item, color)

    def _on_clear_color(self, button: Gtk.Button):
        """Clear the per-instance color (revert to the material default)."""
        self.editor.stock.set_stock_color(self.stock_item, None)

    def _update_color_display(self):
        """Refresh the per-instance color row to match the stock item."""
        material = self.stock_item.material
        tintable = material is not None and material.appearance.tintable
        self.color_row.set_sensitive(tintable)

        # Only show the clear button while a per-instance override exists.
        self._clear_color_button.set_visible(
            tintable and self.stock_item.color is not None
        )

        effective = self.stock_item.get_effective_color()
        if not tintable:
            self.color_row.set_subtitle(_("Set by the material definition"))
        elif self.stock_item.color == StockItem.COLOR_NONE:
            self.color_row.set_subtitle(_("No color"))
        elif self.stock_item.color:
            self.color_row.set_subtitle(
                _("{color} (custom)").format(color=self.stock_item.color)
            )
        elif effective:
            self.color_row.set_subtitle(
                _("{color} (material default)").format(color=effective)
            )
        else:
            self.color_row.set_subtitle(_("Material default (no color)"))

        # Show the effective color in the picker without triggering apply.
        self._updating_color = True
        if effective:
            rgba = Gdk.RGBA()
            if rgba.parse(effective):
                self.color_button.set_rgba(rgba)
        self._updating_color = False
