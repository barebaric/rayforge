import logging
from gi.repository import Gtk, Gdk, Pango
from blinker import Signal
from ...core.doc import Doc
from ...core.stock import StockItem
from ...undo.models.property_cmd import ChangePropertyCommand
from ...shared.util.gtk import apply_css
from ...shared.ui.formatter import format_value
from ...config import config
from ...icons import get_icon
from .stock_properties_dialog import StockPropertiesDialog

logger = logging.getLogger(__name__)


css = """
.stockitemview entry.stockitem-title,
.stockitemview entry.stockitem-title:focus {
    border: none;
    outline: none;
    box-shadow: none;
    background: transparent;
    padding: 0;
    margin: 0;
    min-height: 0;
}

.stock-list-box > row.active-stock-row {
    background-color: @accent_bg_color;
    color: @accent_fg_color;
    border-radius: 6px;
}

.stock-list-box > row.active-stock-row .stockitemview {
    background-color: transparent;
}

.stock-list-box > row.active-stock-row entry {
    caret-color: @accent_fg_color;
}

.stock-list-box > row.active-stock-row .dim-label {
    opacity: 0.7;
}
"""


class StockItemView(Gtk.Box):
    """
    A custom widget representing a single StockItem in a list.
    It displays the stock item's name, thickness, and actions.
    """

    delete_clicked = Signal()

    def __init__(self, doc: Doc, stock_item: StockItem, editor):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        # Apply CSS globally, but only once.
        apply_css(css)
        self.set_margin_start(6)
        self.add_css_class("stockitemview")

        self.doc = doc
        self.stock_item = stock_item
        self.editor = editor

        # Icon
        icon = get_icon("stock-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        self.append(icon)

        # Box for title and subtitle
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_hexpand(True)
        content_box.set_valign(Gtk.Align.CENTER)
        self.append(content_box)

        # Title: An entry styled to look like a label
        self.name_entry = Gtk.Entry()
        self.name_entry.add_css_class("stockitem-title")
        self.name_entry.set_hexpand(False)
        self.name_entry.set_halign(Gtk.Align.START)
        self.name_entry.connect("activate", self.on_name_apply)
        self.name_entry.connect(
            "notify::has-focus", self.on_name_focus_changed
        )
        content_box.append(self.name_entry)

        # Add a key controller to handle the Escape key.
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_name_escape_pressed)
        self.name_entry.add_controller(key_controller)

        # Subtitle: A label for the thickness
        self.subtitle_label = Gtk.Label()
        self.subtitle_label.set_halign(Gtk.Align.START)
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
        content_box.append(self.subtitle_label)

        # Suffix icons
        suffix_box = Gtk.Box(spacing=6)
        suffix_box.set_valign(Gtk.Align.CENTER)
        self.append(suffix_box)

        self.visibility_on_icon = get_icon("visibility-on-symbolic")
        self.visibility_off_icon = get_icon("visibility-off-symbolic")

        properties_icon = get_icon("document-properties-symbolic")
        self.properties_button = Gtk.Button(child=properties_icon)
        self.properties_button.set_tooltip_text(_("Edit stock properties"))
        self.properties_button.connect("clicked", self.on_properties_clicked)
        suffix_box.append(self.properties_button)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.set_tooltip_text(_("Delete this stock item"))
        self.delete_button.connect("clicked", self.on_delete_clicked)
        suffix_box.append(self.delete_button)

        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.set_child(self.visibility_on_icon)
        self.visibility_button.connect("clicked", self.on_button_view_click)
        suffix_box.append(self.visibility_button)

        # Connect to model signals to stay in sync
        self.stock_item.updated.connect(self.on_stock_item_changed)

        # Connect to config changes to update when unit preferences change
        self._config_handler_id = config.changed.connect(
            self.on_config_changed
        )

        # Perform initial UI sync
        self.on_stock_item_changed(self.stock_item)

    def do_destroy(self):
        """Overrides GObject.Object.do_destroy to disconnect signals."""
        self.stock_item.updated.disconnect(self.on_stock_item_changed)
        if hasattr(self, "_config_handler_id"):
            config.changed.disconnect(self._config_handler_id)

    def on_name_escape_pressed(self, controller, keyval, keycode, state):
        """Handler for the 'key-pressed' signal to catch Escape."""
        if keyval == Gdk.KEY_Escape:
            # Revert any changes and remove focus from the entry.
            self.name_entry.set_text(self.stock_item.name)
            list_box = self.get_ancestor(Gtk.ListBox)
            if list_box:
                list_box.grab_focus()
            return True  # Indicate that the event has been handled
        return False  # Allow other key presses to be processed normally

    def on_name_focus_changed(self, entry, gparam):
        # This triggers when focus is lost.
        if not entry.has_focus():
            self.on_name_apply(entry)

    def on_stock_item_changed(self, sender, **kwargs):
        """Updates the UI when the underlying stock item changes."""
        self.update_ui()

    def on_config_changed(self, sender, **kwargs):
        """Updates the UI when config changes (e.g., unit preferences)."""
        self.update_ui()

    def on_delete_clicked(self, button):
        """Emits a signal when the delete button is clicked."""
        self.delete_clicked.send(self)

    def on_name_apply(self, widget, *args):
        """Handles applying the name change from the entry."""
        new_name = self.name_entry.get_text()

        if not new_name.strip() or new_name == self.stock_item.name:
            self.name_entry.set_text(self.stock_item.name)
            return

        command = ChangePropertyCommand(
            target=self.stock_item,
            property_name="name",
            new_value=new_name,
            setter_method_name="set_name",
            name=_("Rename stock item"),
        )
        self.doc.history_manager.execute(command)

    def on_properties_clicked(self, button):
        """Opens the properties dialog for the stock item."""
        # Get the root window
        root = self.get_root()
        if root and isinstance(root, Gtk.Window):
            dialog = StockPropertiesDialog(root, self.stock_item, self.doc)
            dialog.present()

    def on_button_view_click(self, button):
        """Creates an undoable command to toggle the stock item visibility."""
        new_visibility = button.get_active()
        if new_visibility == self.stock_item.visible:
            return

        # Use the editor's stock command
        self.editor.stock.toggle_stock_visibility(self.stock_item)

    def update_ui(self):
        """Synchronizes the widget's state with the stock item data."""
        if not self.name_entry.has_focus():
            self.name_entry.set_text(self.stock_item.name)

        # Update subtitle
        if self.stock_item.thickness is not None:
            # Format the thickness value using the user's preferred unit
            formatted_thickness = format_value(
                self.stock_item.thickness, "length"
            )
            subtitle_text = _("Thickness: {thickness}").format(
                thickness=formatted_thickness
            )
        else:
            subtitle_text = _("No thickness specified")

        self.subtitle_label.set_label(subtitle_text)
        self.subtitle_label.set_tooltip_text(subtitle_text)

        # Sync the visibility button's state and icon with the model.
        self.visibility_button.set_active(self.stock_item.visible)
        if self.stock_item.visible:
            self.visibility_button.set_child(self.visibility_on_icon)
        else:
            self.visibility_button.set_child(self.visibility_off_icon)
