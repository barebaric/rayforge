import logging
from typing import TYPE_CHECKING
from gi.repository import Gtk, Gdk, Pango
from blinker import Signal
from ...core.doc import Doc
from ...core.stock_asset import StockAsset
from ...core.sketcher.sketch import Sketch
from ...shared.ui.formatter import format_value
from ...context import get_context
from ...icons import get_icon
from .stock_properties_dialog import StockPropertiesDialog

if TYPE_CHECKING:
    from ..editor import DocEditor

logger = logging.getLogger(__name__)


class SketchAssetRowWidget(Gtk.Box):
    """
    A custom widget representing a single Sketch asset in a list.
    """

    def __init__(self, doc: Doc, asset: Sketch, editor: "DocEditor"):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.add_css_class("asset-row-view")
        self.add_css_class("sketch-asset-row")

        self.edit_clicked = Signal()
        self.delete_clicked = Signal()

        self.set_margin_start(6)
        self.doc = doc
        self.asset = asset
        self.editor = editor

        # Icon
        icon = get_icon(asset.display_icon_name)
        icon.set_valign(Gtk.Align.CENTER)
        self.append(icon)

        # Box for title and subtitle
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_hexpand(True)
        content_box.set_valign(Gtk.Align.CENTER)
        self.append(content_box)

        # Title: An entry styled to look like a label for renaming the asset
        self.name_entry = Gtk.Entry()
        self.name_entry.add_css_class("asset-title-entry")
        self.name_entry.set_hexpand(False)
        self.name_entry.set_halign(Gtk.Align.START)
        self.name_entry.connect("activate", self.on_name_apply)
        self.name_entry.connect(
            "notify::has-focus", self.on_name_focus_changed
        )
        content_box.append(self.name_entry)

        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_name_escape_pressed)
        self.name_entry.add_controller(key_controller)

        # Subtitle: A label for the sketch parameters count
        self.subtitle_label = Gtk.Label()
        self.subtitle_label.set_halign(Gtk.Align.START)
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
        content_box.append(self.subtitle_label)

        # Suffix icons
        suffix_box = Gtk.Box(spacing=6)
        suffix_box.set_valign(Gtk.Align.CENTER)
        self.append(suffix_box)

        # Edit button
        edit_icon = get_icon("edit-symbolic")
        self.edit_button = Gtk.Button(child=edit_icon)
        self.edit_button.set_tooltip_text(_("Edit this sketch"))
        self.edit_button.connect("clicked", self.on_edit_clicked)
        suffix_box.append(self.edit_button)

        # Delete button
        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.set_tooltip_text(_("Delete this sketch"))
        self.delete_button.connect("clicked", self.on_delete_clicked)
        suffix_box.append(self.delete_button)

        self.update_ui()

    def get_drag_content(self) -> Gdk.ContentProvider:
        """Provides the content for a drag operation."""
        logger.debug(
            "Providing drag content for sketch UID: %s",
            repr(self.asset.uid),
        )
        return Gdk.ContentProvider.new_for_value(str(self.asset.uid))

    def on_edit_clicked(self, button: Gtk.Button):
        self.edit_clicked.send(self)

    def on_delete_clicked(self, button: Gtk.Button):
        self.delete_clicked.send(self)

    def on_name_escape_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self.name_entry.set_text(self.asset.name)
            list_box = self.get_ancestor(Gtk.ListBox)
            if list_box:
                list_box.grab_focus()
            return True
        return False

    def on_name_focus_changed(self, entry: Gtk.Entry, gparam):
        if not entry.has_focus():
            self.on_name_apply(entry)

    def on_name_apply(self, widget: Gtk.Widget, *args):
        new_name = self.name_entry.get_text()
        self.editor.asset.rename_asset(self.asset, new_name)

    def update_ui(self):
        """Synchronizes the widget's state with the sketch data."""
        if not self.name_entry.has_focus():
            self.name_entry.set_text(self.asset.name)
        self.name_entry.set_tooltip_text(
            f"{self.asset.name} ({self.asset.uid})"
        )

        param_count = len(self.asset.input_parameters)
        if param_count == 0:
            subtitle_text = _("No parameters")
        elif param_count == 1:
            subtitle_text = _("1 parameter")
        else:
            subtitle_text = _("{count} parameters").format(count=param_count)

        self.subtitle_label.set_label(subtitle_text)
        self.subtitle_label.set_tooltip_text(subtitle_text)


class StockAssetRowWidget(Gtk.Box):
    """
    A custom widget representing a single StockAsset in a list.
    It has no knowledge of any StockItem instances.
    """

    delete_clicked = Signal()

    def __init__(self, doc: Doc, asset: StockAsset, editor: "DocEditor"):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.add_css_class("asset-row-view")
        self.add_css_class("stock-asset-row")

        self.set_margin_start(6)
        self.doc = doc
        self.asset = asset  # This is the StockAsset
        self.editor = editor

        # Icon
        icon = get_icon(asset.display_icon_name)
        icon.set_valign(Gtk.Align.CENTER)
        self.append(icon)

        # Box for title and subtitle
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_hexpand(True)
        content_box.set_valign(Gtk.Align.CENTER)
        self.append(content_box)

        # Title: An entry styled to look like a label for renaming the asset
        self.name_entry = Gtk.Entry()
        self.name_entry.add_css_class("asset-title-entry")
        self.name_entry.set_hexpand(False)
        self.name_entry.set_halign(Gtk.Align.START)
        self.name_entry.connect("activate", self.on_name_apply)
        self.name_entry.connect(
            "notify::has-focus", self.on_name_focus_changed
        )
        content_box.append(self.name_entry)

        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_name_escape_pressed)
        self.name_entry.add_controller(key_controller)

        # Subtitle
        self.subtitle_label = Gtk.Label()
        self.subtitle_label.set_halign(Gtk.Align.START)
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
        content_box.append(self.subtitle_label)

        # Suffix icons
        suffix_box = Gtk.Box(spacing=6)
        suffix_box.set_valign(Gtk.Align.CENTER)
        self.append(suffix_box)

        properties_icon = get_icon("document-properties-symbolic")
        self.properties_button = Gtk.Button(child=properties_icon)
        self.properties_button.set_tooltip_text(_("Edit stock properties"))
        self.properties_button.connect("clicked", self.on_properties_clicked)
        suffix_box.append(self.properties_button)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.set_tooltip_text(_("Delete this stock asset"))
        self.delete_button.connect("clicked", self.on_delete_clicked)
        suffix_box.append(self.delete_button)

        # Listen to the asset for changes to its data
        self.asset.updated.connect(self.on_asset_changed)

        self._config_handler_id = get_context().config.changed.connect(
            self.on_config_changed
        )
        self.on_asset_changed(self.asset)

    def do_destroy(self):
        self.asset.updated.disconnect(self.on_asset_changed)
        if hasattr(self, "_config_handler_id"):
            get_context().config.changed.disconnect(self._config_handler_id)

    def on_name_escape_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self.name_entry.set_text(self.asset.name)
            list_box = self.get_ancestor(Gtk.ListBox)
            if list_box:
                list_box.grab_focus()
            return True
        return False

    def on_name_focus_changed(self, entry: Gtk.Entry, gparam):
        if not entry.has_focus():
            self.on_name_apply(entry)

    def on_asset_changed(self, sender, **kwargs):
        self.update_ui()

    def on_config_changed(self, sender, **kwargs):
        self.update_ui()

    def on_delete_clicked(self, button: Gtk.Button):
        self.delete_clicked.send(self)

    def on_name_apply(self, widget: Gtk.Widget, *args):
        new_name = self.name_entry.get_text()
        self.editor.asset.rename_asset(self.asset, new_name)

    def on_properties_clicked(self, button: Gtk.Button):
        # Find the FIRST stock item that uses this asset to show properties
        # This is a reasonable compromise for the properties dialog.
        stock_item_instance = next(
            (
                item
                for item in self.doc.stock_items
                if item.stock_asset_uid == self.asset.uid
            ),
            None,
        )
        if not stock_item_instance:
            logger.warning(
                "Properties clicked for a stock asset with no instances."
            )
            # Optionally, show a dialog explaining this. For now, do nothing.
            return

        root = self.get_root()
        if root and isinstance(root, Gtk.Window):
            dialog = StockPropertiesDialog(
                root, stock_item_instance, self.editor
            )
            dialog.present()

    def update_ui(self):
        if not self.name_entry.has_focus():
            self.name_entry.set_text(self.asset.name)

        if self.asset.thickness is not None:
            formatted = format_value(self.asset.thickness, "length")
            subtitle = _("Thickness: {thickness}").format(thickness=formatted)
        else:
            subtitle = _("No thickness specified")

        self.subtitle_label.set_label(subtitle)
        self.subtitle_label.set_tooltip_text(subtitle)
