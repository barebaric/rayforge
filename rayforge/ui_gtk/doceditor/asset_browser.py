import logging
from typing import TYPE_CHECKING, Optional, cast

from gi.repository import Gdk, GLib, Gtk, Pango
from blinker import Signal
from gettext import gettext as _

from ...core.asset import IAsset
from ...core.asset_registry import asset_type_registry
from ...core.doc import Doc
from ...core.stock import StockItem
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.popover_menu import PopoverMenu

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 64
CARD_SIZE = THUMBNAIL_SIZE + 36

css = """
.asset-browser {
    padding: 6px;
}
.asset-flowbox > flowboxchild {
    padding: 0;
    margin: 0;
    background: none;
    border: none;
    outline: none;
    box-shadow: none;
    min-width: 0;
    min-height: 0;
}
.asset-card {
    background: @card_bg_color;
    border-radius: 6px;
    padding: 4px;
}
.asset-card:hover {
    background: alpha(@theme_selected_bg_color, 0.08);
}
.asset-card-label {
    font-size: 13px;
    margin-top: 2px;
}
.asset-type-icon {
    opacity: 0.5;
}
"""


class AssetCard(Gtk.Box):
    """A thumbnail card for a single asset."""

    def __init__(self, asset: IAsset):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.asset = asset
        self.add_css_class("asset-card")
        self._texture: Optional[Gdk.Texture] = None
        self.set_size_request(CARD_SIZE, CARD_SIZE)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.START)
        self.set_hexpand(False)
        self.set_vexpand(False)

        self._picture = Gtk.Picture()
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._picture.set_size_request(THUMBNAIL_SIZE, THUMBNAIL_SIZE)

        self._type_icon = get_icon(asset.display_icon_name)
        self._type_icon.set_halign(Gtk.Align.END)
        self._type_icon.set_valign(Gtk.Align.START)
        self._type_icon.add_css_class("asset-type-icon")
        self._type_icon.set_tooltip_text(asset.type_display_name)

        overlay = Gtk.Overlay()
        overlay.set_child(self._picture)
        overlay.add_overlay(self._type_icon)

        self._icon_fallback = Gtk.Image()
        self._icon_fallback.set_pixel_size(THUMBNAIL_SIZE // 2)
        self._icon_fallback.set_visible(False)

        image_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        image_box.append(overlay)
        image_box.append(self._icon_fallback)

        self._label = Gtk.Label()
        self._label.add_css_class("asset-card-label")
        self._label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self._label.set_max_width_chars(10)

        self.append(image_box)
        self.append(self._label)
        self.refresh()

    def invalidate(self):
        self._texture = None

    def refresh(self):
        if self._texture is None:
            png = self.asset.get_thumbnail(THUMBNAIL_SIZE)
            if png:
                bytes_data = GLib.Bytes.new(png)
                self._texture = Gdk.Texture.new_from_bytes(bytes_data)

        if self._texture:
            self._picture.set_paintable(self._texture)
            self._picture.set_visible(True)
            self._icon_fallback.set_visible(False)
        else:
            self._picture.set_visible(False)
            self._icon_fallback.set_from_icon_name(
                self.asset.display_icon_name
            )
            self._icon_fallback.set_visible(True)

        self._label.set_label(self.asset.name)


class AssetBrowser(Gtk.Box):
    """
    A bottom-panel widget that displays document assets as a grid of
    thumbnails.
    """

    def __init__(self, editor: "DocEditor", **kwargs):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, **kwargs)
        self.add_asset_requested = Signal()
        self.asset_activated = Signal()
        apply_css(css)
        self.add_css_class("asset-browser")
        self.editor = editor
        self.doc = editor.doc

        self._flowbox = Gtk.FlowBox()
        self._flowbox.add_css_class("asset-flowbox")
        self._flowbox.set_column_spacing(6)
        self._flowbox.set_row_spacing(6)
        self._flowbox.set_min_children_per_line(3)
        self._flowbox.set_max_children_per_line(200)
        self._flowbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._flowbox.set_homogeneous(False)
        self._flowbox.set_valign(Gtk.Align.START)
        self._flowbox.connect(
            "selected-children-changed", self._on_selection_changed
        )

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_child(self._flowbox)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        self.append(scrolled)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        toolbar.set_spacing(4)
        toolbar.set_margin_start(4)
        toolbar.set_margin_top(4)

        self._add_btn = Gtk.Button(icon_name="list-add-symbolic")
        self._add_btn.add_css_class("flat")
        self._add_btn.set_tooltip_text(_("Add Asset"))
        self._add_btn.connect("clicked", self._on_add_clicked)
        toolbar.append(self._add_btn)
        self.append(toolbar)

        self._cards: dict[str, list] = {}
        self._connect_signals()
        self._sync_cards(self.doc)

    def set_doc(self, doc: Doc):
        """Updates the widget to track a new document instance."""
        if self.doc == doc:
            return
        self._disconnect_signals()
        self._disconnect_all_asset_signals()
        self.doc = doc
        self._connect_signals()
        self._sync_cards(doc)

    def _connect_signals(self):
        self.doc.updated.connect(self._on_doc_changed)
        self.doc.descendant_added.connect(self._on_doc_changed)
        self.doc.descendant_removed.connect(self._on_doc_changed)

    def _disconnect_signals(self):
        self.doc.updated.disconnect(self._on_doc_changed)
        self.doc.descendant_added.disconnect(self._on_doc_changed)
        self.doc.descendant_removed.disconnect(self._on_doc_changed)

    def _connect_asset_signal(self, card: AssetCard):
        asset = card.asset
        signal = getattr(asset, "updated", None)
        if signal is not None:
            handler = signal.connect(
                lambda sender, uid=asset.uid: self._on_asset_updated(uid)
            )
            self._cards[asset.uid] = [card, signal, handler]

    def _disconnect_all_asset_signals(self):
        for uid, (card, signal, handler) in self._cards.items():
            signal.disconnect(handler)
        self._cards.clear()

    def _on_asset_updated(self, uid: str):
        entry = self._cards.get(uid)
        if entry is None:
            return
        card = entry[0]
        logger.debug("Refreshing thumbnail for asset %s", uid[:8])
        card.invalidate()
        card.refresh()

    def _on_doc_changed(self, sender, **kwargs):
        child = kwargs.get("child")
        if child and not isinstance(child, StockItem):
            return
        self._sync_cards(self.doc)

    def _sync_cards(self, doc: Doc):
        visible = [a for a in doc.get_all_assets() if not a.hidden]

        old_cards: dict[str, AssetCard] = {}
        for uid, (card, signal, handler) in self._cards.items():
            signal.disconnect(handler)
            old_cards[uid] = card
        self._cards.clear()

        child = self._flowbox.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            fb_child = cast(Gtk.FlowBoxChild, child)
            card = fb_child.get_child()
            if card:
                fb_child.set_child(None)
            self._flowbox.remove(fb_child)
            child = next_child

        for asset in visible:
            card = old_cards.pop(asset.uid, None)
            if card is not None:
                card.asset = asset
                card.invalidate()
                card.refresh()
            else:
                card = AssetCard(asset)
                drag_source = Gtk.DragSource()
                drag_source.set_actions(Gdk.DragAction.COPY)
                drag_source.connect("prepare", self._on_drag_prepare)
                card.add_controller(drag_source)

            fb_child = Gtk.FlowBoxChild()
            fb_child.set_halign(Gtk.Align.CENTER)
            fb_child.set_valign(Gtk.Align.START)
            fb_child.set_hexpand(False)
            fb_child.set_vexpand(False)
            fb_child.set_child(card)
            self._flowbox.append(fb_child)
            self._connect_asset_signal(card)

    def _on_drag_prepare(self, source, x, y):
        card = source.get_widget()
        asset = card.asset
        if not getattr(asset, "is_draggable_to_canvas", False):
            return None
        uid = str(asset.uid)
        provider = Gdk.ContentProvider.new_for_value(uid)
        snapshot = Gtk.Snapshot()
        card.do_snapshot(card, snapshot)
        paintable = snapshot.to_paintable()
        if paintable:
            source.set_icon(paintable, int(x), int(y))
        return provider

    def _on_selection_changed(self, flowbox):
        selected = flowbox.get_selected_children()
        if selected:
            child = selected[0]
            card = child.get_child()
            if card:
                self.asset_activated.send(self, asset=card.asset)
            flowbox.unselect_child(child)

    def _on_add_clicked(self, button):
        asset_types = []
        for type_name, asset_class in asset_type_registry.all_types().items():
            if asset_class.is_addable:
                display_name = f"Add {type_name.title()}"
                asset_types.append((_(display_name), type_name))

        popup = PopoverMenu(items=asset_types)
        popup.set_parent(button)
        popup.popup()
        popup.connect("closed", self._on_add_popup_closed)

    def _on_add_popup_closed(self, popup):
        if popup.selected_item:
            self.add_asset_requested.send(self, type_name=popup.selected_item)
