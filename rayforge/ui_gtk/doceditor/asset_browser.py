import copy
import json
import logging
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, cast

from gi.repository import Gdk, Gio, GLib, Graphene, Gtk, Pango
from blinker import Signal
from gettext import gettext as _

from ...core.asset import IAsset
from ...core.asset_registry import asset_type_registry
from ...core.doc import Doc
from ...core.geometry_provider import IGeometryProvider
from ...core.stock import StockItem
from ...core.undo import ListItemCommand
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
    padding: 9px;
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
.asset-flowbox > flowboxchild.selected .asset-card {
    background: alpha(@theme_selected_bg_color, 0.25);
    border: 2px solid @theme_selected_bg_color;
}
.asset-card {
    background: @card_bg_color;
    border-radius: 6px;
    padding: 4px;
    border: 2px solid transparent;
}
.asset-card:hover {
    background: alpha(@theme_selected_bg_color, 0.08);
}
.asset-card-label {
    font-size: 13px;
    margin-top: 2px;
}
.asset-type-icon {
    opacity: 0.9;
    background-color: alpha(@card_bg_color, 0.95);
    border-radius: 4px;
}
.asset-browser-empty {
    padding: 24px;
}
.asset-browser-empty-icon {
    opacity: 0.15;
}
.asset-browser-empty-buttons button {
    padding: 12px 24px;
    font-size: 1.1em;
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

        self._draw_area = Gtk.DrawingArea()
        self._draw_area.set_content_width(THUMBNAIL_SIZE)
        self._draw_area.set_content_height(THUMBNAIL_SIZE)
        self._draw_area.set_draw_func(self._draw_thumbnail)

        self._icon_fallback = Gtk.Image()
        self._icon_fallback.set_pixel_size(THUMBNAIL_SIZE // 2)
        self._icon_fallback.set_visible(False)

        image_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        image_box.append(self._draw_area)
        image_box.append(self._icon_fallback)

        type_icon = get_icon(asset.display_icon_name)
        type_icon.set_pixel_size(12)
        type_icon.set_margin_end(4)
        type_icon.set_tooltip_text(asset.type_display_name)

        self._label = Gtk.Label()
        self._label.add_css_class("asset-card-label")
        self._label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self._label.set_max_width_chars(10)

        label_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        label_box.set_halign(Gtk.Align.CENTER)
        label_box.set_margin_top(4)
        label_box.append(type_icon)
        label_box.append(self._label)

        self.append(image_box)
        self.append(label_box)
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
            self._draw_area.set_visible(True)
            self._icon_fallback.set_visible(False)
        else:
            self._draw_area.set_visible(False)
            self._icon_fallback.set_from_icon_name(
                self.asset.display_icon_name
            )
            self._icon_fallback.set_visible(True)

        self._label.set_label(self.asset.name)
        self.set_tooltip_text(self.asset.name)

    def _draw_thumbnail(self, area, cr, width, height):
        if self._texture is None:
            return
        tex_w = self._texture.get_intrinsic_width()
        tex_h = self._texture.get_intrinsic_height()
        if tex_w <= 0 or tex_h <= 0:
            return
        scale = min(width / tex_w, height / tex_h)
        draw_w = tex_w * scale
        draw_h = tex_h * scale
        x = (width - draw_w) / 2
        y = (height - draw_h) / 2
        snapshot = Gtk.Snapshot()
        rect = Graphene.Rect()
        rect.init(x, y, draw_w, draw_h)
        snapshot.append_texture(self._texture, rect)
        node = snapshot.to_node()
        if node is not None:
            node.draw(cr)


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

        self._main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._main_box.set_hexpand(True)
        self._main_box.set_vexpand(True)
        self.append(self._main_box)

        self._flowbox = Gtk.FlowBox()
        self._flowbox.add_css_class("asset-flowbox")
        self._flowbox.set_column_spacing(6)
        self._flowbox.set_row_spacing(6)
        self._flowbox.set_min_children_per_line(3)
        self._flowbox.set_max_children_per_line(200)
        self._flowbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self._flowbox.set_homogeneous(False)
        self._flowbox.set_valign(Gtk.Align.START)
        self._flowbox.set_activate_on_single_click(False)
        self._flowbox.connect("child-activated", self._on_child_activated)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self._flowbox.add_controller(key_controller)

        click_gesture = Gtk.GestureClick()
        click_gesture.set_button(1)
        click_gesture.connect("pressed", self._on_flowbox_pressed)
        click_gesture.connect("released", self._on_flowbox_released)
        self._flowbox.add_controller(click_gesture)

        self._scrolled = Gtk.ScrolledWindow()
        self._scrolled.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self._scrolled.set_child(self._flowbox)
        self._scrolled.set_hexpand(True)
        self._scrolled.set_vexpand(True)
        self._main_box.append(self._scrolled)

        right_click = Gtk.GestureClick()
        right_click.set_button(Gdk.BUTTON_SECONDARY)
        right_click.connect("pressed", self._on_right_click_pressed)
        self._scrolled.add_controller(right_click)

        self._empty_state = self._create_empty_state()
        self._main_box.append(self._empty_state)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        toolbar.set_spacing(4)
        toolbar.set_margin_start(9)
        toolbar.set_margin_top(9)

        self._add_btn = Gtk.Button(child=get_icon("add-symbolic"))
        self._add_btn.add_css_class("flat")
        self._add_btn.set_tooltip_text(_("Add Asset"))
        self._add_btn.connect("clicked", self._on_add_clicked)
        toolbar.append(self._add_btn)
        self.append(toolbar)

        self._cards: dict[str, list] = {}
        self._selected_uids: set[str] = set()
        self._asset_clipboard: List[Dict] = []
        self._context_popover: Optional[Gtk.PopoverMenu] = None
        self._connect_signals()
        self._sync_cards(self.doc)

    def _create_empty_state(self) -> Gtk.Box:
        empty_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        empty_box.add_css_class("asset-browser-empty")
        empty_box.set_halign(Gtk.Align.CENTER)
        empty_box.set_valign(Gtk.Align.CENTER)
        empty_box.set_margin_top(24)
        empty_box.set_margin_bottom(24)
        empty_box.set_hexpand(True)
        empty_box.set_vexpand(True)

        icon = get_icon("sketch-edit-symbolic")
        icon.set_pixel_size(128)
        icon.add_css_class("asset-browser-empty-icon")
        empty_box.append(icon)

        buttons_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.CENTER,
        )
        buttons_box.add_css_class("asset-browser-empty-buttons")

        add_stock_btn = Gtk.Button(label=_("Add Stock"))
        add_stock_btn.connect("clicked", self._on_empty_add_clicked, "stock")
        buttons_box.append(add_stock_btn)

        add_sketch_btn = Gtk.Button(label=_("Add Sketch"))
        add_sketch_btn.connect("clicked", self._on_empty_add_clicked, "sketch")
        buttons_box.append(add_sketch_btn)

        empty_box.append(buttons_box)

        return empty_box

    def _on_empty_add_clicked(self, button: Gtk.Button, type_name: str):
        self.add_asset_requested.send(self, type_name=type_name)

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
        handler = asset.updated.connect(
            lambda sender, uid=asset.uid: self._on_asset_updated(uid)
        )
        self._cards[asset.uid] = [card, asset.updated, handler]

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
        has_assets = len(visible) > 0

        self._scrolled.set_visible(has_assets)
        self._empty_state.set_visible(not has_assets)

        old_cards: dict[str, AssetCard] = {}
        for uid, (card, signal, handler) in self._cards.items():
            signal.disconnect(handler)
            old_cards[uid] = card
        self._cards.clear()

        child = self._flowbox.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            if isinstance(child, Gtk.FlowBoxChild):
                card = child.get_child()
                if card:
                    child.set_child(None)
                self._flowbox.remove(child)
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
        if not asset.is_draggable_to_canvas:
            return None

        uids: list[str] = []

        if self._selected_uids and asset.uid in self._selected_uids:
            for uid in self._selected_uids:
                entry = self._cards.get(uid)
                if entry and entry[0].asset.is_draggable_to_canvas:
                    uids.append(str(uid))
        else:
            uids.append(str(asset.uid))

        data = json.dumps(uids)
        provider = Gdk.ContentProvider.new_for_value(data)
        paintable = Gtk.WidgetPaintable.new(card)
        source.set_icon(paintable, int(x), int(y))
        return provider

    def _update_selection_visual(self):
        child = self._flowbox.get_first_child()
        while child:
            card = (
                child.get_child()
                if isinstance(child, Gtk.FlowBoxChild)
                else None
            )
            uid = card.asset.uid if isinstance(card, AssetCard) else None
            if uid and uid in self._selected_uids:
                child.add_css_class("selected")
            else:
                child.remove_css_class("selected")
            child = child.get_next_sibling()

    def _on_flowbox_pressed(self, gesture, n_press, x, y):
        modifier = (
            gesture.get_current_event_state()
            if gesture.get_current_event()
            else 0
        )
        ctrl = bool(modifier & Gdk.ModifierType.CONTROL_MASK)
        shift = bool(modifier & Gdk.ModifierType.SHIFT_MASK)

        widget = self._flowbox.pick(x, y, Gtk.PickFlags.DEFAULT)
        clicked_child = None
        while widget and widget != self._flowbox:
            if isinstance(widget, Gtk.FlowBoxChild):
                clicked_child = widget
                break
            widget = widget.get_parent()

        if clicked_child is None:
            if not ctrl and not shift:
                self._selected_uids.clear()
                self._update_selection_visual()
            return

        card = cast(AssetCard, clicked_child.get_child())
        if card is None:
            return

        uid = card.asset.uid
        if n_press == 1:
            if ctrl:
                if uid in self._selected_uids:
                    self._selected_uids.discard(uid)
                else:
                    self._selected_uids.add(uid)
            elif shift and self._selected_uids:
                self._select_range(uid)
            elif uid not in self._selected_uids:
                self._selected_uids = {uid}
            self._update_selection_visual()

    def _select_range(self, uid):
        all_uids = []
        child = self._flowbox.get_first_child()
        while child:
            card = (
                child.get_child()
                if isinstance(child, Gtk.FlowBoxChild)
                else None
            )
            if isinstance(card, AssetCard):
                all_uids.append(card.asset.uid)
            child = child.get_next_sibling()

        if not all_uids or uid not in all_uids:
            return

        anchor_uid = next(iter(self._selected_uids))
        if anchor_uid not in all_uids:
            self._selected_uids = {uid}
            return

        anchor_idx = all_uids.index(anchor_uid)
        click_idx = all_uids.index(uid)
        lo = min(anchor_idx, click_idx)
        hi = max(anchor_idx, click_idx)
        self._selected_uids = set(all_uids[lo : hi + 1])

    def _on_flowbox_released(self, gesture, n_press, x, y):
        pass

    def _on_child_activated(self, flowbox, child):
        card = cast(AssetCard, child.get_child())
        if card:
            self.asset_activated.send(self, asset=card.asset)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Delete:
            self.delete_selected_assets()
            return True
        return False

    def _on_right_click_pressed(self, gesture, n_press, x, y):
        widget = self._scrolled.pick(x, y, Gtk.PickFlags.DEFAULT)
        clicked_child = None
        while widget and widget != self._scrolled:
            if isinstance(widget, Gtk.FlowBoxChild):
                clicked_child = widget
                break
            widget = widget.get_parent()

        if clicked_child is None:
            self._show_empty_context_menu(gesture)
        else:
            card = cast(AssetCard, clicked_child.get_child())
            if card:
                if card.asset.uid not in self._selected_uids:
                    self._selected_uids = {card.asset.uid}
                    self._update_selection_visual()
                self._show_asset_context_menu(gesture, card.asset)

    def _popup_context_menu(self, menu: Gio.Menu, gesture: Gtk.Gesture):
        if self._context_popover:
            self._context_popover.unparent()
        popover = Gtk.PopoverMenu.new_from_model(menu)
        popover.set_parent(self._scrolled)
        popover.set_has_arrow(False)
        popover.set_position(Gtk.PositionType.RIGHT)
        ok, rect = gesture.get_bounding_box()
        if ok:
            popover.set_pointing_to(rect)
        self._context_popover = popover
        popover.popup()

    def _show_empty_context_menu(self, gesture):
        menu = Gio.Menu.new()
        menu.append_item(Gio.MenuItem.new(_("New Sketch"), "win.new_sketch"))
        menu.append_item(Gio.MenuItem.new(_("New Stock"), "win.add-stock"))
        menu.append_section(None, Gio.Menu.new())
        menu.append_item(
            Gio.MenuItem.new(_("Import File\u2026"), "win.import")
        )
        menu.append_section(None, Gio.Menu.new())
        menu.append_item(Gio.MenuItem.new(_("Paste"), "win.asset-paste"))
        self._popup_context_menu(menu, gesture)

    def _show_asset_context_menu(self, gesture, asset: IAsset):
        menu = Gio.Menu.new()

        if isinstance(asset, IGeometryProvider):
            menu.append_item(
                Gio.MenuItem.new(
                    _("Create New Workpiece"),
                    "win.asset-create-workpiece",
                )
            )
            menu.append_section(None, Gio.Menu.new())

        menu.append_item(
            Gio.MenuItem.new(_("Duplicate"), "win.asset-duplicate")
        )
        menu.append_section(None, Gio.Menu.new())
        menu.append_item(Gio.MenuItem.new(_("Copy"), "win.asset-copy"))
        menu.append_item(Gio.MenuItem.new(_("Cut"), "win.asset-cut"))
        menu.append_section(None, Gio.Menu.new())
        menu.append_item(Gio.MenuItem.new(_("Delete"), "win.asset-delete"))
        self._popup_context_menu(menu, gesture)

    def get_selected_assets(self) -> List[IAsset]:
        assets = []
        for uid in self._selected_uids:
            entry = self._cards.get(uid)
            if entry:
                assets.append(entry[0].asset)
        return assets

    def copy_selected_assets(self):
        assets = self.get_selected_assets()
        if not assets:
            return
        self._asset_clipboard = [a.to_dict() for a in assets]
        logger.debug(
            "Copied %d asset(s) to clipboard", len(self._asset_clipboard)
        )

    def cut_selected_assets(self):
        assets = self.get_selected_assets()
        if not assets:
            return
        self._asset_clipboard = [a.to_dict() for a in assets]
        history = self.editor.history_manager
        with history.transaction(_("Cut asset(s)")) as t:
            for asset in assets:
                t.execute(
                    ListItemCommand(
                        owner_obj=self.editor.doc,
                        item=asset,
                        undo_command="add_asset",
                        redo_command="remove_asset",
                        name=_("Cut asset"),
                    )
                )
        self._selected_uids.clear()

    def paste_assets(self):
        if not self._asset_clipboard:
            logger.debug("Paste: clipboard is empty")
            return
        logger.debug(
            "Pasting %d asset(s) from clipboard",
            len(self._asset_clipboard),
        )
        history = self.editor.history_manager
        with history.transaction(_("Paste asset(s)")) as t:
            for asset_dict in self._asset_clipboard:
                data = copy.deepcopy(asset_dict)
                new_uid = str(uuid.uuid4())
                data["uid"] = new_uid
                type_name = data.get("type", "unknown")
                asset_class = asset_type_registry.get(type_name)
                if asset_class:
                    new_asset = asset_class.from_dict(data)
                    t.execute(
                        ListItemCommand(
                            owner_obj=self.editor.doc,
                            item=new_asset,
                            undo_command="remove_asset",
                            redo_command="add_asset",
                            name=_("Paste asset"),
                        )
                    )
                else:
                    logger.warning("Paste: unknown asset type '%s'", type_name)

    def duplicate_selected_assets(self):
        assets = self.get_selected_assets()
        if not assets:
            return
        history = self.editor.history_manager
        with history.transaction(_("Duplicate asset(s)")) as t:
            for asset in assets:
                data = copy.deepcopy(asset.to_dict())
                new_uid = str(uuid.uuid4())
                data["uid"] = new_uid
                data["name"] = asset.name + " copy"
                type_name = data.get("type", "unknown")
                asset_class = asset_type_registry.get(type_name)
                if asset_class:
                    new_asset = asset_class.from_dict(data)
                    t.execute(
                        ListItemCommand(
                            owner_obj=self.editor.doc,
                            item=new_asset,
                            undo_command="remove_asset",
                            redo_command="add_asset",
                            name=_("Duplicate asset"),
                        )
                    )

    def delete_selected_assets(self):
        if not self._selected_uids:
            return
        for uid in list(self._selected_uids):
            entry = self._cards.get(uid)
            if entry:
                self.editor.asset.delete_asset(entry[0].asset)
        self._selected_uids.clear()

    def create_workpiece_from_selected(self):
        assets = self.get_selected_assets()
        if not assets:
            return
        for asset in assets:
            if isinstance(asset, IGeometryProvider):
                self.editor.edit.add_geometry_provider_instance(
                    asset.uid, (0.0, 0.0)
                )

    def can_paste_assets(self) -> bool:
        return len(self._asset_clipboard) > 0

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
