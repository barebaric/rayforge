import logging
from gettext import gettext as _
from typing import List, Set, TYPE_CHECKING
from blinker import Signal
from gi.repository import Gdk, GObject, Gtk
from ...core.doc import Doc
from ...core.layer import Layer
from ..icons import get_icon
from .layer_column import LayerColumn, _LAYER_UID_PREFIX

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class LayersTab(Gtk.Box):
    def __init__(self, editor: "DocEditor"):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.editor = editor
        self.doc = editor.doc
        self._columns = []
        self._layer_drop_index = -1
        self._pan_offset_x = 0.0
        self._selected_items: List = []

        self.edit_item_requested = Signal()
        self.select_items_requested = Signal()

        self.scrolled = Gtk.ScrolledWindow()
        self.scrolled.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER
        )
        self.scrolled.set_hexpand(True)
        self.scrolled.set_vexpand(True)

        self.columns_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=0
        )
        self.columns_box.set_margin_start(9)
        self.columns_box.set_margin_top(9)
        self.columns_box.set_margin_bottom(9)
        self.columns_box.set_valign(Gtk.Align.FILL)
        self.scrolled.set_child(self.columns_box)
        self.append(self.scrolled)

        self._pan_gesture = Gtk.GestureDrag.new()
        self._pan_gesture.set_button(Gdk.BUTTON_MIDDLE)
        self._pan_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self._pan_gesture.connect("drag-begin", self._on_pan_begin)
        self._pan_gesture.connect("drag-update", self._on_pan_update)
        self.scrolled.add_controller(self._pan_gesture)

        drop_target = Gtk.DropTarget.new(
            GObject.TYPE_STRING, Gdk.DragAction.MOVE
        )
        drop_target.connect("accept", self._on_layer_drop_accept)
        drop_target.connect("enter", self._on_layer_drop_enter)
        drop_target.connect("drop", self._on_layer_drop)
        drop_target.connect("motion", self._on_layer_drop_motion)
        drop_target.connect("leave", self._on_layer_drop_leave)
        self.columns_box.add_controller(drop_target)

        add_button = Gtk.Button(child=get_icon("add-symbolic"))
        add_button.add_css_class("flat")
        add_button.set_tooltip_text(_("Add New Layer"))
        add_button.set_valign(Gtk.Align.START)
        add_button.set_margin_top(18)
        add_button.set_margin_start(9)
        add_button.set_margin_end(9)
        add_button.connect("clicked", self._on_add_clicked)
        self.append(add_button)

        self._connect_signals()
        self._rebuild()

    def set_doc(self, doc: Doc):
        if self.doc == doc:
            return
        self._disconnect_signals()
        self.doc = doc
        self._connect_signals()
        self._rebuild()

    def _connect_signals(self):
        self.doc.descendant_added.connect(self._on_structure_changed)
        self.doc.descendant_removed.connect(self._on_structure_changed)
        self.doc.updated.connect(self._on_doc_updated)

    def _disconnect_signals(self):
        self.doc.descendant_added.disconnect(self._on_structure_changed)
        self.doc.descendant_removed.disconnect(self._on_structure_changed)
        self.doc.updated.disconnect(self._on_doc_updated)

    def do_destroy(self):
        self._disconnect_signals()

    def _on_structure_changed(self, sender, **kwargs):
        self._rebuild()

    def _on_doc_updated(self, sender, **kwargs):
        current = [col.layer for col in self._columns]
        if current != list(self.doc.layers):
            self._rebuild()

    def _rebuild(self):
        for col in self._columns:
            self.columns_box.remove(col)
        self._columns.clear()

        can_delete = len(list(self.doc.layers)) > 1
        for child in self.doc.children:
            if not isinstance(child, Layer):
                continue
            col = LayerColumn(
                self.doc, child, self.editor, can_delete=can_delete
            )
            col.edit_item_requested.connect(self._on_column_edit_item)
            col.select_items_requested.connect(self._on_column_select_items)
            col.move_to_layer_requested.connect(
                self._on_column_move_to_layer
            )
            self._columns.append(col)
            self.columns_box.append(col)

    def _on_column_edit_item(self, sender, **kwargs):
        self.edit_item_requested.send(sender, **kwargs)

    def _on_column_select_items(self, sender, **kwargs):
        items = kwargs.get("items", [])
        extend = kwargs.get("extend", False)
        if extend:
            other_layer = [
                i for i in self._selected_items
                if i.layer is not sender.layer
            ]
            same_layer = [
                i for i in self._selected_items
                if i.layer is sender.layer
            ]
            new_uids = {i.uid for i in items}
            for i in same_layer:
                if i.uid not in new_uids:
                    items.append(i)
            items = other_layer + items
        self._selected_items = items
        self.select_items_requested.send(sender, items=items)

    def _on_column_move_to_layer(self, sender, **kwargs):
        items = kwargs.get("items", [])
        target_layer = kwargs.get("target_layer")
        if not target_layer:
            return
        moved_uids = {i.uid for i in items}
        selected_uids = {i.uid for i in self._selected_items}
        if moved_uids & selected_uids:
            items = list(self._selected_items)
        self.editor.layer.move_items_to_layer(items, target_layer)

    def update_row_selection(self, selected_uids: Set):
        all_items = self.get_ordered_items()
        self._selected_items = [i for i in all_items if i.uid in selected_uids]
        for col in self._columns:
            col.update_row_selection(selected_uids)

    def get_selected_items(self) -> List:
        return list(self._selected_items)

    def get_ordered_items(self) -> List:
        items = []
        for col in self._columns:
            child = col.listbox.get_first_child()
            while child:
                if isinstance(child, Gtk.ListBoxRow):
                    item = col._row_items.get(child)
                    if item:
                        items.append(item)
                child = child.get_next_sibling()
        return items

    def _on_add_clicked(self, button):
        self.editor.layer.add_layer_and_set_active()

    def _find_column_at(self, x, y):
        picked = self.columns_box.pick(x, y, Gtk.PickFlags.DEFAULT)
        while picked:
            if isinstance(picked, LayerColumn):
                return picked
            picked = picked.get_parent()
        return None

    def _remove_layer_drop_markers(self):
        LayerColumn._remove_layer_drop_markers_from(self.columns_box)
        self._layer_drop_index = -1

    @staticmethod
    def _parse_layer_uid(value):
        if value and value.startswith(_LAYER_UID_PREFIX):
            return value[len(_LAYER_UID_PREFIX) :]
        return None

    def _on_layer_drop(self, drop_target, value, x, y):
        uid = self._parse_layer_uid(value)
        if not uid:
            return False

        layers = list(self.doc.layers)
        source = None
        for layer in layers:
            if layer.uid == uid:
                source = layer
                break
        if not source:
            self._remove_layer_drop_markers()
            return False

        drop_index = self._layer_drop_index
        self._remove_layer_drop_markers()

        if drop_index == -1 or drop_index > len(layers):
            return False

        source_index = layers.index(source)

        insert_index = drop_index
        if source_index < insert_index:
            insert_index -= 1

        if source_index == insert_index:
            return True

        new_order = list(layers)
        new_order.pop(source_index)
        new_order.insert(insert_index, source)
        self.editor.layer.reorder_layers(new_order)
        return True

    def _on_layer_drop_accept(self, drop_target, drop):
        formats = drop.get_formats() if drop else None
        logger.debug(
            "Accept(columns_box): formats=%s",
            formats.to_string() if formats else None,
        )
        return True

    def _on_layer_drop_enter(self, drop_target, x, y):
        logger.debug("Enter(columns_box): x=%d y=%d", x, y)
        return Gdk.DragAction.MOVE

    def _on_layer_drop_motion(self, drop_target, x, y):
        if not LayerColumn.dragging:
            logger.debug("Motion(columns_box): rejected, not layer drag")
            return 0

        self._remove_layer_drop_markers()

        col = self._find_column_at(x, y)
        if not col:
            return Gdk.DragAction.MOVE

        col_alloc = col.get_allocation()
        col_center_x = col_alloc.x + col_alloc.width / 2

        if x < col_center_x:
            col.add_css_class("drop-left")
            self._layer_drop_index = self._columns.index(col)
        else:
            col.add_css_class("drop-right")
            self._layer_drop_index = self._columns.index(col) + 1

        return Gdk.DragAction.MOVE

    def _on_layer_drop_leave(self, drop_target):
        self._remove_layer_drop_markers()

    def _on_pan_begin(self, gesture, start_x, start_y):
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        adj = self.scrolled.get_hadjustment()
        self._pan_offset_x = adj.get_value()

    def _on_pan_update(self, gesture, offset_x, offset_y):
        adj = self.scrolled.get_hadjustment()
        adj.set_value(self._pan_offset_x - offset_x)
