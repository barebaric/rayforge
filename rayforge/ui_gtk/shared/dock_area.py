from gi.repository import Gtk, Gdk
from blinker import Signal
from .gtk import apply_css


dock_area_css = """
box.dock-area {
    background: @theme_bg_color;
}

box.dock-area > box.dock-icon-strip button {
    min-width: 28px;
    min-height: 28px;
    padding: 2px;
    margin: 1px;
    border-radius: 4px;
    border: none;
    background: transparent;
}

box.dock-area > box.dock-icon-strip button:hover {
    background: alpha(@theme_fg_color, 0.1);
}

box.dock-area > box.dock-icon-strip button.active-tab {
    background: alpha(@theme_selected_bg_color, 0.2);
    color: @theme_selected_bg_color;
}

box.dock-area > box.dock-icon-strip button.drag-highlight-top {
    border-top: 3px solid @theme_selected_bg_color;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
}

box.dock-area > box.dock-icon-strip button.drag-highlight-bottom {
    border-bottom: 3px solid @theme_selected_bg_color;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}

box.dock-area.drag-active {
    background: alpha(@theme_selected_bg_color, 0.06);
    border: 1px dashed alpha(@theme_selected_bg_color, 0.4);
    border-radius: 6px;
}
"""

apply_css(dock_area_css)


class DockArea(Gtk.Box):
    _drag_source_area = None

    def __init__(self, orientation=Gtk.Orientation.HORIZONTAL, **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self.add_css_class("dock-area")
        self.set_vexpand(True)

        self.items = {}
        self._item_order = []
        self._active_item = None
        self._buttons = {}
        self._btn_to_name = {}
        self._drag_source_name = None
        self._last_highlight_btn = None
        self._last_highlight_side = None

        self.layout_changed = Signal()
        self.tab_changed = Signal()
        self.item_dropped = Signal()

        self._icon_strip = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._icon_strip.add_css_class("dock-icon-strip")
        self._icon_strip.set_spacing(2)
        self._icon_strip.set_visible(False)

        self._stack = Gtk.Stack()
        self._stack.set_vexpand(True)
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(150)

        self.append(self._icon_strip)
        self.append(self._stack)

        self._drop_target = Gtk.DropTarget.new(str, Gdk.DragAction.MOVE)
        self._drop_target.connect("motion", self._on_drop_motion)
        self._drop_target.connect("leave", self._on_drop_leave)
        self._drop_target.connect("drop", self._on_drop)
        self.add_controller(self._drop_target)

    def add_item(self, item, position=-1):
        if item.name in self.items:
            return
        parent = item.widget.get_parent()
        if parent is not None:
            parent.remove(item.widget)

        self.items[item.name] = item
        if position < 0 or position >= len(self._item_order):
            self._item_order.append(item.name)
        else:
            self._item_order.insert(position, item.name)

        btn = self._create_tab_button(item)
        if position == 0 and len(self._item_order) > 1:
            self._icon_strip.prepend(btn)
        else:
            self._icon_strip.append(btn)
        self._buttons[item.name] = btn
        self._btn_to_name[btn] = item.name

        self._stack.add_named(item.widget, item.name)

        if self._active_item is None:
            self._activate_item(item.name)
        else:
            self._sync_expand()

        self._update_strip()

    def remove_item(self, name):
        if name not in self.items:
            return None
        item = self.items.pop(name)
        self._item_order.remove(name)

        btn = self._buttons.pop(name, None)
        if btn:
            del self._btn_to_name[btn]
            self._icon_strip.remove(btn)

        widget = self._stack.get_child_by_name(name)
        if widget:
            self._stack.remove(widget)

        if self._active_item == name:
            self._active_item = (
                self._item_order[0] if self._item_order else None
            )
            if self._active_item:
                self._stack.set_visible_child_name(self._active_item)

        self._update_strip()
        self._sync_expand()
        return item

    def has_item(self, name):
        return name in self.items

    def item_count(self):
        return len(self.items)

    def get_item_names(self):
        return list(self._item_order)

    def get_active_item(self):
        return self._active_item

    def set_active_item(self, name):
        if name in self.items and name != self._active_item:
            self._activate_item(name)

    def get_layout(self):
        return {
            "items": list(self._item_order),
            "active": self._active_item,
        }

    def apply_layout(self, layout):
        items = layout.get("items", [])
        known = set(self.items.keys())
        filtered = [n for n in items if n in known]
        missing = [n for n in self._item_order if n not in filtered]
        self._item_order = filtered + missing
        active = layout.get("active")
        if active and active in self.items:
            self._active_item = active
        elif self._item_order:
            self._active_item = self._item_order[0]
        prev = None
        for name in self._item_order:
            btn = self._buttons.get(name)
            if btn is None:
                continue
            self._icon_strip.reorder_child_after(btn, prev)
            prev = btn
        if self._active_item:
            self._stack.set_visible_child_name(self._active_item)
        self._update_strip()

    def _update_strip(self):
        n = len(self._item_order)
        self._icon_strip.set_visible(n >= 1)
        for name, btn in self._buttons.items():
            if n > 1 and name == self._active_item:
                btn.add_css_class("active-tab")
            else:
                btn.remove_css_class("active-tab")

    def _create_tab_button(self, item):
        from ..icons import get_icon

        btn = Gtk.Button(child=get_icon(item.icon_name))
        btn.set_tooltip_text(item.label)
        btn.add_css_class("flat")
        btn.connect("clicked", self._on_button_clicked, item.name)

        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_drag_prepare, item.name)
        drag_source.connect("drag-begin", self._on_drag_begin)
        drag_source.connect("drag-end", self._on_drag_end)
        btn.add_controller(drag_source)

        return btn

    def _on_button_clicked(self, button, name):
        self._activate_item(name)

    def _activate_item(self, name):
        if self._active_item is not None:
            old_btn = self._buttons.get(self._active_item)
            if old_btn is not None:
                old_btn.remove_css_class("active-tab")

        self._active_item = name
        self._stack.set_visible_child_name(name)

        new_btn = self._buttons.get(name)
        if new_btn and len(self._item_order) > 1:
            new_btn.add_css_class("active-tab")

        self._sync_expand()
        self.tab_changed.send(self, name=name)

    def _sync_expand(self):
        expands = any(
            self.items[n].expands for n in self._item_order if n in self.items
        )
        self._stack.set_hexpand(expands)
        self.set_hexpand(expands)

    def _on_drag_prepare(self, source, x, y, name):
        self._drag_source_name = name
        DockArea._drag_source_area = self
        return Gdk.ContentProvider.new_for_value(name)

    def _on_drag_begin(self, source, drag):
        btn = self._buttons.get(self._drag_source_name)
        if btn:
            icon = btn.get_child()
            if icon:
                paintable = Gtk.WidgetPaintable.new(icon)
                source.set_icon(paintable, 0, 0)

    def _on_drag_end(self, source, drag, delete_data):
        self._drag_source_name = None
        DockArea._drag_source_area = None
        self._clear_highlight()

    def _on_drop_motion(self, target, x, y):
        source = DockArea._drag_source_area
        if source is not None and source.get_parent() is None:
            DockArea._drag_source_area = None
            source = None
        if source is None:
            self._clear_highlight()
            self.remove_css_class("drag-active")
            return 0

        if source is self:
            return self._handle_same_area_motion(x, y)

        self._clear_highlight()
        self.add_css_class("drag-active")
        return Gdk.DragAction.MOVE

    def _handle_same_area_motion(self, x, y):
        self.remove_css_class("drag-active")
        strip_alloc = self._icon_strip.get_allocation()
        if strip_alloc.width <= 0:
            self._clear_highlight()
            return 0
        strip_y = y - strip_alloc.y
        tab_name = self._get_tab_name_at_y(strip_y)
        if tab_name is None or tab_name == self._drag_source_name:
            self._clear_highlight()
            return Gdk.DragAction.MOVE
        btn = self._buttons[tab_name]
        btn_alloc = btn.get_allocation()
        mid = btn_alloc.y + btn_alloc.height / 2
        side = "top" if strip_y < mid else "bottom"
        self._set_highlight(btn, side)
        return Gdk.DragAction.MOVE

    def _on_drop_leave(self, target):
        self._clear_highlight()
        self.remove_css_class("drag-active")

    def _on_drop(self, target, value, x, y):
        self._clear_highlight()
        self.remove_css_class("drag-active")
        if not isinstance(value, str):
            return False
        name = value
        source = DockArea._drag_source_area
        if source is not None and source.get_parent() is None:
            DockArea._drag_source_area = None
            source = None

        if source is self and name in self.items:
            return self._handle_same_area_drop(x, y, name)

        if source is not None and name not in self.items:
            self.item_dropped.send(self, name=name)
            return True

        return False

    def _handle_same_area_drop(self, x, y, name):
        strip_alloc = self._icon_strip.get_allocation()
        strip_y = y - strip_alloc.y
        target_name = self._get_tab_name_at_y(strip_y)
        if target_name is None or target_name == name:
            return False
        source_btn = self._buttons[name]
        target_btn = self._buttons[target_name]
        target_btn_alloc = target_btn.get_allocation()
        insert_after = (
            strip_y >= target_btn_alloc.y + target_btn_alloc.height / 2
        )
        if insert_after:
            self._icon_strip.reorder_child_after(source_btn, target_btn)
        else:
            prev_sib = target_btn.get_prev_sibling()
            self._icon_strip.reorder_child_after(source_btn, prev_sib)
        self._item_order = self._get_visual_order()
        self.layout_changed.send(self)
        return True

    def _get_visual_order(self):
        order = []
        child = self._icon_strip.get_first_child()
        while child is not None:
            if child in self._btn_to_name:
                order.append(self._btn_to_name[child])
            child = child.get_next_sibling()
        return order

    def _get_tab_name_at_y(self, y):
        child = self._icon_strip.get_first_child()
        while child is not None:
            if child in self._btn_to_name:
                alloc = child.get_allocation()
                if alloc.y <= y <= alloc.y + alloc.height:
                    return self._btn_to_name[child]
            child = child.get_next_sibling()
        return None

    def _set_highlight(self, btn, side):
        self._clear_highlight()
        btn.add_css_class(f"drag-highlight-{side}")
        self._last_highlight_btn = btn
        self._last_highlight_side = side

    def _clear_highlight(self):
        if self._last_highlight_btn is not None:
            if self._last_highlight_side:
                self._last_highlight_btn.remove_css_class(
                    f"drag-highlight-{self._last_highlight_side}"
                )
            self._last_highlight_btn = None
            self._last_highlight_side = None
