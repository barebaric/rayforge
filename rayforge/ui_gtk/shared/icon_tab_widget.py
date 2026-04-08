from gi.repository import Gtk, Gdk
from blinker import Signal
from ..icons import get_icon
from .gtk import apply_css

css = """
box.icon-tab-strip button {
    min-width: 36px;
    min-height: 36px;
    padding: 4px;
    margin: 2px;
    border-radius: 6px;
    border: none;
    background: transparent;
}

box.icon-tab-strip button:hover {
    background: alpha(@theme_fg_color, 0.1);
}

box.icon-tab-strip button.active-tab {
    background: alpha(@theme_selected_bg_color, 0.2);
    color: @theme_selected_bg_color;
}

box.icon-tab-strip button.drag-highlight-top {
    border-top: 3px solid @theme_selected_bg_color;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
}

box.icon-tab-strip button.drag-highlight-bottom {
    border-bottom: 3px solid @theme_selected_bg_color;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}
"""


class IconTabWidget(Gtk.Box):
    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, **kwargs)
        apply_css(css)

        self.tab_changed = Signal()
        self.tab_order_changed = Signal()
        self._buttons = {}
        self._btn_to_name = {}
        self._active_name = None
        self._drag_source_name = None

        self._icon_strip = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._icon_strip.add_css_class("icon-tab-strip")
        self._icon_strip.set_spacing(2)

        self._stack = Gtk.Stack()
        self._stack.set_hexpand(True)
        self._stack.set_vexpand(True)
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(150)

        self.append(self._icon_strip)
        self.append(self._stack)

        self._dnd_controller = Gtk.DropTarget.new(str, Gdk.DragAction.MOVE)
        self._dnd_controller.connect("motion", self._on_dnd_motion)
        self._dnd_controller.connect("leave", self._on_dnd_leave)
        self._dnd_controller.connect("drop", self._on_dnd_drop)
        self._icon_strip.add_controller(self._dnd_controller)

        self._last_highlight_btn = None
        self._last_highlight_side = None

    def add_tab(self, name, icon_name, widget, tooltip=None, position=-1):
        btn = Gtk.Button(child=get_icon(icon_name))
        btn.set_tooltip_text(tooltip or name)
        btn.add_css_class("flat")
        btn.connect("clicked", self._on_button_clicked, name)

        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_drag_prepare, name)
        drag_source.connect("drag-begin", self._on_drag_begin)
        drag_source.connect("drag-end", self._on_drag_end)
        btn.add_controller(drag_source)

        if position == 0:
            self._icon_strip.prepend(btn)
        else:
            self._icon_strip.append(btn)
        self._stack.add_named(widget, name)
        self._buttons[name] = btn
        self._btn_to_name[btn] = name

        if self._active_name is None:
            self._activate_tab(name)

    def get_tab_order(self):
        order = []
        child = self._icon_strip.get_first_child()
        while child is not None:
            if child in self._btn_to_name:
                order.append(self._btn_to_name[child])
            child = child.get_next_sibling()
        return order

    def set_tab_order(self, order):
        prev = None
        for name in order:
            btn = self._buttons.get(name)
            if btn is None:
                continue
            self._icon_strip.reorder_child_after(btn, prev)
            prev = btn

    def set_current_tab(self, name):
        if name in self._buttons and name != self._active_name:
            self._activate_tab(name)

    def get_current_tab(self):
        return self._active_name

    def tab_count(self):
        return len(self._buttons)

    def remove_tab(self, name):
        if name not in self._buttons:
            return
        btn = self._buttons.pop(name)
        del self._btn_to_name[btn]
        self._icon_strip.remove(btn)
        widget = self._stack.get_child_by_name(name)
        if widget:
            self._stack.remove(widget)
        if self._active_name == name:
            self._active_name = None
            remaining = self.get_tab_order()
            if remaining:
                self._activate_tab(remaining[0])

    def has_tab(self, name):
        return name in self._buttons

    def _iter_children(self):
        child = self._icon_strip.get_first_child()
        while child is not None:
            yield child
            child = child.get_next_sibling()

    def _on_button_clicked(self, button, name):
        self._activate_tab(name)

    def _activate_tab(self, name):
        if self._active_name is not None:
            old_btn = self._buttons.get(self._active_name)
            if old_btn is not None:
                old_btn.remove_css_class("active-tab")

        self._active_name = name
        self._stack.set_visible_child_name(name)

        new_btn = self._buttons[name]
        new_btn.add_css_class("active-tab")

        self.tab_changed.send(self, name=name)

    def _on_drag_prepare(self, source, x, y, name):
        self._drag_source_name = name
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
        self._clear_highlight()

    def _on_dnd_motion(self, target, x, y):
        if self._drag_source_name is None:
            self._clear_highlight()
            return 0

        tab_name = self._get_tab_name_at_y(y)
        if tab_name is None or tab_name == self._drag_source_name:
            self._clear_highlight()
            return Gdk.DragAction.MOVE

        btn = self._buttons[tab_name]
        btn_center = btn.get_allocation().height / 2
        side = "top" if y < btn_center else "bottom"
        self._set_highlight(btn, side)
        return Gdk.DragAction.MOVE

    def _on_dnd_leave(self, target):
        self._clear_highlight()

    def _on_dnd_drop(self, target, value, x, y):
        self._clear_highlight()
        if not isinstance(value, str):
            return False
        source_name = value
        if source_name not in self._buttons:
            return False

        target_name = self._get_tab_name_at_y(y)
        if target_name is None or target_name == source_name:
            return False

        source_btn = self._buttons[source_name]
        target_btn = self._buttons[target_name]
        btn_center = target_btn.get_allocation().height / 2
        insert_after = y >= btn_center

        if insert_after:
            self._icon_strip.reorder_child_after(source_btn, target_btn)
        else:
            prev_sib = target_btn.get_prev_sibling()
            self._icon_strip.reorder_child_after(source_btn, prev_sib)

        self.tab_order_changed.send(self)
        return True

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
