from gi.repository import Gtk
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
"""


class IconTabWidget(Gtk.Box):
    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, **kwargs)
        apply_css(css)

        self.tab_changed = Signal()
        self._buttons = {}
        self._active_name = None

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

    def add_tab(self, name, icon_name, widget, tooltip=None):
        btn = Gtk.Button(child=get_icon(icon_name))
        btn.set_tooltip_text(tooltip or name)
        btn.add_css_class("flat")
        btn.connect("clicked", self._on_button_clicked, name)

        self._icon_strip.append(btn)
        self._stack.add_named(widget, name)
        self._buttons[name] = btn

        if self._active_name is None:
            self._activate_tab(name)

    def set_current_tab(self, name):
        if name in self._buttons and name != self._active_name:
            self._activate_tab(name)

    def get_current_tab(self):
        return self._active_name

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
