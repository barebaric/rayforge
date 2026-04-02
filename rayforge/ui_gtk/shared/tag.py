from gi.repository import Gtk
from .gtk import apply_css


css = """
.tag {
    border-radius: 6px;
    padding: 2px 8px;
    transition: background-color 0.15s;
}

.tag.active {
    background-color: @accent_bg_color;
    color: @accent_fg_color;
}

.tag.inactive {
    background-color: alpha(@view_fg_color, 0.12);
    color: @view_fg_color;
}
"""


class TagWidget(Gtk.Box):
    def __init__(self, active=True, **kwargs):
        super().__init__(**kwargs)
        apply_css(css)
        self.add_css_class("tag")
        self.set_active(active)

    def set_active(self, active: bool):
        if active:
            self.add_css_class("active")
            self.remove_css_class("inactive")
        else:
            self.add_css_class("inactive")
            self.remove_css_class("active")

    def get_active(self) -> bool:
        return self.has_css_class("active")
