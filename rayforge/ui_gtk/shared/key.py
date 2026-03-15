from gi.repository import Gtk
from ..shared.gtk import apply_css


css = """
.key {
    padding: 5px 8px;
    border-radius: 6px;
    background-color: @theme_base_color;
    color: @theme_fg_color;
    border: 1px solid @borders;
    font-size: 12px;
    font-weight: 500;
}
"""


class Key(Gtk.Label):
    def __init__(self, label: str, **kwargs):
        super().__init__(label=label, **kwargs)
        apply_css(css)
        self.add_css_class("key")
        self.set_valign(Gtk.Align.CENTER)
