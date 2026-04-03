from typing import Optional

from gi.repository import Graphene, Gtk, Pango, PangoCairo
from .gtk import apply_css


css = """
.number-badge {
    border-radius: 6px;
    border: 1px solid transparent;
    background-color: @accent_bg_color;
    color: @accent_fg_color;
}
.number-badge.dimmed {
    background-color: alpha(@window_fg_color, 0.15);
    color: @window_fg_color;
}
"""


def _contrast_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


class NumberBadge(Gtk.Widget):
    def __init__(self, number: int = 0, **kwargs):
        super().__init__(**kwargs)
        apply_css(css)
        self.add_css_class("number-badge")
        self._number: int = number
        self._color_class: Optional[str] = None

    def do_measure(self, orientation, for_size):
        return (32, 32, -1, -1)

    def do_snapshot(self, snapshot):
        w = self.get_width()
        h = self.get_height()
        if w == 0 or h == 0:
            return

        layout = self.create_pango_layout(str(self._number))
        font_desc = layout.get_font_description()
        if font_desc is None:
            font_desc = Pango.FontDescription()

        font_size = min(w, h) * 0.55 * Pango.SCALE
        font_desc.set_absolute_size(font_size)
        layout.set_font_description(font_desc)

        lw, lh = layout.get_pixel_size()
        if lw > w - 4:
            font_desc.set_absolute_size(font_size * (w - 4) / lw)
            layout.set_font_description(font_desc)
            lw, lh = layout.get_pixel_size()

        rect = Graphene.Rect().init(0, 0, w, h)
        ctx = snapshot.append_cairo(rect)

        color = self.get_color()
        ctx.set_source_rgba(color.red, color.green, color.blue, color.alpha)
        ctx.move_to((w - lw) / 2, (h - lh) / 2)
        PangoCairo.show_layout(ctx, layout)

    def set_number(self, number: int):
        self._number = number
        self.queue_draw()

    def get_number(self) -> int:
        return self._number

    def set_color(self, hex_color: Optional[str]):
        if self._color_class:
            self.remove_css_class(self._color_class)
            self._color_class = None

        if hex_color is None:
            self.add_css_class("number-badge")
        else:
            self.remove_css_class("number-badge")
            fg = _contrast_color(hex_color)
            class_name = f"badge-{hex_color.lstrip('#').lower()}"
            color_css = (
                f".{class_name} {{"
                f"  border-radius: 6px;"
                f"  border: 1px solid @borders;"
                f"  background-color: {hex_color};"
                f"  color: {fg};"
                "}"
            )
            apply_css(color_css)
            self._color_class = class_name
            self.add_css_class(class_name)
        self.queue_draw()

    def set_dimmed(self, dimmed: bool):
        if dimmed:
            self.add_css_class("dimmed")
        else:
            self.remove_css_class("dimmed")
