from gi.repository import Graphene, Gtk, Pango, PangoCairo
from .gtk import apply_css


css = """
.number-badge {
    border-radius: 6px;
    background-color: @accent_bg_color;
    color: @accent_fg_color;
}
"""


class NumberBadge(Gtk.Widget):
    def __init__(self, number=0, **kwargs):
        super().__init__(**kwargs)
        apply_css(css)
        self.add_css_class("number-badge")
        self._number = number

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

    def set_number(self, number):
        self._number = number
        self.queue_draw()

    def get_number(self):
        return self._number
