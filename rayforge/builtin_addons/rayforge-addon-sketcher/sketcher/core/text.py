import cairo
from gi.repository import Pango, PangoCairo


def create_text_layout(
    ctx: cairo.Context, text: str, font_size: float = 12
) -> Pango.Layout:
    layout = PangoCairo.create_layout(ctx)
    font = Pango.FontDescription()
    font.set_family("Sans")
    font.set_absolute_size(font_size * Pango.SCALE)
    layout.set_font_description(font)
    layout.set_text(text, -1)
    return layout
