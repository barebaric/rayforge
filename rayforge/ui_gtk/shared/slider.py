from gi.repository import Gtk, Adw
from typing import Callable, Optional


SLIDER_TRACK_WIDTH = 200
VALUE_LABEL_WIDTH = 60


def create_slider(
    adjustment: Gtk.Adjustment,
    digits: int = 1,
    draw_value: bool = True,
    on_value_changed: Optional[Callable[[Gtk.Scale], None]] = None,
) -> Gtk.Scale:
    scale = Gtk.Scale(
        orientation=Gtk.Orientation.HORIZONTAL,
        adjustment=adjustment,
        digits=digits,
        draw_value=False,
    )
    scale.set_size_request(SLIDER_TRACK_WIDTH, -1)
    scale.set_valign(Gtk.Align.CENTER)

    if on_value_changed:
        scale.connect("value-changed", on_value_changed)

    return scale


def create_slider_row(
    title: str,
    adjustment: Gtk.Adjustment,
    subtitle: Optional[str] = None,
    digits: int = 1,
    draw_value: bool = True,
    on_value_changed: Optional[Callable[[Gtk.Scale], None]] = None,
    format_suffix: Optional[str] = None,
) -> tuple[Adw.ActionRow, Gtk.Scale]:
    scale = create_slider(
        adjustment=adjustment,
        digits=digits,
        draw_value=draw_value,
        on_value_changed=on_value_changed,
    )

    row = Adw.ActionRow(title=title)
    if subtitle:
        row.set_subtitle(subtitle)

    if draw_value:
        value_label = Gtk.Label()
        value_label.set_width_chars(8)
        value_label.set_xalign(1.0)
        value_label.get_style_context().add_class("dim-label")

        def update_label(s):
            format_str = f"%.{digits}f"
            text = format_str % s.get_value()
            if format_suffix:
                text += format_suffix
            value_label.set_text(text)

        scale.connect("value-changed", update_label)
        update_label(scale)
        row.add_suffix(value_label)

    row.add_suffix(scale)

    return row, scale
