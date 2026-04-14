from gi.repository import Gtk, Adw
from typing import Callable, Optional

from .gtk import apply_css


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
        draw_value=draw_value,
    )
    scale.set_size_request(SLIDER_TRACK_WIDTH, 60 if draw_value else -1)
    scale.set_valign(Gtk.Align.CENTER)

    if on_value_changed:
        scale.connect("value-changed", on_value_changed)

    return scale


_SLIDER_VALUE_ENTRY_CSS = """
.slider-value-entry { outline: none; }
"""


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
        draw_value=False,
        on_value_changed=on_value_changed,
    )

    row = Adw.ActionRow(title=title)
    if subtitle:
        row.set_subtitle(subtitle)

    if draw_value:
        entry = Gtk.Entry()
        entry.set_width_chars(8)
        entry.set_alignment(1.0)
        entry.set_has_frame(False)
        entry.get_style_context().add_class("slider-value-entry")
        apply_css(_SLIDER_VALUE_ENTRY_CSS)

        def format_value(val):
            text = f"%.{digits}f" % val
            if format_suffix:
                text += format_suffix
            return text

        def update_entry(s):
            entry.set_text(format_value(s.get_value()))

        scale.connect("value-changed", update_entry)
        update_entry(scale)

        def commit_entry(e):
            text = e.get_text()
            if format_suffix and text.endswith(format_suffix):
                text = text[: -len(format_suffix)]
            try:
                adjustment.set_value(float(text))
            except ValueError:
                update_entry(scale)

        entry.connect("activate", commit_entry)

        focus_ctrl = Gtk.EventControllerFocus()
        focus_ctrl.connect("leave", lambda c: commit_entry(entry))
        entry.add_controller(focus_ctrl)

        row.add_suffix(entry)

    row.add_suffix(scale)

    return row, scale
