from typing import Optional

from gi.repository import Gtk

from ...shared.util.time_format import format_seconds
from .gtk import apply_css

css = """
.time-estimate-overlay {
    background-color: @theme_bg_color;
    border-radius: 6px;
    padding: 3px 8px;
}
"""


class TimeEstimateOverlay(Gtk.Box):
    def __init__(self, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=4,
            **kwargs,
        )
        apply_css(css)
        self.add_css_class("time-estimate-overlay")
        self.set_halign(Gtk.Align.END)
        self.set_valign(Gtk.Align.END)
        self.set_margin_bottom(6)
        self.set_margin_end(6)

        self._label = Gtk.Label()
        self._label.set_visible(False)
        self.append(self._label)

    def set_estimated_time(self, time_seconds: Optional[float]):
        if time_seconds is None or time_seconds <= 0:
            self._label.set_visible(False)
        else:
            self._label.set_text(
                "~" + format_seconds(time_seconds, compact=True)
            )
            self._label.set_visible(True)
