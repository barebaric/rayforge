from __future__ import annotations

import logging
from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw, Gtk

logger = logging.getLogger(__name__)


class FramePopover(Gtk.Popover):
    """
    Lets the user choose the extent of the frame before it runs.

    The engraving overscan extends raster lines with zero-power travel
    for acceleration room. Framing that path overshoots the artwork
    (issue #563); this popover lets the user frame the burned area
    only, or keep the full machine path including the overscan.
    """

    def __init__(self, on_frame: Callable[[bool], None]):
        super().__init__()
        self.on_frame = on_frame

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(12)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        self.set_child(content_box)

        rows_container = Gtk.ListBox()
        rows_container.set_selection_mode(Gtk.SelectionMode.NONE)
        rows_container.add_css_class("boxed-list")
        content_box.append(rows_container)

        self.overscan_row = Adw.SwitchRow(
            title=_("Include Overscan Margin"),
            subtitle=_(
                "Frame the full machine path, including the extra "
                "engraving travel"
            ),
            active=False,
        )
        rows_container.append(self.overscan_row)

        frame_button = Gtk.Button(label=_("Frame"))
        frame_button.add_css_class("suggested-action")
        frame_button.add_css_class("pill")
        frame_button.connect("clicked", self._on_frame_clicked)
        content_box.append(frame_button)

    def _on_frame_clicked(self, button):
        self.popdown()
        self.on_frame(self.overscan_row.get_active())
