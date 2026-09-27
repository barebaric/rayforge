"""A paned that keeps its children evenly sized until the user drags."""

from gi.repository import GLib, Gtk

from .gtk import apply_css

apply_css("""
.even-split > separator {
    background-color: @borders;
    min-width: 5px;
    min-height: 5px;
}
.even-split > separator:hover {
    background-color: @accent_bg_color;
}
""")


class EvenSplitPaned(Gtk.Paned):
    """A paned that keeps an even split until the handle is dragged."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dragged = False
        self._applying = False
        self._idle_id = 0
        self.add_css_class("even-split")
        self.set_wide_handle(True)
        self.set_resize_start_child(True)
        self.set_resize_end_child(True)
        self.set_shrink_start_child(False)
        self.set_shrink_end_child(False)
        self.connect("notify::position", self._on_position_changed)

    def reset_split(self) -> None:
        """Forget a dragged position so the panes are evened out again."""
        self._dragged = False
        self._queue_even_split()

    def do_size_allocate(self, width: int, height: int, baseline: int):
        """Re-center after GTK settled the size of both children."""
        was_applying = self._applying
        self._applying = True
        try:
            Gtk.Paned.do_size_allocate(self, width, height, baseline)
        finally:
            self._applying = was_applying
        self._queue_even_split()

    def _on_position_changed(self, *_args):
        """Only a change we did not cause can come from the handle."""
        if not self._applying:
            self._dragged = True

    def _queue_even_split(self) -> None:
        if not self._idle_id:
            self._idle_id = GLib.idle_add(self._on_even_split_idle)

    def _on_even_split_idle(self):
        self._idle_id = 0
        self._apply_even_split()
        return GLib.SOURCE_REMOVE

    def _apply_even_split(self) -> None:
        if self._dragged:
            return
        extent = (
            self.get_height()
            if self.get_orientation() == Gtk.Orientation.VERTICAL
            else self.get_width()
        )
        if extent <= 0:
            return
        was_applying = self._applying
        self._applying = True
        try:
            self.set_position(extent // 2)
        finally:
            self._applying = was_applying


__all__ = ["EvenSplitPaned"]
