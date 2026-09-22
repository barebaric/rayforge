from gi.repository import Gtk

from .shortcut import Shortcut


class StatusBar(Gtk.Box):
    """
    Displays context sensitive shortcut hints in a single row.

    Entries are added and removed whenever tool state changes, so their
    combined width varies. The row is therefore hosted in a scrolled
    window with an EXTERNAL horizontal policy: the bar never propagates
    the content width into the window's size request. Otherwise such
    state changes would resize the entire window while dragging in the
    sketcher (observed on macOS, see issue #385). If the bar is too
    narrow, entries are clipped instead of widening the window.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.add_css_class("status-bar")

        self._content = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=24
        )
        self._scroller = Gtk.ScrolledWindow()
        self._scroller.set_policy(
            Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER
        )
        self._scroller.set_child(self._content)
        self._scroller.set_hexpand(True)
        self.append(self._scroller)

    def add_shortcut_entry(
        self,
        keys: list[str],
        description: str | None = None,
        separator: str = "+",
    ):
        """Add a shortcut to the status bar."""
        shortcut = Shortcut(
            keys=keys, description=description, separator=separator
        )
        self._content.append(shortcut)

    def add_separator(self):
        """Add a visual separator between shortcuts."""
        separator = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        separator.set_size_request(1, 16)
        separator.add_css_class("separator")
        separator.get_style_context().add_class("separator")
        self._content.append(separator)

    def clear(self):
        """Remove all shortcuts from the status bar."""
        child = self._content.get_first_child()
        while child is not None:
            self._content.remove(child)
            child = self._content.get_first_child()
