from typing import List, Optional
from gi.repository import Gtk
from .shortcut import Shortcut


class StatusBar(Gtk.Box):
    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, **kwargs)
        self.add_css_class("status-bar")
        self.set_spacing(24)

    def add_shortcut_entry(
        self,
        keys: List[str],
        description: Optional[str] = None,
        separator: str = "+",
    ):
        """Add a shortcut to the status bar."""
        shortcut = Shortcut(
            keys=keys, description=description, separator=separator
        )
        self.append(shortcut)

    def add_separator(self):
        """Add a visual separator between shortcuts."""
        separator = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        separator.set_size_request(1, 16)
        separator.add_css_class("separator")
        separator.get_style_context().add_class("separator")
        self.append(separator)

    def clear(self):
        """Remove all shortcuts from the status bar."""
        child = self.get_first_child()
        while child is not None:
            self.remove(child)
            child = self.get_first_child()
