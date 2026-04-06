from typing import List, Optional
from gi.repository import Gtk
from ..shared.gtk import apply_css
from .key import Key


css = """
.shortcut-description {
    margin-left: 6px;
}
"""


class Shortcut(Gtk.Box):
    def __init__(
        self,
        keys: List[str],
        description: Optional[str] = None,
        separator: str = "+",
        **kwargs,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, **kwargs)
        apply_css(css)
        self.set_spacing(1)

        for i, key in enumerate(keys):
            key_widget = Key(label=key)
            self.append(key_widget)

            if i < len(keys) - 1:
                separator_label = Gtk.Label(label=separator)
                separator_label.add_css_class("caption")
                separator_label.set_opacity(0.7)
                self.append(separator_label)

        if description:
            description_label = Gtk.Label(
                label=description,
                valign=Gtk.Align.CENTER,
                justify=Gtk.Justification.LEFT,
            )
            description_label.add_css_class("shortcut-description")
            self.append(description_label)
