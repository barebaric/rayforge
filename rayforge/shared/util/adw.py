from typing import Any, Iterable
from gi.repository import Gtk, Adw
from ...icons import get_icon
from ...shared.util.gtk import apply_css

css = """
/* 1. Round the top corners of the ListBox to match its .card parent. */
.group-with-button-container > .list-box-in-card {
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}

/* 2. Style the button to connect seamlessly to the ListBox above it. */
.group-with-button-container > button.flat-bottom-button {
    border-top-left-radius: 0;
    border-top-right-radius: 0;
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
    box-shadow: none;
}
"""


class PreferencesGroupWithButton(Adw.PreferencesGroup):
    """
    A reusable, abstract Adw.PreferencesGroup that manages a dynamic list of
    items displayed in a Gtk.ListBox, with an "Add" button at the bottom.

    Subclasses must implement the `create_row_widget` and `_on_add_clicked`
    methods.
    """

    def __init__(self, button_label: str, **kwargs):
        super().__init__(**kwargs)
        apply_css(css)

        container_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        container_box.add_css_class("card")
        container_box.add_css_class("group-with-button-container")
        self.add(container_box)

        self.list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE, show_separators=True
        )
        self.list_box.add_css_class("list-box-in-card")
        self.list_box.get_style_context().add_class("frame")
        container_box.append(self.list_box)

        self.add_button = Gtk.Button()
        self.add_button.add_css_class("darkbutton")
        self.add_button.add_css_class("flat-bottom-button")
        self.add_button.connect("clicked", self._on_add_clicked)
        container_box.append(self.add_button)

        button_content = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            halign=Gtk.Align.CENTER,
            margin_top=10,
            margin_end=12,
            margin_bottom=10,
            margin_start=12,
        )
        self.add_button.set_child(button_content)
        button_content.append(get_icon("add-symbolic"))
        button_content.append(Gtk.Label(label=button_label))

    def set_items(self, items: Iterable):
        """
        Clears and rebuilds the list box with widgets for the given items.
        """
        while child := self.list_box.get_row_at_index(0):
            self.list_box.remove(child)

        for item in items:
            widget = self.create_row_widget(item)
            row = Gtk.ListBoxRow(child=widget, selectable=False)
            self.list_box.append(row)

    def create_row_widget(self, item: Any) -> Gtk.Widget:
        """
        Factory method to be implemented by subclasses.
        Should return a widget to display for a single item in the list.
        """
        raise NotImplementedError(
            "Subclasses of PreferencesGroupWithButton must implement "
            "create_row_widget()"
        )

    def _on_add_clicked(self, button: Gtk.Button):
        """
        Handler for the add button, to be implemented by subclasses.
        Should contain the logic for creating a new item.
        """
        raise NotImplementedError(
            "Subclasses of PreferencesGroupWithButton must implement "
            "_on_add_clicked()"
        )
