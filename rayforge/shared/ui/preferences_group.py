from typing import Any, Iterable
from gi.repository import Gtk, Adw
from ...icons import get_icon
from ..util.gtk import apply_css

css = """
/* 1. Round the top corners of the ListBox to match its .card parent. */
.group-with-button-container > .list-box-in-card {
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}

/* 2. Style the button to connect seamlessly to the ListBox above it. */
.group-with-button-container > .flat-bottom-button,
.group-with-button-container > .flat-bottom-button > .toggle {
    border-top-left-radius: 0;
    border-top-right-radius: 0;
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
    box-shadow: none;
}

/* 3. Round the top corners of a selected row if it's the first child. */
.list-box-in-card row:first-child:selected {
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}
"""


class PreferencesGroupWithButton(Adw.PreferencesGroup):
    """
    A reusable, abstract Adw.PreferencesGroup that manages a dynamic list of
    items displayed in a Gtk.ListBox, with an "Add" button at the bottom.

    Subclasses must implement the `create_row_widget` and `_on_add_clicked`
    methods. They can optionally override `_create_add_button` for custom
    button types like a MenuButton.
    """

    def __init__(
        self,
        button_label: str,
        selection_mode: Gtk.SelectionMode = Gtk.SelectionMode.NONE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        apply_css(css)
        self.add_css_class("pref-group-with-button")

        container_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        container_box.add_css_class("card")
        container_box.add_css_class("group-with-button-container")
        self.add(container_box)

        self.list_box = Gtk.ListBox(
            selection_mode=selection_mode, show_separators=True
        )
        self.list_box.add_css_class("list-box-in-card")
        self.list_box.get_style_context().add_class("frame")
        container_box.append(self.list_box)

        self.add_button = self._create_add_button(button_label)
        self.add_button.add_css_class("darkbutton")
        self.add_button.add_css_class("flat-bottom-button")
        container_box.append(self.add_button)

    def set_items(self, items: Iterable):
        is_selectable = (
            self.list_box.get_selection_mode() != Gtk.SelectionMode.NONE
        )

        while child := self.list_box.get_row_at_index(0):
            self.list_box.remove(child)

        item_list = list(items)

        if not item_list:
            placeholder_label = Gtk.Label(label="No parameters")
            placeholder_label.add_css_class("dim-label")
            placeholder_label.set_halign(Gtk.Align.CENTER)
            placeholder_label.set_margin_top(12)
            placeholder_label.set_margin_bottom(12)
            row = Gtk.ListBoxRow(child=placeholder_label, selectable=False)
            self.list_box.append(row)
        else:
            for item in item_list:
                widget = self.create_row_widget(item)
                row = Gtk.ListBoxRow(child=widget, selectable=is_selectable)
                self.list_box.append(row)

    def create_row_widget(self, item: Any) -> Gtk.Widget:
        raise NotImplementedError(
            "Subclasses must implement create_row_widget()"
        )

    def _create_add_button(self, button_label: str) -> Gtk.Widget:
        """
        Default factory for the button. Creates a simple Gtk.Button.
        Subclasses can override this to return a Gtk.MenuButton or other
        widget.
        """
        button = Gtk.Button()
        button.connect("clicked", self._on_add_clicked)

        button_content = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            halign=Gtk.Align.CENTER,
            margin_top=10,
            margin_end=12,
            margin_bottom=10,
            margin_start=12,
        )
        button.set_child(button_content)
        button_content.append(get_icon("add-symbolic"))
        button_content.append(Gtk.Label(label=button_label))
        return button

    def _on_add_clicked(self, button: Gtk.Button):
        raise NotImplementedError(
            "Subclasses must implement _on_add_clicked()"
        )
