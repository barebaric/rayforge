from typing import List, Callable
from gi.repository import Gtk, Gdk  # type: ignore


css = """
.step-selector-label {
    font-family: 'Roboto', sans-serif;
    font-size: 14px;
    margin: 12px;
}
"""


class StepSelector(Gtk.Popover):
    def __init__(self, step_factories: List[Callable], **kwargs):
        super().__init__(**kwargs)
        self.set_autohide(True)
        self.selected_factory: Callable | None = None

        # Create a ListBox inside the Popover
        self.listbox = Gtk.ListBox()
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(self.listbox)

        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Add step_factories to the ListBox
        for factory_func in step_factories:
            # Create a temporary, parentless step to get its default label.
            # This is a bit of a hack but keeps the UI decoupled.
            temp_step = factory_func(workflow=None)
            label = Gtk.Label(label=temp_step.typelabel)
            label.set_xalign(0)
            label.add_css_class("step-selector-label")
            row = Gtk.ListBoxRow()
            row.set_child(label)
            row.factory = factory_func
            self.listbox.append(row)

        # Connect the row-activated signal to handle factory selection
        self.listbox.connect("row-activated", self.on_row_activated)

    def on_row_activated(self, listbox, row):
        self.selected_factory = row.factory
        self.popdown()
