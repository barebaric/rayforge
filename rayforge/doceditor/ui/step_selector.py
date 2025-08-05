from gi.repository import Gtk, Gdk  # type: ignore


css = """
.step-selector-label {
    font-family: 'Roboto', sans-serif;
    font-size: 14px;
    margin: 12px;
}
"""


class StepSelector(Gtk.Popover):
    def __init__(self, step_classes, **kwargs):
        super().__init__(**kwargs)
        self.set_autohide(True)
        self.selected = None

        # Create a ListBox inside the Popover
        self.listbox = Gtk.ListBox()
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(self.listbox)

        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Add step_classes to the ListBox
        for cls in step_classes:
            label = Gtk.Label(label=cls.typelabel)
            label.set_xalign(0)
            label.add_css_class("step-selector-label")
            row = Gtk.ListBoxRow()
            row.set_child(label)
            row.cls = cls
            self.listbox.append(row)

        # Connect the row-activated signal to handle cls selection
        self.listbox.connect("row-activated", self.on_row_activated)

    def on_row_activated(self, listbox, row):
        self.selected = row.cls
        self.popdown()
