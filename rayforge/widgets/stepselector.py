from gi.repository import Gtk, Gdk


css = """
.workstep-selector-label {
    font-family: 'Roboto', sans-serif;
    font-size: 14px;
    margin: 12px;
}
"""


class WorkStepSelector(Gtk.Popover):
    def __init__(self, workstep_classes, **kwargs):
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

        # Add workstep_classes to the ListBox
        for cls in workstep_classes:
            label = Gtk.Label(label=cls.typelabel)
            label.set_xalign(0)
            label.add_css_class("workstep-selector-label")
            row = Gtk.ListBoxRow()
            row.set_child(label)
            row.cls = cls
            self.listbox.append(row)

        # Connect the row-activated signal to handle cls selection
        self.listbox.connect("row-activated", self.on_row_activated)

    def on_row_activated(self, listbox, row):
        self.selected = row.cls
        self.popdown()
