from typing import List, Callable, TYPE_CHECKING, Optional, Tuple, Any
from gi.repository import Gtk, Gdk

if TYPE_CHECKING:
    from ...context import RayforgeContext


css = """
.popover-menu-label {
    font-family: 'Roboto', sans-serif;
    font-size: 14px;
    margin: 12px;
}
"""


class PopoverMenu(Gtk.Popover):
    def __init__(
        self,
        *,
        step_factories: Optional[List[Callable]] = None,
        items: Optional[List[Tuple[str, Any]]] = None,
        context: Optional["RayforgeContext"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_autohide(True)
        self.selected_item: Any | None = None

        # Create a ListBox inside the Popover
        self.listbox = Gtk.ListBox()
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(self.listbox)

        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        display = Gdk.Display.get_default()
        if display:
            Gtk.StyleContext.add_provider_for_display(
                display,
                provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )

        if step_factories and context:
            # Add step_factories to the ListBox
            for factory_func in step_factories:
                # Create a temporary, parentless step to get its default label.
                # This is a bit of a hack but keeps the UI decoupled.
                temp_step = factory_func(context)
                self._add_row(temp_step.typelabel, factory_func)
        elif items:
            for label_text, item_value in items:
                self._add_row(label_text, item_value)

        # Connect the row-activated signal to handle item selection
        self.listbox.connect("row-activated", self.on_row_activated)

    def _add_row(self, label_text: str, item_value: Any):
        """Helper to create and add a row to the listbox."""
        label = Gtk.Label(label=label_text)
        label.set_xalign(0)
        label.add_css_class("popover-menu-label")
        row = Gtk.ListBoxRow()
        row.set_child(label)
        row.item_value = item_value  # type: ignore
        self.listbox.append(row)

    def on_row_activated(self, listbox, row):
        self.selected_item = row.item_value
        self.popdown()
