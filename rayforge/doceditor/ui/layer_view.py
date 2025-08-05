from gi.repository import Gtk, Gdk, Pango  # type: ignore
from blinker import Signal
from ...core.doc import Doc
from ...core.layer import Layer
from ...undo.models.property_cmd import ChangePropertyCommand


css = """
.layerview entry.layerview-title,
.layerview entry.layerview-title:focus {
    border: none;
    outline: none;
    box-shadow: none;
    background: transparent;
    padding: 0;
    margin: 0;
    min-height: 0; /* Override theme's default min-height */
}

.layer-list-box > row.active-layer-row {
    background-color: @accent_bg_color;
    color: @accent_fg_color;
    border-radius: 6px;
}

.layer-list-box > row.active-layer-row .layerview {
    background-color: transparent;
}

.layer-list-box > row.active-layer-row entry {
    caret-color: @accent_fg_color;
}

.layer-list-box > row.active-layer-row .dim-label {
    opacity: 0.7;
}
"""


class LayerView(Gtk.Box):
    """
    A custom widget representing a single Layer in a list.
    It displays the layer's name, a summary of its contents, and actions.
    """

    delete_clicked = Signal()

    def __init__(self, doc: Doc, layer: Layer):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Apply CSS globally, but only once.
        self.apply_css()
        self.set_margin_start(6)
        self.add_css_class("layerview")

        self.doc = doc
        self.layer = layer

        # Box for title and subtitle, now correctly centered.
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_hexpand(True)
        content_box.set_valign(Gtk.Align.CENTER)
        self.append(content_box)

        # Title: An entry styled to look like a label
        self.name_entry = Gtk.Entry()
        self.name_entry.add_css_class("layerview-title")
        self.name_entry.set_hexpand(False)
        self.name_entry.set_halign(Gtk.Align.START)
        self.name_entry.connect("activate", self.on_name_apply)
        self.name_entry.connect(
            "notify::has-focus", self.on_name_focus_changed
        )
        content_box.append(self.name_entry)

        # Add a key controller to handle the Escape key.
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_name_escape_pressed)
        self.name_entry.add_controller(key_controller)

        # Subtitle: A label for the step list and workpiece count
        self.subtitle_label = Gtk.Label()
        self.subtitle_label.set_halign(Gtk.Align.START)
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
        content_box.append(self.subtitle_label)

        # Suffix icons
        suffix_box = Gtk.Box(spacing=6)
        suffix_box.set_valign(Gtk.Align.CENTER)
        self.append(suffix_box)

        self.visibility_on_icon = Gtk.Image.new_from_icon_name(
            "view-reveal-symbolic"
        )
        self.visibility_off_icon = Gtk.Image.new_from_icon_name(
            "view-conceal-symbolic"
        )

        self.delete_button = Gtk.Button.new_from_icon_name(
            "user-trash-symbolic"
        )
        self.delete_button.set_tooltip_text(_("Delete this layer"))
        self.delete_button.connect("clicked", self.on_delete_clicked)
        suffix_box.append(self.delete_button)

        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.connect("clicked", self.on_button_view_click)
        suffix_box.append(self.visibility_button)

        # Connect to both layer and document changes to stay in sync
        self.layer.changed.connect(self.on_layer_changed)
        self.doc.changed.connect(self.on_layer_changed)

        # Perform initial UI sync
        self.on_layer_changed(self.layer)

    def on_name_escape_pressed(self, controller, keyval, keycode, state):
        """Handler for the 'key-pressed' signal to catch Escape."""
        if keyval == Gdk.KEY_Escape:
            # Revert any changes and remove focus from the entry.
            self.name_entry.set_text(self.layer.name)
            list_box = self.get_ancestor(Gtk.ListBox)
            if list_box:
                list_box.grab_focus()
            return True  # Indicate that the event has been handled
        return False  # Allow other key presses to be processed normally

    def on_name_focus_changed(self, entry, gparam):
        # This triggers when focus is lost.
        if not entry.has_focus():
            self.on_name_apply(entry)

    def set_deletable(self, deletable: bool):
        """Shows or hides the delete button."""
        self.delete_button.set_visible(deletable)

    def on_layer_changed(self, sender, **kwargs):
        """
        Updates the UI when the underlying layer or document model changes.
        """
        self.update_ui()
        self.update_style()

    def on_delete_clicked(self, button):
        """Emits a signal when the delete button is clicked."""
        self.delete_clicked.send(self)

    def on_name_apply(self, widget, *args):
        """Handles applying the name change from the entry."""
        new_name = self.name_entry.get_text()

        if not new_name.strip() or new_name == self.layer.name:
            self.name_entry.set_text(self.layer.name)
            return

        command = ChangePropertyCommand(
            target=self.layer,
            property_name="name",
            new_value=new_name,
            setter_method_name="set_name",
            name=_("Rename layer"),
        )
        self.doc.history_manager.execute(command)

    def on_button_view_click(self, button):
        """Creates an undoable command to toggle the layer's visibility."""
        new_visibility = button.get_active()
        if new_visibility == self.layer.visible:
            return

        command = ChangePropertyCommand(
            target=self.layer,
            property_name="visible",
            new_value=new_visibility,
            setter_method_name="set_visible",
            name=_("Toggle layer visibility"),
        )
        self.doc.history_manager.execute(command)

    def update_style(self):
        """
        Updates the style of the parent ListBoxRow to indicate if the layer
        is active by toggling the 'active-layer-row' class.
        """
        list_box_row = self.get_ancestor(Gtk.ListBoxRow)
        if not list_box_row:
            return  # Failsafe in case we are not in a ListBoxRow

        if self.layer.active:
            list_box_row.add_css_class("active-layer-row")
        else:
            list_box_row.remove_css_class("active-layer-row")

    def update_ui(self):
        """Synchronizes the widget's state with the layer data."""
        if not self.name_entry.has_focus():
            self.name_entry.set_text(self.layer.name)

        # Get step names from the layer's workflow
        step_names = [s.name for s in self.layer.workflow]
        steps_string = ", ".join(step_names)

        # Get workpiece count string
        count = len(self.layer.workpieces)
        if count == 1:
            workpiece_string = _("{count} workpiece").format(count=count)
        else:
            workpiece_string = _("{count} workpieces").format(count=count)

        # Combine them into the final subtitle
        if steps_string:
            # Example: "Outline, Rasterize 4 workpieces"
            subtitle_text = f"{steps_string} {workpiece_string}"
        else:
            # Example: "4 workpieces"
            subtitle_text = workpiece_string

        self.subtitle_label.set_label(subtitle_text)
        self.subtitle_label.set_tooltip_text(subtitle_text)

        # Sync the visibility button's state and icon with the model.
        # Assume 'visible' property exists, default to True for robustness.
        self.visibility_button.set_active(self.layer.visible)
        if self.layer.visible:
            self.visibility_button.set_child(self.visibility_on_icon)
        else:
            self.visibility_button.set_child(self.visibility_off_icon)

    @staticmethod
    def apply_css():
        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
