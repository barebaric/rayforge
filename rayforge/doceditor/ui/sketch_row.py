import logging
from gi.repository import Gtk, Gdk, Pango
from blinker import Signal
from ...core.doc import Doc
from ...core.sketcher.sketch import Sketch
from ...shared.util.gtk import apply_css
from ...icons import get_icon

logger = logging.getLogger(__name__)


css = """
.sketchrowview entry.sketch-title,
.sketchrowview entry.sketch-title:focus {
    border: none;
    outline: none;
    box-shadow: none;
    background: transparent;
    padding: 0;
    margin: 0;
    min-height: 0;
}

.sketch-list-box > row.active-sketch-row {
    background-color: @accent_bg_color;
    color: @accent_fg_color;
    border-radius: 6px;
}

.sketch-list-box > row.active-sketch-row .sketchrowview {
    background-color: transparent;
}

.sketch-list-box > row.active-sketch-row entry {
    caret-color: @accent_fg_color;
}

.sketch-list-box > row.active-sketch-row .dim-label {
    opacity: 0.7;
}
"""


class SketchRowWidget(Gtk.Box):
    """
    A custom widget representing a single Sketch definition in a list.
    It displays the sketch's name and actions.
    """

    def __init__(self, doc: Doc, sketch: Sketch, editor):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        self.edit_clicked = Signal()
        self.delete_clicked = Signal()

        # Apply CSS globally, but only once.
        apply_css(css)
        self.set_margin_start(6)
        self.add_css_class("sketchrowview")

        self.doc = doc
        self.sketch = sketch
        self.editor = editor

        # Icon
        icon = get_icon("sketch-edit-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        self.append(icon)

        # Box for title and subtitle
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_hexpand(True)
        content_box.set_valign(Gtk.Align.CENTER)
        self.append(content_box)

        # Title: A label for the sketch name
        self.name_label = Gtk.Label()
        self.name_label.set_hexpand(False)
        self.name_label.set_halign(Gtk.Align.START)
        self.name_label.set_ellipsize(Pango.EllipsizeMode.END)
        content_box.append(self.name_label)

        # Subtitle: A label for the sketch parameters count
        self.subtitle_label = Gtk.Label()
        self.subtitle_label.set_halign(Gtk.Align.START)
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
        content_box.append(self.subtitle_label)

        # Suffix icons
        suffix_box = Gtk.Box(spacing=6)
        suffix_box.set_valign(Gtk.Align.CENTER)
        self.append(suffix_box)

        # Edit button
        edit_icon = get_icon("edit-symbolic")
        self.edit_button = Gtk.Button(child=edit_icon)
        self.edit_button.set_tooltip_text(_("Edit this sketch"))
        self.edit_button.connect("clicked", self.on_edit_clicked)
        suffix_box.append(self.edit_button)

        # Delete button
        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.set_tooltip_text(_("Delete this sketch"))
        self.delete_button.connect("clicked", self.on_delete_clicked)
        suffix_box.append(self.delete_button)

        # Perform initial UI sync
        self.update_ui()

    def get_drag_content(self) -> Gdk.ContentProvider:
        """
        Provides the content provider for the drag operation initiated by the
        parent list row. This is called by DragListBox.on_drag_prepare.
        """
        logger.debug(
            "Providing drag content for sketch UID: %s",
            repr(self.sketch.uid),
        )
        # Provide the UID as a string, which the Canvas can accept.
        return Gdk.ContentProvider.new_for_value(str(self.sketch.uid))

    def on_edit_clicked(self, button):
        """Emits a signal when the edit button is clicked."""
        self.edit_clicked.send(self)

    def on_delete_clicked(self, button):
        """Emits a signal when the delete button is clicked."""
        self.delete_clicked.send(self)

    def update_ui(self):
        """Synchronizes the widget's state with the sketch data."""
        sketch_name = self.sketch.name or f"Sketch {self.sketch.uid[:8]}"
        self.name_label.set_text(sketch_name)
        self.name_label.set_tooltip_text(
            f"{self.sketch.name} ({self.sketch.uid})"
        )

        # Update subtitle with parameter count
        param_count = len(self.sketch.input_parameters)
        if param_count == 0:
            subtitle_text = _("No parameters")
        elif param_count == 1:
            subtitle_text = _("1 parameter")
        else:
            subtitle_text = _("{count} parameters").format(count=param_count)

        self.subtitle_label.set_label(subtitle_text)
        self.subtitle_label.set_tooltip_text(subtitle_text)
