"""
A widget that displays a collapsible, reorderable list of Sketch
definitions.
"""

import logging
from gi.repository import Gtk, Adw
from blinker import Signal
from typing import cast
from ...core.sketcher.sketch import Sketch
from ...icons import get_icon
from ...shared.ui.draglist import DragListBox
from ...shared.ui.expander import Expander
from ...undo import ListItemCommand
from .sketch_row import SketchRowWidget

logger = logging.getLogger(__name__)


class SketchListWidget(Expander):
    """
    A widget that displays a collapsible, reorderable list of Sketch
    definitions.
    """

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        self.doc = editor.doc

        # Signals
        self.sketch_activated = Signal()
        self.add_clicked = Signal()

        self.set_title(_("Sketches"))
        self.set_expanded(True)

        # A container for all content that will be revealed by the expander
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(content_box)

        # The reorderable list of Sketches goes inside the content box
        self.draglist = DragListBox()
        self.draglist.add_css_class("sketch-list-box")
        self.draglist.reordered.connect(self.on_sketch_reordered)
        self.draglist.connect("row-activated", self.on_row_activated)
        content_box.append(self.draglist)

        # An "Add Sketch" button
        add_button = Gtk.Button()
        add_button.add_css_class("darkbutton")
        add_button.connect("clicked", self.on_button_add_clicked)
        content_box.append(add_button)

        # The button's content is a box with an icon and a label.
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_margin_top(10)
        button_box.set_margin_end(12)
        button_box.set_margin_bottom(10)
        button_box.set_margin_start(12)

        add_icon = get_icon("add-symbolic")
        button_box.append(add_icon)

        lbl = _("Add Sketch")
        add_label = Gtk.Label()
        add_label.set_markup(f"<span weight='normal'>{lbl}</span>")
        add_label.set_xalign(0)
        button_box.append(add_label)
        add_button.set_child(button_box)

        # Connect to document changes and perform initial population
        self.doc.updated.connect(self.on_doc_changed)
        self.on_doc_changed(self.doc)

    def on_doc_changed(self, sender, **kwargs):
        """
        Updates the list and subtitle when the document structure changes.
        """
        count = len(self.doc.sketches)
        self.set_subtitle(
            _("{count} sketch").format(count=count)
            if count == 1
            else _("{count} sketches").format(count=count)
        )
        self.update_list()

    def update_list(self):
        """
        Re-populates the draglist to match the state of the document's
        sketches.
        """
        self.draglist.remove_all()

        for sketch_uid, sketch in self.doc.sketches.items():
            list_box_row = Gtk.ListBoxRow()
            list_box_row.data = sketch  # type: ignore
            sketch_row_widget = SketchRowWidget(self.doc, sketch, self.editor)

            sketch_row_widget.edit_clicked.connect(self.on_edit_sketch_clicked)
            sketch_row_widget.delete_clicked.connect(
                self.on_delete_sketch_clicked
            )
            list_box_row.set_child(sketch_row_widget)
            self.draglist.add_row(list_box_row)

    def on_row_activated(self, listbox, row):
        """Handles user clicks to edit a sketch."""
        if row and row.data:
            sketch = cast(Sketch, row.data)
            # Send a signal for other parts of the UI (e.g., MainWindow)
            self.sketch_activated.send(self, sketch=sketch)

    def on_button_add_clicked(self, button):
        """Emits a signal to indicate a new sketch should be added."""
        self.add_clicked.send(self)

    def on_edit_sketch_clicked(self, sketch_row_widget):
        """Handles editing a sketch."""
        sketch = sketch_row_widget.sketch
        self.sketch_activated.send(self, sketch=sketch)

    def on_delete_sketch_clicked(self, sketch_row_widget):
        """Handles deletion of a sketch definition with an undoable command."""
        sketch_to_delete = sketch_row_widget.sketch

        # Check if any workpieces are using this sketch
        workpieces_using_sketch = [
            wp
            for wp in self.doc.all_workpieces
            if wp.sketch_uid == sketch_to_delete.uid
        ]

        root = self.get_root()
        parent_window = (
            cast(Gtk.Window, root) if isinstance(root, Gtk.Window) else None
        )

        if workpieces_using_sketch:
            dialog = Adw.MessageDialog(
                transient_for=parent_window,
                heading=_("Cannot Delete Sketch"),
                body=_(
                    "This sketch is still in use by {count} workpiece(s) on "
                    "the canvas. Please delete those workpieces first."
                ).format(count=len(workpieces_using_sketch)),
            )
            dialog.add_response("ok", _("OK"))
            dialog.set_default_response("ok")
            dialog.connect("response", lambda d, r: d.close())
            dialog.present()
            return

        # If not in use, confirm deletion
        dialog = Adw.MessageDialog(
            transient_for=parent_window,
            heading=_("Delete '{name}'?").format(name=sketch_to_delete.name),
            body=_(
                "This sketch definition will be permanently removed from the "
                "project. This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )

        def on_response(d, response_id):
            if response_id == "delete":
                command = ListItemCommand(
                    owner_obj=self.doc,
                    item=sketch_to_delete,
                    undo_command="add_sketch",
                    redo_command="remove_sketch",
                    name=_("Delete Sketch Definition"),
                )
                self.editor.doc.history_manager.execute(command)
            d.close()

        dialog.connect("response", on_response)
        dialog.present()

    def on_sketch_reordered(self, sender):
        """Handles reordering of sketches."""
        # For now, we don't support reordering sketches as they're stored in
        # a dict. This could be implemented in the future.
        pass
