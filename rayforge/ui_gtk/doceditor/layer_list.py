import logging
from gi.repository import Gtk
from blinker import Signal
from typing import cast, TYPE_CHECKING

from ...core.layer import Layer
from ..shared.draglist import DragListBox
from .layer_view import LayerView
from ..shared.expander import Expander
from ..icons import get_icon

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class LayerListView(Expander):
    """
    A widget that displays a collapsible, reorderable list of Layers.
    """

    layer_activated = Signal()

    def __init__(self, editor: "DocEditor", **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        self.doc = editor.doc

        self.set_title(_("Workpiece Layers"))
        self.set_expanded(True)

        # A container for all content that will be revealed by the expander
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(content_box)

        # The reorderable list of Layers goes inside the content box
        self.draglist = DragListBox()
        self.draglist.add_css_class("layer-list-box")
        self.draglist.reordered.connect(self.on_layers_reordered)
        self.draglist.connect("row-activated", self.on_row_activated)
        content_box.append(self.draglist)

        # An "Add" button
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

        lbl = _("Add New Layer")
        add_label = Gtk.Label()
        add_label.set_markup(f"<span weight='normal'>{lbl}</span>")
        add_label.set_xalign(0)
        button_box.append(add_label)
        add_button.set_child(button_box)

        # Connect to document changes and perform initial population
        self.doc.updated.connect(self.on_doc_changed)
        self.doc.descendant_added.connect(self.on_doc_changed)
        self.doc.descendant_removed.connect(self.on_doc_changed)
        self.doc.active_layer_changed.connect(self.on_active_layer_changed)
        self.on_doc_changed(self.doc)

    def on_doc_changed(self, sender, **kwargs):
        """
        Updates the list and subtitle when the document structure changes.
        """
        count = len(self.doc.layers)
        self.set_subtitle(
            _("{count} layer").format(count=count)
            if count == 1
            else _("{count} Layers").format(count=count)
        )
        self.update_list()

    def on_active_layer_changed(self, sender, **kwargs):
        """
        Updates only the styles when the active layer changes, avoiding
        expensive list rebuilding.
        """
        # Just update styles instead of rebuilding the entire list
        for row in self.draglist:
            if isinstance(row, Gtk.ListBoxRow) and hasattr(row, "data"):
                layer_view = row.get_child()
                from .layer_view import LayerView

                if isinstance(layer_view, LayerView):
                    layer_view.update_style()

    def update_list(self):
        """
        Re-populates the draglist to match the state of the document's
        layers and ensures the initial active state is correctly displayed.
        """
        self.draglist.remove_all()
        # You can only delete a regular layer if there is more than one.
        can_delete_regular_layer = len(self.doc.layers) > 1

        for layer in self.doc.children:
            if not isinstance(layer, Layer):
                continue

            list_box_row = Gtk.ListBoxRow()
            list_box_row.data = layer  # type: ignore
            layer_view = LayerView(self.doc, layer, self.editor)

            is_deletable = can_delete_regular_layer
            layer_view.set_deletable(is_deletable)

            layer_view.delete_clicked.connect(self.on_delete_layer_clicked)
            list_box_row.set_child(layer_view)
            self.draglist.add_row(list_box_row)

            # The LayerView now has a parent. Manually call
            # update_style() here to guarantee the initial CSS class is set
            # correctly based on the model's state at creation time.
            layer_view.update_style()

    def on_row_activated(self, listbox, row):
        """Handles user clicks to change the active layer."""
        if row and row.data:
            layer = cast(Layer, row.data)

            # Update the model. This fires the `active_layer_changed` signal,
            # which all LayerView widgets (including this one) are listening
            # to. They will then update their own styles automatically.
            if self.doc.active_layer is not layer:
                self.editor.layer.set_active_layer(layer)

            # Send a signal for other parts of the UI (e.g., MainWindow)
            self.layer_activated.send(self, layer=layer)

    def on_button_add_clicked(self, button):
        """Handles creation of a new layer with an undoable command."""
        self.editor.layer.add_layer_and_set_active()

    def on_delete_layer_clicked(self, layer_view):
        """Handles deletion of a layer with an undoable command."""
        layer_to_delete = layer_view.layer
        try:
            self.editor.layer.delete_layer(layer_to_delete)
        except ValueError as e:
            logger.warning(
                "Layer deletion prevented by model validation: %s", e
            )
            # Optionally, show a toast to the user here. The model state did
            # not change, so the UI will remain correct after the next
            # scheduled redraw.

    def on_layers_reordered(self, sender):
        """Handles reordering of Layers with an undoable command."""
        new_order = [row.data for row in self.draglist]  # type: ignore
        self.editor.layer.reorder_layers(new_order)
