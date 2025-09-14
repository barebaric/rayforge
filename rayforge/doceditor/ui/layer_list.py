import logging
from gi.repository import Gtk
from blinker import Signal
from typing import cast

from ...core.stocklayer import StockLayer
from ...core.doc import Doc
from ...undo.models.list_cmd import ReorderListCommand
from ...core.layer import Layer
from ...shared.ui.draglist import DragListBox
from .layer_view import LayerView
from ...shared.ui.expander import Expander
from ...icons import get_icon

logger = logging.getLogger(__name__)


class LayerListView(Expander):
    """
    A widget that displays a collapsible, reorderable list of Layers.
    """

    layer_activated = Signal()

    def __init__(self, doc: Doc, **kwargs):
        super().__init__(**kwargs)
        self.doc = doc

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

        # An "Add" button, styled like in WorkflowView
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

    def update_list(self):
        """
        Re-populates the draglist to match the state of the document's
        layers and ensures the initial active state is correctly displayed.
        """
        self.draglist.remove_all()
        # You can only delete a regular layer if there is more than one.
        can_delete_regular_layer = (
            sum(1 for la in self.doc.layers if not isinstance(la, StockLayer))
            > 1
        )

        for layer in self.doc.children:
            if not isinstance(layer, Layer):
                continue

            list_box_row = Gtk.ListBoxRow()
            list_box_row.data = layer  # type: ignore
            layer_view = LayerView(self.doc, layer)

            is_deletable = can_delete_regular_layer and not isinstance(
                layer, StockLayer
            )
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
                self.doc.active_layer = layer

            # Send a signal for other parts of the UI (e.g., MainWindow)
            self.layer_activated.send(self, layer=layer)

    def on_button_add_clicked(self, button):
        """Handles creation of a new layer with an undoable command."""
        # Find a unique default name for the new layer
        base_name = _("Layer")
        existing_names = {layer.name for layer in self.doc.layers}
        highest_num = 0
        for name in existing_names:
            if name.startswith(base_name):
                try:
                    num_part = name[len(base_name) :].strip()
                    if num_part.isdigit():
                        highest_num = max(highest_num, int(num_part))
                except ValueError:
                    continue  # Ignore names that don't parse correctly

        new_name = f"{base_name} {highest_num + 1}"

        new_layer = Layer(name=new_name)

        new_list = self.doc.layers + [new_layer]
        command = ReorderListCommand(
            target_obj=self.doc,
            list_property_name="layers",
            new_list=new_list,
            setter_method_name="set_layers",
            name=_("Add layer '{name}'").format(name=new_layer.name),
        )
        self.doc.history_manager.execute(command)
        self.doc.active_layer = new_layer

    def on_delete_layer_clicked(self, layer_view):
        """Handles deletion of a layer with an undoable command."""
        layer_to_delete = layer_view.layer
        new_list = [g for g in self.doc.layers if g is not layer_to_delete]
        command = ReorderListCommand(
            target_obj=self.doc,
            list_property_name="layers",
            new_list=new_list,
            setter_method_name="set_layers",
            name=_("Remove layer '{name}'").format(name=layer_to_delete.name),
        )
        try:
            self.doc.history_manager.execute(command)
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
        command = ReorderListCommand(
            target_obj=self.doc,
            list_property_name="layers",
            new_list=new_order,
            setter_method_name="set_layers",
            name=_("Reorder layers"),
        )
        self.doc.history_manager.execute(command)
