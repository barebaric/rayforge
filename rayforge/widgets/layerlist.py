from gi.repository import Gtk  # type: ignore
from blinker import Signal
from ..models.doc import Doc
from ..undo.list_cmd import ReorderListCommand
from .draglist import DragListBox
from .layerview import LayerView
from .expander import Expander


class LayerListView(Expander):
    """
    A widget that displays a collapsible, reorderable list of Layers.
    """

    layer_activated = Signal()

    def __init__(self, doc: Doc, **kwargs):
        super().__init__(**kwargs)
        self.doc = doc

        self.set_title(_("Workpiece Layers"))
        self.set_expanded(True)  # Expanded by default

        # A container for all content that will be revealed by the expander
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(content_box)

        # The reorderable list of Layers goes inside the content box
        self.draglist = DragListBox()
        self.draglist.add_css_class("layer-list-box")
        self.draglist.reordered.connect(self.on_layers_reordered)
        self.draglist.connect("row-activated", self.on_row_activated)
        content_box.append(self.draglist)

        # Connect to document changes and perform initial population
        self.doc.changed.connect(self.on_doc_changed)
        self.on_doc_changed(self.doc)

    def on_doc_changed(self, sender, **kwargs):
        """Updates the list and subtitle when the document changes."""
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
        layers.
        """
        deletable = len(self.doc.layers) > 1
        self.draglist.remove_all()

        for layer in self.doc.layers:
            list_box_row = Gtk.ListBoxRow()
            list_box_row.data = layer
            layer_view = LayerView(self.doc, layer)
            # Control delete button visibility from the list view
            layer_view.set_deletable(deletable)
            layer_view.delete_clicked.connect(self.on_delete_layer_clicked)
            list_box_row.set_child(layer_view)
            self.draglist.add_row(list_box_row)
            layer_view.update_style()

    def on_row_activated(self, listbox, row):
        """Emits a signal when a layer row is clicked/activated."""
        if row and row.data:
            self.doc.set_active_layer(row.data)
            self.layer_activated.send(self, layer=row.data)

    def on_delete_layer_clicked(self, layer_view):
        """Handles deletion of a layer with an undoable command."""
        layer_to_delete = layer_view.layer
        new_list = [
            g for g in self.doc.layers if g is not layer_to_delete
        ]
        command = ReorderListCommand(
            target_obj=self.doc,
            list_property_name="layers",
            new_list=new_list,
            setter_method_name="set_layers",
            name=_("Remove layer '{name}'").format(name=layer_to_delete.name),
        )
        self.doc.history_manager.execute(command)

    def on_layers_reordered(self, sender):
        """Handles reordering of Layers with an undoable command."""
        new_order = [row.data for row in self.draglist]
        command = ReorderListCommand(
            target_obj=self.doc,
            list_property_name="layers",
            new_list=new_order,
            setter_method_name="set_layers",
            name=_("Reorder layers"),
        )
        self.doc.history_manager.execute(command)
