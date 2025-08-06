from gi.repository import Gtk  # type: ignore
from blinker import Signal
from ...core.doc import Doc
from ...undo.models.list_cmd import ReorderListCommand
from ...core.layer import Layer
from ...shared.ui.draglist import DragListBox
from .layer_view import LayerView
from ...shared.ui.expander import Expander


class LayerListView(Expander):
    """
    A widget that displays a collapsible, reorderable list of Layers.
    """

    layer_activated = Signal()

    def __init__(self, doc: Doc, **kwargs):
        super().__init__(**kwargs)
        self.doc = doc

        self.set_title(_("Workpiece Layers"))
        self.set_expanded(False)

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

        add_icon = Gtk.Image.new_from_icon_name("list-add-symbolic")
        button_box.append(add_icon)

        lbl = _('Add New Layer')
        add_label = Gtk.Label()
        add_label.set_markup(
            f"<span weight='normal'>{lbl}</span>"
        )
        add_label.set_xalign(0)
        button_box.append(add_label)
        add_button.set_child(button_box)

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
            self.doc.active_layer = row.data
            self.layer_activated.send(self, layer=row.data)

    def on_button_add_clicked(self, button):
        """Handles creation of a new layer with an undoable command."""
        # Find a unique default name for the new layer
        base_name = _("Layer")
        existing_names = {layer.name for layer in self.doc.layers}
        next_num_to_try = len(self.doc.layers) + 1
        while True:
            new_name = f"{base_name} {next_num_to_try}"
            if new_name not in existing_names:
                break
            next_num_to_try += 1

        new_layer = Layer(self.doc, name=new_name)

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
