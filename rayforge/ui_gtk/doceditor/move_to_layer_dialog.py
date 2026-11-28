"""A dialog that chooses the target layer for moving items."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk

from ...core.doc import Doc
from ...core.group import Group
from ...core.item import DocItem
from ...core.workpiece import WorkPiece


class MoveToLayerDialog(Gtk.Dialog):
    """Lists the layers an item can be moved to.

    Layers have no "up" or "down", so moving items between them asks
    for the target explicitly. :attr:`layer_selected` reports the
    chosen layer.
    """

    def __init__(self, parent: Gtk.Window, doc: Doc, items: list[DocItem]):
        super().__init__(transient_for=parent)
        self.layer_selected = Signal()

        self.set_title(_("Move to Layer"))
        self.set_modal(True)
        self.add_button(_("_Cancel"), Gtk.ResponseType.CANCEL)
        self._ok_button = self.add_button(_("_Move"), Gtk.ResponseType.OK)
        self._ok_button.add_css_class("suggested")
        self._ok_button.set_sensitive(False)

        movable = [
            item for item in items if isinstance(item, (WorkPiece, Group))
        ]
        current = {item.layer for item in movable}
        self._layers = [layer for layer in doc.layers if layer not in current]

        content = self.get_content_area()
        content.set_spacing(6)
        intro = Gtk.Label(
            label=_("Choose the layer to move the selected items to."),
            wrap=True,
            xalign=0,
            max_width_chars=45,
        )
        content.append(intro)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        listbox.connect("row-selected", self._on_row_selected)
        for layer in self._layers:
            row = Gtk.ListBoxRow()
            row.set_child(Gtk.Label(label=layer.name, xalign=0))
            row.set_activatable(True)
            listbox.append(row)
        content.append(listbox)
        self._listbox = listbox

        self.connect("response", self._on_response)

    def _on_row_selected(self, listbox, row):
        self._ok_button.set_sensitive(row is not None)

    def _on_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            row = self._listbox.get_selected_row()
            if row is not None:
                self.layer_selected.send(
                    self, layer=self._layers[row.get_index()]
                )
        self.destroy()
