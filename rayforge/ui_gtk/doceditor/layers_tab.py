import logging
from gettext import gettext as _
from typing import TYPE_CHECKING
from gi.repository import Gtk
from ...core.doc import Doc
from ...core.layer import Layer
from ..icons import get_icon
from .layer_column import LayerColumn

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class LayersTab(Gtk.Box):
    def __init__(self, editor: "DocEditor"):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.editor = editor
        self.doc = editor.doc
        self._columns = []

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)

        self.columns_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=0
        )
        self.columns_box.set_margin_start(9)
        self.columns_box.set_margin_top(9)
        self.columns_box.set_margin_bottom(9)
        self.columns_box.set_valign(Gtk.Align.FILL)
        scrolled.set_child(self.columns_box)
        self.append(scrolled)

        add_button = Gtk.Button(child=get_icon("add-symbolic"))
        add_button.add_css_class("flat")
        add_button.set_tooltip_text(_("Add New Layer"))
        add_button.set_valign(Gtk.Align.START)
        add_button.set_margin_top(18)
        add_button.set_margin_start(9)
        add_button.set_margin_end(9)
        add_button.connect("clicked", self._on_add_clicked)
        self.append(add_button)

        self._connect_signals()
        self._rebuild()

    def set_doc(self, doc: Doc):
        if self.doc == doc:
            return
        self._disconnect_signals()
        self.doc = doc
        self._connect_signals()
        self._rebuild()

    def _connect_signals(self):
        self.doc.descendant_added.connect(self._on_structure_changed)
        self.doc.descendant_removed.connect(self._on_structure_changed)

    def _disconnect_signals(self):
        self.doc.descendant_added.disconnect(self._on_structure_changed)
        self.doc.descendant_removed.disconnect(self._on_structure_changed)

    def do_destroy(self):
        self._disconnect_signals()

    def _on_structure_changed(self, sender, **kwargs):
        self._rebuild()

    def _rebuild(self):
        for col in self._columns:
            self.columns_box.remove(col)
        self._columns.clear()

        can_delete = len(list(self.doc.layers)) > 1
        for child in self.doc.children:
            if not isinstance(child, Layer):
                continue
            col = LayerColumn(
                self.doc, child, self.editor, can_delete=can_delete
            )
            self._columns.append(col)
            self.columns_box.append(col)

    def _on_add_clicked(self, button):
        self.editor.layer.add_layer_and_set_active()
