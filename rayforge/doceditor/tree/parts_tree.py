"""The Parts tab: the stock, the groups and the parts."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk

from ...core.doc import Doc
from ...core.group import Group
from ...core.layer import Layer
from ...core.workpiece import WorkPiece
from .group_node import GroupNode
from .node import show_row_menu
from .stock_node import StockNode
from .tree import TreeWidget
from .workpiece_node import WorkPieceNode


class PartsTree(Gtk.Box):
    """Shows the document's stock, groups and parts.

    The tree owns the expansion and selection state and re-derives the
    rows from the document on every change. Visibility toggles are
    reported via :attr:`visibility_toggled`; applying them to the model
    is the caller's job, and the refresh happens through the document
    signals.
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.selection_changed = Signal()
        self.rename_requested = Signal()
        self.visibility_toggled = Signal()

        self._doc: Doc | None = None
        self._layer: Layer | None = None
        self._expanded: set = set()
        self._selected: set = set()
        self._tree = TreeWidget()
        self.append(self._tree)

    def set_layer(self, layer: Layer | None):
        """Restricts the tree to one layer; None shows all layers."""
        if self._layer is layer:
            return
        self._layer = layer
        self.refresh()

    def set_selected_items(self, items):
        """Highlights the given model items, expanding their groups."""
        for item in items:
            parent = item.parent
            while parent is not None:
                if isinstance(parent, Group):
                    self._expanded.add(parent.uid)
                parent = parent.parent
        uids = {item.uid for item in items}
        if uids == self._selected:
            return
        self._selected = uids
        self.refresh()

    def set_doc(self, doc: Doc | None):
        if self._doc is doc:
            return
        if self._doc is not None:
            self._disconnect(self._doc)
        self._doc = doc
        if doc is not None:
            self._connect(doc)
        self._selected.clear()
        self.refresh()

    def do_destroy(self):
        if self._doc is not None:
            self._disconnect(self._doc)
            self._doc = None

    def _connect(self, doc: Doc):
        doc.descendant_added.connect(self._on_doc_changed)
        doc.descendant_removed.connect(self._on_doc_changed)
        doc.descendant_updated.connect(self._on_doc_changed)
        doc.descendant_renamed.connect(self._on_doc_changed)
        doc.updated.connect(self._on_doc_changed)

    def _disconnect(self, doc: Doc):
        doc.descendant_added.disconnect(self._on_doc_changed)
        doc.descendant_removed.disconnect(self._on_doc_changed)
        doc.descendant_updated.disconnect(self._on_doc_changed)
        doc.descendant_renamed.disconnect(self._on_doc_changed)
        doc.updated.disconnect(self._on_doc_changed)

    def _on_doc_changed(self, sender, **kwargs):
        self.refresh()

    def expand(self, uid: str):
        """Expands the row with the given uid and refreshes."""
        self._expanded.add(uid)
        self.refresh()

    def refresh(self):
        if self._doc is None:
            self._tree.show([])
            return
        rows: list[Gtk.Widget] = []
        layers = [self._layer] if self._layer is not None else self._doc.layers
        for layer in layers:
            for stock in layer.stock_items:
                rows.append(StockNode(stock, 0, self._selected, self))
            for item in layer.content_items:
                if isinstance(item, Group):
                    rows.extend(
                        GroupNode.build_rows(
                            item, 0, self._expanded, self._selected, self
                        )
                    )
                elif isinstance(item, WorkPiece):
                    rows.extend(
                        WorkPieceNode.build_rows(
                            item, 0, self._expanded, self._selected, self
                        )
                    )
        self._tree.show(rows)

    def on_row_activated(self, spec):
        self._selected = {spec.uid}
        self.refresh()
        self.selection_changed.send(self, spec=spec)

    def on_eye_toggled(self, spec):
        if self._doc is None:
            return
        item = self._doc.find_descendant_by_uid(spec.uid)
        if item is not None:
            self.visibility_toggled.send(self, item=item, on=not spec.on)

    def on_row_renamed(self, spec, name: str):
        self.rename_requested.send(self, uid=spec.uid, name=name)

    def on_row_context_menu(self, spec, row, x, y):
        show_row_menu(
            row,
            [(_("Rename"), lambda: self.start_rename(spec.uid))],
            x=x,
            y=y,
        )

    def rows(self):
        """Yields the visible node rows."""
        return self._tree.rows()

    def find_row(self, uid: str):
        """Returns the row with the given uid, or None."""
        return self._tree.find_row(uid)

    def start_rename(self, uid: str) -> bool:
        """Starts inline renaming of the row with the given uid."""
        return self._tree.start_rename(uid)

    def on_twisty_toggled(self, spec):
        if spec.uid in self._expanded:
            self._expanded.remove(spec.uid)
        else:
            self._expanded.add(spec.uid)
        self.refresh()
