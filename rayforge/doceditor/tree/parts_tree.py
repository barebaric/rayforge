"""The Parts tab: the stock, the groups and the parts."""

from blinker import Signal
from gi.repository import Gtk

from ...core.doc import Doc
from ...core.group import Group
from ...core.workpiece import WorkPiece
from .group_node import GroupNode
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
        self._expanded: set = set()
        self._selected: set = set()
        self._tree = TreeWidget()
        self.append(self._tree)

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
        for stock in self._doc.stock_items:
            rows.append(StockNode(stock, 0, self._selected, self))
        for layer in self._doc.layers:
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

    def start_rename(self, uid: str) -> bool:
        """Starts inline renaming of the row with the given uid."""
        return self._tree.start_rename(uid)

    def on_twisty_toggled(self, spec):
        if spec.uid in self._expanded:
            self._expanded.remove(spec.uid)
        else:
            self._expanded.add(spec.uid)
        self.refresh()
