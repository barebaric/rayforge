"""The generic tree container used by the Parts and Operations tabs."""

from gi.repository import Gtk

from ...ui_gtk.shared.gtk import apply_css

_CSS = """
.rf-tree-list {
    background-color: transparent;
}
"""


class TreeWidget(Gtk.ScrolledWindow):
    """Hosts the rows of a tree; stateless beyond its child widgets.

    Callers rebuild the row list on every model change; rows carry
    their interactions through the callbacks object they were
    constructed with.
    """

    def __init__(self):
        super().__init__()
        apply_css(_CSS)
        self.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.set_hexpand(True)
        self.set_vexpand(True)

        self._list = Gtk.ListBox()
        self._list.add_css_class("rf-tree-list")
        self._list.set_selection_mode(Gtk.SelectionMode.NONE)

        self._tail_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        outer.append(self._list)
        outer.append(self._tail_box)
        self.set_child(outer)

    def show(self, rows: list[Gtk.Widget], tail: Gtk.Widget | None = None):
        """Replaces the rows and the trailing widget below them."""
        child = self._list.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._list.remove(child)
            child = next_child

        for row in rows:
            self._list.append(row)

        child = self._tail_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._tail_box.remove(child)
            child = next_child
        if tail is not None:
            self._tail_box.append(tail)

    def rows(self):
        """Yields the tree's node rows."""
        from .node import NodeRow

        child = self._list.get_first_child()
        while child:
            if isinstance(child, NodeRow):
                yield child
            child = child.get_next_sibling()

    def find_row(self, uid: str):
        """Returns the row with the given uid, or None."""
        for row in self.rows():
            if row.spec.uid == uid:
                return row
        return None

    def start_rename(self, uid: str) -> bool:
        """Starts inline renaming of the row with the given uid."""
        row = self.find_row(uid)
        if row is None:
            return False
        row.start_rename()
        return True
