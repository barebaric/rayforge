from typing import Protocol, runtime_checkable
from gi.repository import Gtk, Gdk
from blinker import Signal
from ...icons import get_icon
from ..util.gtk import apply_css


css = """
.material-list {
    background-color: transparent;
    padding: 0;
}

.material-list>row {
    background-color: transparent;
    transition: background-color 0.2s ease;
    border-bottom: 1px solid #00000020;
}
.material-list>row:last-child {
    border: 0;
}
.material-list>row:hover {
}
.material-list>row:drop(active) {
    outline: none;
    box-shadow: none;
}
.material-list>row.drop-above {
    border: 1px solid #f00;
    border-width: 2px 0px 0px 0px;
}
.material-list>row.drop-below {
    border: 1px solid #f00;
    border-width: 0px 0px 2px 0px;
}
.material-list>row:active {
}
.drag-handle {
    opacity: 0.5;
}
.material-list>row:hover .drag-handle {
    opacity: 1;
}
"""


@runtime_checkable
class Draggable(Protocol):
    """
    A protocol for widgets that can provide content for a drag operation.

    This is typically used for dragging an item out of a list view onto another
    widget, like a canvas. Any widget that implements the `get_drag_content`
    method will satisfy this protocol.
    """

    def get_drag_content(self) -> Gdk.ContentProvider:
        """Provides the content for a drag operation."""
        ...


class DragListBox(Gtk.ListBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_selection_mode(Gtk.SelectionMode.NONE)
        self.add_css_class("material-list")
        apply_css(css)
        self.reordered = Signal()
        self.drag_source_row = None
        self.potential_drop_index = -1

    def add_row(self, row):
        # Get original content widget from the row
        original_child = row.get_child()
        if original_child:
            row.set_child(None)  # Detach to re-parent it

        # Create a container box with a handle and the original content
        hbox = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_start=6,
            margin_end=6,
            margin_top=6,
            margin_bottom=6,
        )

        # Create drag handle
        handle = get_icon("drag-handle-symbolic")
        handle.add_css_class("drag-handle")
        handle.set_valign(Gtk.Align.CENTER)

        hbox.append(handle)

        if original_child:
            original_child.set_hexpand(True)
            hbox.append(original_child)

        row.set_child(hbox)
        self.append(row)
        self.make_row_draggable(row)

    def make_row_draggable(self, row):
        # Drag source is attached to the entire row. Clicks on interactive
        # children (buttons, entries) are consumed by them and won't
        # start a drag, which is the desired behavior.
        drag_source = Gtk.DragSource()
        # Allow MOVE for reordering and COPY for dragging content out
        # (e.g. to canvas)
        drag_source.set_actions(Gdk.DragAction.MOVE | Gdk.DragAction.COPY)
        drag_source.connect("prepare", self.on_drag_prepare, row)
        drag_source.connect("drag-end", self.on_drag_end)
        # Attach the drag source to the entire row. Clicks on interactive
        # children (buttons, entries) are consumed by them and won't
        # start a drag, which is the desired behavior.
        row.add_controller(drag_source)

        # Drop target is also on the entire row.
        drop_target = Gtk.DropTarget.new(Gtk.ListBoxRow, Gdk.DragAction.MOVE)
        drop_target.connect("drop", self.on_drop)
        drop_target.connect("motion", self.on_drag_motion)
        drop_target.connect("leave", self.on_drag_leave)
        row.add_controller(drop_target)

    def _remove_drop_marker(self):
        row = self.get_first_child()
        while row:
            row.remove_css_class("drop-above")
            row.remove_css_class("drop-below")
            row = row.get_next_sibling()

    def on_drag_prepare(self, source, x, y, row):
        snapshot = Gtk.Snapshot()
        row.do_snapshot(row, snapshot)
        paintable = snapshot.to_paintable()

        source.set_icon(paintable, x, y)

        self.drag_source_row = row
        self.potential_drop_index = -1

        # Default provider for reordering within the list
        reorder_provider = Gdk.ContentProvider.new_for_value(row)
        providers = [reorder_provider]

        # Check if the row's content widget provides custom drag content for
        # dropping on external widgets (like the canvas).
        hbox = row.get_child()
        if hbox and isinstance(hbox, Gtk.Box):
            # The actual content widget is the last child in our hbox layout.
            content_widget = hbox.get_last_child()

            # Use a type-safe protocol check instead of hasattr
            if isinstance(content_widget, Draggable):
                # This is usually a provider for a string (e.g., sketch UID)
                drag_out_provider = content_widget.get_drag_content()
                if drag_out_provider:
                    providers.append(drag_out_provider)

        # If we have multiple providers, unite them. Otherwise, just use the
        # one.
        if len(providers) > 1:
            return Gdk.ContentProvider.new_union(providers)
        else:
            return providers[0]

    def on_drag_motion(self, drop_target, x, y):
        # This handler is called on the *target* list. We only want to handle
        # drags that originated from *this* list. `self.drag_source_row` is
        # only set on the source list in `on_drag_prepare`.
        target_row = drop_target.get_widget()
        if not self.drag_source_row:
            return Gdk.DragAction(0)  # Reject drops from other lists

        self._remove_drop_marker()

        # Determine drop position and update visual marker
        if y < (target_row.get_height() / 2):
            target_row.add_css_class("drop-above")
            drop_index = target_row.get_index()
        else:
            target_row.add_css_class("drop-below")
            drop_index = target_row.get_index() + 1

        # Adjust index for the removal of the source row
        assert self.drag_source_row
        source_index = self.drag_source_row.get_index()
        if source_index < drop_index:
            drop_index -= 1

        self.potential_drop_index = drop_index
        return Gdk.DragAction.MOVE

    def on_drag_leave(self, drop_target):
        self._remove_drop_marker()

    def on_drag_end(self, source, drag, delete_data):
        # `delete_data` is True if `on_drop` returned True, meaning the drop
        # happened on a valid target.
        # If `delete_data` is False, we check if we have a last known valid
        # position.
        if delete_data or (self.potential_drop_index != -1):
            assert self.drag_source_row
            source_index = self.drag_source_row.get_index()
            # Only perform the move if the position is different
            if source_index != self.potential_drop_index:
                self.remove(self.drag_source_row)
                self.insert(self.drag_source_row, self.potential_drop_index)
                self.reordered.send(self)

        self._remove_drop_marker()
        self.drag_source_row = None
        self.potential_drop_index = -1

    def on_drop(self, drop_target, value, x, y):
        # This handler is called on the *target* list. We only want to handle
        # drags that originated from *this* list.
        if not self.drag_source_row:
            return False  # Reject drop

        # We just signal that the drop is accepted if a valid position was
        # found.
        # The actual reordering is handled in `on_drag_end`.
        return self.potential_drop_index != -1

    def __iter__(self):
        """
        Provides a Pythonic way to iterate over the rows of the ListBox,
        which is platform-independent.
        """
        child = self.get_first_child()
        while child:
            yield child
            child = child.get_next_sibling()


if __name__ == "__main__":

    class DragListWindow(Gtk.ApplicationWindow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.set_title("Reorderable List Example")
            self.set_default_size(300, 400)
            listview = DragListBox()
            self.set_child(listview)

            # Add some rows
            for i in range(5):
                label = Gtk.Label(label=f"Item {i + 1}")
                label.set_xalign(0)
                row = Gtk.ListBoxRow()
                row.set_child(label)
                listview.add_row(row)

    def on_activate(app):
        win = DragListWindow(application=app)
        win.present()

    app = Gtk.Application(application_id="org.example.DragListBox")
    app.connect("activate", on_activate)
    app.run(None)
