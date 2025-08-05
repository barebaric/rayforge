from gi.repository import Gtk, Gdk  # type: ignore
from blinker import Signal


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


class DragListBox(Gtk.ListBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_selection_mode(Gtk.SelectionMode.NONE)
        self.add_css_class("material-list")
        self.apply_css()
        self.reordered = Signal()
        self.drag_source_row = None
        self.potential_drop_index = -1

    def apply_css(self):
        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def add_row(self, row):
        row.add_css_class("material-row")

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
        handle = Gtk.Image.new_from_icon_name("drag-handle-symbolic")
        handle.add_css_class("drag-handle")
        handle.set_valign(Gtk.Align.CENTER)

        hbox.append(handle)

        if original_child:
            original_child.set_hexpand(True)
            hbox.append(original_child)

        row.set_child(hbox)
        self.append(row)
        self.make_row_draggable(row, handle)

    def make_row_draggable(self, row, handle):
        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self.on_drag_prepare, row)
        drag_source.connect("drag-end", self.on_drag_end, row)
        # Attach the drag source to the entire row. Clicks on interactive
        # children (buttons, entries) are consumed by them and won't
        # start a drag, which is the desired behavior.
        row.add_controller(drag_source)

        drop_target = Gtk.DropTarget.new(Gtk.ListBoxRow, Gdk.DragAction.MOVE)
        drop_target.connect("drop", self.on_drop, row)
        drop_target.connect("motion", self.on_drag_motion, row)
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
        return Gdk.ContentProvider.new_for_value(row)

    def on_drag_motion(self, drop_target, x, y, target_row):
        # This handler is called on the *target* list. We only want to handle
        # drags that originated from *this* list. `self.drag_source_row` is
        # only set on the source list in `on_drag_prepare`.
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

    def on_drag_leave(self, drag, row):
        row.remove_css_class("drop-above")
        row.remove_css_class("drop-below")

    def on_drag_end(self, source, drag, delete_data, row):
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

    def on_drop(self, drop_target, value, x, y, target_row):
        # This handler is called on the *target* list. We only want to handle
        # drags that originated from *this* list.
        if not self.drag_source_row:
            return False  # Reject drop

        # We just signal that the drop is accepted if a valid position was
        # found.
        # The actual reordering is handled in `on_drag_end`.
        return self.potential_drop_index != -1


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
