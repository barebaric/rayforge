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
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def add_row(self, row):
        row.add_css_class("material-row")
        self.append(row)
        self.make_row_draggable(row)

    def make_row_draggable(self, row):
        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self.on_drag_prepare, row)
        drag_source.connect("drag-end", self.on_drag_end, row)
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
        source.set_icon(paintable, x, row.get_height()/2)
        self.drag_source_row = row
        self.potential_drop_index = -1
        return Gdk.ContentProvider.new_for_value(row)

    def on_drag_motion(self, drop_target, x, y, target_row):
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

    app = Gtk.Application(application_id='org.example.DragListBox')
    app.connect('activate', on_activate)
    app.run(None)
