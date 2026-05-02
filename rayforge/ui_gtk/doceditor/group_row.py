import logging
from gi.repository import Gdk, Gtk, Pango
from ...core.group import Group
from ..icons import get_icon

logger = logging.getLogger(__name__)


class GroupRow(Gtk.Box):
    def __init__(self, group: Group):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.group = group
        self.set_margin_start(6)
        self.set_margin_end(6)
        self.set_margin_top(4)
        self.set_margin_bottom(4)

        self.icon = get_icon("layer-symbolic")
        self.icon.set_valign(Gtk.Align.CENTER)
        self.append(self.icon)

        self.name_label = Gtk.Label()
        self.name_label.set_hexpand(True)
        self.name_label.set_halign(Gtk.Align.START)
        self.name_label.set_valign(Gtk.Align.CENTER)
        self.name_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.append(self.name_label)

        self._update_ui()

        group.updated.connect(self._on_group_updated)

    def do_destroy(self):
        self.group.updated.disconnect(self._on_group_updated)

    def get_drag_content(self) -> Gdk.ContentProvider:
        return Gdk.ContentProvider.new_for_value(self.group.uid)

    def _update_ui(self):
        self.name_label.set_text(self.group.name)

    def _on_group_updated(self, sender, **kwargs):
        self._update_ui()

    def _on_drag_prepare(self, drag_source, x, y):
        snapshot = Gtk.Snapshot()
        GroupRow.do_snapshot(self, snapshot)
        paintable = snapshot.to_paintable()
        if paintable:
            drag_source.set_icon(paintable, x, y)
        return self.get_drag_content()
