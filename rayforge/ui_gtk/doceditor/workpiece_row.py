import logging
from gi.repository import Gdk, Gtk, Pango
from ...core.workpiece import WorkPiece
from ..icons import get_icon

logger = logging.getLogger(__name__)

_ICON_MAP = {
    ".svg": "file-svg-generic-symbolic",
    ".png": "file-png-generic-symbolic",
    ".jpg": "file-jpg-generic-symbolic",
    ".jpeg": "file-jpg-generic-symbolic",
    ".dxf": "file-dxf-generic-symbolic",
    ".pdf": "file-pdf-generic-symbolic",
    ".rd": "file-rd-generic-symbolic",
}


class WorkpieceRow(Gtk.Box):
    def __init__(self, workpiece: WorkPiece):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.workpiece = workpiece
        self.set_margin_start(6)
        self.set_margin_end(6)
        self.set_margin_top(4)
        self.set_margin_bottom(4)

        icon_name = self._get_icon_name()
        self.icon = get_icon(icon_name)
        self.icon.set_valign(Gtk.Align.CENTER)
        self.append(self.icon)

        self.name_label = Gtk.Label()
        self.name_label.set_hexpand(True)
        self.name_label.set_halign(Gtk.Align.START)
        self.name_label.set_valign(Gtk.Align.CENTER)
        self.name_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.append(self.name_label)

        self._update_ui()

        self._drag_source = Gtk.DragSource()
        self._drag_source.set_actions(Gdk.DragAction.MOVE)
        self._drag_source.connect("prepare", self._on_drag_prepare)
        self.add_controller(self._drag_source)

        workpiece.updated.connect(self._on_workpiece_updated)

    def do_destroy(self):
        self.workpiece.updated.disconnect(self._on_workpiece_updated)

    def get_drag_content(self) -> Gdk.ContentProvider:
        return Gdk.ContentProvider.new_for_value(self.workpiece.uid)

    def _get_icon_name(self) -> str:
        source = self.workpiece.source
        if source and source.source_file:
            suffix = source.source_file.suffix.lower()
            return _ICON_MAP.get(suffix, "image-x-generic-symbolic")
        return "image-x-generic-symbolic"

    def _update_ui(self):
        source = self.workpiece.source
        display_name = self.workpiece.name
        if source and source.name:
            display_name = source.name
        self.name_label.set_text(display_name)

    def _on_workpiece_updated(self, sender, **kwargs):
        self._update_ui()

    def _on_drag_prepare(self, drag_source, x, y):
        logger.debug(
            "DragPrepare(%s): uid=%s",
            self.workpiece.name,
            self.workpiece.uid[:8],
        )
        snapshot = Gtk.Snapshot()
        WorkpieceRow.do_snapshot(self, snapshot)
        paintable = snapshot.to_paintable()
        if paintable:
            drag_source.set_icon(paintable, x, y)
        return self.get_drag_content()
