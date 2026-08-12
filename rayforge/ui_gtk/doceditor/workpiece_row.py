import logging

from gi.repository import Gdk, Gtk, Pango

from ...core.workpiece import WorkPiece
from ..icons import get_icon
from ..shared.gtk import apply_css

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

_RENAME_CSS = """
.layer-workpiece-list .layer-rename-entry {
    min-height: 0;
    padding: 1px 6px;
}
"""


class WorkpieceRow(Gtk.Box):
    def __init__(self, workpiece: WorkPiece, on_rename=None):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.workpiece = workpiece
        self._on_rename = on_rename
        self._rename_entry = None
        self._rename_click_controller = None
        self.set_margin_start(6)
        self.set_margin_end(6)
        self.set_margin_top(4)
        self.set_margin_bottom(4)

        apply_css(_RENAME_CSS)

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

        click = Gtk.GestureClick()
        click.set_button(Gdk.BUTTON_PRIMARY)
        click.connect("pressed", self._on_double_clicked)
        self.add_controller(click)

        self._update_ui()

        workpiece.updated.connect(self._on_workpiece_updated)

    def do_destroy(self):
        self.workpiece.updated.disconnect(self._on_workpiece_updated)
        self._remove_rename_click_controller()

    def get_drag_content(self) -> Gdk.ContentProvider:
        return Gdk.ContentProvider.new_for_value(self.workpiece.uid)

    def _get_icon_name(self) -> str:
        source = self.workpiece.source
        if source and source.source_file:
            suffix = source.source_file.suffix.lower()
            return _ICON_MAP.get(suffix, "image-x-generic-symbolic")
        return "image-x-generic-symbolic"

    def _get_display_name(self) -> str:
        display_name = self.workpiece.name
        if not display_name:
            source = self.workpiece.source
            if source and source.name:
                display_name = source.name
        return display_name

    def _update_ui(self):
        self.name_label.set_text(self._get_display_name())

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

    def _on_double_clicked(self, gesture, n_press, x, y):
        if n_press != 2:
            return
        if self.workpiece.geometry_provider_uid:
            return
        self.start_rename()

    def start_rename(self):
        """Starts in-place editing of the item name."""
        if self._rename_entry is not None:
            return
        entry = Gtk.Entry()
        entry.set_text(self._get_display_name())
        entry.select_region(0, -1)
        entry.set_hexpand(True)
        entry.set_halign(Gtk.Align.START)
        entry.set_valign(Gtk.Align.CENTER)
        entry.add_css_class("layer-rename-entry")
        entry.connect("activate", self._on_rename_committed)
        focus_controller = Gtk.EventControllerFocus.new()
        focus_controller.connect("leave", self._on_rename_focus_out)
        entry.add_controller(focus_controller)
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self._on_rename_key_pressed)
        entry.add_controller(key_controller)
        self._rename_entry = entry
        self.remove(self.name_label)
        self.append(entry)
        entry.grab_focus()
        self._install_rename_click_capture()

    def _install_rename_click_capture(self):
        """Closes the editor when clicking anywhere outside the entry."""
        root = self.get_ancestor(Gtk.Window)
        if root is None:
            return
        controller = Gtk.GestureClick()
        controller.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        controller.connect("pressed", self._on_rename_root_click)
        root.add_controller(controller)
        self._rename_click_controller = controller

    def _remove_rename_click_controller(self):
        if self._rename_click_controller is None:
            return
        widget = self._rename_click_controller.get_widget()
        if widget:
            widget.remove_controller(self._rename_click_controller)
        self._rename_click_controller = None

    def _on_rename_root_click(self, gesture, n_press, x, y):
        if self._rename_entry is None:
            return
        entry = self._rename_entry
        root = self.get_ancestor(Gtk.Window)
        picked = root.pick(x, y, Gtk.PickFlags.DEFAULT) if root else None
        while picked is not None and picked is not root:
            if picked is entry:
                return
            picked = picked.get_parent()
        self._finish_rename(entry)

    def _on_rename_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self._cancel_rename()
            return True
        return False

    def _on_rename_committed(self, entry):
        self._finish_rename(entry)

    def _on_rename_focus_out(self, *args):
        self._finish_rename(self._rename_entry)

    def _cancel_rename(self):
        if self._rename_entry is None:
            return
        self._replace_name_widget()
        self._update_ui()

    def _finish_rename(self, entry):
        if self._rename_entry is None:
            return
        new_name = entry.get_text().strip()
        self._replace_name_widget()
        if new_name and new_name != self.workpiece.name:
            if self._on_rename:
                self._on_rename(self.workpiece, new_name)
            else:
                self._update_ui()

    def _replace_name_widget(self):
        if self._rename_entry is None:
            return
        self._remove_rename_click_controller()
        self.remove(self._rename_entry)
        self._rename_entry = None
        self.append(self.name_label)
