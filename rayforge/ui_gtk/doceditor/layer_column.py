import logging
from gettext import gettext as _
from typing import Optional, TYPE_CHECKING
from gi.repository import Gdk, GObject, Gtk, Pango
from ...core.doc import Doc
from ...core.layer import Layer
from ...core.workpiece import WorkPiece
from ..icons import get_icon
from ..shared.gtk import apply_css
from .layer_settings_dialog import LayerSettingsDialog
from .workpiece_row import WorkpieceRow

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

css = """
.layer-column {
    background-color: alpha(@theme_fg_color, 0.03);
    border-radius: 8px;
    border: 1px solid @borders;
    min-width: 160px;
}
.layer-column.active-layer-column {
    border-color: @accent_bg_color;
    background-color: alpha(@accent_bg_color, 0.05);
}
.layer-column-header {
    padding: 6px 8px;
    border-bottom: 1px solid @borders;
    border-radius: 8px 8px 0 0;
    background-color: alpha(@theme_fg_color, 0.05);
}
.layer-column.active-layer-column .layer-column-header {
    background-color: alpha(@accent_bg_color, 0.1);
}
.layer-column-header button.flat {
    min-width: 28px;
    min-height: 28px;
    padding: 2px;
}
.layer-workpiece-list {
    background-color: transparent;
    padding: 0;
}
.layer-workpiece-list > row {
    background-color: transparent;
    border-radius: 4px;
    margin: 1px 4px;
}
.layer-workpiece-list > row:drop(active) {
    background-color: alpha(@accent_bg_color, 0.2);
    outline: 2px solid @accent_bg_color;
    outline-offset: -2px;
}
"""


class LayerColumn(Gtk.Box):
    def __init__(
        self,
        doc: Doc,
        layer: Layer,
        editor: "DocEditor",
        can_delete: bool = True,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        apply_css(css)
        self.add_css_class("layer-column")
        self.set_margin_end(6)

        self.doc = doc
        self.layer = layer
        self.editor = editor
        self._row_workpieces = {}

        self._build_header(can_delete)
        self._build_workpiece_list()

        self._click_gesture = Gtk.GestureClick()
        self._click_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self._click_gesture.connect("pressed", self._on_column_clicked)
        self.add_controller(self._click_gesture)

        self._connect_signals()
        self._update_style()

    def _build_header(self, can_delete: bool):
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        header.add_css_class("layer-column-header")
        header.set_hexpand(True)

        self.icon_container = Gtk.Box()
        self.icon_container.set_valign(Gtk.Align.CENTER)
        header.append(self.icon_container)

        name_label = Gtk.Label()
        name_label.set_text(self.layer.name)
        name_label.set_hexpand(True)
        name_label.set_halign(Gtk.Align.START)
        name_label.set_valign(Gtk.Align.CENTER)
        name_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.name_label = name_label
        header.append(name_label)

        self.settings_button = Gtk.Button(child=get_icon("settings-symbolic"))
        self.settings_button.add_css_class("flat")
        self.settings_button.set_tooltip_text(_("Layer Settings"))
        self.settings_button.connect("clicked", self._on_settings_clicked)
        header.append(self.settings_button)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.add_css_class("flat")
        self.delete_button.set_tooltip_text(_("Delete this layer"))
        self.delete_button.set_visible(can_delete)
        self.delete_button.connect("clicked", self._on_delete_clicked)
        header.append(self.delete_button)

        self.visibility_on_icon = get_icon("visibility-on-symbolic")
        self.visibility_off_icon = get_icon("visibility-off-symbolic")

        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.set_active(self.layer.visible)
        self.visibility_button.set_child(self.visibility_on_icon)
        self.visibility_button.add_css_class("flat")
        self.visibility_button.set_tooltip_text(_("Toggle layer visibility"))
        self.visibility_button.connect("clicked", self._on_visibility_clicked)
        header.append(self.visibility_button)

        self.append(header)
        self._update_icon()

    def _build_workpiece_list(self):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)

        self.listbox = Gtk.ListBox()
        self.listbox.add_css_class("layer-workpiece-list")
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)

        drop_target = Gtk.DropTarget.new(
            GObject.TYPE_STRING, Gdk.DragAction.MOVE
        )
        drop_target.connect("drop", self._on_drop)
        drop_target.connect("motion", self._on_drop_motion)
        drop_target.connect("leave", self._on_drop_leave)
        self.listbox.add_controller(drop_target)

        scrolled.set_child(self.listbox)
        self.append(scrolled)

        self._rebuild_workpiece_list()

    def _rebuild_workpiece_list(self):
        child = self.listbox.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self.listbox.remove(child)
            child = next_child
        self._row_workpieces.clear()

        for wp in self.layer.workpieces:
            row = Gtk.ListBoxRow()
            wp_row = WorkpieceRow(wp)
            row.set_child(wp_row)
            self._row_workpieces[row] = wp
            self.listbox.append(row)

    def _update_icon(self):
        if old := self.icon_container.get_first_child():
            self.icon_container.remove(old)
        icon_name = (
            "rotary-symbolic"
            if self.layer.rotary_enabled
            else "layer-symbolic"
        )
        self.icon_container.append(get_icon(icon_name))

    def _update_style(self):
        if self.layer.active:
            self.add_css_class("active-layer-column")
        else:
            self.remove_css_class("active-layer-column")

    def _update_ui(self):
        self.name_label.set_text(self.layer.name)
        self._update_icon()
        self.visibility_button.set_active(self.layer.visible)
        if self.layer.visible:
            self.visibility_button.set_child(self.visibility_on_icon)
        else:
            self.visibility_button.set_child(self.visibility_off_icon)

    def _connect_signals(self):
        self.layer.updated.connect(self._on_layer_updated)
        self.layer.descendant_added.connect(self._on_layer_structure_changed)
        self.layer.descendant_removed.connect(self._on_layer_structure_changed)
        self.doc.active_layer_changed.connect(self._on_active_layer_changed)

    def do_destroy(self):
        self.layer.updated.disconnect(self._on_layer_updated)
        self.layer.descendant_added.disconnect(
            self._on_layer_structure_changed
        )
        self.layer.descendant_removed.disconnect(
            self._on_layer_structure_changed
        )
        self.doc.active_layer_changed.disconnect(self._on_active_layer_changed)

    def _on_layer_updated(self, sender, **kwargs):
        self._update_ui()

    def _on_layer_structure_changed(self, sender, **kwargs):
        self._rebuild_workpiece_list()

    def _on_active_layer_changed(self, sender, **kwargs):
        self._update_style()

    def _on_settings_clicked(self, button):
        toplevel = self.get_ancestor(Gtk.Window)
        if not toplevel:
            return
        dialog = LayerSettingsDialog(self.layer, transient_for=toplevel)
        dialog.present()

    def _on_delete_clicked(self, button):
        self.editor.layer.delete_layer(self.layer)

    def _on_visibility_clicked(self, button):
        new_visibility = button.get_active()
        if new_visibility == self.layer.visible:
            return
        self.editor.layer.set_layer_visibility(self.layer, new_visibility)

    def _on_column_clicked(self, gesture, n_press, x, y):
        picked = self.pick(x, y, Gtk.PickFlags.DEFAULT)
        if picked is not None:
            widget = picked
            while widget and widget is not self:
                if isinstance(widget, (Gtk.Button, WorkpieceRow)):
                    gesture.set_state(Gtk.EventSequenceState.DENIED)
                    return
                widget = widget.get_parent()
        if self.doc.active_layer is not self.layer:
            self.editor.layer.set_active_layer(self.layer)

    def _on_drop(self, drop_target, value, x, y):
        uid = value
        if not uid:
            return False

        wp = self._find_workpiece_by_uid(uid)
        if not wp:
            return False
        if not wp.layer or wp.layer is self.layer:
            return False

        self.editor.layer.move_workpieces_to_layer([wp], self.layer)
        return True

    def _on_drop_motion(self, drop_target, x, y):
        return Gdk.DragAction.MOVE

    def _on_drop_leave(self, drop_target):
        pass

    def _find_workpiece_by_uid(self, uid: str) -> Optional[WorkPiece]:
        for layer in self.doc.layers:
            for wp in layer.all_workpieces:
                if wp.uid == uid:
                    return wp
        return None
