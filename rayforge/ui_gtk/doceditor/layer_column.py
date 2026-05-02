import json
import logging
from gettext import gettext as _
from typing import List, Optional, Set, TYPE_CHECKING, cast
from blinker import Signal
from gi.repository import Adw, Gdk, Gio, GObject, Gtk, Pango
from ...context import get_context
from ...core.doc import Doc
from ...core.group import Group
from ...core.item import DocItem
from ...core.layer import Layer
from ...core.source_asset import SourceAsset
from ...core.stock_asset import StockAsset
from ...core.workpiece import WorkPiece
from ..icons import get_icon
from ..shared.gtk import apply_css
from . import import_handler
from .group_row import GroupRow
from .layer_settings_dialog import LayerSettingsDialog
from .workflow_row import WorkflowRow
from .workpiece_row import WorkpieceRow

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor
    from ...ui_gtk.mainwindow import MainWindow

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
.layer-column-header .dim-label {
    font-size: smaller;
}
.layer-workpiece-list {
    background-color: transparent;
    padding: 0;
}
.layer-workpiece-list > row {
    background-color: transparent;
    border-radius: 4px;
    padding: 1px 4px;
    margin: 0;
    border: none;
}
.layer-workpiece-list > row > * {
    margin: -1px -4px;
    padding: 1px 4px;
}
.layer-workpiece-list > row:drop(active) {
    background-color: transparent;
    outline: none;
}
.layer-workpiece-list > row.drop-above {
    box-shadow: inset 0 2px 0 0 @accent_bg_color;
}
.layer-workpiece-list > row.drop-below {
    box-shadow: inset 0 -2px 0 0 @accent_bg_color;
}
.layer-workpiece-list > row.selected-row {
    background-color: alpha(@accent_bg_color, 0.2);
}
.layer-column.drop-left {
    box-shadow: inset 3px 0 0 0 @accent_bg_color;
}
.layer-column.drop-right {
    box-shadow: inset -3px 0 0 0 @accent_bg_color;
}
"""

_LAYER_UID_PREFIX = "layer:"


class LayerColumn(Gtk.Box):
    dragging = False

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
        self.set_hexpand(False)

        self.doc = doc
        self.layer = layer
        self.editor = editor
        self._row_items = {}
        self._ordered_items: List = []
        self._selected_uids: Set = set()
        self._selection_anchor = None
        self._potential_drop_index = -1
        self._drop_shift_held = False

        self.edit_item_requested = Signal()
        self.select_items_requested = Signal()
        self.move_to_layer_requested = Signal()

        self._build_header(can_delete)
        self._build_workflow_row()
        self._build_workpiece_list()
        self._setup_layer_drag_source()

        self._click_gesture = Gtk.GestureClick()
        self._click_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self._click_gesture.connect("pressed", self._on_column_clicked)
        self._click_gesture.connect("released", self._on_column_released)
        self.add_controller(self._click_gesture)

        self._context_popover: Optional[Gtk.PopoverMenu] = None
        self._row_drag_happened: bool = False

        right_click = Gtk.GestureClick()
        right_click.set_button(Gdk.BUTTON_SECONDARY)
        right_click.connect("pressed", self._on_right_click_pressed)
        self.add_controller(right_click)

        self._connect_signals()
        self._update_style()
        self._update_subtitle()

    def do_measure(self, orientation, for_size):
        min_, nat, min_bl, nat_bl = super().do_measure(orientation, for_size)
        if orientation == Gtk.Orientation.HORIZONTAL:
            nat = min(nat, 400)
        return min_, nat, min_bl, nat_bl

    def _build_header(self, can_delete: bool):
        self.header = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=4
        )
        self.header.add_css_class("layer-column-header")
        self.header.set_hexpand(True)

        self.drag_label = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=4
        )

        self.icon_container = Gtk.Box()
        self.icon_container.set_valign(Gtk.Align.CENTER)
        self.icon_container.set_margin_start(3)
        self.icon_container.set_margin_end(3)
        self.drag_label.append(self.icon_container)

        self.name_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.name_box.set_hexpand(True)
        self.name_box.set_halign(Gtk.Align.START)
        self.name_box.set_valign(Gtk.Align.CENTER)

        name_label = Gtk.Label()
        name_label.set_text(self.layer.name)
        name_label.set_halign(Gtk.Align.START)
        name_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.name_label = name_label
        self.name_box.append(name_label)

        subtitle_label = Gtk.Label()
        subtitle_label.set_halign(Gtk.Align.START)
        subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
        subtitle_label.add_css_class("dim-label")
        self.subtitle_label = subtitle_label
        self.name_box.append(subtitle_label)

        self.drag_label.append(self.name_box)

        self.header.append(self.drag_label)

        self.settings_button = Gtk.Button(child=get_icon("settings-symbolic"))
        self.settings_button.add_css_class("flat")
        self.settings_button.set_tooltip_text(_("Layer Settings"))
        self.settings_button.connect("clicked", self._on_settings_clicked)
        self.header.append(self.settings_button)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.add_css_class("flat")
        self.delete_button.set_tooltip_text(_("Delete this layer"))
        self.delete_button.set_visible(can_delete)
        self.delete_button.connect("clicked", self._on_delete_clicked)
        self.header.append(self.delete_button)

        self.visibility_on_icon = get_icon("visibility-on-symbolic")
        self.visibility_off_icon = get_icon("visibility-off-symbolic")

        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.set_active(self.layer.visible)
        self.visibility_button.set_child(
            self.visibility_on_icon
            if self.layer.visible
            else self.visibility_off_icon
        )
        self.visibility_button.add_css_class("flat")
        self.visibility_button.set_tooltip_text(_("Toggle layer visibility"))
        self.visibility_button.connect("clicked", self._on_visibility_clicked)
        self.header.append(self.visibility_button)

        self.append(self.header)
        self._update_icon()

    def _build_workflow_row(self):
        self.workflow_row = WorkflowRow(self.editor, self.layer)
        self.append(self.workflow_row)

    def _build_workpiece_list(self):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)

        self.listbox = Gtk.ListBox()
        self.listbox.add_css_class("layer-workpiece-list")
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)

        drop_target = Gtk.DropTarget.new(
            GObject.TYPE_STRING,
            Gdk.DragAction.MOVE | Gdk.DragAction.COPY,
        )
        drop_target.connect("accept", self._on_drop_accept)
        drop_target.connect("enter", self._on_drop_enter)
        drop_target.connect("motion", self._on_drop_motion)
        drop_target.connect("drop", self._on_drop)
        drop_target.connect("leave", self._on_drop_leave)
        self.add_controller(drop_target)

        scrolled.set_child(self.listbox)
        self.append(scrolled)

        self._rebuild_workpiece_list()

    def _setup_layer_drag_source(self):
        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_layer_drag_prepare)
        drag_source.connect("drag-begin", self._on_layer_drag_begin)
        drag_source.connect("drag-end", self._on_layer_drag_end)
        drag_source.connect("drag-cancel", self._on_layer_drag_end)
        self.header.add_controller(drag_source)

    @staticmethod
    def _setup_row_drag_source(row, item_row):
        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", item_row._on_drag_prepare)
        row.add_controller(drag_source)

    @staticmethod
    def _on_layer_drag_begin(drag_source, drag):
        LayerColumn.dragging = True

    @staticmethod
    def _on_layer_drag_end(drag_source, drag, delete):
        LayerColumn.dragging = False

    @staticmethod
    def _on_layer_drag_cancel(drag_source, drag, reason):
        LayerColumn.dragging = False

    def _on_layer_drag_prepare(self, drag_source, x, y):
        snapshot = Gtk.Snapshot()
        Gtk.Widget.do_snapshot(self.drag_label, snapshot)
        paintable = snapshot.to_paintable()
        if paintable:
            drag_source.set_icon(paintable, x, y)
        return Gdk.ContentProvider.new_for_value(
            _LAYER_UID_PREFIX + self.layer.uid
        )

    @staticmethod
    def _parse_layer_uid(value):
        if value and value.startswith(_LAYER_UID_PREFIX):
            return value[len(_LAYER_UID_PREFIX) :]
        return None

    @staticmethod
    def _remove_layer_drop_markers_from(widget):
        child = widget.get_first_child()
        while child:
            child.remove_css_class("drop-left")
            child.remove_css_class("drop-right")
            child = child.get_next_sibling()

    def _rebuild_workpiece_list(self):
        child = self.listbox.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self.listbox.remove(child)
            child = next_child
        self._row_items.clear()
        self._ordered_items.clear()

        for item in self.layer.get_content_items():
            row = Gtk.ListBoxRow()
            if isinstance(item, Group):
                item_row = GroupRow(item)
            elif isinstance(item, WorkPiece):
                item_row = WorkpieceRow(item)
            else:
                continue
            row.set_child(item_row)
            self._row_items[row] = item
            self._ordered_items.append(item)
            self._setup_row_drag_source(row, item_row)
            self.listbox.append(row)

    def update_row_selection(self, selected_uids: Set):
        self._selected_uids = {
            uid
            for uid in selected_uids
            if uid in {i.uid for i in self._ordered_items}
        }
        if self._selected_uids:
            for item in reversed(self._ordered_items):
                if item.uid in self._selected_uids:
                    self._selection_anchor = item
                    break
        child = self.listbox.get_first_child()
        while child:
            if isinstance(child, Gtk.ListBoxRow):
                item = self._row_items.get(child)
                if item and item.uid in selected_uids:
                    child.add_css_class("selected-row")
                else:
                    child.remove_css_class("selected-row")
            child = child.get_next_sibling()

    def _update_icon(self):
        if old := self.icon_container.get_first_child():
            self.icon_container.remove(old)
        icon_name = (
            "rotary-symbolic"
            if self.layer.rotary_enabled
            else "layer-symbolic"
        )
        icon = get_icon(icon_name)
        rgba = Gdk.RGBA()
        rgba.parse(self.layer.color)
        dark = Adw.StyleManager.get_default().get_dark()
        alpha = 0.35 if dark else 0.9
        bg = (
            f"rgba({int(rgba.red * 255)},{int(rgba.green * 255)},"
            f"{int(rgba.blue * 255)},{alpha})"
        )
        css_class = f"layer-icon-{self.layer.uid[:8]}"
        icon.set_css_classes([css_class])
        apply_css(
            f".{css_class} "
            f"{{ background: {bg}; border-radius: 4px; "
            f"padding: 4px; }}"
        )
        self.icon_container.append(icon)

    def _update_style(self):
        if self.layer.active:
            self.add_css_class("active-layer-column")
        else:
            self.remove_css_class("active-layer-column")

    def _update_ui(self):
        self.name_label.set_text(self.layer.name)
        self._update_icon()
        self._update_subtitle()
        self.visibility_button.set_active(self.layer.visible)
        if self.layer.visible:
            self.visibility_button.set_child(self.visibility_on_icon)
        else:
            self.visibility_button.set_child(self.visibility_off_icon)

    def _update_subtitle(self):
        machine = get_context().machine
        module_name = None
        if machine and self.layer.rotary_module_uid:
            rm = machine.get_rotary_module_by_uid(self.layer.rotary_module_uid)
            if rm:
                module_name = rm.name
        self.subtitle_label.set_text(self.layer.get_subtitle(module_name))

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
        self._rebuild_workpiece_list()

    def _on_layer_structure_changed(self, sender, **kwargs):
        self.workflow_row.refresh()
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
        self._pressed_clicked_item = None
        picked = self.pick(x, y, Gtk.PickFlags.DEFAULT)
        if picked is not None:
            widget = picked
            while widget and widget is not self:
                if isinstance(widget, Gtk.ListBoxRow):
                    item = self._row_items.get(widget)
                    if item is None:
                        break
                    if n_press == 1:
                        if not self._handle_item_click(item, gesture):
                            return
                    elif n_press == 2 and isinstance(item, WorkPiece):
                        self._on_workpiece_double_clicked(item)
                    gesture.set_state(Gtk.EventSequenceState.DENIED)
                    return
                if isinstance(widget, Gtk.Button):
                    gesture.set_state(Gtk.EventSequenceState.DENIED)
                    return
                widget = widget.get_parent()
        if self.doc.active_layer is not self.layer:
            self.editor.layer.set_active_layer(self.layer)

    def _on_column_released(self, gesture, n_press, x, y):
        if (
            self._pressed_clicked_item
            and self._pressed_clicked_item.uid in self._selected_uids
            and len(self._selected_uids) > 1
        ):
            self._selection_anchor = self._pressed_clicked_item
            self.select_items_requested.send(
                self,
                items=[self._pressed_clicked_item],
                extend=False,
            )
        self._pressed_clicked_item = None

    def _handle_item_click(self, item, gesture) -> bool:
        event = gesture.get_current_event()
        modifiers = (
            event.get_modifier_state() if event else Gdk.ModifierType(0)
        )
        shift = bool(modifiers & Gdk.ModifierType.SHIFT_MASK)
        ctrl = bool(modifiers & Gdk.ModifierType.CONTROL_MASK)

        if shift:
            if (
                not self._selection_anchor
                or self._selection_anchor not in self._ordered_items
            ):
                self._selection_anchor = item
            if item in self._ordered_items:
                anchor_idx = self._ordered_items.index(self._selection_anchor)
                click_idx = self._ordered_items.index(item)
                lo = min(anchor_idx, click_idx)
                hi = max(anchor_idx, click_idx)
                selected = self._ordered_items[lo : hi + 1]
            else:
                selected = [item]
            self.select_items_requested.send(
                self,
                items=selected,
                extend=True,
            )
            return True
        elif ctrl:
            self._selection_anchor = item
            selected = [
                i for i in self._ordered_items if i.uid in self._selected_uids
            ]
            if item in selected:
                selected = [i for i in selected if i is not item]
            else:
                selected.append(item)
            self.select_items_requested.send(
                self,
                items=selected,
                extend=True,
            )
            return True
        elif item.uid not in self._selected_uids:
            self._selection_anchor = item
            self.select_items_requested.send(
                self,
                items=[item],
                extend=False,
            )
            return True
        elif len(self._selected_uids) > 1:
            self._pressed_clicked_item = item
            return False
        return False

    def _on_workpiece_double_clicked(self, wp):
        if not wp.geometry_provider_uid:
            return
        asset = self.doc.get_asset_by_uid(wp.geometry_provider_uid)
        if not asset:
            return
        action_name = type(asset).edit_item_action
        if not action_name:
            return
        self.edit_item_requested.send(self, item=wp, action_name=action_name)

    def _on_right_click_pressed(self, gesture, n_press, x, y):
        widget = self.pick(x, y, Gtk.PickFlags.DEFAULT)
        clicked_item = None
        while widget and widget is not self:
            if isinstance(widget, Gtk.ListBoxRow):
                clicked_item = self._row_items.get(widget)
                break
            widget = widget.get_parent()

        if clicked_item is None:
            self._show_empty_context_menu(gesture)
        else:
            if clicked_item.uid not in self._selected_uids:
                self.select_items_requested.send(
                    self,
                    items=[clicked_item],
                    extend=False,
                )
            self._show_item_context_menu(gesture)

    def _popup_context_menu(self, menu: Gio.Menu, gesture: Gtk.Gesture):
        if self._context_popover:
            self._context_popover.unparent()
        popover = Gtk.PopoverMenu.new_from_model(menu)
        popover.set_parent(self)
        popover.set_has_arrow(False)
        ok, rect = gesture.get_bounding_box()
        if ok:
            popover.set_pointing_to(rect)
        self._context_popover = popover
        popover.popup()

    def _show_empty_context_menu(self, gesture):
        menu = Gio.Menu.new()
        menu.append_item(Gio.MenuItem.new(_("Paste"), "win.paste"))
        self._popup_context_menu(menu, gesture)

    def _show_item_context_menu(self, gesture):
        menu = Gio.Menu.new()
        menu.append_item(Gio.MenuItem.new(_("Duplicate"), "win.duplicate"))
        menu.append_section(None, Gio.Menu.new())
        menu.append_item(Gio.MenuItem.new(_("Copy"), "win.copy"))
        menu.append_item(Gio.MenuItem.new(_("Cut"), "win.cut"))
        menu.append_section(None, Gio.Menu.new())
        menu.append_item(Gio.MenuItem.new(_("Delete"), "win.remove"))
        self._popup_context_menu(menu, gesture)

    def _on_drop(self, drop_target, value, x, y):
        if not value:
            logger.debug("Drop: rejected, empty value")
            return False

        asset_uids = self._parse_asset_uids(value)
        if asset_uids is not None:
            self._remove_drop_markers()
            return self._handle_asset_drop(asset_uids)

        dragged_item = self._find_item_by_uid(value)
        if not dragged_item:
            self._remove_drop_markers()
            logger.debug("Drop: rejected, item not found uid=%r", value[:8])
            return False

        drop_index = self._potential_drop_index
        self._remove_drop_markers()
        self._drop_shift_held = False

        if dragged_item.uid in self._selected_uids:
            items_to_move = [
                i for i in self._ordered_items if i.uid in self._selected_uids
            ]
        else:
            items_to_move = [dragged_item]

        if not items_to_move:
            return False

        source_layer = cast(WorkPiece, items_to_move[0]).layer
        if source_layer is self.layer:
            return self._handle_reorder_drop(items_to_move, drop_index)

        self.move_to_layer_requested.send(
            self, items=items_to_move, target_layer=self.layer
        )
        return True

    def _handle_reorder_drop(self, items, drop_index):
        current_items = list(self.layer.get_content_items())
        if drop_index == -1:
            drop_index = len(current_items)

        new_order = [i for i in current_items if i not in items]
        for i in reversed(items):
            if drop_index >= len(new_order):
                new_order.append(i)
            else:
                new_order.insert(drop_index, i)
        self.editor.layer.reorder_content_items(self.layer, new_order)
        return True

    def _remove_drop_markers(self):
        child = self.listbox.get_first_child()
        while child:
            child.remove_css_class("drop-above")
            child.remove_css_class("drop-below")
            child = child.get_next_sibling()
        self._potential_drop_index = -1

    def _parse_asset_uids(self, value: str):
        try:
            uids = json.loads(value)
            if isinstance(uids, list) and len(uids) > 0:
                return uids
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _handle_asset_drop(self, asset_uids: list) -> bool:
        for asset_uid in asset_uids:
            asset = self.doc.get_asset_by_uid(asset_uid)
            if isinstance(asset, StockAsset):
                return False
            if not asset or not asset.is_draggable_to_canvas:
                continue

        success = False
        for asset_uid in asset_uids:
            asset = self.doc.get_asset_by_uid(asset_uid)
            if not asset or not asset.is_draggable_to_canvas:
                continue
            pos = self._get_center_position()
            try:
                if isinstance(asset, SourceAsset):
                    self._create_source_instance(asset, pos)
                    success = True
                else:
                    self.editor.edit.add_geometry_provider_instance(
                        asset_uid, pos, target_layer=self.layer
                    )
                    success = True
            except Exception:
                logger.exception(
                    "Error creating instance from asset %s", asset_uid[:8]
                )
        return success

    def _create_source_instance(
        self,
        asset: SourceAsset,
        pos: tuple,
    ):
        win = self.get_ancestor(Gtk.Window)
        if not win:
            return
        import_handler.start_reimport(
            cast("MainWindow", win),
            self.editor,
            asset,
            pos,
            target_layer=self.layer,
        )

    def _get_center_position(self) -> tuple:
        machine = get_context().machine
        if machine:
            work_area = machine.work_area
            wa_w, wa_h = work_area[2], work_area[3]
            origin_x, origin_y = machine.get_reference_position_world()
            space = machine.get_coordinate_space()
            bl_x, bl_y = space.world_position_from_origin(
                origin_x, origin_y, (wa_w, wa_h)
            )
            return (bl_x + wa_w / 2, bl_y + wa_h / 2)
        return (50.0, 50.0)

    def _on_drop_accept(self, drop_target, drop):
        formats = drop.get_formats() if drop else None
        logger.debug(
            "Accept(%s): formats=%s",
            self.layer.name,
            formats.to_string() if formats else None,
        )
        return True

    def _on_drop_enter(self, drop_target, x, y):
        logger.debug("Enter(%s): x=%d y=%d", self.layer.name, x, y)
        return Gdk.DragAction.MOVE

    def _on_drop_motion(self, drop_target, x, y):
        if LayerColumn.dragging:
            return 0

        drop = drop_target.get_drop()
        if drop and drop.get_actions() & Gdk.DragAction.COPY:
            self._remove_drop_markers()
            return Gdk.DragAction.COPY

        self._remove_drop_markers()

        fallback_index = len(self.layer.get_content_items())
        coords = self.translate_coordinates(self.listbox, x, y)
        if not coords:
            logger.debug(
                "Motion(%s): translate_coordinates returned None, "
                "x=%d y=%d fallback=%d",
                self.layer.name,
                x,
                y,
                fallback_index,
            )
            self._potential_drop_index = fallback_index
            return Gdk.DragAction.MOVE

        lb_x, lb_y = coords
        target_row = self._find_row_at(lb_x, lb_y)
        if not target_row:
            logger.debug(
                "Motion(%s): no row at (%d, %d), fallback=%d",
                self.layer.name,
                lb_x,
                lb_y,
                fallback_index,
            )
            self._potential_drop_index = fallback_index
            return Gdk.DragAction.MOVE

        row_alloc = target_row.get_allocation()
        row_center = row_alloc.y + row_alloc.height / 2

        if lb_y < row_center:
            target_row.add_css_class("drop-above")
            self._potential_drop_index = target_row.get_index()
        else:
            target_row.add_css_class("drop-below")
            self._potential_drop_index = target_row.get_index() + 1

        return Gdk.DragAction.MOVE

    def _on_drop_leave(self, drop_target):
        logger.debug("Leave(%s)", self.layer.name)
        self._remove_drop_markers()

    def _find_row_at(self, x, y):
        picked = self.listbox.pick(x, y, Gtk.PickFlags.DEFAULT)
        while picked:
            if isinstance(picked, Gtk.ListBoxRow):
                return picked
            picked = picked.get_parent()
        return None

    def _find_item_by_uid(self, uid: str) -> Optional[DocItem]:
        for layer in self.doc.layers:
            for item in layer.get_content_items():
                if item.uid == uid:
                    return item
            for wp in layer.all_workpieces:
                if wp.uid == uid:
                    return wp
        return None
