"""The Parts/Operations sidebar shown next to the canvas."""

from __future__ import annotations

from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING

from blinker import Signal
from gi.repository import Gtk

from ...core.doc import Doc
from ...core.item import DocItem
from ...core.layer import Layer
from ...core.step import Step
from ...core.stock import StockItem
from ...core.workflow import Workflow
from ...ui_gtk.doceditor.layer_settings_dialog import (
    LayerSettingsDialog,
)
from ...ui_gtk.icons import get_icon
from ...ui_gtk.shared.gtk import apply_css
from .operations_tree import OperationsTree
from .parts_tree import PartsTree

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


class Sidebar(Gtk.Box):
    """The document sidebar with Parts and Operations tabs.

    The layer selector at the top drives the active layer; the Parts
    tab shows that layer's stock, groups and parts, while the
    Operations tab shows every workflow in run order. Model edits go
    through the editor's undoable commands; the current selection is
    reported via :attr:`item_selected`.
    """

    _CSS = """
    .rf-layer-button {
        min-width: 34px;
        min-height: 34px;
        padding: 0;
        border-radius: 6px;
    }
    """

    def __init__(self, editor: DocEditor):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        apply_css(self._CSS)
        self.editor = editor
        self.item_selected = Signal()

        self._doc: Doc | None = None
        self._layer_signature: tuple[tuple[str, str], ...] | None = None
        self._syncing_layers = False
        self._layer_settings_dialog: LayerSettingsDialog | None = None
        self._layer_buttons_by_layer: list[tuple[Gtk.ToggleButton, Layer]] = []

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_margin_start(8)
        header.set_margin_end(8)
        header.set_margin_top(8)
        self._layer_buttons = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=4
        )
        self._layer_buttons.set_hexpand(True)
        header.append(self._layer_buttons)
        self._layer_settings_button = Gtk.Button(
            child=get_icon("settings-symbolic")
        )
        self._layer_settings_button.add_css_class("flat")
        self._layer_settings_button.set_tooltip_text(_("Layer Settings"))
        self._layer_settings_button.connect(
            "clicked", self._on_layer_settings_clicked
        )
        header.append(self._layer_settings_button)
        self.append(header)

        self._stack = Gtk.Stack()
        self._stack.set_vexpand(True)
        switcher = Gtk.StackSwitcher()
        switcher.set_stack(self._stack)
        switcher.set_halign(Gtk.Align.CENTER)
        switcher.set_margin_top(4)
        switcher.set_margin_bottom(4)
        self.append(switcher)

        self.parts_tree = PartsTree()
        self.operations_tree = OperationsTree()
        self._stack.add_titled(self.parts_tree, "parts", _("Parts"))
        self._stack.add_titled(
            self.operations_tree, "operations", _("Operations")
        )
        self.append(self._stack)

        self.parts_tree.selection_changed.connect(self._on_selection_changed)
        self.operations_tree.selection_changed.connect(
            self._on_selection_changed
        )
        self.parts_tree.rename_requested.connect(self._on_rename_requested)
        self.operations_tree.rename_requested.connect(
            self._on_rename_requested
        )
        self.parts_tree.visibility_toggled.connect(self._on_visibility_toggled)
        self.operations_tree.workflow_requested.connect(
            self._on_workflow_requested
        )
        self.operations_tree.enabled_toggled.connect(
            self._on_workflow_enabled_toggled
        )
        self.operations_tree.workflow_color_requested.connect(
            self._on_workflow_color_requested
        )
        self.operations_tree.workflow_delete_requested.connect(
            self._on_workflow_delete_requested
        )
        self.operations_tree.workflow_parts_changed.connect(
            self._on_workflow_parts_changed
        )
        self.operations_tree.workflow_reorder_requested.connect(
            self._on_workflow_reorder_requested
        )

        self.set_size_request(220, -1)

    # ── Document ──────────────────────────────────────────────────

    def set_doc(self, doc: Doc | None):
        if self._doc is doc:
            return
        if self._doc is not None:
            self._disconnect(self._doc)
        self._doc = doc
        self.parts_tree.set_doc(doc)
        self.operations_tree.set_doc(doc)
        if doc is not None:
            doc.descendant_added.connect(self._on_layers_changed)
            doc.descendant_removed.connect(self._on_layers_changed)
            doc.descendant_renamed.connect(self._on_layers_changed)
            doc.active_layer_changed.connect(self._on_active_layer_changed)
        self._layer_signature = None
        self._sync_layers()

    def _disconnect(self, doc: Doc):
        doc.descendant_added.disconnect(self._on_layers_changed)
        doc.descendant_removed.disconnect(self._on_layers_changed)
        doc.descendant_renamed.disconnect(self._on_layers_changed)
        doc.active_layer_changed.disconnect(self._on_active_layer_changed)

    def do_destroy(self):
        self._close_layer_settings()
        if self._doc is not None:
            self._disconnect(self._doc)
            self._doc = None

    def refresh(self):
        """Rebuilds both trees and the layer list from the document."""
        self.parts_tree.refresh()
        self.operations_tree.refresh()
        self._sync_layers()

    # ── Layer selector ────────────────────────────────────────────

    def _on_layers_changed(self, sender, **kwargs):
        self._sync_layers()

    def _on_active_layer_changed(self, sender):
        self._close_layer_settings()
        self._sync_layers()

    def _sync_layers(self):
        doc = self._doc
        layers = doc.layers if doc is not None else []
        signature = tuple((layer.uid, layer.name) for layer in layers)
        if signature != self._layer_signature:
            self._layer_signature = signature
            self._rebuild_layer_buttons(layers)
        self._syncing_layers = True
        for button, layer in self._layer_buttons_by_layer:
            button.set_active(layer is doc.active_layer if doc else False)
        self._syncing_layers = False
        self.parts_tree.set_layer(
            doc.active_layer if doc is not None and doc.layers else None
        )

    def _rebuild_layer_buttons(self, layers):
        child = self._layer_buttons.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._layer_buttons.remove(child)
            child = next_child
        self._layer_buttons_by_layer = []
        for index, layer in enumerate(layers):
            button = Gtk.ToggleButton(label=str(index + 1))
            button.add_css_class("rf-layer-button")
            button.set_tooltip_text(layer.name)
            button.connect("toggled", self._on_layer_button_toggled, layer)
            self._layer_buttons.append(button)
            self._layer_buttons_by_layer.append((button, layer))

        add_button = Gtk.Button(child=get_icon("add-symbolic"))
        add_button.add_css_class("flat")
        add_button.set_tooltip_text(_("Add Layer"))
        add_button.connect("clicked", self._on_add_layer_clicked)
        self._layer_buttons.append(add_button)

    def _on_layer_button_toggled(self, button, layer):
        if self._syncing_layers:
            return
        if not button.get_active():
            # The active layer button must stay pressed.
            self._syncing_layers = True
            button.set_active(
                self._doc is not None and layer is self._doc.active_layer
            )
            self._syncing_layers = False
            return
        self._close_layer_settings()
        self.editor.layer.set_active_layer(layer)

    def _on_layer_button_context(self, gesture, n_press, x, y, button, layer):
        from .node import show_row_menu

        items: list[tuple[str, Callable[[], object]]] = [
            (
                _("Layer Settings\u2026"),
                lambda: self._open_layer_settings(layer),
            )
        ]
        if self._doc is not None and len(self._doc.layers) > 1:
            items.append(
                (
                    _("Delete Layer"),
                    lambda: self.editor.layer.delete_layer(layer),
                )
            )
        show_row_menu(button, items, x=x, y=y)

    def _on_add_layer_clicked(self, button):
        if self._doc is None:
            return
        self._close_layer_settings()
        self.editor.layer.add_layer_and_set_active()

    def _on_layer_settings_clicked(self, button):
        if self._doc is None or not self._doc.layers:
            return
        self._open_layer_settings(self._doc.active_layer)

    def _open_layer_settings(self, layer):
        if self._layer_settings_dialog is not None:
            self._layer_settings_dialog.present()
            return
        root = self.get_root()
        self._layer_settings_dialog = LayerSettingsDialog(
            layer=layer,
            transient_for=root if isinstance(root, Gtk.Window) else None,
            editor=self.editor,
        )
        self._layer_settings_dialog.connect(
            "destroy", self._on_layer_settings_destroyed
        )
        self._layer_settings_dialog.present()

    def _on_layer_settings_destroyed(self, *_args):
        self._layer_settings_dialog = None

    def _close_layer_settings(self):
        if self._layer_settings_dialog is not None:
            self._layer_settings_dialog.close()
            self._layer_settings_dialog = None

    # ── Selection ─────────────────────────────────────────────────

    def set_selected_items(self, items: list[DocItem]):
        """Highlights the given items in both trees."""
        self.parts_tree.set_selected_items(items)
        self.operations_tree.set_selected_items(items)

    def select_item(self, item: DocItem):
        """Activates the item's layer, selects it and reports it."""
        layer = self._layer_of(item)
        if layer is not None and self._doc is not None:
            self.editor.layer.set_active_layer(layer)
        if isinstance(item, Workflow):
            self.show_page("operations")
            self.operations_tree.expand(item.uid)
        elif isinstance(item, Step) and isinstance(item.parent, Workflow):
            self.operations_tree.expand(item.parent.uid)
        self.set_selected_items([item])
        self.item_selected.send(self, item=item)

    def start_rename(self, uid: str) -> bool:
        """Starts inline renaming of a row in either tab."""
        return self.parts_tree.start_rename(
            uid
        ) or self.operations_tree.start_rename(uid)

    def show_page(self, name: str):
        """Switches the sidebar to the "parts" or "operations" tab."""
        self._stack.set_visible_child_name(name)

    # ── Tree callbacks ────────────────────────────────────────────

    def _on_selection_changed(self, sender, spec):
        if self._doc is None:
            return
        item = self._doc.find_descendant_by_uid(spec.uid)
        if item is None:
            return
        layer = self._layer_of(item)
        if (
            layer is not None
            and layer is not self._doc.active_layer
            and layer in self._doc.layers
        ):
            self.editor.layer.set_active_layer(layer)
        self.item_selected.send(self, item=item)

    def _on_rename_requested(self, sender, uid: str, name: str):
        if self._doc is None:
            return
        item = self._doc.find_descendant_by_uid(uid)
        if item is not None:
            self.editor.edit.rename_item(item, name)

    def _on_visibility_toggled(self, sender, item, on: bool):
        if isinstance(item, StockItem):
            self.editor.stock.toggle_stock_visibility(item)
        else:
            self.editor.edit.set_item_visibility(item, on)

    def _on_workflow_requested(self, sender, color: str):
        if self._doc is None or not self._doc.layers:
            return
        layer = self._doc.active_layer
        workflow = Workflow(
            _("Workflow {number}").format(number=len(layer.workflows) + 1)
        )
        workflow.set_color(color)
        self.editor.workflow.add_workflow(layer, workflow)
        self.select_item(workflow)

    def _on_workflow_enabled_toggled(
        self, sender, workflow: Workflow, enabled: bool
    ):
        self.editor.workflow.set_workflow_enabled(workflow, enabled)

    def _on_workflow_color_requested(
        self, sender, workflow: Workflow, color: str
    ):
        self.editor.workflow.set_workflow_color(workflow, color)

    def _on_workflow_parts_changed(
        self, sender, workflow: Workflow, workpiece_uids: list[str]
    ):
        self.editor.workflow.set_workflow_workpieces(workflow, workpiece_uids)

    def _on_workflow_reorder_requested(
        self, sender, layer: Layer, workflows: list[Workflow]
    ):
        self.editor.workflow.reorder_workflows(layer, workflows)

    def _on_workflow_delete_requested(self, sender, workflow: Workflow):
        self.editor.workflow.delete_workflow(workflow)
        self.set_selected_items([])
        self.item_selected.send(self, item=None)

    @staticmethod
    def _layer_of(item: DocItem) -> Layer | None:
        current = item
        while current is not None:
            if isinstance(current, Layer):
                return current
            current = current.parent
        return None
