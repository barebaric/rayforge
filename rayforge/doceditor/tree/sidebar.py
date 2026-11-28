"""The Parts/Operations sidebar shown next to the canvas."""

from __future__ import annotations

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

    def __init__(self, editor: DocEditor):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.editor = editor
        self.item_selected = Signal()

        self._doc: Doc | None = None
        self._layer_signature: tuple[tuple[str, str], ...] | None = None
        self._syncing_layers = False
        self._layer_settings_dialog: LayerSettingsDialog | None = None

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_margin_start(8)
        header.set_margin_end(8)
        header.set_margin_top(8)
        label = Gtk.Label(label=_("Layer"), xalign=0)
        self._layer_dropdown = Gtk.DropDown()
        self._layer_dropdown.set_hexpand(True)
        self._layer_dropdown.set_tooltip_text(_("Active layer"))
        self._layer_dropdown.connect(
            "notify::selected", self._on_layer_selected
        )
        header.append(label)
        header.append(self._layer_dropdown)
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
        self.operations_tree.workflow_activated.connect(
            self._on_workflow_activated
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
        if doc is None:
            self._layer_signature = None
            self._syncing_layers = True
            self._layer_dropdown.set_model(Gtk.StringList.new([]))
            self._syncing_layers = False
            self._layer_dropdown.set_sensitive(False)
            self.parts_tree.set_layer(None)
            return
        layers = doc.layers
        signature = tuple((layer.uid, layer.name) for layer in layers)
        if signature != self._layer_signature:
            self._layer_signature = signature
            self._syncing_layers = True
            self._layer_dropdown.set_model(
                Gtk.StringList.new([layer.name for layer in layers])
            )
            self._syncing_layers = False
        self._layer_dropdown.set_sensitive(bool(layers))
        if not layers:
            self.parts_tree.set_layer(None)
            return
        active = doc.active_layer
        index = next(
            (i for i, layer in enumerate(layers) if layer is active), 0
        )
        self._syncing_layers = True
        self._layer_dropdown.set_selected(index)
        self._syncing_layers = False
        self.parts_tree.set_layer(active)

    def _on_layer_selected(self, dropdown, _pspec):
        if self._syncing_layers or self._doc is None:
            return
        index = dropdown.get_selected()
        layers = self._doc.layers
        if 0 <= index < len(layers):
            self._close_layer_settings()
            self.editor.layer.set_active_layer(layers[index])

    def _on_layer_settings_clicked(self, button):
        if self._doc is None or not self._doc.layers:
            return
        if self._layer_settings_dialog is not None:
            self._layer_settings_dialog.present()
            return
        root = self.get_root()
        self._layer_settings_dialog = LayerSettingsDialog(
            layer=self._doc.active_layer,
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

    def _on_workflow_activated(self, sender, workflow: Workflow):
        self.select_item(workflow)

    @staticmethod
    def _layer_of(item: DocItem) -> Layer | None:
        current = item
        while current is not None:
            if isinstance(current, Layer):
                return current
            current = current.parent
        return None
