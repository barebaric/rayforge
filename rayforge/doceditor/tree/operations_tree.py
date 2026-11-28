"""The Operations tab: the workflows in run order."""

from collections.abc import Callable
from gettext import gettext as _

from blinker import Signal
from gi.repository import Gdk, Gtk

from ...core.color import COLOR_PALETTE
from ...core.doc import Doc
from .add_workflow_button import WorkflowBar
from .layer_separator_node import LayerSeparatorNode
from .node import show_row_menu
from .tree import TreeWidget
from .workflow_node import WorkflowNode


class OperationsTree(Gtk.Box):
    """Shows the document's workflows and the parts they operate on.

    Like :class:`PartsTree`, the tree owns expansion and selection and
    re-derives the rows from the document on every change. Picking a
    color for a new workflow is reported via :attr:`workflow_requested`;
    picking an existing workflow's swatch via
    :attr:`workflow_activated`. Creating the workflow is the caller's
    job.
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.selection_changed = Signal()
        self.rename_requested = Signal()
        self.workflow_requested = Signal()
        self.enabled_toggled = Signal()
        self.workflow_color_requested = Signal()
        self.workflow_delete_requested = Signal()
        self.workflow_parts_changed = Signal()
        self.workflow_reorder_requested = Signal()

        self._doc: Doc | None = None
        self._expanded: set = set()
        self._selected: set = set()
        self._tree = TreeWidget()
        self._bar = WorkflowBar()
        self._bar.workflow_requested.connect(self._on_workflow_requested)
        self.append(self._tree)

    def set_selected_items(self, items):
        """Highlights the given model items."""
        uids = {item.uid for item in items}
        if uids == self._selected:
            return
        self._selected = uids
        self.refresh()

    def set_doc(self, doc: Doc | None):
        if self._doc is doc:
            return
        if self._doc is not None:
            self._disconnect(self._doc)
        self._doc = doc
        if doc is not None:
            self._connect(doc)
        self._selected.clear()
        self.refresh()

    def do_destroy(self):
        if self._doc is not None:
            self._disconnect(self._doc)
            self._doc = None

    def _connect(self, doc: Doc):
        doc.descendant_added.connect(self._on_doc_changed)
        doc.descendant_removed.connect(self._on_doc_changed)
        doc.descendant_updated.connect(self._on_doc_changed)
        doc.descendant_renamed.connect(self._on_doc_changed)
        doc.updated.connect(self._on_doc_changed)

    def _disconnect(self, doc: Doc):
        doc.descendant_added.disconnect(self._on_doc_changed)
        doc.descendant_removed.disconnect(self._on_doc_changed)
        doc.descendant_updated.disconnect(self._on_doc_changed)
        doc.descendant_renamed.disconnect(self._on_doc_changed)
        doc.updated.disconnect(self._on_doc_changed)

    def _on_doc_changed(self, sender, **kwargs):
        self.refresh()

    def expand(self, uid: str):
        """Expands the row with the given uid and refreshes."""
        self._expanded.add(uid)
        self.refresh()

    def refresh(self):
        if self._doc is None:
            self._tree.show([])
            return
        rows: list[Gtk.Widget] = []
        for layer in self._doc.layers:
            rows.append(LayerSeparatorNode(layer, self))
            for workflow in layer.workflows:
                rows.extend(
                    WorkflowNode.build_rows(
                        workflow, 0, self._expanded, self._selected, self
                    )
                )
        self._tree.show(rows, tail=self._bar)
        self._install_drag_and_drop(rows)

    def on_row_activated(self, spec):
        self._selected = {spec.uid}
        self.refresh()
        self.selection_changed.send(self, spec=spec)

    def on_row_renamed(self, spec, name: str):
        self.rename_requested.send(self, uid=spec.uid, name=name)

    def on_eye_toggled(self, spec):
        workflow = self._workflow_by_uid(spec.uid)
        if workflow is not None:
            self.enabled_toggled.send(
                self, workflow=workflow, enabled=not spec.on
            )

    def on_row_context_menu(self, spec, row, x, y):
        if spec.kind == "workflow":
            self._show_workflow_menu(spec, row, x, y)
            return
        from .part_node import PartNode

        if isinstance(row, PartNode):
            self._show_part_menu(spec, row, x, y)

    def _show_workflow_menu(self, spec, row, x, y):
        workflow = self._workflow_by_uid(spec.uid)
        if workflow is None:
            return
        show_row_menu(
            row,
            [
                (_("Rename"), lambda: self.start_rename(spec.uid)),
                (
                    _("Delete"),
                    lambda: self.workflow_delete_requested.send(
                        self, workflow=workflow
                    ),
                ),
            ],
            x=x,
            y=y,
            colors=COLOR_PALETTE,
            on_color=lambda color: self.workflow_color_requested.send(
                self, workflow=workflow, color=color
            ),
        )

    def _show_part_menu(self, spec, row, x, y):
        workflow = row.workflow
        if workflow is None:
            return
        parts = workflow.workpieces
        remaining = [wp for wp in parts if wp.uid != spec.uid]
        items: list[tuple[str, Callable[[], object]]] = [
            (_("Rename"), lambda: self.start_rename(spec.uid))
        ]
        if remaining:
            if workflow.workpiece_uids:
                uids = [
                    uid for uid in workflow.workpiece_uids if uid != spec.uid
                ]
            else:
                uids = [wp.uid for wp in remaining]
            items.append(
                (
                    _("Remove from Workflow"),
                    lambda: self.workflow_parts_changed.send(
                        self, workflow=workflow, workpiece_uids=uids
                    ),
                )
            )
        if workflow.workpiece_uids:
            items.append(
                (
                    _("Run for All Parts"),
                    lambda: self.workflow_parts_changed.send(
                        self, workflow=workflow, workpiece_uids=[]
                    ),
                )
            )
        show_row_menu(row, items, x=x, y=y)

    def _workflow_by_uid(self, uid: str):
        if self._doc is None:
            return None
        for layer in self._doc.layers:
            for workflow in layer.workflows:
                if workflow.uid == uid:
                    return workflow
        return None

    # ── Drag and drop ─────────────────────────────────────────────

    def _install_drag_and_drop(self, rows):
        from .part_node import PartNode

        for row in rows:
            if isinstance(row, WorkflowNode):
                self._make_workflow_draggable(row)
            elif isinstance(row, PartNode):
                self._make_part_draggable(row)

    def _make_workflow_draggable(self, row: WorkflowNode):
        source = Gtk.DragSource(actions=Gdk.DragAction.MOVE)
        source.connect("prepare", self._on_drag_prepare, row)
        source.connect("drag-end", self._on_drag_end)
        row.add_controller(source)

        target = Gtk.DropTarget.new(Gtk.ListBoxRow, Gdk.DragAction.MOVE)
        target.connect("motion", self._on_workflow_motion, row)
        target.connect("leave", self._on_drag_leave, row)
        target.connect("drop", self._on_workflow_drop, row)
        row.add_controller(target)

    def _make_part_draggable(self, row):
        source = Gtk.DragSource(actions=Gdk.DragAction.MOVE)
        source.connect("prepare", self._on_drag_prepare, row)
        source.connect("drag-end", self._on_drag_end)
        row.add_controller(source)

        target = Gtk.DropTarget.new(Gtk.ListBoxRow, Gdk.DragAction.MOVE)
        target.connect("motion", self._on_part_motion, row)
        target.connect("leave", self._on_drag_leave, row)
        target.connect("drop", self._on_part_drop, row)
        row.add_controller(target)

    def _on_drag_prepare(self, source, x, y, row):
        return Gdk.ContentProvider.new_for_value(row)

    def _on_drag_end(self, source, drag, delete_data):
        self._clear_drop_markers()

    def _clear_drop_markers(self):
        for row in self._tree.rows():
            row.remove_css_class("drop-above")
            row.remove_css_class("drop-below")

    def _mark_drop(self, row, above: bool):
        self._clear_drop_markers()
        row.add_css_class("drop-above" if above else "drop-below")

    def _on_drag_leave(self, target, row):
        self._clear_drop_markers()

    def _on_workflow_motion(self, target, x, y, row):
        self._mark_drop(row, y < row.get_height() / 2)
        return Gdk.DragAction.MOVE

    def _on_part_motion(self, target, x, y, row):
        self._mark_drop(row, y < row.get_height() / 2)
        return Gdk.DragAction.MOVE

    def _on_workflow_drop(self, target, value, x, y, row):
        self._clear_drop_markers()
        if not isinstance(value, WorkflowNode):
            return False
        source_workflow = self._workflow_by_uid(value.spec.uid)
        target_workflow = self._workflow_by_uid(row.spec.uid)
        if source_workflow is None or target_workflow is None:
            return False
        layer = source_workflow.layer
        if layer is None or target_workflow.layer is not layer:
            return False
        order = list(layer.workflows)
        if source_workflow not in order:
            return False
        order.remove(source_workflow)
        target_index = order.index(target_workflow)
        if y >= row.get_height() / 2:
            target_index += 1
        order.insert(target_index, source_workflow)
        self.workflow_reorder_requested.send(
            self, layer=layer, workflows=order
        )
        return True

    def _on_part_drop(self, target, value, x, y, row):
        from .part_node import PartNode

        self._clear_drop_markers()
        if not isinstance(value, PartNode) or not isinstance(row, PartNode):
            return False
        workflow = row.workflow
        if workflow is None or value.workflow is not workflow:
            return False
        layer = workflow.layer
        if layer is None:
            return False
        parts = list(workflow.workpieces)
        source_uid = value.spec.uid
        target_uid = row.spec.uid
        if source_uid == target_uid:
            return False
        source = next((wp for wp in parts if wp.uid == source_uid), None)
        if source is None:
            return False
        parts.remove(source)
        target_index = next(
            i for i, wp in enumerate(parts) if wp.uid == target_uid
        )
        if y >= row.get_height() / 2:
            target_index += 1
        parts.insert(target_index, source)
        uids = [wp.uid for wp in parts]
        self.workflow_parts_changed.send(
            self, workflow=workflow, workpiece_uids=uids
        )
        return True

    def rows(self):
        """Yields the visible node rows."""
        return self._tree.rows()

    def find_row(self, uid: str):
        """Returns the row with the given uid, or None."""
        return self._tree.find_row(uid)

    def start_rename(self, uid: str) -> bool:
        """Starts inline renaming of the row with the given uid."""
        return self._tree.start_rename(uid)

    def on_twisty_toggled(self, spec):
        if spec.uid in self._expanded:
            self._expanded.remove(spec.uid)
        else:
            self._expanded.add(spec.uid)
        self.refresh()

    def _on_workflow_requested(self, bar, color):
        self.workflow_requested.send(self, color=color)
