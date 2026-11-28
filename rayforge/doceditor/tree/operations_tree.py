"""The Operations tab: the workflows in run order."""

from blinker import Signal
from gi.repository import Gtk

from ...core.doc import Doc
from .add_workflow_button import WorkflowBar
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
        self.workflow_requested = Signal()
        self.workflow_activated = Signal()

        self._doc: Doc | None = None
        self._expanded: set = set()
        self._selected: set = set()
        self._tree = TreeWidget()
        self._bar = WorkflowBar()
        self._bar.workflow_requested.connect(self._on_workflow_requested)
        self._bar.workflow_activated.connect(self._on_workflow_activated)
        self.append(self._tree)

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
        doc.updated.connect(self._on_doc_changed)

    def _disconnect(self, doc: Doc):
        doc.descendant_added.disconnect(self._on_doc_changed)
        doc.descendant_removed.disconnect(self._on_doc_changed)
        doc.descendant_updated.disconnect(self._on_doc_changed)
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
        workflows = [
            workflow
            for setup in self._doc.setups
            for workflow in setup.workflows
        ]
        for workflow in workflows:
            rows.extend(
                WorkflowNode.build_rows(
                    workflow, 0, self._expanded, self._selected, self
                )
            )
        self._tree.show(rows, tail=self._bar)
        self._bar.set_workflows(workflows)

    def on_row_activated(self, spec):
        self._selected = {spec.uid}
        self.refresh()
        self.selection_changed.send(self, spec=spec)

    def on_twisty_toggled(self, spec):
        if spec.uid in self._expanded:
            self._expanded.remove(spec.uid)
        else:
            self._expanded.add(spec.uid)
        self.refresh()

    def _on_workflow_requested(self, bar, color):
        self.workflow_requested.send(self, color=color)

    def _on_workflow_activated(self, bar, workflow):
        self.workflow_activated.send(self, workflow=workflow)
