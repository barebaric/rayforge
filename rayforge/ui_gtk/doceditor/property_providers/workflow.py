"""Property provider for a selected workflow."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk

from ....core.item import DocItem
from ....core.workflow import Workflow
from ...shared.gtk import apply_css
from ..workflow_view import WorkflowView
from .base import PropertyProvider

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

_CSS = """
.rf-part-chip {
    border-radius: 999px;
    padding: 2px 10px;
}
"""


class WorkflowPropertyProvider(PropertyProvider):
    """Edits a workflow's name, steps, and the parts it runs for."""

    priority = 10

    def can_handle(self, items: list[DocItem]) -> bool:
        return len(items) == 1 and isinstance(items[0], Workflow)

    def create_widgets(self) -> list[Gtk.Widget]:
        apply_css(_CSS)
        self.name_row = Adw.EntryRow(title=_("Workflow Name"))
        self.name_row.set_show_apply_button(False)
        self.name_row.connect("apply", self._on_name_applied)

        self.parts_label = Gtk.Label(
            label=_("Parts this workflow runs for"), xalign=0
        )
        self.parts_label.add_css_class("dim-label")
        self.parts_label.set_margin_top(6)

        self.parts_box = Gtk.FlowBox()
        self.parts_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.parts_box.set_margin_top(4)
        self.parts_box.set_margin_bottom(6)
        self.parts_box.set_row_spacing(4)
        self.parts_box.set_column_spacing(4)

        parts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        parts_box.set_margin_start(12)
        parts_box.set_margin_end(12)
        parts_box.append(self.parts_label)
        parts_box.append(self.parts_box)
        self.parts_row = Gtk.ListBoxRow()
        self.parts_row.set_child(parts_box)

        self.workflow_view_holder = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL
        )
        self._workflow_view: WorkflowView | None = None
        self._workflow_view_workflow: Workflow | None = None

        return [
            self.name_row,
            self.parts_row,
            self.workflow_view_holder,
        ]

    def update_widgets(self, editor: "DocEditor", items: list[DocItem]):
        self.editor = editor
        self.items = items
        workflow = items[0]
        assert isinstance(workflow, Workflow)
        self.workflow = workflow
        self._in_update = True
        try:
            if self.name_row.get_text() != workflow.name:
                self.name_row.set_text(workflow.name)
            self._rebuild_part_chips(workflow)
            self._sync_workflow_view(editor, workflow)
        finally:
            self._in_update = False

    def _sync_workflow_view(self, editor: "DocEditor", workflow: Workflow):
        """Shows the standard workflow panel for the selected workflow."""
        if self._workflow_view is None:
            self._workflow_view = WorkflowView(editor, workflow)
            self.workflow_view_holder.append(self._workflow_view)
            self._workflow_view_workflow = workflow
            return
        if workflow is not self._workflow_view_workflow:
            self._workflow_view.set_workflow(workflow)
            self._workflow_view_workflow = workflow

    def _rebuild_part_chips(self, workflow: Workflow):
        child = self.parts_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self.parts_box.remove(child)
            child = next_child

        layer = workflow.layer
        if layer is None:
            self.parts_row.set_visible(False)
            return

        workpieces = layer.all_workpieces
        selected = workflow.selected_workpieces(workpieces)
        selected_uids = {workpiece.uid for workpiece in selected}
        self.parts_row.set_visible(bool(workpieces))
        for workpiece in workpieces:
            button = Gtk.ToggleButton(label=workpiece.name)
            button.add_css_class("rf-part-chip")
            button.set_active(workpiece.uid in selected_uids)
            button.connect(
                "toggled", self._on_part_toggled, workflow, workpiece
            )
            self.parts_box.append(button)

    def _on_name_applied(self, row: Adw.EntryRow):
        if self._in_update:
            return
        name = row.get_text().strip()
        if not name or name == self.workflow.name:
            return
        self.editor.workflow.rename_workflow(self.workflow, name)

    def _on_part_toggled(
        self, button: Gtk.ToggleButton, workflow: Workflow, workpiece
    ):
        if self._in_update:
            return
        layer = workflow.layer
        if layer is None:
            return
        workpieces = layer.all_workpieces
        chosen_uids = {
            wp.uid for wp in workflow.selected_workpieces(workpieces)
        }
        if button.get_active():
            chosen_uids.add(workpiece.uid)
        else:
            chosen_uids.discard(workpiece.uid)
        if not chosen_uids:
            self._in_update = True
            button.set_active(True)
            self._in_update = False
            return
        uids = [wp.uid for wp in workpieces if wp.uid in chosen_uids]
        if len(uids) == len(workpieces):
            uids = []
        self.editor.workflow.set_workflow_workpieces(workflow, uids)
