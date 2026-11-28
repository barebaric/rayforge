from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..core.color import pick_unused_color
from ..core.layer import Layer
from ..core.undo import (
    ChangePropertyCommand,
    DictItemCommand,
)
from ..core.undo.list_cmd import ListItemCommand, ReorderListCommand
from ..core.workflow import Workflow
from ..core.workpiece import WorkPiece
from ..image.assembler import ItemAssembler
from ..image.structures import SourceLayerInfo

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class WorkflowCmd:
    """Handles commands related to workflow manipulation."""

    def __init__(self, editor: DocEditor):
        self._editor = editor

    def create_for_source_layer(
        self,
        layer: Layer,
        source_layer: Any,
        workpieces: list[WorkPiece],
    ) -> Workflow | None:
        """Builds a workflow for one imported source layer.

        The workflow is NOT added to the document; the caller commits
        it undoable. When the source layer carried importer settings,
        a matching step is created on the workflow so the reload keeps
        the configured operation.
        """
        if not isinstance(source_layer, SourceLayerInfo):
            return None
        workflow = Workflow(name=source_layer.name)
        if source_layer.color:
            workflow.set_color(source_layer.color)
        workflow.set_workpiece_uids([wp.uid for wp in workpieces])
        if source_layer.settings:
            ItemAssembler.apply_settings_to_workflow(
                workflow, source_layer.name, source_layer.settings
            )
        return workflow

    def add_workflow(self, layer: Layer, workflow: Workflow):
        """Adds a workflow to a layer with an undoable command."""
        cmd = ListItemCommand(
            owner_obj=layer,
            item=workflow,
            undo_command="remove_child",
            redo_command="add_child",
            name=_("Add workflow '{name}'").format(name=workflow.name),
        )
        self._editor.history_manager.execute(cmd)

    def delete_workflow(self, workflow: Workflow):
        """Deletes a workflow with an undoable command."""
        parent = workflow.parent
        if parent is None:
            return
        cmd = ListItemCommand(
            owner_obj=parent,
            item=workflow,
            undo_command="add_child",
            redo_command="remove_child",
            name=_("Remove workflow '{name}'").format(name=workflow.name),
        )
        self._editor.history_manager.execute(cmd)

    def reorder_workflows(self, layer: Layer, new_order: list[Workflow]):
        """Reorders a layer's workflows with an undoable command."""
        cmd = ReorderListCommand(
            target_obj=layer,
            list_property_name="workflows",
            new_list=new_order,
            setter_method_name="reorder_workflows",
            name=_("Reorder workflows"),
        )
        self._editor.history_manager.execute(cmd)

    def rename_workflow(self, workflow: Workflow, new_name: str):
        """Renames a workflow with an undoable command."""
        if new_name == workflow.name:
            return
        cmd = ChangePropertyCommand(
            target=workflow,
            property_name="name",
            new_value=new_name,
            name=_("Rename workflow"),
        )
        self._editor.history_manager.execute(cmd)

    def set_workflow_color(self, workflow: Workflow, color: str):
        """Sets a workflow's color with an undoable command."""
        if color == workflow.color:
            return
        cmd = ChangePropertyCommand(
            target=workflow,
            property_name="color",
            new_value=color,
            setter_method_name="set_color",
            name=_("Change workflow color"),
        )
        self._editor.history_manager.execute(cmd)

    def set_workflow_enabled(self, workflow: Workflow, enabled: bool):
        """Sets a workflow's enabled flag with an undoable command."""
        if enabled == workflow.enabled:
            return
        cmd = ChangePropertyCommand(
            target=workflow,
            property_name="enabled",
            new_value=enabled,
            setter_method_name="set_enabled",
            name=_("Toggle workflow"),
        )
        self._editor.history_manager.execute(cmd)

    def set_workflow_post_processors(
        self, workflow: Workflow, post_processors_dicts: list[dict]
    ):
        """Sets a workflow's post processors with an undoable command."""
        cmd = ChangePropertyCommand(
            target=workflow,
            property_name="post_processors_dicts",
            new_value=list(post_processors_dicts),
            setter_method_name="set_post_processors_dicts",
            name=_("Change workflow post processors"),
        )
        self._editor.history_manager.execute(cmd)

    def set_workflow_post_processor_param(
        self,
        target_dict: dict,
        key: str,
        new_value: Any,
        name: str,
    ):
        """Sets one parameter of a workflow post processor dict,
        undoable."""
        if isinstance(new_value, float):
            old_value = target_dict.get(key)
            if (
                isinstance(old_value, (int, float))
                and abs(new_value - old_value) < 1e-6
            ):
                return
        elif new_value == target_dict.get(key):
            return
        cmd = DictItemCommand(
            target_dict=target_dict,
            key=key,
            new_value=new_value,
            name=name,
        )
        self._editor.history_manager.execute(cmd)

    def next_unused_color(self) -> str:
        """Picks a palette color no workflow in the document uses."""
        used = {
            workflow.color
            for layer in self._editor.doc.layers
            for workflow in layer.workflows
        }
        return pick_unused_color(used)
