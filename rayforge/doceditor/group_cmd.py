import logging
from typing import TYPE_CHECKING, List, Dict, Optional
from ..undo.models.command import Command
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.group import Group
from ..core.matrix import Matrix

if TYPE_CHECKING:
    from .editor import DocEditor
    from ..pipeline.generator import OpsGenerator

logger = logging.getLogger(__name__)


class _CreateGroupCommand(Command):
    """An undoable command to group a list of DocItems into a new Group."""

    def __init__(
        self,
        layer: Layer,
        items_to_group: List[DocItem],
        ops_generator: "OpsGenerator",
        name: str = "Group Items",
    ):
        super().__init__(name)
        self.layer = layer
        self.items_to_group = list(items_to_group)
        self.ops_generator = ops_generator
        self.new_group: Optional[Group] = None
        self._original_parents: Dict[str, DocItem] = {
            item.uid: item.parent
            for item in self.items_to_group
            if item.parent
        }
        self._original_matrices: Dict[str, Matrix] = {
            item.uid: item.matrix.copy() for item in self.items_to_group
        }

    def execute(self) -> None:
        """Performs the grouping operation."""
        with self.ops_generator.paused():
            result = Group.create_from_items(self.items_to_group, self.layer)
            if not result:
                return

            self.new_group = result.new_group
            self.layer.add_child(self.new_group)

            for item in self.items_to_group:
                # Reparent the item
                if item.parent:
                    item.parent.remove_child(item)
                self.new_group.add_child(item)

                # Set its new local matrix *after* reparenting
                item.matrix = result.child_matrices[item.uid]

    def undo(self) -> None:
        """Reverts the grouping operation."""
        if not self.new_group:
            return

        # Move children back to original parents first
        with self.ops_generator.paused():
            for item in self.items_to_group:
                original_parent = self._original_parents.get(item.uid)
                if original_parent:
                    self.new_group.remove_child(item)
                    original_parent.add_child(item)
                    item.matrix = self._original_matrices[item.uid]
            # Then remove the now-empty group
            self.layer.remove_child(self.new_group)


class _UngroupCommand(Command):
    """An undoable command to dissolve one or more Groups."""

    def __init__(
        self,
        groups_to_ungroup: List[Group],
        ops_generator: "OpsGenerator",
        name: str = "Ungroup Items",
    ):
        super().__init__(name)
        self.groups_to_ungroup = list(groups_to_ungroup)
        self.ops_generator = ops_generator
        self._undo_data = []
        for group in self.groups_to_ungroup:
            if group.parent:
                self._undo_data.append(
                    {
                        "group_uid": group.uid,
                        "group_matrix": group.matrix.copy(),
                        "parent": group.parent,
                        "children": list(group.children),
                        "child_matrices": {
                            c.uid: c.matrix.copy() for c in group.children
                        },
                    }
                )

    def _calculate_ungroup_transforms(self, group: Group) -> Dict[str, Matrix]:
        """
        Calculates the new local matrices for a group's children if they were
        re-parented to the group's own parent. This is a pure helper function.
        """
        if not group.parent:
            return {}

        group_world_transform = group.get_world_transform()
        parent_inv_world = group.parent.get_world_transform().invert()

        new_child_matrices = {}
        for child in group.children:
            child_world_transform = group_world_transform @ child.matrix
            new_child_matrices[child.uid] = (
                parent_inv_world @ child_world_transform
            )

        return new_child_matrices

    def execute(self) -> None:
        """Performs the ungrouping operation."""
        with self.ops_generator.paused():
            for group in self.groups_to_ungroup:
                parent = group.parent
                if not parent:
                    continue

                new_child_matrices = self._calculate_ungroup_transforms(group)
                children_to_move = list(group.children)

                parent.remove_child(group)
                for child in children_to_move:
                    parent.add_child(child)
                    child.matrix = new_child_matrices[child.uid]

    def undo(self) -> None:
        """Reverts the ungrouping by re-creating the original groups."""
        with self.ops_generator.paused():
            for data in reversed(self._undo_data):
                parent = data["parent"]
                children = data["children"]
                group = next(
                    (
                        g
                        for g in self.groups_to_ungroup
                        if g.uid == data["group_uid"]
                    ),
                    None,
                )
                if not group:
                    continue

                # Add group back to its parent and restore its matrix
                parent.add_child(group)
                group.matrix = data["group_matrix"]

                # Move children back into the re-created group
                for child in children:
                    parent.remove_child(child)
                    group.add_child(child)
                    child.matrix = data["child_matrices"][child.uid]


class GroupCmd:
    """Handles grouping and ungrouping of document items."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def group_items(
        self, layer: Layer, items_to_group: List[DocItem]
    ) -> Optional[Group]:
        """
        Creates and executes an undoable command to group items.

        Returns:
            The newly created Group, or None if grouping failed.
        """
        command = _CreateGroupCommand(
            layer=layer,
            items_to_group=items_to_group,
            ops_generator=self._editor.ops_generator,
        )
        self._editor.history_manager.execute(command)
        return command.new_group

    def ungroup_items(self, groups_to_ungroup: List[Group]):
        """Creates and executes an undoable command to ungroup items."""
        command = _UngroupCommand(
            groups_to_ungroup=groups_to_ungroup,
            ops_generator=self._editor.ops_generator,
        )
        self._editor.history_manager.execute(command)
