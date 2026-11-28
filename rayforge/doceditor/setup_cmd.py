from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..context import get_context
from ..core.group import Group
from ..core.item import DocItem
from ..core.setup import Setup
from ..core.undo import (
    ChangePropertyCommand,
    Command,
)
from ..core.undo.list_cmd import ReorderListCommand
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from ..ui_gtk.canvas2d.surface import WorkSurface
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class MoveItemsSetupCommand(Command):
    """
    An undoable command to move one or more DocItems (workpieces or
    groups) to a different setup.
    """

    def __init__(
        self,
        items: list[DocItem] | list[WorkPiece],
        new_setup: Setup,
        old_setup: Setup,
        name: str | None = None,
    ):
        super().__init__(name)
        self.items = list(items)
        self.new_setup = new_setup
        self.old_setup = old_setup
        if not name:
            self.name = _("Move to another setup")

    def _move(self, from_setup: Setup, to_setup: Setup):
        for item in self.items:
            to_setup.add_child(item)

    def execute(self):
        self._move(self.old_setup, self.new_setup)

    def undo(self):
        self._move(self.new_setup, self.old_setup)


class AddSetupAndSetActiveCommand(Command):
    """
    An undoable command to add a new setup and set it as the active
    setup.
    """

    def __init__(
        self,
        editor: DocEditor,
        new_setup: Setup | None = None,
        name: str = "Add setup",
    ):
        super().__init__(name=name)
        self._editor = editor
        self.new_setup = new_setup or self._create_default_setup()
        self._old_active_setup: Setup | None = None

    def _create_default_setup(self) -> Setup:
        """Creates a new setup with a default, unique name."""
        base_name = _("Setup")
        existing_names = {setup.name for setup in self._editor.doc.setups}
        highest_num = 0
        for name in existing_names:
            if name.startswith(base_name):
                try:
                    num_part = name[len(base_name) :].strip()
                    if num_part.isdigit():
                        highest_num = max(highest_num, int(num_part))
                except ValueError:
                    continue  # Ignore names that don't parse correctly

        new_name = f"{base_name} {highest_num + 1}"
        setup = Setup(name=new_name)
        material = get_context().material_mgr.get_default_material()
        if material is not None:
            setup.set_stock_material_uid(material.uid)
        return setup

    def execute(self):
        """Adds the setup and makes it active."""
        self._old_active_setup = self._editor.doc.active_setup
        new_list = self._editor.doc.setups + [self.new_setup]
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="setups",
            new_list=new_list,
            setter_method_name="set_setups",
        )
        cmd.execute()
        self._editor.doc.active_setup = self.new_setup

    def undo(self):
        """Removes the setup and restores the previous active setup."""
        new_list = [
            s for s in self._editor.doc.setups if s is not self.new_setup
        ]
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="setups",
            new_list=new_list,
            setter_method_name="set_setups",
        )
        cmd.execute()
        if self._old_active_setup in self._editor.doc.setups:
            self._editor.doc.active_setup = self._old_active_setup


class SetupCmd:
    """Handles commands related to setup manipulation."""

    def __init__(self, editor: DocEditor):
        self._editor = editor

    def set_active_setup(self, setup: Setup):
        """Sets the active setup."""
        if self._editor.doc.active_setup is setup:
            return
        old_setup = self._editor.doc.active_setup
        cmd = ChangePropertyCommand(
            target=self._editor.doc,
            property_name="active_setup",
            new_value=setup,
            old_value=old_setup,
            name=_("Set active setup"),
        )
        self._editor.history_manager.execute(cmd)

    def add_setup_and_set_active(self, new_setup: Setup | None = None):
        """Adds a new setup to the document and sets it as active."""
        cmd = AddSetupAndSetActiveCommand(self._editor, new_setup)
        self._editor.history_manager.execute(cmd)

    def rename_setup(self, setup: Setup, new_name: str):
        """Renames a setup with an undoable command."""
        if new_name == setup.name:
            return
        cmd = ChangePropertyCommand(
            target=setup,
            property_name="name",
            new_value=new_name,
            setter_method_name="set_name",
            name=_("Rename setup"),
        )
        self._editor.history_manager.execute(cmd)

    def set_setup_stock_material(self, setup: Setup, material_uid: str | None):
        """Sets a setup's rotary stock material with an undoable
        command."""
        if material_uid == setup.stock_material_uid:
            return
        cmd = ChangePropertyCommand(
            target=setup,
            property_name="stock_material_uid",
            new_value=material_uid,
            setter_method_name="set_stock_material_uid",
            name=_("Change setup stock material"),
        )
        self._editor.history_manager.execute(cmd)

    def delete_setup(self, setup: Setup):
        """Deletes a setup with an undoable command."""
        new_list = [s for s in self._editor.doc.setups if s is not setup]
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="setups",
            new_list=new_list,
            setter_method_name="set_setups",
            name=_("Remove setup '{name}'").format(name=setup.name),
        )
        self._editor.history_manager.execute(cmd)

    def reorder_setups(self, new_order: list[Setup]):
        """Reorders setups with an undoable command."""
        cmd = ReorderListCommand(
            target_obj=self._editor.doc,
            list_property_name="setups",
            new_list=new_order,
            setter_method_name="set_setups",
        )
        self._editor.history_manager.execute(cmd)

    def move_selected_to_adjacent_setup(
        self, surface: WorkSurface, direction: int
    ):
        """
        Creates an undoable command to move selected workpieces to the
        next or previous setup, preserving the selection.

        Args:
            surface: The WorkSurface instance containing the selection.
            direction: 1 for the next setup, -1 for the previous.
        """
        selected_wps = surface.get_selected_workpieces()
        if not selected_wps:
            return

        doc = self._editor.doc
        setups = list(doc.setups)

        if len(setups) <= 1:
            # Not enough setups to move between.
            return

        # Assume all selected workpieces are on the same setup, which is
        # a reasonable constraint for this operation.
        current_setup = selected_wps[0].setup
        if not current_setup:
            return

        try:
            current_index = setups.index(current_setup)

            # Wrap around the setup list.
            new_index = (current_index + direction + len(setups)) % len(setups)
            new_setup = setups[new_index]

            # 1. Create the model-only command.
            cmd = MoveItemsSetupCommand(selected_wps, new_setup, current_setup)

            # 2. Execute the command. The history manager updates the
            #    model, which triggers signals that cause the UI to
            #    destructively rebuild the moved elements in the new
            #    setup element.
            self._editor.history_manager.execute(cmd)

            # 3. After the model and UI have been updated, explicitly
            #    re-apply the selection to the newly created UI elements.
            surface.select_items(selected_wps)

        except ValueError:
            logger.warning(
                f"Setup '{current_setup.name}' not found in document's "
                "setup list."
            )

    def move_items_to_setup(self, items: list[DocItem], target_setup: Setup):
        """Creates an undoable command to move items to a specific
        setup."""
        if not items:
            return

        by_setup: dict[Setup, list[DocItem]] = {}
        for item in items:
            if isinstance(item, (WorkPiece, Group)):
                setup = item.setup
            else:
                continue
            if not setup or setup is target_setup:
                continue
            by_setup.setdefault(setup, []).append(item)

        if not by_setup:
            return

        history = self._editor.history_manager
        with history.transaction(_("Move to another setup")) as t:
            for source_setup, setup_items in by_setup.items():
                cmd = MoveItemsSetupCommand(
                    setup_items, target_setup, source_setup
                )
                t.execute(cmd)

    def reorder_content_items(self, setup: Setup, new_order: list[DocItem]):
        """Reorders content items within a setup with an undoable
        command."""
        cmd = ReorderListCommand(
            target_obj=setup,
            list_property_name="content_items",
            new_list=new_order,
            setter_method_name="reorder_content_items",
            name=_("Reorder items"),
        )
        self._editor.history_manager.execute(cmd)
