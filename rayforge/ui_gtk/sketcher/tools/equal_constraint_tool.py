import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ....core.sketcher.commands import AddItemsCommand, RemoveItemsCommand
from ....core.sketcher.constraints import EqualLengthConstraint
from ....core.sketcher.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ....core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class EqualConstraintTool(SketchTool):
    ICON = "sketch-constrain-equal-symbolic"
    LABEL = _("Equal")
    SHORTCUT = ("e", _("Equal"))

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return self.element.sketch.supports_constraint(
            "equal", sel.point_ids, sel.entity_ids
        )

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._add_constraint()
        self.element.set_tool("select")

    def _add_constraint(self):
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not editor:
            return

        selected_ids = set(sel.entity_ids)
        existing_constraints_to_merge = []
        final_ids = set(selected_ids)

        for constr in sketch.constraints:
            if isinstance(constr, EqualLengthConstraint):
                if not selected_ids.isdisjoint(constr.entity_ids):
                    existing_constraints_to_merge.append(constr)
                    final_ids.update(constr.entity_ids)

        remove_cmd = RemoveItemsCommand(
            sketch, "", constraints=existing_constraints_to_merge
        )
        new_constr = EqualLengthConstraint(list(final_ids))
        add_cmd = AddItemsCommand(
            sketch, _("Add Equal Constraint"), constraints=[new_constr]
        )

        remove_cmd._do_execute()

        original_add_undo = add_cmd._do_undo

        def composite_undo():
            original_add_undo()
            remove_cmd._do_undo()

        add_cmd._do_undo = composite_undo
        self.element.execute_command(add_cmd)
