import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Union

from ...core.commands import UnstickJunctionCommand
from ...core.commands.items import RemoveItemsCommand
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class DeleteTool(SketchTool):
    ICON = "delete-symbolic"
    LABEL = _("Delete")

    def is_available(
        self,
        target: Optional[Union[Point, Entity, "Constraint"]],
        target_type: Optional[str],
    ) -> bool:
        sel = self.element.selection
        return bool(
            sel.point_ids
            or sel.entity_ids
            or sel.constraint_idx is not None
            or sel.junction_pid is not None
        )

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._delete_selection()
        self.element.set_tool("select")

    def _delete_selection(self) -> bool:
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not editor:
            return False

        if sel.junction_pid is not None:
            cmd = UnstickJunctionCommand(sketch, sel.junction_pid)
            self.element.execute_command(cmd)
            sel.clear()
            return True

        (
            points_to_del,
            entities_to_del,
            constraints_to_del,
        ) = RemoveItemsCommand.calculate_dependencies(sketch, sel)

        did_work = bool(points_to_del or entities_to_del or constraints_to_del)
        if did_work:
            cmd = RemoveItemsCommand(
                sketch,
                _("Delete Selection"),
                points=points_to_del,
                entities=entities_to_del,
                constraints=constraints_to_del,
            )
            self.element.execute_command(cmd)
            sel.clear()

        return did_work
