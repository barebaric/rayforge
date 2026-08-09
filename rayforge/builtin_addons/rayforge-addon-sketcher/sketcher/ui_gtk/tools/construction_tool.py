import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, ClassVar, Union

from ...core.commands import ToggleConstructionCommand
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class ConstructionTool(SketchTool):
    ICON = "sketch-construction-symbolic"
    LABEL = _("Construction")
    SHORTCUTS: ClassVar[list[str]] = ["gn"]

    def is_available(
        self,
        target: Union[Point, Entity, "Constraint"] | None,
        target_type: str | None,
    ) -> bool:
        return len(self.element.selection.entity_ids) > 0

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        self._toggle_construction()
        self.element.set_tool("select")

    def _toggle_construction(self):
        sel = self.element.selection
        sketch = self.element.sketch
        editor = self.element.editor

        if not sel.entity_ids or not editor:
            return

        cmd = ToggleConstructionCommand(
            sketch, _("Toggle Construction"), sel.entity_ids
        )
        self.element.execute_command(cmd)
