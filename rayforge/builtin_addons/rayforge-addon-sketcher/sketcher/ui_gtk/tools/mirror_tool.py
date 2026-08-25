from gettext import gettext as _
from typing import TYPE_CHECKING, ClassVar, Union

from ...core.commands import MirrorCommand, MirrorDirection
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint


class MirrorVerticalTool(SketchTool):
    ICON = "flip-vertical-symbolic"
    LABEL = _("Mirror Vertically")
    SHORTCUTS: ClassVar[list[str]] = ["mv"]
    SHOW_IN_PIE = False

    def is_available(
        self,
        target: Union[Point, Entity, "Constraint"] | None,
        target_type: str | None,
    ) -> bool:
        sel = self.element.selection
        return bool(sel.entity_ids or sel.point_ids)

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        sel = self.element.selection
        cmd = MirrorCommand(self.element.sketch, sel, MirrorDirection.VERTICAL)
        self.element.execute_command(cmd)
        self.element.set_tool("select")


class MirrorHorizontalTool(SketchTool):
    ICON = "flip-horizontal-symbolic"
    LABEL = _("Mirror Horizontally")
    SHORTCUTS: ClassVar[list[str]] = ["mh"]
    SHOW_IN_PIE = False

    def is_available(
        self,
        target: Union[Point, Entity, "Constraint"] | None,
        target_type: str | None,
    ) -> bool:
        sel = self.element.selection
        return bool(sel.entity_ids or sel.point_ids)

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        sel = self.element.selection
        cmd = MirrorCommand(
            self.element.sketch, sel, MirrorDirection.HORIZONTAL
        )
        self.element.execute_command(cmd)
        self.element.set_tool("select")
