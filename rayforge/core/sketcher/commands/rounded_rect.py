from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any
from ..entities import Point, Line, Arc
from ..constraints import (
    HorizontalConstraint,
    VerticalConstraint,
    TangentConstraint,
    EqualLengthConstraint,
    EqualDistanceConstraint,
)
from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..sketch import Sketch


class RoundedRectCommand(SketchChangeCommand):
    """A smart command to create a fully constrained rounded rectangle."""

    def __init__(
        self,
        sketch: Sketch,
        start_pid: int,
        end_pos: Tuple[float, float],
        radius: float,
        is_start_temp: bool = False,
    ):
        super().__init__(sketch, _("Add Rounded Rectangle"))
        self.start_pid = start_pid
        self.end_pos = end_pos
        self.radius = radius
        self.is_start_temp = is_start_temp
        self.add_cmd: Optional[AddItemsCommand] = None

    @staticmethod
    def calculate_geometry(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
    ) -> Optional[Dict[str, Any]]:
        """Calculates geometry for a rounded rectangle."""
        width, height = abs(x2 - x1), abs(y2 - y1)
        if width < 1e-6 or height < 1e-6:
            return None

        radius = min(radius, width / 2.0, height / 2.0)
        sx, sy = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1)

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        points = {
            "t1": Point(next_temp_id(), x1 + sx * radius, y1),
            "t2": Point(next_temp_id(), x2 - sx * radius, y1),
            "t3": Point(next_temp_id(), x2, y1 + sy * radius),
            "t4": Point(next_temp_id(), x2, y2 - sy * radius),
            "t5": Point(next_temp_id(), x2 - sx * radius, y2),
            "t6": Point(next_temp_id(), x1 + sx * radius, y2),
            "t7": Point(next_temp_id(), x1, y2 - sy * radius),
            "t8": Point(next_temp_id(), x1, y1 + sy * radius),
            "c1": Point(next_temp_id(), x1 + sx * radius, y1 + sy * radius),
            "c2": Point(next_temp_id(), x2 - sx * radius, y1 + sy * radius),
            "c3": Point(next_temp_id(), x2 - sx * radius, y2 - sy * radius),
            "c4": Point(next_temp_id(), x1 + sx * radius, y2 - sy * radius),
        }

        is_cw = sx * sy < 0
        entities = [
            Line(next_temp_id(), points["t1"].id, points["t2"].id),
            Line(next_temp_id(), points["t3"].id, points["t4"].id),
            Line(next_temp_id(), points["t5"].id, points["t6"].id),
            Line(next_temp_id(), points["t7"].id, points["t8"].id),
            Arc(
                next_temp_id(),
                points["t8"].id,
                points["t1"].id,
                points["c1"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t2"].id,
                points["t3"].id,
                points["c2"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t4"].id,
                points["t5"].id,
                points["c3"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t6"].id,
                points["t7"].id,
                points["c4"].id,
                clockwise=is_cw,
            ),
        ]

        constraints = [
            HorizontalConstraint(points["t1"].id, points["t2"].id),
            VerticalConstraint(points["t3"].id, points["t4"].id),
            HorizontalConstraint(points["t5"].id, points["t6"].id),
            VerticalConstraint(points["t7"].id, points["t8"].id),
            TangentConstraint(entities[0].id, entities[4].id),
            TangentConstraint(entities[3].id, entities[4].id),
            TangentConstraint(entities[0].id, entities[5].id),
            TangentConstraint(entities[1].id, entities[5].id),
            TangentConstraint(entities[1].id, entities[6].id),
            TangentConstraint(entities[2].id, entities[6].id),
            TangentConstraint(entities[2].id, entities[7].id),
            TangentConstraint(entities[3].id, entities[7].id),
            EqualLengthConstraint([e.id for e in entities[4:]]),
            EqualDistanceConstraint(
                points["c1"].id,
                points["t8"].id,
                points["c1"].id,
                points["t1"].id,
            ),
            EqualDistanceConstraint(
                points["c2"].id,
                points["t2"].id,
                points["c2"].id,
                points["t3"].id,
            ),
            EqualDistanceConstraint(
                points["c3"].id,
                points["t4"].id,
                points["c3"].id,
                points["t5"].id,
            ),
            EqualDistanceConstraint(
                points["c4"].id,
                points["t6"].id,
                points["c4"].id,
                points["t7"].id,
            ),
        ]
        return {
            "points": list(points.values()),
            "entities": entities,
            "constraints": constraints,
        }

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        reg = self.sketch.registry
        try:
            start_p = reg.get_point(self.start_pid)
        except IndexError:
            return

        result = self.calculate_geometry(
            start_p.x, start_p.y, self.end_pos[0], self.end_pos[1], self.radius
        )
        if not result:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_pid)
            return

        points_to_add = result["points"]
        if self.is_start_temp:
            reg.points.remove(start_p)
            # Unlike Rectangle, RoundedRect doesn't use the start_pid in its
            # final geometry, so we just remove it.

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
            if self.is_start_temp and self.start_pid in self._state_snapshot:
                start_x, start_y = self._state_snapshot[self.start_pid]
                new_p = Point(self.start_pid, start_x, start_y)
                self.sketch.registry.points.append(new_p)
