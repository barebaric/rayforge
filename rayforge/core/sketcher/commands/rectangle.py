from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from .base import SketchChangeCommand
from .items import AddItemsCommand
from ..entities import Point, Line
from ..constraints import HorizontalConstraint, VerticalConstraint

if TYPE_CHECKING:
    from ..sketch import Sketch


class RectangleCommand(SketchChangeCommand):
    """A smart command to create a fully constrained rectangle."""

    def __init__(
        self,
        sketch: Sketch,
        start_pid: int,
        end_pos: Tuple[float, float],
        end_pid: Optional[int] = None,
        is_start_temp: bool = False,
    ):
        super().__init__(sketch, _("Add Rectangle"))
        self.start_pid = start_pid
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.add_cmd: Optional[AddItemsCommand] = None

    @staticmethod
    def calculate_geometry(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        start_pid: int,
        end_pid: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Calculates the points, entities, and constraints for a rectangle."""
        if abs(x2 - x1) < 1e-6 or abs(y2 - y1) < 1e-6:
            return None

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        p3_id = end_pid if end_pid is not None else next_temp_id()

        points = {
            "p1_id": start_pid,
            "p2": Point(next_temp_id(), x2, y1),
            "p3": Point(p3_id, x2, y2),
            "p4": Point(next_temp_id(), x1, y2),
        }

        entities = [
            Line(next_temp_id(), points["p1_id"], points["p2"].id),
            Line(next_temp_id(), points["p2"].id, points["p3"].id),
            Line(next_temp_id(), points["p3"].id, points["p4"].id),
            Line(next_temp_id(), points["p4"].id, points["p1_id"]),
        ]

        constraints = [
            HorizontalConstraint(points["p1_id"], points["p2"].id),
            VerticalConstraint(points["p2"].id, points["p3"].id),
            HorizontalConstraint(points["p4"].id, points["p3"].id),
            VerticalConstraint(points["p1_id"], points["p4"].id),
        ]
        return {
            "points": points,
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

        final_mx, final_my = self.end_pos
        if self.end_pid is not None:
            try:
                end_p = reg.get_point(self.end_pid)
                final_mx, final_my = end_p.x, end_p.y
            except IndexError:
                pass  # Use mouse coords if pid is invalid

        result = self.calculate_geometry(
            start_p.x,
            start_p.y,
            final_mx,
            final_my,
            self.start_pid,
            self.end_pid,
        )
        if not result:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_pid)
            return

        points_dict = result["points"]
        points_to_add = []
        # These points are always new
        points_to_add.extend([points_dict["p2"], points_dict["p4"]])

        # Add p3 only if it wasn't an existing snapped point
        if self.end_pid is None:
            points_to_add.append(points_dict["p3"])

        # If the start point was temporary, remove it from the registry
        # and add its object to the command to be re-added properly.
        if self.is_start_temp:
            reg.points.remove(start_p)
            points_to_add.append(start_p)

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
            # If the start point was temporary, it was removed from the sketch
            # before the command ran. The command's undo removes the *new*
            # version. We must restore the original temp point.
            if self.is_start_temp and self.start_pid in self._state_snapshot:
                start_x, start_y = self._state_snapshot[self.start_pid]
                new_p = Point(self.start_pid, start_x, start_y)
                self.sketch.registry.points.append(new_p)
