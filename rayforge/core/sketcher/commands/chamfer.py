from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, List, Optional

from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand
from ..entities import Line, Point
from ..constraints import (
    Constraint,
    EqualDistanceConstraint,
    CollinearConstraint,
)

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class ChamferCommand(SketchChangeCommand):
    """Command to add a chamfer to a corner by replacing the corner lines."""

    def __init__(
        self,
        sketch: Sketch,
        corner_pid: int,
        line1_id: int,
        line2_id: int,
        distance: float,
    ):
        super().__init__(sketch, _("Add Chamfer"))
        self.corner_pid = corner_pid
        self.line1_id = line1_id
        self.line2_id = line2_id
        self.distance = distance

        # State for undo/redo
        self.added_points: List[Point] = []
        self.added_entities: List[Line] = []
        self.added_constraints: List[Constraint] = []
        self.removed_entities: List[Line] = []
        self.corner_point: Optional[Point] = None

    def _do_execute(self) -> None:
        sketch = self.sketch
        reg = sketch.registry

        # On first run, prepare all new/old items
        if not self.added_points:
            line1 = reg.get_entity(self.line1_id)
            line2 = reg.get_entity(self.line2_id)
            if not isinstance(line1, Line) or not isinstance(line2, Line):
                return
            self.removed_entities = [line1, line2]

            try:
                self.corner_point = reg.get_point(self.corner_pid)
                other1_pid = (
                    line1.p2_idx
                    if line1.p1_idx == self.corner_pid
                    else line1.p1_idx
                )
                other2_pid = (
                    line2.p2_idx
                    if line2.p1_idx == self.corner_pid
                    else line2.p1_idx
                )
                other1_pt = reg.get_point(other1_pid)
                other2_pt = reg.get_point(other2_pid)
            except IndexError:
                return

            v1 = (
                other1_pt.x - self.corner_point.x,
                other1_pt.y - self.corner_point.y,
            )
            len1 = math.hypot(v1[0], v1[1])
            u1 = (v1[0] / len1, v1[1] / len1) if len1 > 1e-9 else (0.0, 0.0)
            p_new1_pos = (
                self.corner_point.x + self.distance * u1[0],
                self.corner_point.y + self.distance * u1[1],
            )

            v2 = (
                other2_pt.x - self.corner_point.x,
                other2_pt.y - self.corner_point.y,
            )
            len2 = math.hypot(v2[0], v2[1])
            u2 = (v2[0] / len2, v2[1] / len2) if len2 > 1e-9 else (0.0, 0.0)
            p_new2_pos = (
                self.corner_point.x + self.distance * u2[0],
                self.corner_point.y + self.distance * u2[1],
            )

            # Define new items with temporary IDs
            p1 = Point(-1, p_new1_pos[0], p_new1_pos[1])
            p2 = Point(-2, p_new2_pos[0], p_new2_pos[1])
            self.added_points = [p1, p2]

            self.added_entities = [
                Line(-3, p1.id, p2.id),  # chamfer_line
                Line(-4, other1_pid, p1.id),  # new_segment1
                Line(-5, other2_pid, p2.id),  # new_segment2
            ]

            self.added_constraints = [
                CollinearConstraint(other1_pid, self.corner_pid, p1.id),
                CollinearConstraint(other2_pid, self.corner_pid, p2.id),
                EqualDistanceConstraint(
                    self.corner_pid, p1.id, self.corner_pid, p2.id
                ),
            ]

        # --- Apply changes ---
        # 1. Remove original lines
        remove_cmd = RemoveItemsCommand(
            sketch, "", entities=self.removed_entities
        )
        remove_cmd._do_execute()

        # 2. Add new geometry and constraints
        add_cmd = AddItemsCommand(
            sketch,
            "",
            points=self.added_points,
            entities=self.added_entities,
            constraints=self.added_constraints,
        )
        add_cmd._do_execute()

    def _do_undo(self) -> None:
        # 1. Remove new geometry
        remove_cmd = RemoveItemsCommand(
            self.sketch,
            "",
            points=self.added_points,
            entities=self.added_entities,
            constraints=self.added_constraints,
        )
        remove_cmd._do_execute()

        # 2. Add original lines back
        add_cmd = AddItemsCommand(
            self.sketch, "", entities=self.removed_entities
        )
        add_cmd._do_execute()
