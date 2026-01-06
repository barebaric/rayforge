from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Optional, Dict, Any

from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand
from ..entities import Line, Point, Arc
from ..constraints import (
    TangentConstraint,
    CollinearConstraint,
    EqualDistanceConstraint,
)

if TYPE_CHECKING:
    from ..sketch import Sketch
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


class FilletCommand(SketchChangeCommand):
    """Command to add a fillet (rounded corner) between two lines."""

    def __init__(
        self,
        sketch: Sketch,
        corner_pid: int,
        line1_id: int,
        line2_id: int,
        radius: float,
    ):
        super().__init__(sketch, _("Add Fillet"))
        self.corner_pid = corner_pid
        self.line1_id = line1_id
        self.line2_id = line2_id
        self.radius = radius

        # State for undo/redo
        self.add_cmd: Optional[AddItemsCommand] = None
        self.remove_cmd: Optional[RemoveItemsCommand] = None
        self._prepared = False

    @staticmethod
    def calculate_geometry(
        reg: "EntityRegistry",
        corner_pid: int,
        line1_id: int,
        line2_id: int,
        radius: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates the points, entities, and constraints for a fillet.
        This is a pure function for testability and reusability.
        """
        line1 = reg.get_entity(line1_id)
        line2 = reg.get_entity(line2_id)
        if not isinstance(line1, Line) or not isinstance(line2, Line):
            return None

        try:
            corner_point = reg.get_point(corner_pid)
            other1_pid = (
                line1.p2_idx if line1.p1_idx == corner_pid else line1.p1_idx
            )
            other2_pid = (
                line2.p2_idx if line2.p1_idx == corner_pid else line2.p1_idx
            )
            other1_pt = reg.get_point(other1_pid)
            other2_pt = reg.get_point(other2_pid)
        except IndexError:
            return None

        v1 = (other1_pt.x - corner_point.x, other1_pt.y - corner_point.y)
        v2 = (other2_pt.x - corner_point.x, other2_pt.y - corner_point.y)
        len1, len2 = math.hypot(v1[0], v1[1]), math.hypot(v2[0], v2[1])

        if len1 < 1e-9 or len2 < 1e-9:
            return None

        u1 = (v1[0] / len1, v1[1] / len1)
        u2 = (v2[0] / len2, v2[1] / len2)
        dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        angle = math.acos(dot)

        if angle < 1e-3 or abs(angle - math.pi) < 1e-3:
            return None

        tan_half = math.tan(angle / 2.0)
        if abs(tan_half) < 1e-9:
            return None
        dist_to_tangent = radius / tan_half

        if dist_to_tangent > len1 or dist_to_tangent > len2:
            return None  # Fillet too large for lines

        p_tan1_pos = (
            corner_point.x + dist_to_tangent * u1[0],
            corner_point.y + dist_to_tangent * u1[1],
        )
        p_tan2_pos = (
            corner_point.x + dist_to_tangent * u2[0],
            corner_point.y + dist_to_tangent * u2[1],
        )

        bisector_len = math.hypot(u1[0] + u2[0], u1[1] + u2[1])
        if bisector_len < 1e-9:
            return None

        u_bisector = (
            (u1[0] + u2[0]) / bisector_len,
            (u1[1] + u2[1]) / bisector_len,
        )
        dist_to_center = radius / math.sin(angle / 2.0)
        p_center_pos = (
            corner_point.x + dist_to_center * u_bisector[0],
            corner_point.y + dist_to_center * u_bisector[1],
        )

        cross = u1[0] * u2[1] - u1[1] * u2[0]
        is_cw = cross > 0

        p_tan1 = Point(-1, p_tan1_pos[0], p_tan1_pos[1])
        p_tan2 = Point(-2, p_tan2_pos[0], p_tan2_pos[1])
        p_center = Point(-3, p_center_pos[0], p_center_pos[1])

        new_line1 = Line(-4, other1_pid, p_tan1.id)
        new_line2 = Line(-5, other2_pid, p_tan2.id)
        fillet_arc = Arc(
            -6, p_tan1.id, p_tan2.id, p_center.id, clockwise=is_cw
        )

        added_constraints = [
            TangentConstraint(new_line1.id, fillet_arc.id),
            TangentConstraint(new_line2.id, fillet_arc.id),
            CollinearConstraint(other1_pid, corner_pid, p_tan1.id),
            CollinearConstraint(other2_pid, corner_pid, p_tan2.id),
            EqualDistanceConstraint(
                corner_pid, p_tan1.id, corner_pid, p_tan2.id
            ),
        ]

        return {
            "points": [p_tan1, p_tan2, p_center],
            "entities": [new_line1, new_line2, fillet_arc],
            "constraints": added_constraints,
            "removed_entities": [line1, line2],
        }

    def _prepare(self) -> bool:
        """Prepares internal commands on first execution."""
        if self._prepared:
            return True

        result = self.calculate_geometry(
            self.sketch.registry,
            self.corner_pid,
            self.line1_id,
            self.line2_id,
            self.radius,
        )

        if result is None:
            return False

        self.remove_cmd = RemoveItemsCommand(
            self.sketch, "", entities=result["removed_entities"]
        )
        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=result["points"],
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self._prepared = True
        return True

    def _do_execute(self) -> None:
        if not self._prepare():
            return

        if self.remove_cmd:
            self.remove_cmd._do_execute()
        if self.add_cmd:
            self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if not self.add_cmd or not self.remove_cmd:
            return

        self.add_cmd._do_undo()
        self.remove_cmd._do_undo()
