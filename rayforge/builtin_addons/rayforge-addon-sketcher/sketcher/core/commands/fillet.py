from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.sketch.constraints.collinear import CollinearConstraint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.constraints.tangent import TangentConstraint
from raygeo.sketch.entities.arc import Arc
from raygeo.sketch.entities.line import Line

from ..types import EntityID
from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class FilletCommand(SketchChangeCommand):
    """Command to add a fillet (rounded corner) between two lines."""

    def __init__(
        self,
        sketch: Sketch,
        corner_pid: EntityID,
        line1_id: EntityID,
        line2_id: EntityID,
        radius: float,
    ):
        super().__init__(sketch, _("Add Fillet"))
        self.corner_pid = corner_pid
        self.line1_id = line1_id
        self.line2_id = line2_id
        self.radius = radius

        # State for undo/redo
        self.add_cmd: AddItemsCommand | None = None
        self.remove_cmd: RemoveItemsCommand | None = None
        self._prepared = False

    @staticmethod
    def calculate_geometry(
        reg: EntityRegistry,
        corner_pid: EntityID,
        line1_id: EntityID,
        line2_id: EntityID,
        radius: float,
        add_point=None,
    ) -> dict[str, Any] | None:
        """
        Calculates the fillet geometry and allocates its points
        through ``add_point``. This is a pure function for
        testability and reusability.
        """
        temp_points: list = []
        if add_point is None:
            from raygeo.sketch.entities.point import SketchPoint

            def add_point(x, y):
                pid = len(temp_points)
                temp_points.append(SketchPoint(pid, x, y))
                return pid

        line1 = reg.get_entity_object(line1_id)
        line2 = reg.get_entity_object(line2_id)
        if not isinstance(line1, Line) or not isinstance(line2, Line):
            return None

        corner_point = reg.get_point(corner_pid)
        other1_pid = (
            line1.p2_idx if line1.p1_idx == corner_pid else line1.p1_idx
        )
        other2_pid = (
            line2.p2_idx if line2.p1_idx == corner_pid else line2.p1_idx
        )
        other1_pt = reg.get_point(other1_pid)
        other2_pt = reg.get_point(other2_pid)
        if corner_point is None or other1_pt is None or other2_pt is None:
            return None

        v1 = (other1_pt[0] - corner_point[0], other1_pt[1] - corner_point[1])
        v2 = (other2_pt[0] - corner_point[0], other2_pt[1] - corner_point[1])
        len1, len2 = math.hypot(v1[0], v1[1]), math.hypot(v2[0], v2[1])

        if len1 < 1e-6 or len2 < 1e-6:
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
            corner_point[0] + dist_to_tangent * u1[0],
            corner_point[1] + dist_to_tangent * u1[1],
        )
        p_tan2_pos = (
            corner_point[0] + dist_to_tangent * u2[0],
            corner_point[1] + dist_to_tangent * u2[1],
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
            corner_point[0] + dist_to_center * u_bisector[0],
            corner_point[1] + dist_to_center * u_bisector[1],
        )

        cross = u1[0] * u2[1] - u1[1] * u2[0]
        is_cw = cross > 0

        p_tan1 = add_point(*p_tan1_pos)
        p_tan2 = add_point(*p_tan2_pos)
        p_center = add_point(*p_center_pos)

        new_line1 = Line(other1_pid, p_tan1)
        new_line2 = Line(other2_pid, p_tan2)
        fillet_arc = Arc(p_tan1, p_tan2, p_center, clockwise=is_cw)

        added_constraints = [
            TangentConstraint(
                other1_pid, p_tan1, p_center, p_tan1
            ),
            TangentConstraint(
                other2_pid, p_tan2, p_center, p_tan1
            ),
            CollinearConstraint(other1_pid, corner_pid, p_tan1),
            CollinearConstraint(other2_pid, corner_pid, p_tan2),
            EqualDistanceConstraint(
                corner_pid, p_tan1, corner_pid, p_tan2
            ),
        ]

        return {
            "points": temp_points,
            "entities": [new_line1, new_line2, fillet_arc],
            "constraints": added_constraints,
            "removed_entity_ids": [line1_id, line2_id],
            "removed_entities": [line1_id, line2_id],
        }

    def _prepare(self) -> bool:
        """Prepares internal commands on first execution."""
        if self._prepared:
            return True

        add_cmd = AddItemsCommand(self.sketch, "")
        result = self.calculate_geometry(
            self.sketch.registry,
            self.corner_pid,
            self.line1_id,
            self.line2_id,
            self.radius,
            add_cmd.add_point,
        )

        if result is None:
            return False

        for entity in result["entities"]:
            add_cmd.add_entity(entity)
        for constraint in result["constraints"]:
            add_cmd.add_constraint(constraint)

        self.remove_cmd = RemoveItemsCommand(
            self.sketch,
            "",
            entity_ids=result["removed_entity_ids"],
        )
        self.add_cmd = add_cmd
        self._prepared = True
        return True

    def _do_execute(self) -> None:
        if not self._prepare():
            return

        if self.remove_cmd:
            self.remove_cmd._do_execute()
        if self.add_cmd:
            self.add_cmd._redo()

    def _do_undo(self) -> None:
        if not self.add_cmd or not self.remove_cmd:
            return

        self.add_cmd._do_undo()
        self.remove_cmd._do_undo()
