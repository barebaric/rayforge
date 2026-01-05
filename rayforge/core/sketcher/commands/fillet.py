from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, List, Optional
from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand
from ..entities import Line, Point, Arc
from ..constraints import (
    Constraint,
    TangentConstraint,
    CollinearConstraint,
    EqualDistanceConstraint,
)

if TYPE_CHECKING:
    from ..sketch import Sketch

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
        self.added_points: List[Point] = []
        self.added_entities: List[Line | Arc] = []
        self.added_constraints: List[Constraint] = []
        self.removed_entities: List[Line] = []
        self.corner_point: Optional[Point] = None

    def _do_execute(self) -> None:
        sketch = self.sketch
        reg = sketch.registry

        # On first run, calculate geometry and prepare items
        if not self.added_points:
            line1 = reg.get_entity(self.line1_id)
            line2 = reg.get_entity(self.line2_id)
            if not isinstance(line1, Line) or not isinstance(line2, Line):
                return
            self.removed_entities = [line1, line2]

            try:
                self.corner_point = reg.get_point(self.corner_pid)
                # Identify the "other" points (far ends of the lines)
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

            # Vector math to find tangent points and center
            v1 = (
                other1_pt.x - self.corner_point.x,
                other1_pt.y - self.corner_point.y,
            )
            v2 = (
                other2_pt.x - self.corner_point.x,
                other2_pt.y - self.corner_point.y,
            )

            len1 = math.hypot(v1[0], v1[1])
            len2 = math.hypot(v2[0], v2[1])

            if len1 < 1e-9 or len2 < 1e-9:
                return  # Degenerate lines

            # Unit vectors pointing AWAY from corner
            u1 = (v1[0] / len1, v1[1] / len1)
            u2 = (v2[0] / len2, v2[1] / len2)

            # Calculate angle between vectors
            dot = u1[0] * u2[0] + u1[1] * u2[1]
            # Clamp to avoid domain errors
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)

            # Avoid parallel lines (angle 0 or 180)
            if angle < 1e-3 or abs(angle - math.pi) < 1e-3:
                return

            # Distance from corner to tangent points: d = r / tan(theta/2)
            tan_half = math.tan(angle / 2.0)
            if abs(tan_half) < 1e-9:
                return
            dist_to_tangent = self.radius / tan_half

            # Coordinates of tangent points
            p_tan1_pos = (
                self.corner_point.x + dist_to_tangent * u1[0],
                self.corner_point.y + dist_to_tangent * u1[1],
            )
            p_tan2_pos = (
                self.corner_point.x + dist_to_tangent * u2[0],
                self.corner_point.y + dist_to_tangent * u2[1],
            )

            # Calculate Arc Center
            # The center lies on the angle bisector.
            bisector_x = u1[0] + u2[0]
            bisector_y = u1[1] + u2[1]
            bisector_len = math.hypot(bisector_x, bisector_y)

            if bisector_len < 1e-9:
                return

            u_bisector = (bisector_x / bisector_len, bisector_y / bisector_len)

            # Distance from corner to center: h = r / sin(theta/2)
            sin_half = math.sin(angle / 2.0)
            dist_to_center = self.radius / sin_half

            p_center_pos = (
                self.corner_point.x + dist_to_center * u_bisector[0],
                self.corner_point.y + dist_to_center * u_bisector[1],
            )

            # Determine winding:
            # If turning from u1 to u2 is CCW (Left turn), the arc (Tan1->Tan2)
            # must be drawn Clockwise in local coordinates to fit inside.
            cross = u1[0] * u2[1] - u1[1] * u2[0]
            is_cw = cross > 0

            # Create new Items
            # IDs: -1: Tan1, -2: Tan2, -3: Center
            # IDs: -4: Line1, -5: Line2, -6: Arc
            p_tan1 = Point(-1, p_tan1_pos[0], p_tan1_pos[1])
            p_tan2 = Point(-2, p_tan2_pos[0], p_tan2_pos[1])
            p_center = Point(-3, p_center_pos[0], p_center_pos[1])

            self.added_points = [p_tan1, p_tan2, p_center]

            # New truncated lines connecting original endpoints to tangent
            # points
            new_line1 = Line(-4, other1_pid, p_tan1.id)
            new_line2 = Line(-5, other2_pid, p_tan2.id)

            # The fillet arc
            fillet_arc = Arc(
                -6,
                start_idx=p_tan1.id,
                end_idx=p_tan2.id,
                center_idx=p_center.id,
                clockwise=is_cw,
            )

            self.added_entities = [new_line1, new_line2, fillet_arc]

            # Constraints
            # 1. Tangent: Ensures smooth transition from line to arc.
            # 2. Collinear: Keeps the new line segments aligned with the
            # virtual intersection point (the original corner).
            # 3. EqualDistance: Ensures the fillet is symmetric relative to the
            #    virtual corner, preventing the tangent points from sliding
            #    independently along the lines.
            self.added_constraints = [
                TangentConstraint(new_line1.id, fillet_arc.id),
                TangentConstraint(new_line2.id, fillet_arc.id),
                CollinearConstraint(other1_pid, self.corner_pid, p_tan1.id),
                CollinearConstraint(other2_pid, self.corner_pid, p_tan2.id),
                EqualDistanceConstraint(
                    self.corner_pid, p_tan1.id, self.corner_pid, p_tan2.id
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
