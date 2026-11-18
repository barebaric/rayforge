"""The sketcher point model.

The point state lives on the compiled raygeo class; this subclass
adds the bezier-waypoint constraint propagation the commands drive.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Self, cast

from raygeo.sketch.entities import SketchPoint

if TYPE_CHECKING:
    from raygeo.sketch.entities import Bezier

    from .registry import EntityRegistry
    from .sketch import Sketch


class WaypointType(str, Enum):
    SHARP = "sharp"
    SMOOTH = "smooth"
    SYMMETRIC = "symmetric"


class SketchPointOps(SketchPoint):
    """Bezier-waypoint constraint propagation for sketch points."""

    def apply_constraint(
        self,
        registry: EntityRegistry,
        bezier: Bezier,
        cp_index: int,
        sketch: Sketch | None = None,
    ) -> None:
        if self.is_sharp():
            return

        b1, b2 = self.get_paired_beziers(registry, sketch)
        if b1 is None or b2 is None:
            return

        other_bezier = b2 if bezier == b1 else b1

        point_ids = {self.id}
        if sketch is not None:
            point_ids.update(sketch.get_coincident_points(self.id))

        if bezier.start_idx in point_ids and cp_index == 1:
            cp_out = bezier.cp1
            if cp_out is not None:
                if other_bezier.end_idx in point_ids:
                    other_bezier.cp2 = self._compute_constrained_cp(
                        cp_out,
                        other_bezier.cp2,
                        symmetric=self.is_symmetric(),
                    )
                elif other_bezier.start_idx in point_ids:
                    other_bezier.cp1 = self._compute_constrained_cp(
                        cp_out,
                        other_bezier.cp1,
                        symmetric=self.is_symmetric(),
                    )
        elif bezier.end_idx in point_ids and cp_index == 2:
            cp_in = bezier.cp2
            if cp_in is not None:
                if other_bezier.start_idx in point_ids:
                    other_bezier.cp1 = self._compute_constrained_cp(
                        cp_in,
                        other_bezier.cp1,
                        symmetric=self.is_symmetric(),
                    )
                elif other_bezier.end_idx in point_ids:
                    other_bezier.cp2 = self._compute_constrained_cp(
                        cp_in,
                        other_bezier.cp2,
                        symmetric=self.is_symmetric(),
                    )

    def _compute_constrained_cp(
        self,
        modified_cp: tuple[float, float],
        other_cp: tuple[float, float] | None,
        symmetric: bool,
    ) -> tuple[float, float]:
        if symmetric:
            return (-modified_cp[0], -modified_cp[1])
        if other_cp is None:
            return (-modified_cp[0], -modified_cp[1])
        other_length = math.sqrt(other_cp[0] ** 2 + other_cp[1] ** 2)
        if other_length < 1e-10:
            return (-modified_cp[0], -modified_cp[1])
        modified_length = math.sqrt(modified_cp[0] ** 2 + modified_cp[1] ** 2)
        if modified_length < 1e-10:
            return (0.0, 0.0)
        direction = (
            -modified_cp[0] / modified_length,
            -modified_cp[1] / modified_length,
        )
        return (
            direction[0] * other_length,
            direction[1] * other_length,
        )

    def enforce_constraint(
        self,
        registry: EntityRegistry,
        sketch: Sketch | None = None,
    ) -> None:
        if self.is_sharp():
            return

        beziers = self.get_connected_beziers(registry, sketch)
        if len(beziers) < 2:
            return

        point_ids = {self.id}
        if sketch is not None:
            point_ids.update(sketch.get_coincident_points(self.id))

        cp_data = []
        for b in beziers:
            if b.start_idx in point_ids and b.cp1 is not None:
                cp_data.append((b, "cp1", b.cp1))
            elif b.end_idx in point_ids and b.cp2 is not None:
                cp_data.append((b, "cp2", b.cp2))

        if len(cp_data) < 2:
            return

        if self.is_symmetric():
            avg_length = sum(
                math.sqrt(cp[0] ** 2 + cp[1] ** 2) for _, _, cp in cp_data
            ) / len(cp_data)
            if avg_length < 1e-10:
                return

            sum_dx = 0.0
            sum_dy = 0.0
            for _, attr, cp in cp_data:
                sign = 1.0 if attr == "cp1" else -1.0
                sum_dx += sign * cp[0]
                sum_dy += sign * cp[1]
            length = math.sqrt(sum_dx**2 + sum_dy**2)
            if length < 1e-10:
                return
            direction = (sum_dx / length, sum_dy / length)

            for b, attr, _ in cp_data:
                sign = 1.0 if attr == "cp1" else -1.0
                setattr(
                    b,
                    attr,
                    (
                        sign * direction[0] * avg_length,
                        sign * direction[1] * avg_length,
                    ),
                )


class Point(SketchPointOps):
    """The sketcher point: compiled state plus waypoint propagation."""

    def __new__(
        cls,
        id,
        x,
        y,
        fixed=False,
        waypoint_type: WaypointType | str = "sharp",
    ) -> Self:
        point = cast(Self, super().__new__(cls, id, x, y, fixed))
        if waypoint_type != WaypointType.SHARP:
            point.waypoint_type = (
                waypoint_type.value
                if isinstance(waypoint_type, WaypointType)
                else waypoint_type
            )
        return point

    def __init__(self, *args, **kwargs):
        # State is fully initialized by the compiled constructor and
        # __new__; the reference __init__ has no further work.
        pass

    @property
    def waypoint_type(self) -> WaypointType:
        value = SketchPoint.waypoint_type.__get__(self)
        return WaypointType(value)

    @waypoint_type.setter
    def waypoint_type(self, value) -> None:
        if isinstance(value, WaypointType):
            value = value.value
        SketchPoint.waypoint_type.__set__(self, value)

    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Point:
        point = cls(
            data["id"],
            data["x"],
            data["y"],
            data.get("fixed", False),
        )
        waypoint_type = data.get("waypoint_type", "sharp")
        if waypoint_type != "sharp":
            point.waypoint_type = waypoint_type
        return point
