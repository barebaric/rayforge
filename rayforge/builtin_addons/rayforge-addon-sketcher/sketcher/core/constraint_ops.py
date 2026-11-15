"""Rebuilding of raygeo constraint objects with remapped points.

Constraint objects are immutable snapshots: changing which points a
constraint references means constructing a new object of the same
type. This module carries the per-type constructor knowledge used by
commands that duplicate or re-own geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Constraint

from raygeo.sketch.constraints.angle import AngleConstraint
from raygeo.sketch.constraints.aspect_ratio import AspectRatioConstraint
from raygeo.sketch.constraints.coincident import CoincidentConstraint
from raygeo.sketch.constraints.collinear import CollinearConstraint
from raygeo.sketch.constraints.diameter import DiameterConstraint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.drag import DragConstraint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.constraints.equal_length import EqualLengthConstraint
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.parallelogram import ParallelogramConstraint
from raygeo.sketch.constraints.perpendicular import (
    PerpendicularConstraint,
    PerpendicularTarget,
)
from raygeo.sketch.constraints.point_on_curve import (
    PointOnCurveConstraint,
)
from raygeo.sketch.constraints.point_on_line import (
    PointOnLineConstraint,
    PointOnTarget,
)
from raygeo.sketch.constraints.radius import RadiusConstraint
from raygeo.sketch.constraints.rotational import RotationalConstraint
from raygeo.sketch.constraints.symmetry import (
    SymmetryConstraint,
    SymmetryTarget,
)
from raygeo.sketch.constraints.tangent import TangentConstraint
from raygeo.sketch.constraints.vertical import VerticalConstraint

from .constraint_info import get_expression


def rebuild(constraint: Constraint, pid_map: dict[int, int]) -> Constraint:
    """Returns a new constraint of the same type with its referenced
    point ids remapped through ``pid_map``. Value, expression, ratio
    and user-visibility are carried over."""
    mapped = _map(constraint, pid_map)
    _restore_state(constraint, mapped)
    return mapped


def _pid(pid: int, pid_map: dict[int, int]) -> int:
    return pid_map.get(pid, pid)


def _map(constraint: Constraint, pid_map: dict[int, int]) -> Constraint:
    if isinstance(constraint, (HorizontalConstraint, VerticalConstraint)):
        return type(constraint)(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, CoincidentConstraint):
        return CoincidentConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, CollinearConstraint):
        return CollinearConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            _pid(constraint.p3, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, DistanceConstraint):
        return DistanceConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            constraint.value,
            constraint.user_visible,
        )
    if isinstance(constraint, AngleConstraint):
        return AngleConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            _pid(constraint.p3, pid_map),
            _pid(constraint.p4, pid_map),
            constraint.value,
            constraint.e1_far,
            constraint.e2_far,
            constraint.user_visible,
        )
    if isinstance(constraint, AspectRatioConstraint):
        return AspectRatioConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            _pid(constraint.p3, pid_map),
            _pid(constraint.p4, pid_map),
            constraint.ratio,
            constraint.user_visible,
        )
    if isinstance(constraint, RadiusConstraint):
        return RadiusConstraint(
            _pid(constraint.center, pid_map),
            _pid(constraint.radius_point, pid_map),
            constraint.value,
            constraint.user_visible,
        )
    if isinstance(constraint, DiameterConstraint):
        return DiameterConstraint(
            _pid(constraint.center, pid_map),
            _pid(constraint.radius_point, pid_map),
            constraint.value,
            constraint.user_visible,
        )
    if isinstance(constraint, TangentConstraint):
        return TangentConstraint(
            _pid(constraint.l1, pid_map),
            _pid(constraint.l2, pid_map),
            _pid(constraint.center, pid_map),
            _pid(constraint.radius_point, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, EqualLengthConstraint):
        return EqualLengthConstraint(
            [
                [_pid(pid, pid_map) for pid in pair]
                for pair in constraint.pairs
            ],
            constraint.user_visible,
        )
    if isinstance(constraint, EqualDistanceConstraint):
        return EqualDistanceConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            _pid(constraint.p3, pid_map),
            _pid(constraint.p4, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, PointOnLineConstraint):
        return PointOnLineConstraint(
            _pid(constraint.point, pid_map),
            _map_point_on_target(constraint.target, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, PointOnCurveConstraint):
        return PointOnCurveConstraint(
            _pid(constraint.point, pid_map),
            _pid(constraint.start, pid_map),
            _pid(constraint.end, pid_map),
            tuple(constraint.cp1_offset),
            tuple(constraint.cp2_offset),
            constraint.is_line,
            constraint.user_visible,
        )
    if isinstance(constraint, SymmetryConstraint):
        return SymmetryConstraint(
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            _map_symmetry_target(constraint.target, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, PerpendicularConstraint):
        return PerpendicularConstraint(
            _map_perpendicular_target(constraint.target, pid_map),
            constraint.user_visible,
        )
    if isinstance(constraint, ParallelogramConstraint):
        return ParallelogramConstraint(
            _pid(constraint.p_origin, pid_map),
            _pid(constraint.p_width, pid_map),
            _pid(constraint.p_height, pid_map),
            _pid(constraint.p4, pid_map),
        )
    if isinstance(constraint, RotationalConstraint):
        return RotationalConstraint(
            _pid(constraint.center, pid_map),
            _pid(constraint.p1, pid_map),
            _pid(constraint.p2, pid_map),
            constraint.angle_rad,
            constraint.user_visible,
        )
    if isinstance(constraint, DragConstraint):
        return DragConstraint(
            _pid(constraint.point, pid_map),
            constraint.target_x,
            constraint.target_y,
            constraint.weight,
        )
    raise ValueError(f"Unsupported constraint type: {type(constraint)}")


def _map_point_on_target(
    target: PointOnTarget, pid_map: dict[int, int]
) -> PointOnTarget:
    if isinstance(target, PointOnTarget.Line):
        return PointOnTarget.Line(
            l1=_pid(target.l1, pid_map), l2=_pid(target.l2, pid_map)
        )
    return PointOnTarget.RadiusCurve(
        center=_pid(target.center, pid_map),
        radius_point=_pid(target.radius_point, pid_map),
    )


def _map_symmetry_target(
    target: SymmetryTarget, pid_map: dict[int, int]
) -> SymmetryTarget:
    if isinstance(target, SymmetryTarget.Point):
        return SymmetryTarget.Point(center=_pid(target.center, pid_map))
    return SymmetryTarget.Line(
        l1=_pid(target.l1, pid_map), l2=_pid(target.l2, pid_map)
    )


def _map_perpendicular_target(
    target: PerpendicularTarget, pid_map: dict[int, int]
) -> PerpendicularTarget:
    if isinstance(target, PerpendicularTarget.LineLine):
        return PerpendicularTarget.LineLine(
            a1=_pid(target.a1, pid_map),
            a2=_pid(target.a2, pid_map),
            b1=_pid(target.b1, pid_map),
            b2=_pid(target.b2, pid_map),
        )
    if isinstance(target, PerpendicularTarget.LineCenter):
        return PerpendicularTarget.LineCenter(
            l1=_pid(target.l1, pid_map),
            l2=_pid(target.l2, pid_map),
            center=_pid(target.center, pid_map),
        )
    return PerpendicularTarget.Circles(
        c1=_pid(target.c1, pid_map),
        r1_point=_pid(target.r1_point, pid_map),
        c2=_pid(target.c2, pid_map),
        r2_point=_pid(target.r2_point, pid_map),
    )


def _restore_state(constraint: Constraint, mapped: Constraint) -> None:
    """Carries expression and user-visibility onto the rebuilt
    constraint; status resets to its default (constraints are
    re-evaluated by the next solve)."""
    expression = get_expression(constraint)
    if expression and hasattr(mapped, "expression"):
        mapped.expression = expression
    if hasattr(mapped, "set_user_visible") and hasattr(
        constraint, "user_visible"
    ):
        mapped.set_user_visible(constraint.user_visible)
