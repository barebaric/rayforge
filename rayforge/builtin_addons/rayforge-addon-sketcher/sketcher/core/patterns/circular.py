from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..constraints import RadiusConstraint
from ..constraints.rotational import RotationalConstraint
from ..entities import Circle, Point
from .base import InstancePlacement, PatternStrategy, PlacementKind

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity


class CircularPatternStrategy(PatternStrategy):
    """
    Distributes copies along a circular arc around a center point.

    Copies are parametrically linked to the template member through
    RotationalConstraints: editing any member updates the whole pattern,
    while deleting a member only removes its own constraints. The
    construction guide circle acts as the master that carries the
    pattern definition and can be double-clicked to edit the array.
    """

    needs_center_point = True

    def calculate_placements(
        self, seed_center: tuple[float, float]
    ) -> list[InstancePlacement]:
        count = max(int(self.params.count), 1)
        total = math.radians(self.params.total_angle_deg)
        cx, cy = self.params.center

        placements: list[InstancePlacement] = []
        for j in range(1, count):
            theta = total * j / count
            if self.params.rotate_copies:
                placements.append(
                    InstancePlacement(
                        kind=PlacementKind.ROTATION,
                        angle=theta,
                        center=(cx, cy),
                    )
                )
            else:
                vx = seed_center[0] - cx
                vy = seed_center[1] - cy
                nx = cx + math.cos(theta) * vx - math.sin(theta) * vy
                ny = cy + math.sin(theta) * vx + math.cos(theta) * vy
                placements.append(
                    InstancePlacement(
                        kind=PlacementKind.TRANSLATION,
                        delta=(
                            nx - seed_center[0],
                            ny - seed_center[1],
                        ),
                    )
                )
        return placements

    def create_master_geometry(
        self,
        center_pid: int | None,
        radius_pt_pid: int | None,
    ) -> tuple[list[Point], list[Entity], list[Constraint]]:
        if center_pid is None or radius_pt_pid is None:
            return [], [], []
        if self.params.radius <= 0.0:
            return [], [], []

        circle_temp_id = -(abs(radius_pt_pid) + 1)
        guide_circle = Circle(
            circle_temp_id,
            center_idx=center_pid,
            radius_pt_idx=radius_pt_pid,
            construction=True,
        )

        constraints: list[Constraint] = [
            RadiusConstraint(circle_temp_id, value=self.params.radius)
        ]

        radius_point = Point(
            radius_pt_pid,
            self.params.center[0] + self.params.radius,
            self.params.center[1],
        )
        return [radius_point], [guide_circle], constraints

    def build_linkage_constraints(
        self,
        instances: list[tuple[int, dict[int, int]]],
        center_pid: int | None,
    ) -> list[Constraint]:
        if not self.params.rotate_copies or center_pid is None:
            return []

        count = max(int(self.params.count), 1)
        total = math.radians(self.params.total_angle_deg)
        step = total / count

        constraints: list[Constraint] = []
        for slot, mapping in instances:
            for src_pid, copy_pid in mapping.items():
                constraints.append(
                    RotationalConstraint(
                        center=center_pid,
                        p1=src_pid,
                        p2=copy_pid,
                        value=step * slot,
                    )
                )
        return constraints
