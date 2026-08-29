from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..constraints import RadiusConstraint
from ..entities import Circle, Point
from ..entity_group import EntityGroup
from .base import (
    Array,
    ArrayStrategy,
    InstancePlacement,
    PlacementKind,
)

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity
    from ..registry import EntityRegistry


class CircularArrayStrategy(ArrayStrategy):
    """
    Distributes copies around a center point on a constructed guide
    circle.

    Position 0 is the drawn position of the template itself (the
    identity placement): the guide circle is constructed around the
    template's center with the dialog's radius, and the copy of slot
    j is the template rotated by total_angle * j / count about the
    center. The copies are static baked geometry (no constraints,
    fixed points, outside the solver). The guide circle's radius
    constraint is the single source of truth for the orbit: the
    radius drives the member placement, never the other way around —
    editing the radius re-projects every member radially onto the
    new orbit (shape and angular position preserved), while editing
    the template never touches the circle.
    """

    needs_center_point = True

    def __init__(
        self,
        count: int = 6,
        rotate_copies: bool = True,
        total_angle_deg: float = 360.0,
        center: tuple[float, float] = (0.0, 0.0),
        radius: float = 0.0,
    ):
        self.count = count
        self.rotate_copies = rotate_copies
        self.total_angle_deg = total_angle_deg
        self.center = center
        self.radius = radius

    def template_placement(
        self,
        template_center: tuple[float, float],
        registry: Any | None = None,
    ) -> InstancePlacement:
        """Places the template onto the guide circle: a radial
        translation putting its center on the circle at the angle it
        was drawn at, shape preserved."""
        cx, cy = self.center
        vx = template_center[0] - cx
        vy = template_center[1] - cy
        d = math.hypot(vx, vy)
        if d < 1e-9 or self.radius <= 0.0:
            return InstancePlacement(
                kind=PlacementKind.TRANSLATION, delta=(0.0, 0.0)
            )
        k = self.radius / d
        return InstancePlacement(
            kind=PlacementKind.TRANSLATION,
            delta=(
                cx + vx * k - template_center[0],
                cy + vy * k - template_center[1],
            ),
        )

    def member_placements(
        self,
        template_center: tuple[float, float],
        registry: Any | None = None,
    ) -> list[InstancePlacement]:
        count = max(int(self.count), 1)
        total = math.radians(self.total_angle_deg)
        cx, cy = self.center

        placements: list[InstancePlacement] = []
        for j in range(1, count):
            theta = total * j / count
            if self.rotate_copies:
                placements.append(
                    InstancePlacement(
                        kind=PlacementKind.ROTATION,
                        angle=theta,
                        center=(cx, cy),
                    )
                )
            else:
                vx = template_center[0] - cx
                vy = template_center[1] - cy
                nx = cx + math.cos(theta) * vx - math.sin(theta) * vy
                ny = cy + math.sin(theta) * vx + math.cos(theta) * vy
                placements.append(
                    InstancePlacement(
                        kind=PlacementKind.TRANSLATION,
                        delta=(
                            nx - template_center[0],
                            ny - template_center[1],
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
        if self.radius <= 0.0:
            return [], [], []

        circle_temp_id = -(abs(radius_pt_pid) + 1)
        guide_circle = Circle(
            circle_temp_id,
            center_idx=center_pid,
            radius_pt_idx=radius_pt_pid,
            construction=True,
        )

        constraints: list[Constraint] = [
            RadiusConstraint(circle_temp_id, value=self.radius)
        ]

        radius_point = Point(
            radius_pt_pid,
            self.center[0] + self.radius,
            self.center[1],
        )
        return [radius_point], [guide_circle], constraints

    def capture_master_frame(
        self, registry: EntityRegistry, array_def: Array
    ) -> tuple[tuple[float, float], float] | None:
        circle = registry.get_entity(array_def.guide_circle_id)
        if not isinstance(circle, Circle):
            return None
        try:
            center = registry.get_point(circle.center_idx)
            radius_pt = registry.get_point(circle.radius_pt_idx)
        except IndexError:
            return None
        return (
            center.x,
            center.y,
        ), math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)

    def apply_frame(
        self,
        registry: EntityRegistry,
        array_def: Array,
        old_frame: tuple[tuple[float, float], float],
        constraints: list,
        frame_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Moves the array into the edited guide frame: translates the
        members to the new center and, when the radius changed,
        re-projects every member radially onto the new orbit. A
        member keeps its shape and its angular position — the radius
        drives the member placement, never the other way around.
        """
        (ocx, ocy), _old_radius = old_frame
        ncx, ncy = self.center

        dx, dy = ncx - ocx, ncy - ocy
        if abs(dx) > 1e-12 or abs(dy) > 1e-12:
            # Entities of the array share points (e.g. an ellipse's
            # helper lines are built on the ellipse's own points), so
            # the group translates unique points only: translating per
            # reference would move shared points once per referencing
            # entity and tear the members apart.
            whole = EntityGroup(
                registry, array_def.living_entity_ids(registry)
            )
            whole.translate(dx, dy)

        # The template center always sits on the guide circle (the
        # radius drives the member placement): re-projecting is
        # idempotent for on-circle members and pulls a member that a
        # drag moved off the circle back onto its orbit.
        self._reproject_members(registry, array_def, (ncx, ncy), self.radius)

        return self._pin_guide_circle(
            registry, array_def, constraints, frame_state
        )

    def _reproject_members(
        self,
        registry: EntityRegistry,
        array_def: Array,
        center: tuple[float, float],
        radius: float,
    ) -> None:
        """Translates every living member radially so its center sits
        on the circle of the given radius, shape and angle preserved."""
        ccx, ccy = center
        for _slot, eids in array_def.living_members(registry):
            EntityGroup(registry, eids).radial_project((ccx, ccy), radius)

    def _pin_guide_circle(
        self,
        registry: EntityRegistry,
        array_def: Array,
        constraints: list,
        frame_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Writes the master circle geometry and its radius constraint
        exactly onto the target frame."""
        circle = registry.get_entity(array_def.guide_circle_id)
        radius_constraint: RadiusConstraint | None = None
        old_radius_value: float | None = None
        if isinstance(circle, Circle):
            center_pt = registry.get_point(circle.center_idx)
            center_pt.x, center_pt.y = self.center
            radius_pt = registry.get_point(circle.radius_pt_idx)
            radius_pt.x = self.center[0] + self.radius
            radius_pt.y = self.center[1]

        for constr in constraints:
            if (
                isinstance(constr, RadiusConstraint)
                and constr.entity_id == array_def.guide_circle_id
            ):
                radius_constraint = constr
                if frame_state is not None:
                    old_radius_value = frame_state.get("old_radius_value")
                if old_radius_value is None:
                    old_radius_value = constr.value
                constr.value = self.radius
                break

        return {
            "old_radius_value": old_radius_value,
            "radius_constraint": radius_constraint,
        }


class CircularArray(Array):
    """
    Persistent definition of a circular (polar) array. Carries the
    mode-specific state: the sweep angle covered by the copies.
    """

    MODE = "circular"
    STRATEGY = CircularArrayStrategy

    def __init__(
        self,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]] | None = None,
        count: int = 6,
        total_angle_deg: float = 360.0,
        rotate_copies: bool = True,
    ):
        super().__init__(
            uid,
            guide_circle_id,
            members=members,
            count=count,
            rotate_copies=rotate_copies,
        )
        self.total_angle_deg = total_angle_deg

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["total_angle_deg"] = self.total_angle_deg
        return data

    def snapshot(self) -> dict[str, Any]:
        state = super().snapshot()
        state["total_angle_deg"] = self.total_angle_deg
        state["rotate_copies"] = self.rotate_copies
        return state

    def restore(self, state: dict[str, Any]) -> None:
        super().restore(state)
        self.total_angle_deg = state["total_angle_deg"]
        self.rotate_copies = state["rotate_copies"]

    def commit(self, strategy: ArrayStrategy) -> None:
        assert isinstance(strategy, CircularArrayStrategy)
        super().commit(strategy)
        self.total_angle_deg = strategy.total_angle_deg
        self.rotate_copies = strategy.rotate_copies

    def params_changed(self, strategy: ArrayStrategy) -> bool:
        assert isinstance(strategy, CircularArrayStrategy)
        return (
            super().params_changed(strategy)
            or strategy.rotate_copies != self.rotate_copies
            or strategy.total_angle_deg != self.total_angle_deg
        )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> CircularArray:
        return cls(
            uid=data["uid"],
            guide_circle_id=data["guide_circle_id"],
            members=cls._parse_members(data),
            count=data.get("count", 6),
            total_angle_deg=data.get("total_angle_deg", 360.0),
            rotate_copies=data.get("rotate_copies", True),
        )

    def make_strategy(self, registry: EntityRegistry) -> CircularArrayStrategy:
        """Reads the orbit's live geometry — the guide circle's center
        and radius — from the registry; the stored radius constraint
        is the single source of truth for the orbit."""
        circle = registry.get_entity(self.guide_circle_id)
        assert isinstance(circle, Circle)
        center = registry.get_point(circle.center_idx)
        radius_pt = registry.get_point(circle.radius_pt_idx)
        return CircularArrayStrategy(
            count=self.count,
            total_angle_deg=self.total_angle_deg,
            center=(center.x, center.y),
            radius=math.hypot(radius_pt.x - center.x, radius_pt.y - center.y),
            rotate_copies=self.rotate_copies,
        )

    @classmethod
    def _from_strategy(
        cls,
        strategy: ArrayStrategy,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]],
        count: int,
        template_anchor: tuple[tuple[float, float], float] | None = None,
    ) -> CircularArray:
        assert isinstance(strategy, CircularArrayStrategy)
        # Circular arrays hold the template on their guide via
        # constraints instead of a stored anchor.
        return cls(
            uid=uid,
            guide_circle_id=guide_circle_id,
            members=members,
            count=count,
            total_angle_deg=strategy.total_angle_deg,
            rotate_copies=strategy.rotate_copies,
        )
