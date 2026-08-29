import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry
from raygeo.geo.shape.arc import (
    does_arc_intersect_rect,
    get_arc_bounds,
    get_arc_midpoint,
    is_angle_between,
)
from raygeo.geo.shape.rect import does_rect_contain_rect
from raygeo.geo.types import Point, Polygon, Rect

from ..types import EntityID
from .entity import Entity

if TYPE_CHECKING:
    from ..commands.mirror import MirrorAxis
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class Arc(Entity):
    def __init__(
        self,
        id: EntityID,
        start_idx: EntityID,
        end_idx: EntityID,
        center_idx: EntityID,
        clockwise: bool = False,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.start_idx: EntityID = start_idx
        self.end_idx: EntityID = end_idx
        self.center_idx: EntityID = center_idx
        self.clockwise = clockwise
        self.type = "arc"

    def get_state(self) -> dict[str, Any] | None:
        state = super().get_state() or {}
        state["clockwise"] = self.clockwise
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        if "clockwise" in state:
            self.clockwise = state["clockwise"]

    def mirror(self, axis: "MirrorAxis") -> None:
        # Reflection reverses chirality: a clockwise arc becomes CCW and
        # vice versa. Point positions are mirrored by the command.
        self.clockwise = not self.clockwise

    def geometry_signature(self, registry: "EntityRegistry") -> tuple:
        """Extends the point signature with the arc's chirality, which
        changes the shape without moving any defining point."""
        return (*super().geometry_signature(registry), self.clockwise)

    def get_point_ids(self) -> list[EntityID]:
        return [self.start_idx, self.end_idx, self.center_idx]

    def get_endpoint_ids(self) -> list[EntityID]:
        return [self.start_idx, self.end_idx]

    def get_junction_point_ids(self) -> list[EntityID]:
        return [self.start_idx, self.end_idx, self.center_idx]

    def get_drag_anchor_points(self, point_id: EntityID) -> list[EntityID]:
        if point_id in (self.start_idx, self.end_idx):
            return [self.center_idx]
        return []

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        center = registry.get_point(self.center_idx)
        start = registry.get_point(self.start_idx)
        if not (center and start):
            return False

        radius = math.hypot(start.x - center.x, start.y - center.y)
        if radius == 0.0:
            return False

        dist_mouse = math.hypot(mx - center.x, my - center.y)
        if abs(dist_mouse - radius) >= threshold:
            return False

        angle_mouse = math.atan2(my - center.y, mx - center.x)
        return self.is_angle_within_sweep(angle_mouse, registry)

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        s = registry.get_point(self.start_idx)
        e = registry.get_point(self.end_idx)
        c = registry.get_point(self.center_idx)
        self.constrained = s.constrained and e.constrained and c.constrained

    def _get_bbox(self, registry: "EntityRegistry") -> Rect:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)

        # Reuse core primitive utility for exact arc bounding box
        # Note: primitive expects center_offset relative to start, so:
        # center = start + offset. Here center is absolute.
        # offset = center - start.
        return get_arc_bounds(
            start.pos(),
            end.pos(),
            (center.x - start.x, center.y - start.y),
            self.clockwise,
        )

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        # For an arc to be strictly inside, its entire bounding box must be
        # inside
        arc_box = self._get_bbox(registry)
        return does_rect_contain_rect(rect, arc_box)

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)
        return does_arc_intersect_rect(
            start.pos(), end.pos(), center.pos(), self.clockwise, rect
        )

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        """Converts the arc to a Geometry object."""
        geo = Geometry()
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)
        i = center.x - start.x
        j = center.y - start.y
        geo.move_to(start.x, start.y)
        geo.arc_to(end.x, end.y, i, j, clockwise=self.clockwise)
        return geo

    def append_to_geometry(
        self,
        geo: Geometry,
        registry: "EntityRegistry",
        forward: bool,
    ) -> None:
        """Appends this arc to an existing geometry object."""
        arc_start_pt = registry.get_point(self.start_idx)
        arc_end_pt = registry.get_point(self.end_idx)
        center_pt = registry.get_point(self.center_idx)

        target_pt = arc_end_pt if forward else arc_start_pt
        current_pt = arc_start_pt if forward else arc_end_pt

        offset_x = center_pt.x - current_pt.x
        offset_y = center_pt.y - current_pt.y

        is_cw = self.clockwise if forward else not self.clockwise

        geo.arc_to(
            target_pt.x,
            target_pt.y,
            offset_x,
            offset_y,
            clockwise=is_cw,
        )

    def to_polygon_vertices(
        self,
        registry: "EntityRegistry",
        forward: bool,
    ) -> Polygon:
        start_pt = registry.get_point(self.start_idx)
        end_pt = registry.get_point(self.end_idx)
        center_pt = registry.get_point(self.center_idx)
        if not (start_pt and end_pt and center_pt):
            return []

        radius = math.hypot(start_pt.x - center_pt.x, start_pt.y - center_pt.y)
        start_a = math.atan2(
            start_pt.y - center_pt.y, start_pt.x - center_pt.x
        )
        end_a = math.atan2(end_pt.y - center_pt.y, end_pt.x - center_pt.x)

        vertices: Polygon = []
        # Scale the sample count by the swept angle so short arcs stay
        # coarse and ~360° arcs get enough resolution.
        num_segments = 16
        max_segments = 64

        if forward:
            if self.clockwise:
                if end_a > start_a:
                    end_a -= 2 * math.pi
            else:
                if end_a < start_a:
                    end_a += 2 * math.pi
            sweep = abs(end_a - start_a)
            num_segments = max(
                2, min(max_segments, int(num_segments * sweep / math.pi))
            )
            for i in range(num_segments + 1):
                t = i / num_segments
                a = start_a + t * (end_a - start_a)
                px = center_pt.x + radius * math.cos(a)
                py = center_pt.y + radius * math.sin(a)
                vertices.append((px, py))
        else:
            if self.clockwise:
                if start_a > end_a:
                    start_a -= 2 * math.pi
            else:
                if start_a < end_a:
                    start_a += 2 * math.pi
            sweep = abs(start_a - end_a)
            num_segments = max(
                2, min(max_segments, int(num_segments * sweep / math.pi))
            )
            for i in range(num_segments + 1):
                t = i / num_segments
                a = end_a + t * (start_a - end_a)
                px = center_pt.x + radius * math.cos(a)
                py = center_pt.y + radius * math.sin(a)
                vertices.append((px, py))

        return vertices

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Arc to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "start_idx": self.start_idx,
                "end_idx": self.end_idx,
                "center_idx": self.center_idx,
                "clockwise": self.clockwise,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Arc":
        """Deserializes a dictionary into an Arc instance."""
        return cls(
            id=data["id"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            center_idx=data["center_idx"],
            clockwise=data.get("clockwise", False),
            construction=data.get("construction", False),
        )

    def get_midpoint(self, registry: "EntityRegistry") -> Point | None:
        """
        Calculates the midpoint coordinates along the arc's circumference.
        """
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)
        if not (start and end and center):
            return None
        return get_arc_midpoint(
            start.pos(), end.pos(), center.pos(), self.clockwise
        )

    def is_angle_within_sweep(
        self, angle: float, registry: "EntityRegistry"
    ) -> bool:
        """Checks if a given angle is within the arc's sweep."""
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)
        if not (start and end and center):
            return False

        start_angle = math.atan2(start.y - center.y, start.x - center.x)
        end_angle = math.atan2(end.y - center.y, end.x - center.x)

        return is_angle_between(angle, start_angle, end_angle, self.clockwise)

    def __repr__(self) -> str:
        return (
            f"Arc(id={self.id}, start={self.start_idx}, end={self.end_idx}, "
            f"center={self.center_idx}, cw={self.clockwise}, "
            f"construction={self.construction})"
        )
