import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry
from raygeo.geo.shape.bezier import get_bezier_closest_point
from raygeo.geo.shape.line import (
    does_line_segment_intersect_rect,
    get_line_segment_closest_point,
)
from raygeo.geo.shape.rect import does_rect_contain_rect
from raygeo.geo.types import Point as GeoPoint
from raygeo.geo.types import Polygon, Rect

from ..types import EntityID
from .entity import Entity, quantize

if TYPE_CHECKING:
    from ..commands.mirror import MirrorAxis
    from ..constraints import Constraint
    from ..entity_group import PlacementTransform
    from ..registry import EntityRegistry
    from .point import Point


class Bezier(Entity):
    def __init__(
        self,
        id: EntityID,
        start_idx: EntityID,
        end_idx: EntityID,
        construction: bool = False,
        cp1: GeoPoint | None = None,
        cp2: GeoPoint | None = None,
    ):
        super().__init__(id, construction)
        self.start_idx: EntityID = start_idx
        self.end_idx: EntityID = end_idx
        self.type = "bezier"
        self.cp1 = cp1
        self.cp2 = cp2

    def get_state(self) -> dict[str, Any] | None:
        state = super().get_state() or {}
        state["cp1"] = self.cp1
        state["cp2"] = self.cp2
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        if "cp1" in state:
            self.cp1 = state["cp1"]
        if "cp2" in state:
            self.cp2 = state["cp2"]

    def mirror(self, axis: "MirrorAxis") -> None:
        # Control points are stored as (dx, dy) offsets relative to the
        # start/end points. Since the endpoints are mirrored by the command,
        # we must mirror the deltas too so the curve bulges correctly.
        if self.cp1 is not None:
            self.cp1 = axis.flip_offset(self.cp1)
        if self.cp2 is not None:
            self.cp2 = axis.flip_offset(self.cp2)

    def transform_offsets(self, placement: "PlacementTransform") -> None:
        # Control points are stored as (dx, dy) offsets relative to
        # the start/end points; transform them with the placement so
        # the curve follows translation and rotation placements.
        if self.cp1 is not None:
            self.cp1 = placement.transform_offset(*self.cp1)
        if self.cp2 is not None:
            self.cp2 = placement.transform_offset(*self.cp2)

    def rewrite_offsets_from(
        self, template: "Entity", placement: "PlacementTransform"
    ) -> None:
        # The copy's own offsets may lag behind (its endpoints moved
        # independently); take the template's, transformed like its
        # points.
        if not isinstance(template, Bezier):
            return
        if template.cp1 is not None:
            self.cp1 = placement.transform_offset(*template.cp1)
        if template.cp2 is not None:
            self.cp2 = placement.transform_offset(*template.cp2)

    def geometry_signature(self, registry: "EntityRegistry") -> tuple:
        """Extends the point signature with the quantized
        control-point offsets, which shape the curve without moving
        any defining point."""
        return (
            *super().geometry_signature(registry),
            (
                (quantize(self.cp1[0]), quantize(self.cp1[1]))
                if self.cp1 is not None
                else None
            ),
            (
                (quantize(self.cp2[0]), quantize(self.cp2[1]))
                if self.cp2 is not None
                else None
            ),
        )

    def get_control_points(
        self, registry: "EntityRegistry"
    ) -> tuple[float | None, float | None, float | None, float | None]:
        cp1_x, cp1_y = None, None
        cp2_x, cp2_y = None, None
        if self.cp1 is not None:
            start = registry.get_point(self.start_idx)
            if start:
                cp1_x = start.x + self.cp1[0]
                cp1_y = start.y + self.cp1[1]
        if self.cp2 is not None:
            end = registry.get_point(self.end_idx)
            if end:
                cp2_x = end.x + self.cp2[0]
                cp2_y = end.y + self.cp2[1]
        return cp1_x, cp1_y, cp2_x, cp2_y

    def get_control_points_or_endpoints(
        self, registry: "EntityRegistry"
    ) -> tuple[float, float, float, float]:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        cp1_x_opt, cp1_y_opt, cp2_x_opt, cp2_y_opt = self.get_control_points(
            registry
        )
        cp1_x: float = cp1_x_opt if cp1_x_opt is not None else start.x
        cp1_y: float = cp1_y_opt if cp1_y_opt is not None else start.y
        cp2_x: float = cp2_x_opt if cp2_x_opt is not None else end.x
        cp2_y: float = cp2_y_opt if cp2_y_opt is not None else end.y
        return cp1_x, cp1_y, cp2_x, cp2_y

    def is_line(self, registry: "EntityRegistry") -> bool:
        cp1_x, _cp1_y, cp2_x, _cp2_y = self.get_control_points(registry)
        return cp1_x is None and cp2_x is None

    def get_point_ids(self) -> list[EntityID]:
        return [self.start_idx, self.end_idx]

    def get_endpoint_ids(self) -> list[EntityID]:
        return [self.start_idx, self.end_idx]

    def is_edge_entity(self) -> bool:
        return True

    def characteristic_length_pairs(
        self,
    ) -> list[tuple[EntityID, EntityID]]:
        return [(self.start_idx, self.end_idx)]

    def tangent_at(
        self, registry: "EntityRegistry", point_id: EntityID
    ) -> tuple[float, float]:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return (1.0, 0.0)
        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        if point_id == start.id:
            return (cp1_x - start.x, cp1_y - start.y)
        return (end.x - cp2_x, end.y - cp2_y)

    def signed_distance_to(
        self, point: "Point", registry: "EntityRegistry"
    ) -> float:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return 0.0

        if self.is_line(registry):
            _, _, dist_sq = get_line_segment_closest_point(
                (start.x, start.y),
                (end.x, end.y),
                point.x,
                point.y,
            )
            return math.sqrt(dist_sq)

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        start_x, start_y = start.x, start.y
        end_x, end_y = end.x, end.y
        return self._closest_point_dist(
            start_x, start_y, cp1_x, cp1_y, cp2_x, cp2_y, end_x, end_y, point
        )

    @staticmethod
    def _closest_point_dist(
        start_x: float,
        start_y: float,
        cp1_x: float,
        cp1_y: float,
        cp2_x: float,
        cp2_y: float,
        end_x: float,
        end_y: float,
        point: "Point",
    ) -> float:
        result = get_bezier_closest_point(
            (start_x, start_y, 0.0),
            (cp1_x, cp1_y, 0.0),
            (cp2_x, cp2_y, 0.0),
            (end_x, end_y, 0.0),
            point.x,
            point.y,
        )
        if result is None:
            return 0.0
        _t, _pt, dist_sq = result
        return math.sqrt(dist_sq)

    def get_junction_point_ids(self) -> list[EntityID]:
        return [self.start_idx, self.end_idx]

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return False

        if self.is_line(registry):
            _, _, dist_sq = get_line_segment_closest_point(
                (start.x, start.y), (end.x, end.y), mx, my
            )
            return dist_sq < threshold**2

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        points = self._sample_bezier(
            start.x, start.y, cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y, 20
        )

        min_dist_sq = float("inf")
        for i in range(len(points) - 1):
            _, _, dist_sq = get_line_segment_closest_point(
                points[i], points[i + 1], mx, my
            )
            min_dist_sq = min(min_dist_sq, dist_sq)

        return min_dist_sq < threshold**2

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        self.constrained = start.constrained and end.constrained

    def _get_bbox(self, registry: "EntityRegistry") -> Rect:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return (0.0, 0.0, 0.0, 0.0)

        if self.is_line(registry):
            min_x = min(start.x, end.x)
            max_x = max(start.x, end.x)
            min_y = min(start.y, end.y)
            max_y = max(start.y, end.y)
            return (min_x, min_y, max_x, max_y)

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        points = self._sample_bezier(
            start.x, start.y, cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y, 20
        )
        if not points:
            return (0.0, 0.0, 0.0, 0.0)

        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        return (min_x, min_y, max_x, max_y)

    def _sample_bezier(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        num_samples: int,
    ) -> list[tuple]:
        points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            t2 = t * t
            t3 = t2 * t

            x = mt3 * x0 + 3 * mt2 * t * x1 + 3 * mt * t2 * x2 + t3 * x3
            y = mt3 * y0 + 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t3 * y3
            points.append((x, y))
        return points

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        bezier_box = self._get_bbox(registry)
        return does_rect_contain_rect(rect, bezier_box)

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return False

        if self.is_line(registry):
            return does_line_segment_intersect_rect(
                start.pos(), end.pos(), rect
            )

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        points = self._sample_bezier(
            start.x, start.y, cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y, 20
        )

        for i in range(len(points) - 1):
            if does_line_segment_intersect_rect(
                points[i], points[i + 1], rect
            ):
                return True

        min_x, min_y, max_x, max_y = rect
        for px, py in points:
            if min_x <= px <= max_x and min_y <= py <= max_y:
                return True

        return False

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        geo = Geometry()
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return geo

        geo.move_to(start.x, start.y)

        if self.is_line(registry):
            geo.line_to(end.x, end.y)
        else:
            cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
                registry
            )
            geo.bezier_to(end.x, end.y, cp1_x, cp1_y, cp2_x, cp2_y)
        return geo

    def append_to_geometry(
        self,
        geo: Geometry,
        registry: "EntityRegistry",
        forward: bool,
    ) -> None:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return

        if self.is_line(registry):
            if forward:
                geo.line_to(end.x, end.y)
            else:
                geo.line_to(start.x, start.y)
        else:
            cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
                registry
            )
            if forward:
                geo.bezier_to(end.x, end.y, cp1_x, cp1_y, cp2_x, cp2_y)
            else:
                geo.bezier_to(start.x, start.y, cp2_x, cp2_y, cp1_x, cp1_y)

    def to_polygon_vertices(
        self,
        registry: "EntityRegistry",
        forward: bool,
    ) -> Polygon:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        if not (start and end):
            return []

        if self.is_line(registry):
            start_pid = self.start_idx if forward else self.end_idx
            p = registry.get_point(start_pid)
            return [(p.x, p.y)]

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        points = self._sample_bezier(
            start.x, start.y, cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y, 20
        )
        if not forward:
            points = list(reversed(points))
        return points

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "start_idx": self.start_idx,
                "end_idx": self.end_idx,
            }
        )
        if self.cp1 is not None:
            data["cp1_dx"] = self.cp1[0]
            data["cp1_dy"] = self.cp1[1]
        if self.cp2 is not None:
            data["cp2_dx"] = self.cp2[0]
            data["cp2_dy"] = self.cp2[1]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bezier":
        cp1 = None
        if "cp1_dx" in data and "cp1_dy" in data:
            cp1 = (data["cp1_dx"], data["cp1_dy"])
        cp2 = None
        if "cp2_dx" in data and "cp2_dy" in data:
            cp2 = (data["cp2_dx"], data["cp2_dy"])
        return cls(
            id=data["id"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            construction=data.get("construction", False),
            cp1=cp1,
            cp2=cp2,
        )

    def __repr__(self) -> str:
        return (
            f"Bezier(id={self.id}, start={self.start_idx}, "
            f"end={self.end_idx}, construction={self.construction}, "
            f"cp1={self.cp1}, cp2={self.cp2})"
        )
