from typing import Dict, Any, List, Sequence, TYPE_CHECKING, Optional, Tuple
from ...geo import Geometry, Point as GeoPoint, Polygon, Rect, primitives
from ...geo.primitives import find_closest_point_on_line_segment
from ..types import EntityID
from .entity import Entity

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class Bezier(Entity):
    def __init__(
        self,
        id: EntityID,
        start_idx: EntityID,
        end_idx: EntityID,
        construction: bool = False,
        cp1: Optional[GeoPoint] = None,
        cp2: Optional[GeoPoint] = None,
    ):
        super().__init__(id, construction)
        self.start_idx: EntityID = start_idx
        self.end_idx: EntityID = end_idx
        self.type = "bezier"
        self.cp1 = cp1
        self.cp2 = cp2

    def get_control_points(
        self, registry: "EntityRegistry"
    ) -> Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float]
    ]:
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
    ) -> Tuple[float, float, float, float]:
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
        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points(registry)
        return cp1_x is None and cp2_x is None

    def get_point_ids(self) -> List[EntityID]:
        return [self.start_idx, self.end_idx]

    def get_endpoint_ids(self) -> List[EntityID]:
        return [self.start_idx, self.end_idx]

    def get_junction_point_ids(self) -> List[EntityID]:
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
            _, _, dist_sq = find_closest_point_on_line_segment(
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
            _, _, dist_sq = find_closest_point_on_line_segment(
                points[i], points[i + 1], mx, my
            )
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

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
    ) -> List[tuple]:
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
        return primitives.rect_a_contains_rect_b(rect, bezier_box)

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
            return primitives.line_segment_intersects_rect(
                start.pos(), end.pos(), rect
            )

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        points = self._sample_bezier(
            start.x, start.y, cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y, 20
        )

        for i in range(len(points) - 1):
            if primitives.line_segment_intersects_rect(
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
            if forward:
                return [(end.x, end.y)]
            else:
                return [(start.x, start.y)]

        cp1_x, cp1_y, cp2_x, cp2_y = self.get_control_points_or_endpoints(
            registry
        )
        points = self._sample_bezier(
            start.x, start.y, cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y, 20
        )
        if not forward:
            points = list(reversed(points))
        return points

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Bezier":
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
