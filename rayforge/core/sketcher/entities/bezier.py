from typing import Dict, Any, List, Sequence, TYPE_CHECKING
from ...geo import Geometry, Rect, primitives
from ...geo.primitives import find_closest_point_on_line_segment
from .entity import Entity

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class Bezier(Entity):
    def __init__(
        self,
        id: int,
        start_idx: int,
        cp1_idx: int,
        cp2_idx: int,
        end_idx: int,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.start_idx = start_idx
        self.cp1_idx = cp1_idx
        self.cp2_idx = cp2_idx
        self.end_idx = end_idx
        self.type = "bezier"

    def get_point_ids(self) -> List[int]:
        return [self.start_idx, self.cp1_idx, self.cp2_idx, self.end_idx]

    def get_junction_point_ids(self) -> List[int]:
        return [self.start_idx, self.cp1_idx, self.cp2_idx, self.end_idx]

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        start = registry.get_point(self.start_idx)
        cp1 = registry.get_point(self.cp1_idx)
        cp2 = registry.get_point(self.cp2_idx)
        end = registry.get_point(self.end_idx)
        if not (start and cp1 and cp2 and end):
            return False

        points = self._sample_bezier(
            start.x,
            start.y,
            cp1.x,
            cp1.y,
            cp2.x,
            cp2.y,
            end.x,
            end.y,
            20,
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
        cp1 = registry.get_point(self.cp1_idx)
        cp2 = registry.get_point(self.cp2_idx)
        end = registry.get_point(self.end_idx)
        self.constrained = (
            start.constrained
            and cp1.constrained
            and cp2.constrained
            and end.constrained
        )

    def _get_bbox(self, registry: "EntityRegistry") -> Rect:
        start = registry.get_point(self.start_idx)
        cp1 = registry.get_point(self.cp1_idx)
        cp2 = registry.get_point(self.cp2_idx)
        end = registry.get_point(self.end_idx)

        points = self._sample_bezier(
            start.x, start.y, cp1.x, cp1.y, cp2.x, cp2.y, end.x, end.y, 20
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
        cp1 = registry.get_point(self.cp1_idx)
        cp2 = registry.get_point(self.cp2_idx)
        end = registry.get_point(self.end_idx)

        points = self._sample_bezier(
            start.x,
            start.y,
            cp1.x,
            cp1.y,
            cp2.x,
            cp2.y,
            end.x,
            end.y,
            20,
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
        cp1 = registry.get_point(self.cp1_idx)
        cp2 = registry.get_point(self.cp2_idx)
        end = registry.get_point(self.end_idx)
        geo.move_to(start.x, start.y)
        geo.bezier_to(end.x, end.y, cp1.x, cp1.y, cp2.x, cp2.y)
        return geo

    def append_to_geometry(
        self,
        geo: Geometry,
        registry: "EntityRegistry",
        forward: bool,
    ) -> None:
        start = registry.get_point(self.start_idx)
        cp1 = registry.get_point(self.cp1_idx)
        cp2 = registry.get_point(self.cp2_idx)
        end = registry.get_point(self.end_idx)

        if forward:
            geo.bezier_to(end.x, end.y, cp1.x, cp1.y, cp2.x, cp2.y)
        else:
            geo.bezier_to(
                start.x,
                start.y,
                cp2.x,
                cp2.y,
                cp1.x,
                cp1.y,
            )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "start_idx": self.start_idx,
                "cp1_idx": self.cp1_idx,
                "cp2_idx": self.cp2_idx,
                "end_idx": self.end_idx,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bezier":
        return cls(
            id=data["id"],
            start_idx=data["start_idx"],
            cp1_idx=data["cp1_idx"],
            cp2_idx=data["cp2_idx"],
            end_idx=data["end_idx"],
            construction=data.get("construction", False),
        )

    def __repr__(self) -> str:
        return (
            f"Bezier(id={self.id}, start={self.start_idx}, "
            f"cp1={self.cp1_idx}, cp2={self.cp2_idx}, end={self.end_idx}, "
            f"construction={self.construction})"
        )
