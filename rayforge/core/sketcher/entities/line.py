from typing import List, Dict, Any, Sequence, TYPE_CHECKING
from ...geo import Geometry, Polygon, Rect
from ...geo.primitives import (
    find_closest_point_on_line_segment,
    line_segment_intersects_rect,
)
from ..types import EntityID
from .entity import Entity

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class Line(Entity):
    def __init__(
        self,
        id: EntityID,
        p1_idx: EntityID,
        p2_idx: EntityID,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.p1_idx: EntityID = p1_idx
        self.p2_idx: EntityID = p2_idx
        self.type = "line"

    def get_point_ids(self) -> List[EntityID]:
        return [self.p1_idx, self.p2_idx]

    def get_endpoint_ids(self) -> List[EntityID]:
        return [self.p1_idx, self.p2_idx]

    def get_junction_point_ids(self) -> List[EntityID]:
        return [self.p1_idx, self.p2_idx]

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        p1 = registry.get_point(self.p1_idx)
        p2 = registry.get_point(self.p2_idx)
        if not (p1 and p2):
            return False
        _, _, dist_sq = find_closest_point_on_line_segment(
            (p1.x, p1.y), (p2.x, p2.y), mx, my
        )
        return dist_sq < threshold**2

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        p1 = registry.get_point(self.p1_idx)
        p2 = registry.get_point(self.p2_idx)
        self.constrained = p1.constrained and p2.constrained

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        p1 = registry.get_point(self.p1_idx)
        p2 = registry.get_point(self.p2_idx)
        return p1.is_in_rect(rect) and p2.is_in_rect(rect)

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        p1 = registry.get_point(self.p1_idx)
        p2 = registry.get_point(self.p2_idx)
        return line_segment_intersects_rect(p1.pos(), p2.pos(), rect)

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        """Converts the line to a Geometry object."""
        geo = Geometry()
        p1 = registry.get_point(self.p1_idx)
        p2 = registry.get_point(self.p2_idx)
        geo.move_to(p1.x, p1.y)
        geo.line_to(p2.x, p2.y)
        return geo

    def append_to_geometry(
        self,
        geo: Geometry,
        registry: "EntityRegistry",
        forward: bool,
    ) -> None:
        """Appends this line to an existing geometry object."""
        p_ids = self.get_point_ids()
        end_pid = p_ids[1] if forward else p_ids[0]
        end_pt = registry.get_point(end_pid)
        geo.line_to(end_pt.x, end_pt.y)

    def to_polygon_vertices(
        self,
        registry: "EntityRegistry",
        forward: bool,
    ) -> Polygon:
        p_ids = self.get_endpoint_ids()
        start_pid = p_ids[0] if forward else p_ids[1]
        p = registry.get_point(start_pid)
        return [(p.x, p.y)]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Line to a dictionary."""
        data = super().to_dict()
        data.update({"p1_idx": self.p1_idx, "p2_idx": self.p2_idx})
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Line":
        """Deserializes a dictionary into a Line instance."""
        return cls(
            id=data["id"],
            p1_idx=data["p1_idx"],
            p2_idx=data["p2_idx"],
            construction=data.get("construction", False),
        )

    def __repr__(self) -> str:
        return (
            f"Line(id={self.id}, p1={self.p1_idx}, p2={self.p2_idx}, "
            f"construction={self.construction})"
        )
