import math
from typing import List, Tuple, Dict, Optional, Any, Sequence, TYPE_CHECKING
from ...geo import primitives
from .entity import Entity

if TYPE_CHECKING:
    from .registry import EntityRegistry
    from ..constraints import Constraint


class Arc(Entity):
    def __init__(
        self,
        id: int,
        start_idx: int,
        end_idx: int,
        center_idx: int,
        clockwise: bool = False,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.center_idx = center_idx
        self.clockwise = clockwise
        self.type = "arc"

    def get_point_ids(self) -> List[int]:
        return [self.start_idx, self.end_idx, self.center_idx]

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        s = registry.get_point(self.start_idx)
        e = registry.get_point(self.end_idx)
        c = registry.get_point(self.center_idx)
        self.constrained = s.constrained and e.constrained and c.constrained

    def _get_bbox(
        self, registry: "EntityRegistry"
    ) -> Tuple[float, float, float, float]:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)

        # Reuse core primitive utility for exact arc bounding box
        # Note: primitive expects center_offset relative to start, so:
        # center = start + offset. Here center is absolute.
        # offset = center - start.
        return primitives.get_arc_bounding_box(
            start.pos(),
            end.pos(),
            (center.x - start.x, center.y - start.y),
            self.clockwise,
        )

    def is_contained_by(
        self,
        rect: Tuple[float, float, float, float],
        registry: "EntityRegistry",
    ) -> bool:
        # For an arc to be strictly inside, its entire bounding box must be
        # inside
        arc_box = self._get_bbox(registry)
        return primitives.rect_a_contains_rect_b(rect, arc_box)

    def intersects_rect(
        self,
        rect: Tuple[float, float, float, float],
        registry: "EntityRegistry",
    ) -> bool:
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)
        return primitives.arc_intersects_rect(
            start.pos(), end.pos(), center.pos(), self.clockwise, rect
        )

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Arc":
        """Deserializes a dictionary into an Arc instance."""
        return cls(
            id=data["id"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            center_idx=data["center_idx"],
            clockwise=data.get("clockwise", False),
            construction=data.get("construction", False),
        )

    def get_midpoint(
        self, registry: "EntityRegistry"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the midpoint coordinates along the arc's circumference.
        """
        start = registry.get_point(self.start_idx)
        end = registry.get_point(self.end_idx)
        center = registry.get_point(self.center_idx)
        if not (start and end and center):
            return None
        return primitives.get_arc_midpoint(
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

        return primitives.is_angle_between(
            angle, start_angle, end_angle, self.clockwise
        )

    def __repr__(self) -> str:
        return (
            f"Arc(id={self.id}, start={self.start_idx}, end={self.end_idx}, "
            f"center={self.center_idx}, cw={self.clockwise}, "
            f"construction={self.construction})"
        )
