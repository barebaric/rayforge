import math
from typing import List, Tuple, Dict, Optional, Any, Sequence, TYPE_CHECKING
from ...geo import primitives
from .entity import Entity

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class Circle(Entity):
    def __init__(
        self,
        id: int,
        center_idx: int,
        radius_pt_idx: int,
        construction: bool = False,
    ):
        super().__init__(id, construction)
        self.center_idx = center_idx
        self.radius_pt_idx = radius_pt_idx
        self.type = "circle"

    def get_point_ids(self) -> List[int]:
        return [self.center_idx, self.radius_pt_idx]

    def get_ignorable_unconstrained_points(self) -> List[int]:
        """
        If the circle is geometrically constrained, the radius point (which
        acts only as a handle for the radius value) does not need to be
        constrained rotationally.
        """
        if self.constrained:
            return [self.radius_pt_idx]
        return []

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        center_pt = registry.get_point(self.center_idx)
        radius_pt = registry.get_point(self.radius_pt_idx)

        # A circle's geometry is defined by its center and radius.
        center_is_constrained = center_pt.constrained

        # The radius is defined if:
        # 1. The radius point itself is fully constrained.
        # 2. Or, a constraint explicitly defines the radius.
        radius_is_defined = radius_pt.constrained
        if not radius_is_defined:
            for constr in constraints:
                if constr.constrains_radius(registry, self.id):
                    radius_is_defined = True
                    break

        self.constrained = center_is_constrained and radius_is_defined

    def is_contained_by(
        self,
        rect: Tuple[float, float, float, float],
        registry: "EntityRegistry",
    ) -> bool:
        center = registry.get_point(self.center_idx)
        radius_pt = registry.get_point(self.radius_pt_idx)
        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        return primitives.circle_is_contained_by_rect(
            center.pos(), radius, rect
        )

    def intersects_rect(
        self,
        rect: Tuple[float, float, float, float],
        registry: "EntityRegistry",
    ) -> bool:
        center = registry.get_point(self.center_idx)
        radius_pt = registry.get_point(self.radius_pt_idx)
        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)

        # A circle that is contained by a rect also intersects it.
        # The primitive for intersection seems to miss this case, so we check
        # for containment first.
        if primitives.circle_is_contained_by_rect(center.pos(), radius, rect):
            return True

        return primitives.circle_intersects_rect(center.pos(), radius, rect)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Circle to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "center_idx": self.center_idx,
                "radius_pt_idx": self.radius_pt_idx,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Circle":
        """Deserializes a dictionary into a Circle instance."""
        return cls(
            id=data["id"],
            center_idx=data["center_idx"],
            radius_pt_idx=data["radius_pt_idx"],
            construction=data.get("construction", False),
        )

    def get_midpoint(
        self, registry: "EntityRegistry"
    ) -> Optional[Tuple[float, float]]:
        """Returns a point on the circumference (the radius point)."""
        radius_pt = registry.get_point(self.radius_pt_idx)
        if not radius_pt:
            return None
        return radius_pt.pos()

    def __repr__(self) -> str:
        return (
            f"Circle(id={self.id}, center={self.center_idx}, "
            f"radius_pt={self.radius_pt_idx}, "
            f"construction={self.construction})"
        )
