from typing import List, Tuple, Dict, Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class Entity:
    """Base class for geometric primitives."""

    def __init__(self, id: int, construction: bool = False):
        self.id = id
        self.construction = construction
        self.type = "entity"
        # Constrained state is calculated by solver
        self.constrained = False

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        """
        Updates self.constrained based on the status of defining points
        and relevant constraints.
        """
        self.constrained = False

    def get_point_ids(self) -> List[int]:
        """Returns IDs of all control points used by this entity."""
        return []

    def get_ignorable_unconstrained_points(self) -> List[int]:
        """
        Returns IDs of points that can remain unconstrained if this entity
        is constrained (e.g. radius handles).
        """
        return []

    def is_contained_by(
        self,
        rect: Tuple[float, float, float, float],
        registry: "EntityRegistry",
    ) -> bool:
        """
        Returns True if the entity is fully strictly contained within the rect.
        Used for Window Selection.
        """
        return False

    def intersects_rect(
        self,
        rect: Tuple[float, float, float, float],
        registry: "EntityRegistry",
    ) -> bool:
        """
        Returns True if the entity intersects the rect or is contained by it.
        Used for Crossing Selection.
        """
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Base serialization method for entities."""
        return {
            "id": self.id,
            "type": self.type,
            "construction": self.construction,
        }

    def __repr__(self) -> str:
        return f"Entity(id={self.id}, type={self.type})"
