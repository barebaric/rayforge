from __future__ import annotations
from enum import Enum, auto
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Callable,
    Optional,
    TYPE_CHECKING,
    Set,
)
from ...expression import safe_evaluate

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class ConstraintStatus(Enum):
    """Represents the validation status of a constraint."""

    VALID = auto()
    EXPRESSION_BASED = auto()
    ERROR = auto()


class Constraint:
    """Base class for all geometric constraints."""

    # These attributes are expected on dimensional constraints
    value: float = 0.0
    expression: Optional[str] = None
    status: ConstraintStatus = ConstraintStatus.VALID

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Union[float, Tuple[float, ...], List[float]]:
        """Calculates the error of the constraint."""
        return 0.0

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        Calculates the partial derivatives (Jacobian entries) of the error.
        Returns a map: point_id -> list of (d_error/dx, d_error/dy).
        The list length matches the number of scalar errors returned by
        error().
        """
        return {}

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: int
    ) -> bool:
        """
        Returns True if this constraint explicitly defines or links the
        radius/length of the specified entity.
        Used by the Solver to determine visual feedback (green color).
        The registry is provided to allow checking related point status.
        """
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the constraint to a dictionary."""
        return {}  # Default for non-serializable constraints like Drag

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        """Checks if the constraint's visual representation is hit."""
        return False

    def update_from_context(self, context: Dict[str, Any]):
        """
        Re-evaluates the expression (if present) using the provided context
        and updates self.value and self.status.
        """
        if self.expression:
            try:
                self.value = safe_evaluate(self.expression, context)
                self.status = ConstraintStatus.EXPRESSION_BASED
            except (ValueError, SyntaxError, NameError, TypeError):
                # Keep old value on failure to prevent geometry collapse
                # during invalid typing. Set status to error.
                self.status = ConstraintStatus.ERROR
        else:
            # If there's no expression, it's just a valid numeric constraint.
            self.status = ConstraintStatus.VALID

    def depends_on_points(self, point_ids: Set[int]) -> bool:
        """Checks if the constraint references any of the given point IDs."""
        for attr in ["p1", "p2", "p3", "p4", "center", "point_id"]:
            if hasattr(self, attr):
                pid = getattr(self, attr)
                if pid is not None and pid in point_ids:
                    return True
        return False

    def depends_on_entities(self, entity_ids: Set[int]) -> bool:
        """Checks if the constraint references any of the given entity IDs."""
        for attr in [
            "e1_id",
            "e2_id",
            "line_id",
            "shape_id",
            "entity_id",
            "circle_id",
            "axis",
        ]:
            if hasattr(self, attr):
                eid = getattr(self, attr)
                if eid is not None and eid in entity_ids:
                    return True
        # Special case for lists of entities
        for attr in ["entity_ids"]:
            if hasattr(self, attr):
                eids = getattr(self, attr)
                if eids and not entity_ids.isdisjoint(eids):
                    return True
        return False
