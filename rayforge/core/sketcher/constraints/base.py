from __future__ import annotations
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Callable,
    Optional,
    TYPE_CHECKING,
)
from ..evaluator import safe_evaluate

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class Constraint:
    """Base class for all geometric constraints."""

    # These attributes are expected on dimensional constraints
    value: float = 0.0
    expression: Optional[str] = None

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
        and updates self.value.
        """
        if self.expression:
            try:
                self.value = safe_evaluate(self.expression, context)
            except ValueError:
                # Keep old value on failure to prevent geometry collapse
                # during invalid typing
                pass
