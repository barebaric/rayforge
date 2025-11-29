# constraints/base.py

from __future__ import annotations
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Callable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ..entities import EntityRegistry
    from ..params import ParameterContext


class Constraint:
    """Base class for all geometric constraints."""

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
