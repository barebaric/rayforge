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
from locale import format_string
from ...expression import safe_evaluate
from ...geo import Point
from ..types import EntityID

if TYPE_CHECKING:
    import cairo
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class ConstraintStatus(Enum):
    """Represents the validation status of a constraint."""

    VALID = auto()
    EXPRESSION_BASED = auto()
    ERROR = auto()
    CONFLICTING = auto()


class Constraint:
    """Base class for all geometric constraints."""

    # These attributes are expected on dimensional constraints
    value: float = 0.0
    expression: Optional[str] = None
    status: ConstraintStatus = ConstraintStatus.VALID
    user_visible: bool = True

    def __init__(self, user_visible: bool = True):
        self.user_visible = user_visible

    @classmethod
    def can_apply_to(
        cls, selection: "SketchSelection", sketch: Optional["Sketch"] = None
    ) -> bool:
        """
        Returns True if this constraint can be applied to the current
        selection.
        Subclasses should override this method.
        """
        return False

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        raise NotImplementedError()

    def targets_segment(
        self, p1: EntityID, p2: EntityID, entity_id: Optional[EntityID]
    ) -> bool:
        """
        Returns True if this constraint restricts the length/distance of the
        segment defined by points (p1, p2) or the given entity_id.
        """
        return False

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Union[float, Tuple[float, ...], List[float]]:
        """Calculates the error of the constraint."""
        return 0.0

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[EntityID, List[Point]]:
        """
        Calculates the partial derivatives (Jacobian entries) of the error.
        Returns a map: point_id -> list of (d_error/dx, d_error/dy).
        The list length matches the number of scalar errors returned by
        error().
        """
        return {}

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: EntityID
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
        to_screen: Callable[[Point], Point],
        element: Any,
        threshold: float,
    ) -> bool:
        """Checks if the constraint's visual representation is hit."""
        return False

    def _set_color(self, ctx: "cairo.Context", is_hovered: bool) -> None:
        """
        Sets the standard drawing color for constraints based on hover and
        status.
        """
        if is_hovered:
            ctx.set_source_rgb(1.0, 0.8, 0.0)  # Yellow for hover
        elif self.status == ConstraintStatus.CONFLICTING:
            ctx.set_source_rgb(1.0, 0.2, 0.2)  # Red for conflicting
        elif self.status == ConstraintStatus.ERROR:
            ctx.set_source_rgb(1.0, 0.2, 0.2)  # Red for error
        elif self.status == ConstraintStatus.EXPRESSION_BASED:
            ctx.set_source_rgb(1.0, 0.6, 0.0)  # Orange for expression
        else:  # VALID
            ctx.set_source_rgb(0.0, 0.6, 0.0)  # Green for valid

    def _draw_selection_underlay(
        self, ctx: "cairo.Context", width_scale: float = 3.0
    ) -> None:
        """Draws a semi-transparent blue underlay for the current path."""
        ctx.save()
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
        ctx.set_line_width(ctx.get_line_width() * width_scale)
        ctx.stroke_preserve()
        ctx.restore()

    def _draw_conflict_underlay(
        self, ctx: "cairo.Context", width_scale: float = 3.5
    ) -> None:
        """Draws a semi-transparent red underlay for conflicting items."""
        ctx.save()
        ctx.set_source_rgba(1.0, 0.2, 0.2, 0.5)
        ctx.set_line_width(ctx.get_line_width() * width_scale)
        ctx.stroke_preserve()
        ctx.restore()

    def _format_value(self) -> str:
        """Helper to format the value string for constraints."""
        return f"{float(self.value):.1f}"

    def get_title(self) -> str:
        """
        Returns a human-readable title for this constraint.
        Subclasses should override to include the value.
        """
        return self.get_type_name()

    def get_subtitle(self, registry: "EntityRegistry") -> str:
        """
        Returns a human-readable subtitle describing the constrained entities.
        Subclasses should override to provide meaningful descriptions.
        """
        return ""

    def _format_coord(self, x: float, y: float) -> str:
        """Formats coordinates respecting the user's locale."""
        return format_string("%.1f/%.1f", (x, y), grouping=True)

    def draw(
        self,
        ctx: "cairo.Context",
        registry: "EntityRegistry",
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        """
        Draws the visual representation of the constraint on the canvas.
        Default implementation does nothing.
        """
        pass

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

    def depends_on_points(self, point_ids: Set[EntityID]) -> bool:
        """Checks if the constraint references any of the given point IDs."""
        for attr in ["p1", "p2", "p3", "p4", "center", "point_id"]:
            if hasattr(self, attr):
                pid = getattr(self, attr)
                if pid is not None and pid in point_ids:
                    return True
        return False

    def depends_on_entities(self, entity_ids: Set[EntityID]) -> bool:
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

    def get_draggable_point(self) -> Optional[EntityID]:
        """
        Returns a point ID that can be dragged to manipulate this constraint.

        Override in subclasses that represent point-like constraints.
        Returns None by default.
        """
        return None
