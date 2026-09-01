from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from gettext import gettext as _
from locale import format_string
from typing import (
    TYPE_CHECKING,
    Any,
)

from raygeo.geo.types import Point

from rayforge.core.expression import safe_evaluate

from ..types import EntityID

if TYPE_CHECKING:
    import cairo

    from ..commands.mirror import MirrorAxis
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
    expression: str | None = None
    status: ConstraintStatus = ConstraintStatus.VALID
    user_visible: bool = True

    def __init__(self, user_visible: bool = True):
        self.user_visible = user_visible

    def is_world_anchored(self) -> bool:
        """Returns True if this constraint pins geometry to global axes.

        World-anchored constraints (e.g. horizontal/vertical) should be
        stripped from array templates, since they conflict with the
        rotations applied to array copies.
        """
        return False

    @classmethod
    def can_apply_to(
        cls, selection: SketchSelection, sketch: Sketch | None = None
    ) -> bool:
        """
        Returns True if this constraint can be applied to the current
        selection.
        Subclasses should override this method.
        """
        return False

    @classmethod
    def get_type_key(cls) -> str | None:
        """
        Returns the string key used to identify this constraint type.
        Returns None for constraints that cannot be created by users.
        """
        return None

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        raise NotImplementedError()

    def targets_segment(
        self, p1: EntityID, p2: EntityID, entity_id: EntityID | None
    ) -> bool:
        """
        Returns True if this constraint restricts the length/distance of the
        segment defined by points (p1, p2) or the given entity_id.
        """
        return False

    def error(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> float | tuple[float, ...] | list[float]:
        """Calculates the error of the constraint."""
        return 0.0

    def gradient(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> dict[EntityID, list[Point]]:
        """
        Calculates the partial derivatives (Jacobian entries) of the error.
        Returns a map: point_id -> list of (d_error/dx, d_error/dy).
        The list length matches the number of scalar errors returned by
        error().
        """
        return {}

    def constrains_radius(
        self, registry: EntityRegistry, entity_id: EntityID
    ) -> bool:
        """
        Returns True if this constraint explicitly defines or links the
        radius/length of the specified entity.
        Used by the Solver to determine visual feedback (green color).
        The registry is provided to allow checking related point status.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serializes the constraint to a dictionary."""
        return {}  # Default for non-serializable constraints like Drag

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: EntityRegistry,
        to_screen: Callable[[Point], Point],
        element: Any,
        threshold: float,
    ) -> bool:
        """Checks if the constraint's visual representation is hit."""
        return False

    def _set_color(self, ctx: cairo.Context, is_hovered: bool) -> None:
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
        self, ctx: cairo.Context, width_scale: float = 3.0
    ) -> None:
        """Draws a semi-transparent blue underlay for the current path."""
        ctx.save()
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
        ctx.set_line_width(ctx.get_line_width() * width_scale)
        ctx.stroke_preserve()
        ctx.restore()

    def _draw_conflict_underlay(
        self, ctx: cairo.Context, width_scale: float = 3.5
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

    def get_subtitle(self, registry: EntityRegistry) -> str:
        """
        Returns a human-readable subtitle describing the constrained entities.
        Subclasses should override to provide meaningful descriptions.
        """
        return ""

    def get_edit_subtitle(self) -> str:
        """
        Returns a user-facing hint string for the constraint edit dialog.
        Subclasses should override to describe the expected input.
        """
        return _("Enter value or expression.")

    def _format_coord(self, x: float, y: float) -> str:
        """Formats coordinates respecting the user's locale."""
        return format_string("%.1f/%.1f", (x, y), grouping=True)

    def draw(
        self,
        ctx: cairo.Context,
        registry: EntityRegistry,
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        """
        Draws the visual representation of the constraint on the canvas.
        Default implementation does nothing.
        """

    def update_from_context(self, context: dict[str, Any]):
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

    def depends_on_points(self, point_ids: set[EntityID]) -> bool:
        """Checks if the constraint references any of the given point IDs."""
        return not self.get_referenced_point_ids().isdisjoint(point_ids)

    def depends_on_entities(self, entity_ids: set[EntityID]) -> bool:
        """Checks if the constraint references any of the given entity IDs."""
        return not self.get_referenced_entity_ids().isdisjoint(entity_ids)

    def get_referenced_point_ids(self) -> set[EntityID]:
        """
        Returns the set of point IDs this constraint references.

        Subclasses that reference points via attributes not covered by the
        default ``p1``..``p4``, ``center``, ``point_id`` names must override
        this.
        """
        ids: set[EntityID] = set()
        for attr in ("p1", "p2", "p3", "p4", "center", "point_id"):
            if hasattr(self, attr):
                pid = getattr(self, attr)
                if pid is not None:
                    ids.add(pid)
        return ids

    def get_referenced_entity_ids(self) -> set[EntityID]:
        """
        Returns the set of entity IDs this constraint references.

        Subclasses that reference entities via attributes not covered by the
        default names must override this.
        """
        ids: set[EntityID] = set()
        for attr in (
            "e1_id",
            "e2_id",
            "line_id",
            "shape_id",
            "entity_id",
            "circle_id",
            "axis",
        ):
            if hasattr(self, attr):
                eid = getattr(self, attr)
                if eid is not None:
                    ids.add(eid)
        entity_ids_attr = getattr(self, "entity_ids", None)
        if entity_ids_attr:
            ids.update(entity_ids_attr)
        return ids

    def is_mirror_compatible(self) -> bool:
        """
        Returns True if this constraint can be preserved when *all* its
        referenced entities/points are mirrored together.

        The default is True: most geometric constraints are invariant under
        reflection. Subclasses that encode chirality or a signed value
        (e.g. AngleConstraint with an expression) should override this.
        """
        return True

    def mirror(self, axis: MirrorAxis) -> None:
        """
        Updates internal state for a mirror transform applied to all
        referenced geometry. The default is a no-op: most constraints are
        mirror-invariant when their references move together.

        Subclasses that encode chirality or a signed value should override
        this (see AngleConstraint).
        """

    def get_draggable_point(self) -> EntityID | None:
        """
        Returns a point ID that can be dragged to manipulate this constraint.

        Override in subclasses that represent point-like constraints.
        Returns None by default.
        """
        return None
