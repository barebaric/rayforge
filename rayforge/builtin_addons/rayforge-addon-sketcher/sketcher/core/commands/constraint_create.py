from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.sketch.constraints.diameter import DiameterConstraint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.radius import RadiusConstraint
from raygeo.sketch.entities.arc import Arc
from raygeo.sketch.entities.circle import Circle
from raygeo.sketch.entities.line import Line

from rayforge.core.undo.command import Command

from .base import SketchChangeCommand
from .constraint import ModifyConstraintCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from typing import Any

    from ..sketch import Sketch


class CreateOrEditConstraintCommand(SketchChangeCommand):
    """
    Creates a constraint for an entity, or returns existing one for editing.

    This command is used for double-click interactions on entities where
    the user wants to add or edit a dimensional constraint.
    """

    def __init__(
        self,
        sketch: Sketch,
        entity: Any,
        name: str = _("Add Any"),
    ):
        super().__init__(sketch, name)
        self.entity = entity
        self._existing_constraint: Any | None = None
        self._created_constraint: Any | None = None
        self._add_cmd: AddItemsCommand | None = None

    def _get_command_label(self, constraint: Any) -> str:
        if isinstance(constraint, DistanceConstraint):
            return _("Add Distance")
        if isinstance(constraint, RadiusConstraint):
            return _("Add Radius")
        if isinstance(constraint, DiameterConstraint):
            return _("Add Diameter")
        return _("Add Any")

    @staticmethod
    def get_constraint_for_entity(
        sketch: Sketch,
        entity: Any,
    ) -> Any | None:
        """
        Returns existing constraint for entity, or None.

        Args:
            sketch: The sketch containing the entity.
            entity: The entity to find a constraint for.

        Returns:
            The existing constraint if found, or None.
        """
        registry = sketch.registry
        constraints = sketch.constraints

        if isinstance(entity, Arc):
            for constr in constraints:
                if isinstance(constr, RadiusConstraint) and (
                    constr.constrains_radius(registry, entity.id)
                ):
                    return constr

        elif isinstance(entity, Line):
            p1_id, p2_id = entity.p1_idx, entity.p2_idx
            for constr in constraints:
                if isinstance(constr, DistanceConstraint) and {
                    constr.p1,
                    constr.p2,
                } == {p1_id, p2_id}:
                    return constr

        elif isinstance(entity, Circle):
            for constr in constraints:
                if isinstance(constr, DiameterConstraint) and (
                    constr.constrains_radius(registry, entity.id)
                ):
                    return constr

        return None

    @staticmethod
    def create_constraint_for_entity(
        sketch: Sketch,
        entity: Any,
        initial_value: float | None = None,
    ) -> Any | None:
        """
        Creates and returns a new dimensional constraint for an entity.

        Args:
            sketch: The sketch containing the entity.
            entity: The entity to create a constraint for.
            initial_value: Optional initial value for the constraint.
                If None, the value is calculated from current geometry.

        Returns:
            The constraint object, or None if the entity type doesn't
            support constraint creation.
        """
        registry = sketch.registry

        if isinstance(entity, Arc):
            start = registry.get_point(entity.start_idx)
            center = registry.get_point(entity.center_idx)
            if start and center:
                radius = math.hypot(start[0] - center[0], start[1] - center[1])
                value = initial_value if initial_value is not None else radius
                return RadiusConstraint(
                    entity.center_idx, entity.start_idx, value
                )

        elif isinstance(entity, Line):
            p1 = registry.get_point(entity.p1_idx)
            p2 = registry.get_point(entity.p2_idx)
            if p1 and p2:
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                value = initial_value if initial_value is not None else dist
                return DistanceConstraint(entity.p1_idx, entity.p2_idx, value)

        elif isinstance(entity, Circle):
            center = registry.get_point(entity.center_idx)
            radius_pt = registry.get_point(entity.radius_pt_idx)
            if center and radius_pt:
                radius = math.hypot(
                    radius_pt[0] - center[0], radius_pt[1] - center[1]
                )
                value = (
                    initial_value if initial_value is not None else radius * 2
                )
                return DiameterConstraint(
                    entity.center_idx, entity.radius_pt_idx, value
                )

        return None

    @property
    def constraint(self) -> Any | None:
        """
        Returns the constraint involved in this operation.

        After execute(), this returns either the existing constraint
        (if one was found) or the newly created constraint.
        """
        if self._existing_constraint is not None:
            return self._existing_constraint
        return self._created_constraint

    @property
    def is_new_constraint(self) -> bool:
        """Returns True if a new constraint was created, False if existing."""
        return (
            self._existing_constraint is None
            and self._created_constraint is not None
        )

    def _do_execute(self) -> None:
        if self._add_cmd is not None:
            return self._add_cmd._redo()

        existing = self.get_constraint_for_entity(self.sketch, self.entity)

        if existing is not None:
            self._existing_constraint = existing
            return

        constraint = self.create_constraint_for_entity(
            self.sketch, self.entity
        )

        if constraint is None:
            return

        label = self._get_command_label(constraint)
        self._add_cmd = AddItemsCommand(self.sketch, label)
        self._created_constraint = self._add_cmd.add_constraint(constraint)

    def _do_undo(self) -> None:
        if self._add_cmd is not None:
            self._add_cmd._do_undo()

    def should_skip_undo(self) -> bool:
        # Skip history for no-op case where an existing constraint was
        # found and no new one was created.
        return self._add_cmd is None and self._created_constraint is None

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, ModifyConstraintCommand)
            and self._created_constraint is not None
            and next_command.constraint is self._created_constraint
        )

    def coalesce_with(self, next_command: Command) -> bool:
        if not self.can_coalesce_with(next_command):
            return False
        # Adopt the new value from the modify command so that undo
        # correctly removes the constraint entirely (via _add_cmd._do_undo)
        # rather than reverting to the initial default value.
        self.timestamp = next_command.timestamp
        return True
