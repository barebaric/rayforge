from __future__ import annotations
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional

from ..constraints import (
    DiameterConstraint,
    DistanceConstraint,
    RadiusConstraint,
)
from ..entities import Arc, Circle, Entity, Line
from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
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
        entity: Entity,
        name: str = _("Add Constraint"),
    ):
        super().__init__(sketch, name)
        self.entity = entity
        self._existing_constraint: Optional[Constraint] = None
        self._created_constraint: Optional[Constraint] = None
        self._add_cmd: Optional[AddItemsCommand] = None

    @staticmethod
    def get_constraint_for_entity(
        sketch: Sketch,
        entity: Entity,
    ) -> Optional[Constraint]:
        """
        Returns existing constraint for entity, or None.

        Args:
            sketch: The sketch containing the entity.
            entity: The entity to find a constraint for.

        Returns:
            The existing constraint if found, or None.
        """
        constraints = sketch.constraints or []

        if isinstance(entity, Arc):
            for constr in constraints:
                if (
                    isinstance(constr, RadiusConstraint)
                    and constr.entity_id == entity.id
                ):
                    return constr

        elif isinstance(entity, Line):
            p1_id, p2_id = entity.p1_idx, entity.p2_idx
            for constr in constraints:
                if isinstance(constr, DistanceConstraint):
                    if {constr.p1, constr.p2} == {p1_id, p2_id}:
                        return constr

        elif isinstance(entity, Circle):
            for constr in constraints:
                if (
                    isinstance(constr, DiameterConstraint)
                    and constr.circle_id == entity.id
                ):
                    return constr

        return None

    @staticmethod
    def create_constraint_for_entity(
        sketch: Sketch,
        entity: Entity,
        initial_value: Optional[float] = None,
    ) -> Optional[Constraint]:
        """
        Creates and returns a new constraint for the entity.

        Args:
            sketch: The sketch containing the entity.
            entity: The entity to create a constraint for.
            initial_value: Optional initial value for the constraint.
                If None, the value is calculated from current geometry.

        Returns:
            The newly created constraint, or None if entity type
            doesn't support constraint creation.
        """
        registry = sketch.registry

        if isinstance(entity, Arc):
            start = registry.get_point(entity.start_idx)
            center = registry.get_point(entity.center_idx)
            if start and center:
                radius = math.hypot(start.x - center.x, start.y - center.y)
                value = initial_value if initial_value is not None else radius
                return RadiusConstraint(entity.id, value)

        elif isinstance(entity, Line):
            p1 = registry.get_point(entity.p1_idx)
            p2 = registry.get_point(entity.p2_idx)
            if p1 and p2:
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                value = initial_value if initial_value is not None else dist
                return DistanceConstraint(entity.p1_idx, entity.p2_idx, value)

        elif isinstance(entity, Circle):
            center = registry.get_point(entity.center_idx)
            radius_pt = registry.get_point(entity.radius_pt_idx)
            if center and radius_pt:
                radius = math.hypot(
                    radius_pt.x - center.x, radius_pt.y - center.y
                )
                value = (
                    initial_value if initial_value is not None else radius * 2
                )
                return DiameterConstraint(entity.id, value)

        return None

    @property
    def constraint(self) -> Optional[Constraint]:
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
            return self._add_cmd._do_execute()

        existing = self.get_constraint_for_entity(self.sketch, self.entity)

        if existing is not None:
            self._existing_constraint = existing
            return

        new_constr = self.create_constraint_for_entity(
            self.sketch, self.entity
        )

        if new_constr is None:
            return

        self._created_constraint = new_constr

        label = self._get_command_label(new_constr)
        self._add_cmd = AddItemsCommand(
            self.sketch,
            label,
            constraints=[new_constr],
        )
        self._add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self._add_cmd is not None:
            self._add_cmd._do_undo()

    def _get_command_label(self, constraint: Constraint) -> str:
        return _("Add {}").format(constraint.get_type_name())
