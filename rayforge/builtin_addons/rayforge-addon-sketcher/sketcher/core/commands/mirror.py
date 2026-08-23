from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..entities import Ellipse, TextBoxEntity
from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..selection import SketchSelection
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class MirrorDirection(Enum):
    """
    Direction of a mirror reflection.

    VERTICAL: flip in the vertical direction (top↔bottom) → flips Y
    coordinates, mirror axis is horizontal.
    HORIZONTAL: flip in the horizontal direction (left↔right) → flips X
    coordinates, mirror axis is vertical.
    """

    VERTICAL = auto()
    HORIZONTAL = auto()


@dataclass(frozen=True)
class MirrorAxis:
    """
    Defines a mirror reflection axis.

    - VERTICAL: horizontal axis at y=position → flips Y coords.
    - HORIZONTAL: vertical axis at x=position → flips X coords.
    """

    direction: MirrorDirection
    position: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Mirrors a point (x, y) across this axis."""
        if self.direction == MirrorDirection.VERTICAL:
            return (x, 2.0 * self.position - y)
        return (2.0 * self.position - x, y)

    def flip_offset(self, offset: tuple[float, float]) -> tuple[float, float]:
        """Mirrors a (dx, dy) offset vector across this axis."""
        if self.direction == MirrorDirection.VERTICAL:
            return (offset[0], -offset[1])
        return (-offset[0], offset[1])


class MirrorCommand(SketchChangeCommand):
    """
    Mirrors the current selection in-place across an axis through the
    bounding-box center of the selected points.

    Points are mirrored centrally (shared points are de-duplicated).
    Entity-specific state (bezier control-point deltas, arc chirality) is
    handled polymorphically via ``Entity.mirror()``. Constraints that
    reference geometry outside the selection are dropped. Internal
    constraints are preserved; chirality-sensitive constraints (e.g.
    AngleConstraint) are updated via ``Constraint.mirror()`` or dropped if
    incompatible (expression-based angle).
    """

    def __init__(
        self,
        sketch: Sketch,
        selection: SketchSelection,
        direction: MirrorDirection,
    ):
        super().__init__(sketch, _("Mirror Selection"))
        self._selection = selection
        self._direction: MirrorDirection = direction
        self._axis: MirrorAxis | None = None
        self._dropped_constraints: list[Constraint] = []
        # Save state of internal constraints that were mutated by mirror()
        # so undo can restore them (e.g. AngleConstraint.value negation).
        self._mirrored_constraint_states: dict[int, dict] = {}

    @staticmethod
    def prepare(
        sketch: Sketch,
        selection: SketchSelection,
        direction: MirrorDirection,
    ) -> (
        tuple[
            list[EntityID],
            set[EntityID],
            list[Constraint],
            MirrorAxis,
        ]
        | None
    ):
        """
        Pure function that resolves the mirror operation parameters.

        Returns:
            A tuple of (entity_ids_to_mirror, point_ids_to_mirror,
            constraints_to_drop, mirror_axis), or None if the selection
            is empty or contains no mirrorable geometry.
        """
        registry = sketch.registry

        # 1. Resolve entity set (including compound helpers)
        entity_id_set: set[EntityID] = set(selection.entity_ids)
        point_id_set: set[EntityID] = set(selection.point_ids)

        # Pull in helper lines for compound entities (same unity logic
        # as RemoveItemsCommand.calculate_dependencies).
        changed = True
        while changed:
            changed = False
            for e in registry.entities:
                helper_ids = None
                if isinstance(e, TextBoxEntity):
                    helper_ids = e.construction_line_ids
                elif isinstance(e, Ellipse):
                    helper_ids = e.helper_line_ids

                if helper_ids and (
                    e.id in entity_id_set
                    or not entity_id_set.isdisjoint(helper_ids)
                ):
                    for hid in helper_ids:
                        if hid not in entity_id_set:
                            entity_id_set.add(hid)
                            changed = True

        # 2. Resolve point set from entities
        for eid in entity_id_set:
            e = registry.get_entity(eid)
            if e:
                point_id_set.update(e.get_point_ids())

        if not point_id_set and not entity_id_set:
            return None

        if not point_id_set:
            return None

        # 3. Compute mirror axis = bbox center of selected points
        xs = []
        ys = []
        for pid in point_id_set:
            p = registry.get_point(pid)
            if p:
                xs.append(p.x)
                ys.append(p.y)

        if not xs:
            return None

        if direction == MirrorDirection.VERTICAL:
            axis_pos = (min(ys) + max(ys)) / 2.0
        else:
            axis_pos = (min(xs) + max(xs)) / 2.0

        axis = MirrorAxis(direction=direction, position=axis_pos)

        # 4. Find constraints to drop:
        #    - Constraints with at least one ref inside AND at least one
        #      ref outside the selection.
        #    - Constraints that are not mirror-compatible (e.g. expression-
        #      based AngleConstraint) with all refs inside.
        constraints_to_drop: list[Constraint] = []
        for constr in sketch.constraints:
            ref_points = constr.get_referenced_point_ids()
            ref_entities = constr.get_referenced_entity_ids()

            has_internal = bool(ref_points & point_id_set) or bool(
                ref_entities & entity_id_set
            )
            has_external = bool(ref_points - point_id_set) or bool(
                ref_entities - entity_id_set
            )

            if (
                has_internal
                and has_external
                or (
                    has_internal
                    and not has_external
                    and (not constr.is_mirror_compatible())
                )
            ):
                constraints_to_drop.append(constr)

        entity_ids = sorted(entity_id_set)
        return entity_ids, point_id_set, constraints_to_drop, axis

    def _do_execute(self) -> None:
        result = self.prepare(self.sketch, self._selection, self._direction)
        if result is None:
            return

        entity_ids, point_ids, dropped, axis = result
        self._axis = axis
        self._dropped_constraints = list(dropped)

        registry = self.sketch.registry

        # Mirror points
        for pid in point_ids:
            p = registry.get_point(pid)
            if p and not p.fixed:
                p.x, p.y = axis.apply(p.x, p.y)

        # Mirror entity-specific state
        for eid in entity_ids:
            e = registry.get_entity(eid)
            if e:
                e.mirror(axis)

        # Mirror internal constraints (e.g. negate angle values)
        # Constraints that were dropped are skipped.
        dropped_set = {id(c) for c in self._dropped_constraints}
        for constr in self.sketch.constraints:
            if id(constr) in dropped_set:
                continue
            ref_points = constr.get_referenced_point_ids()
            ref_entities = constr.get_referenced_entity_ids()
            has_internal = bool(ref_points & point_ids) or bool(
                ref_entities & set(entity_ids)
            )
            has_external = bool(ref_points - point_ids) or bool(
                ref_entities - set(entity_ids)
            )
            if has_internal and not has_external:
                # Save value before mirroring so undo can restore it
                # (e.g. AngleConstraint.mirror negates self.value)
                if hasattr(constr, "value"):
                    self._mirrored_constraint_states[id(constr)] = {
                        "value": constr.value
                    }
                constr.mirror(axis)

        # Drop incompatible constraints
        for c in self._dropped_constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)

    def _do_undo(self) -> None:
        # Restore constraint values that were mutated by mirror()
        # (e.g. AngleConstraint.value negation).
        for constr in self.sketch.constraints:
            state = self._mirrored_constraint_states.get(id(constr))
            if state and hasattr(constr, "value"):
                constr.value = state["value"]

        # Re-add dropped constraints.
        for c in self._dropped_constraints:
            if c not in self.sketch.constraints:
                self.sketch.constraints.append(c)
