from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from .. import constraint_info
from ..entity_group import EntityGroup, points_bbox_center
from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
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
        self._dropped_constraints: list[Any] = []
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
            list[Any],
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
            for eid in registry.entity_ids():
                helper_ids = registry.helper_ids(eid)

                if helper_ids and (
                    eid in entity_id_set
                    or not entity_id_set.isdisjoint(helper_ids)
                ):
                    for hid in helper_ids:
                        if hid not in entity_id_set:
                            entity_id_set.add(hid)
                            changed = True

        # 2. Resolve point set from entities
        point_id_set.update(
            EntityGroup(registry, sorted(entity_id_set)).point_ids()
        )

        if not point_id_set:
            return None

        # 3. Compute mirror axis = bbox center of the selected points
        # (plain bbox: unlike an array group's logical center, a
        # selected circle contributes its defining points only).
        pts = []
        for pid in sorted(point_id_set):
            position = registry.get_point(pid)
            if position is not None:
                pts.append(position)
        bbox_cx, bbox_cy = points_bbox_center(pts)
        if direction == MirrorDirection.VERTICAL:
            axis_pos = bbox_cy
        else:
            axis_pos = bbox_cx

        axis = MirrorAxis(direction=direction, position=axis_pos)

        # 4. Find constraints to drop:
        #    - Constraints with at least one ref inside AND at least one
        #      ref outside the selection.
        #    - Constraints that are not mirror-compatible (e.g. expression-
        #      based AngleConstraint) with all refs inside.
        constraints_to_drop: list[Any] = []
        for constr in sketch.constraints:
            ref_points = constr.get_referenced_point_ids()
            ref_entities = constraint_info.referenced_entity_ids(
                registry, constr
            )

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
                    and (not constraint_info.is_mirror_compatible(constr))
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
        group = EntityGroup(
            registry, entity_ids, flags=self.sketch.flags
        )
        direction = (
            "vertical"
            if self._direction is MirrorDirection.VERTICAL
            else "horizontal"
        )

        # Mirror points centrally (fixed points, e.g. the origin,
        # stay put). The set also carries bare selected points, which
        # belong to no entity and are therefore not in the group.
        for pid in point_ids:
            position = registry.get_point(pid)
            if position is not None and not registry.point_is_fixed(pid):
                x, y = axis.apply(*position)
                registry.set_point_position(pid, x, y)

        # Mirror entity-specific state (arc chirality, bezier control
        # offsets, polygon ring chirality)
        for entity in group.entities():
            registry.mirror(entity.id, direction, axis.position)

        # Mirror internal constraints (e.g. negate angle values)
        # Constraints that were dropped are skipped.
        dropped_set = {id(c) for c in self._dropped_constraints}
        entity_id_set = set(entity_ids)
        for constr in self.sketch.constraints:
            if id(constr) in dropped_set:
                continue
            ref_points = constr.get_referenced_point_ids()
            ref_entities = constraint_info.referenced_entity_ids(
                registry, constr
            )
            has_internal = bool(ref_points & point_ids) or bool(
                ref_entities & entity_id_set
            )
            has_external = bool(ref_points - point_ids) or bool(
                ref_entities - entity_id_set
            )
            if has_internal and not has_external:
                expression = getattr(constr, "expression", None)
                if callable(expression):
                    self._mirrored_constraint_states[id(constr)] = {
                        "expression": expression()
                    }
                mirror_fn = getattr(constr, "mirror", None)
                if callable(mirror_fn):
                    mirror_fn()

        # Drop incompatible constraints
        self.sketch.remove_constraints(self._dropped_constraints)

    def _do_undo(self) -> None:
        # Restore constraint expressions that were dropped because
        # they do not survive mirroring.
        self.sketch.add_constraints(self._dropped_constraints)
