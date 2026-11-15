"""Ephemeral views over entity groups: a set of entities plus their
defining points, treated as one piece of geometry.

An ``EntityGroup`` is cheap to construct and never stored: commands,
arrays and tools build one for the duration of a single operation,
read the group's geometry and apply transforms through it. Placement
transforms are accepted structurally (the ``PlacementTransform``
protocol), so this module never imports the ``arrays`` package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .entity_flags import EntityFlags
from .entity_ops import (
    PlacementTransform,
    apply_placement_to_state,
    rewrite_state_from,
)

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..types import Entity


def points_bbox_center(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


class EntityGroup:
    """
    An ephemeral view over ``(registry, entity_ids)``: the surviving
    entities plus their unique defining points, manipulable as one
    piece of geometry.

    The group holds no back-references beyond the registry passed to
    it, is never stored and has no serialization impact. It mutates
    only point coordinates and entity-internal state (bezier control
    offsets, polygon ring chirality); all higher-level semantics —
    which entities belong together, which constraints follow them,
    who bookkeeps undo state — stay with the callers.
    """

    def __init__(
        self,
        registry: EntityRegistry,
        entity_ids: Iterable[int],
        flags: EntityFlags | None = None,
    ) -> None:
        self.registry = registry
        self.flags = flags
        self.entity_ids: list[int] = list(entity_ids)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def entities(self) -> list[Entity]:
        """Returns the group's surviving entities, in group order.
        References to deleted entities are skipped."""
        entities: list[Entity] = []
        for eid in self.entity_ids:
            entity = self.registry.get_entity_object(eid)
            if entity is not None:
                entities.append(entity)
        return entities

    def point_ids(self) -> list[int]:
        """Returns the group's unique point IDs in stable order:
        entities in group order, each entity's points in entity
        order, duplicates dropped."""
        pids: list[int] = []
        for entity in self.entities():
            for pid in entity.get_point_ids():
                if pid not in pids:
                    pids.append(pid)
        for eid in self.entity_ids:
            if (
                self.registry.get_entity_object(eid) is None
                and self.registry.get_point(eid) is not None
                and eid not in pids
            ):
                pids.append(eid)
        return pids

    def points(self) -> list[tuple[int, tuple[float, float]]]:
        """Returns the group's unique defining points as ``(pid, (x,
        y))`` pairs, in the same stable order as ``point_ids``."""
        points = []
        for pid in self.point_ids():
            point = self.registry.get_point_object(pid)
            if point is not None:
                points.append(point)
        return points

    def point_position(self, pid: int) -> tuple[float, float] | None:
        return self.registry.get_point(pid)

    def move_point(self, pid: int, x: float, y: float) -> None:
        self.registry.set_point_position(pid, x, y)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def center(self) -> tuple[float, float]:
        """
        Returns the logical center of the group's entities.

        When the group contains exactly one Circle or Ellipse (plus
        any attached helper geometry) that entity's explicit center
        point is the logical center — the defining points of an
        ellipse only span a quarter of its area, so a bbox over them
        is wrong. For anything else the bbox center of all defining
        points is used.
        """
        shapes = [
            entity
            for entity in self.entities()
            if entity.type_name() in ("circle", "ellipse")
        ]
        if len(shapes) == 1:
            position = self.registry.get_point(shapes[0].center_idx)
            if position is not None:
                return position
        return points_bbox_center([(p.x, p.y) for p in self.points()])

    def apply_placement(self, placement: PlacementTransform) -> None:
        """
        Rigidly moves the group's defining points and entity-internal
        relative state (bezier control-point offsets, polygon
        chirality) by the placement transform.

        Points shared between entities (e.g. the joined edges of a
        rounded rectangle) are moved exactly once: every new position
        is computed from the pre-move geometry before anything is
        written.
        """
        old: dict[int, tuple[float, float]] = {}
        for p in self.points():
            old[p.id] = (p.x, p.y)
        for pid, (x, y) in old.items():
            nx, ny = placement.transform_point(x, y)
            self.registry.set_point_position(pid, nx, ny)
        for entity in self.entities():
            apply_placement_to_state(self.registry, entity.id, placement)

    def apply_rigid_motion(self, motion: PlacementTransform) -> None:
        """
        Applies a rigid motion — a rotation about a center followed
        by a translation — e.g. the re-anchor motion that places a
        template member onto its guide's position 0. Same application
        rules as ``apply_placement``; the separate name documents
        that the transform is shape-preserving.
        """
        self.apply_placement(motion)

    def translate(self, dx: float, dy: float) -> None:
        """Moves every point of the group by the given delta. Points
        shared between entities are moved exactly once."""
        for p in self.points():
            self.registry.set_point_position(p.id, p.x + dx, p.y + dy)

    def radial_project(
        self,
        center: tuple[float, float],
        radius: float,
        pin: tuple[float, float] | None = None,
    ) -> None:
        """
        Translates the group radially so its center sits on the circle
        of ``radius`` around ``center``, shape and angular position
        preserved. No-op when the group's center already sits there.

        ``pin`` overrides the projected point. Callers that move a
        group extended with helper geometry (``with_helpers``) pass
        the plain group's center, so the projection matches the array
        placement math instead of the helper-widened bounding box.
        """
        if pin is None:
            if not self.points():
                return
            mcx, mcy = self.center()
        else:
            mcx, mcy = pin
        vx, vy = mcx - center[0], mcy - center[1]
        d = math.hypot(vx, vy)
        if d < 1e-9:
            return
        scale = (radius - d) / d
        self.translate(vx * scale, vy * scale)

    def rewrite_copy_from(
        self,
        copy: EntityGroup,
        placement: PlacementTransform,
        extra_point_pairs: Iterable[tuple[int, int]] = (),
    ) -> list[tuple[int, float, float]]:
        """
        Rewrites an existing copy group's geometry to the placement
        applied to this (template) group: entities are paired
        positionally, points are paired positionally and moved onto
        the transform of their template point, and each copy entity's
        internal state is re-derived from its template counterpart
        (bezier control-point offsets, polygon ring chirality).
        ``extra_point_pairs`` pairs the groups' standalone points
        ((template pid, copy pid) in matching order); each is moved
        onto the transform of its template point, exactly like entity
        points. Returns the applied copy-point positions so the
        caller can bookkeep them for undo/redo.
        """
        applied: list[tuple[int, float, float]] = []
        for tpl_entity, copy_entity in zip(self.entities(), copy.entities()):
            for tpl_pid, copy_pid in zip(
                tpl_entity.get_point_ids(), copy_entity.get_point_ids()
            ):
                tpl_pt = self.registry.get_point(tpl_pid)
                copy_pt = self.registry.get_point(copy_pid)
                if tpl_pt is None or copy_pt is None:
                    continue
                nx, ny = placement.transform_point(tpl_pt[0], tpl_pt[1])
                self.registry.set_point_position(copy_pid, nx, ny)
                applied.append((copy_pid, nx, ny))
            rewrite_state_from(
                self.registry, copy_entity.id, tpl_entity.id, placement
            )
        for tpl_pid, copy_pid in extra_point_pairs:
            tpl_pt = self.registry.get_point(tpl_pid)
            copy_pt = self.registry.get_point(copy_pid)
            if tpl_pt is None or copy_pt is None:
                continue
            nx, ny = placement.transform_point(tpl_pt[0], tpl_pt[1])
            self.registry.set_point_position(copy_pid, nx, ny)
            applied.append((copy_pid, nx, ny))
        return applied

    def snapshot_positions(self) -> list[tuple[int, float, float]]:
        """Captures the current position of every point of the group."""
        return [(p.id, p.x, p.y) for p in self.points()]

    @staticmethod
    def restore_positions(
        snapshot: Iterable[tuple[int, float, float]],
        registry: EntityRegistry,
    ) -> None:
        """Restores the point positions captured by
        ``snapshot_positions``."""
        for pid, x, y in snapshot:
            registry.set_point_position(pid, x, y)

    def polylines(self) -> list[list[tuple[float, float]]]:
        """Samples every entity of the group into polylines (in
        model coordinates), e.g. for preview rendering. Multi-contour
        entities (e.g. text glyphs) contribute one polyline per
        contour."""
        return [
            polyline
            for entity_id in self.entity_ids
            if self.registry.get_entity_object(entity_id) is not None
            for polyline in self.registry.to_polylines(entity_id)
        ]

    # ------------------------------------------------------------------
    # Membership semantics
    # ------------------------------------------------------------------

    def helper_ids(self) -> list[int]:
        """
        Returns the helper geometry belonging to the group's
        entities: registered helpers (an ellipse's helper lines, a
        text box's construction lines) plus any construction or
        invisible entity that is fully attached to the group's points
        (e.g. an ellipse's visible axis lines, which are not
        registered as helpers). The helpers share the group's points
        and must be extracted — and placed — along with it.
        """
        helper_ids: list[int] = []
        for entity in self.entities():
            helper_ids.extend(self.registry.helper_ids(entity.id))
        template_pids = set(self.point_ids())
        taken = set(helper_ids) | {entity.id for entity in self.entities()}
        for eid in self.registry.entity_ids():
            if eid in taken:
                continue
            entity = self.registry.get_entity_object(eid)
            if entity is None:
                continue
            invisible = self.flags.is_invisible(eid) if self.flags else False
            if not (entity.construction or invisible):
                continue
            point_ids = entity.get_point_ids()
            if point_ids and set(point_ids) <= template_pids:
                helper_ids.append(eid)
        return helper_ids

    def with_helpers(self) -> EntityGroup:
        """
        Returns a view extended with this group's helper entities (see
        ``helper_ids``). Rigid motions must be applied to the extended
        view: a helper's points complete the member's shape (a text
        box's fourth frame corner is referenced only by its
        construction lines), so leaving them behind tears the member
        apart and leaves its constraints violated. The returned group
        is ephemeral like the group itself.
        """
        return EntityGroup(
            self.registry,
            self.entity_ids + self.helper_ids(),
            flags=self.flags,
        )
