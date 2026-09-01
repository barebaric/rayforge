"""
Ephemeral views over entity groups: a set of entities plus their
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
from typing import TYPE_CHECKING, Any, Protocol

from .entities import Circle, Ellipse, TextBoxEntity

if TYPE_CHECKING:
    from .entities import Entity, Point
    from .registry import EntityRegistry


class PlacementTransform(Protocol):
    """Anything that can map absolute points and relative offsets
    (e.g. an array placement or a mirror axis) into new coordinates."""

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transforms an absolute point."""
        ...

    def transform_offset(self, dx: float, dy: float) -> tuple[float, float]:
        """Transforms a relative offset (e.g. a bezier control point)."""
        ...


def points_bbox_center(points: list[Point]) -> tuple[float, float]:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def remap_point_refs(obj: Any, pid_map: dict[int, int]) -> None:
    """Rewrites point ID references on an entity or constraint."""
    for attr, value in vars(obj).items():
        # Note: bool is an int subclass; never remap flag attributes.
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in pid_map
        ):
            setattr(obj, attr, pid_map[value])


class EntityGroup:
    """
    An ephemeral view over ``(registry, entity_ids)``: the surviving
    entities plus their unique defining points, manipulable as one
    piece of geometry.

    The group holds no back-references beyond the registry passed to
    it, is never stored and has no serialization impact. It mutates
    only point coordinates and entity-internal geometry (bezier
    control offsets); all higher-level semantics — which entities
    belong together, which constraints follow them, who bookkeeps
    undo state — stay with the callers.
    """

    def __init__(
        self, registry: EntityRegistry, entity_ids: Iterable[int]
    ) -> None:
        self.registry = registry
        self.entity_ids: list[int] = list(entity_ids)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def entities(self) -> list[Entity]:
        """Returns the group's surviving entities, in group order.
        References to deleted entities are skipped."""
        entities: list[Entity] = []
        for eid in self.entity_ids:
            entity = self.registry.get_entity(eid)
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
        return pids

    def points(self) -> list[Point]:
        """Returns the group's unique defining points, in the same
        stable order as ``point_ids``."""
        points: list[Point] = []
        for pid in self.point_ids():
            pt = self._get_point(pid)
            if pt is not None:
                points.append(pt)
        return points

    def _get_point(self, pid: int) -> Point | None:
        try:
            return self.registry.get_point(pid)
        except IndexError:
            return None

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
            if isinstance(entity, (Circle, Ellipse))
        ]
        if len(shapes) == 1:
            cpt = self._get_point(shapes[0].center_idx)
            if cpt is not None:
                return (cpt.x, cpt.y)
        return points_bbox_center(self.points())

    def apply_placement(self, placement: PlacementTransform) -> None:
        """
        Rigidly moves the group's defining points and Bezier
        control-point offsets by the placement transform.

        Points shared between entities (e.g. the joined edges of a
        rounded rectangle) are moved exactly once: every new position
        is computed from the pre-move geometry before anything is
        written.
        """
        old: dict[int, tuple[float, float]] = {}
        for pid in self.point_ids():
            pt = self._get_point(pid)
            if pt is not None:
                old[pid] = (pt.x, pt.y)
        for pid, (x, y) in old.items():
            pt = self.registry.get_point(pid)
            pt.x, pt.y = placement.transform_point(x, y)
        for entity in self.entities():
            entity.transform_offsets(placement)

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
        for pt in self.points():
            pt.x += dx
            pt.y += dy

    def radial_project(
        self, center: tuple[float, float], radius: float
    ) -> None:
        """
        Translates the group radially so its center sits on the
        circle of ``radius`` around ``center``, shape and angular
        position preserved. No-op when the group's center already
        sits there.
        """
        if not self.points():
            return
        mcx, mcy = self.center()
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
    ) -> list[tuple[Point, float, float]]:
        """
        Rewrites an existing copy group's geometry to the placement
        applied to this (template) group: entities are paired
        positionally, points are paired positionally and moved onto
        the transform of their template point, and each entity's
        internal state is re-derived from its template counterpart
        (``Entity.rewrite_offsets_from``, e.g. bezier control-point
        offsets). ``extra_point_pairs`` pairs the groups' standalone
        points ((template pid, copy pid) in matching order); each is
        moved onto the transform of its template point, exactly like
        entity points. Returns the applied copy-point positions so the
        caller can bookkeep them for undo/redo.
        """
        applied: list[tuple[Point, float, float]] = []
        for tpl_entity, copy_entity in zip(self.entities(), copy.entities()):
            for tpl_pid, copy_pid in zip(
                tpl_entity.get_point_ids(), copy_entity.get_point_ids()
            ):
                tpl_pt = self._get_point(tpl_pid)
                copy_pt = copy._get_point(copy_pid)
                if tpl_pt is None or copy_pt is None:
                    continue
                copy_pt.x, copy_pt.y = placement.transform_point(
                    tpl_pt.x, tpl_pt.y
                )
                applied.append((copy_pt, copy_pt.x, copy_pt.y))
            copy_entity.rewrite_offsets_from(tpl_entity, placement)
        for tpl_pid, copy_pid in extra_point_pairs:
            tpl_pt = self._get_point(tpl_pid)
            copy_pt = copy._get_point(copy_pid)
            if tpl_pt is None or copy_pt is None:
                continue
            copy_pt.x, copy_pt.y = placement.transform_point(
                tpl_pt.x, tpl_pt.y
            )
            applied.append((copy_pt, copy_pt.x, copy_pt.y))
        return applied

    def snapshot_positions(self) -> list[tuple[Point, float, float]]:
        """Captures the current position of every point of the group.
        The snapshot keeps references to the live ``Point`` objects,
        so restoring works even after the group itself is gone."""
        return [(p, p.x, p.y) for p in self.points()]

    @staticmethod
    def restore_positions(
        snapshot: Iterable[tuple[Point, float, float]],
    ) -> None:
        """Restores the point positions captured by
        ``snapshot_positions``."""
        for pt, x, y in snapshot:
            pt.x = x
            pt.y = y

    def polylines(self) -> list[list[tuple[float, float]]]:
        """Samples every entity of the group into a polyline (in
        model coordinates), e.g. for preview rendering. Delegates to
        the entities' polymorphic ``to_polyline``."""
        return [
            entity.to_polyline(self.registry) for entity in self.entities()
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
            if isinstance(entity, Ellipse):
                helper_ids.extend(entity.helper_line_ids)
            elif isinstance(entity, TextBoxEntity):
                helper_ids.extend(entity.construction_line_ids)
        template_pids = set(self.point_ids())
        taken = set(helper_ids) | {entity.id for entity in self.entities()}
        for entity in self.registry.entities:
            if entity.id in taken:
                continue
            if not (entity.construction or entity.invisible):
                continue
            point_ids = entity.get_point_ids()
            if point_ids and set(point_ids) <= template_pids:
                helper_ids.append(entity.id)
        return helper_ids

    def remap_point_refs(self, pid_map: dict[int, int]) -> None:
        """Rewrites the group's entities' point ID references by the
        given map (used when the group's points are re-owned)."""
        for entity in self.entities():
            remap_point_refs(entity, pid_map)
