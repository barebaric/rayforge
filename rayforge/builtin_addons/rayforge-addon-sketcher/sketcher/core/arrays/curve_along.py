from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from ..entities import Arc, Bezier, Line
from .base import (
    Array,
    ArrayStrategy,
    InstancePlacement,
    PlacementKind,
    apply_placement_to_entities,
    resolve_template_center,
)

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity
    from ..entities.point import Point
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


class CurveAlongArrayStrategy(ArrayStrategy):
    """
    Distributes copies along a guide path (a Line, Arc or Bezier).

    The path is sampled at equal arc-length intervals (after an
    optional offset from the start). The template member is placed
    onto the first sample, rotated to the path tangent there (when
    ``align_to_tangent`` is on); the copy of slot j is the rigid
    motion of the template that carries the first sample onto sample
    j. Preview, create, edit and sync all derive members through
    ``template_placement``/``member_placements``, so the live
    template is the single source of truth. Copies are static baked
    geometry: they carry no constraints and their points are fixed,
    so the solver never touches them — they are updated only by
    re-deriving them from the template. When the path or the template
    is edited the array auto-regenerates via
    ``Sketch.sync_arrays()``.
    """

    uses_template_anchor = True

    def __init__(
        self,
        count: int = 6,
        rotate_copies: bool = True,
        path_entity_id: int = -1,
        align_to_tangent: bool = True,
        offset_to_start: float = 0.0,
        spacing: float = 0.0,
    ):
        self.count = count
        self.rotate_copies = rotate_copies
        self.path_entity_id = path_entity_id
        self.align_to_tangent = align_to_tangent
        self.offset_to_start = offset_to_start
        self.spacing = spacing

    def existing_master_id(self) -> int | None:
        # The guide path entity is created by the user before invoking
        # the tool, so the array reuses it as master instead of
        # generating new construction geometry.
        return self.path_entity_id

    def _samples(self, registry: EntityRegistry | None):
        if registry is None or self.path_entity_id < 0:
            return []
        count = _resolve_count(registry, self)
        if count <= 1:
            return []
        samples = sample_path(
            registry,
            self.path_entity_id,
            count,
            self.offset_to_start,
        )
        if len(samples) < 2:
            return []
        return samples

    def _slot_angle(self, tangent_angle: float) -> float:
        return tangent_angle if self.align_to_tangent else 0.0

    def template_placement(
        self,
        template_center: tuple[float, float],
        registry: Any | None = None,
    ) -> InstancePlacement:
        """Position 0 of the guide is the path start: the template's
        center is mapped to the first sample and rotated to its
        tangent. Without a usable path the identity placement is
        returned (creating an array from it fails elsewhere)."""
        samples = self._samples(registry) if registry is not None else []
        if not samples:
            return InstancePlacement(
                kind=PlacementKind.TRANSLATION, delta=(0.0, 0.0)
            )
        point, tangent_angle = samples[0]
        return InstancePlacement(
            kind=PlacementKind.CURVE_ALIGNED,
            angle=self._slot_angle(tangent_angle),
            center=template_center,
            target_center=point,
        )

    def member_placements(
        self,
        template_center: tuple[float, float],
        registry: Any | None = None,
    ) -> list[InstancePlacement]:
        """Computes the placements deriving slots 1..N-1 from the
        template (slot 0).

        Each placement is the rigid motion carrying the template from
        sample 0 onto sample j: a rotation by the tangent difference
        about sample 0 followed by the translation to sample j. Being
        relative to the template, user edits of the template shape
        propagate to every copy.
        """
        if registry is None:
            return []
        samples = self._samples(registry)
        if not samples:
            return []
        origin, origin_angle = samples[0]
        placements: list[InstancePlacement] = []
        for j in range(1, len(samples)):
            point, tangent_angle = samples[j]
            placements.append(
                InstancePlacement(
                    kind=PlacementKind.CURVE_ALIGNED,
                    angle=(
                        self._slot_angle(tangent_angle)
                        - self._slot_angle(origin_angle)
                    ),
                    center=origin,
                    target_center=point,
                )
            )
        return placements

    def create_master_geometry(
        self,
        center_pid: int | None,
        radius_pt_pid: int | None,
    ) -> tuple[list[Point], list[Entity], list[Constraint]]:
        # The guide path entity is created by the user before invoking
        # the tool, so the array reuses it as master instead of
        # generating new construction geometry here.
        return [], [], []


# ----------------------------------------------------------------------
# Path sampling
# ----------------------------------------------------------------------


def sample_path(
    registry: EntityRegistry,
    path_entity_id: int,
    count: int,
    offset_to_start: float = 0.0,
) -> list[tuple[tuple[float, float], float]]:
    """
    Returns ``count`` (point, tangent_angle) samples taken at equal
    arc-length spacing along the path, after an optional leading
    ``offset_to_start`` (in the same units as the path length, i.e.
    model space distance). The first sample is the path start (offset
    aside) and the last is the path end. Sample 0 is position 0,
    where the template sits; the copy of slot j lands on sample j.
    """
    entity = registry.get_entity(path_entity_id)
    if entity is None:
        return []

    polyline = _polyline_for(entity, registry)
    if len(polyline) < 2:
        return []

    cum = _cumulative_lengths(polyline)
    total = cum[-1]
    if total <= 1e-9:
        return []

    offset = max(0.0, min(offset_to_start, total * (1.0 - 1e-6)))
    usable = total - offset
    if count <= 1 or usable <= 0.0:
        return [(polyline[0], 0.0)]

    samples: list[tuple[tuple[float, float], float]] = []
    for j in range(count):
        s = offset + usable * j / (count - 1)
        point, tangent = _point_at_arclength(polyline, cum, s)
        angle = math.atan2(tangent[1], tangent[0])
        samples.append((point, angle))
    return samples


def _polyline_for(
    entity: Entity, registry: EntityRegistry
) -> list[tuple[float, float]]:
    def p(pid):
        pt = registry.get_point(pid)
        return (pt.x, pt.y)

    if isinstance(entity, Line):
        return [p(entity.p1_idx), p(entity.p2_idx)]

    if isinstance(entity, Arc):
        start = p(entity.start_idx)
        end = p(entity.end_idx)
        c = p(entity.center_idx)
        radius = math.hypot(start[0] - c[0], start[1] - c[1])
        start_a = math.atan2(start[1] - c[1], start[0] - c[0])
        end_a = math.atan2(end[1] - c[1], end[0] - c[0])
        sweep = end_a - start_a
        if entity.clockwise:
            while sweep >= 0:
                sweep -= 2 * math.pi
        else:
            while sweep <= 0:
                sweep += 2 * math.pi
        segments = max(8, int(abs(sweep) / (2 * math.pi) * 48))
        return [
            (
                c[0] + radius * math.cos(start_a + sweep * i / segments),
                c[1] + radius * math.sin(start_a + sweep * i / segments),
            )
            for i in range(segments + 1)
        ]

    if isinstance(entity, Bezier):
        start = p(entity.start_idx)
        end = p(entity.end_idx)
        cp1 = start
        if entity.cp1 is not None:
            cp1 = (start[0] + entity.cp1[0], start[1] + entity.cp1[1])
        cp2 = end
        if entity.cp2 is not None:
            cp2 = (end[0] + entity.cp2[0], end[1] + entity.cp2[1])
        segments = 32
        pts = []
        for i in range(segments + 1):
            t = i / segments
            u = 1.0 - t
            x = (
                u * u * u * start[0]
                + 3 * u * u * t * cp1[0]
                + 3 * u * t * t * cp2[0]
                + t * t * t * end[0]
            )
            y = (
                u * u * u * start[1]
                + 3 * u * u * t * cp1[1]
                + 3 * u * t * t * cp2[1]
                + t * t * t * end[1]
            )
            pts.append((x, y))
        return pts

    return []


def _cumulative_lengths(
    polyline: list[tuple[float, float]],
) -> list[float]:
    cum = [0.0]
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i - 1][0]
        dy = polyline[i][1] - polyline[i - 1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    return cum


def _point_at_arclength(
    polyline: list[tuple[float, float]],
    cum: list[float],
    s: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Returns (point, unit_tangent) at arc length ``s`` along the path."""
    if s <= 0.0:
        dx = polyline[1][0] - polyline[0][0]
        dy = polyline[1][1] - polyline[0][1]
        n = math.hypot(dx, dy) or 1.0
        return polyline[0], (dx / n, dy / n)
    if s >= cum[-1]:
        a = polyline[-2]
        b = polyline[-1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        return b, (dx / n, dy / n)

    # Binary search for the segment containing s.
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < s:
            lo = mid + 1
        else:
            hi = mid
    seg = max(1, lo)
    a = polyline[seg - 1]
    b = polyline[seg]
    seg_len = cum[seg] - cum[seg - 1]
    t = (s - cum[seg - 1]) / seg_len if seg_len > 0 else 0.0
    point = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return point, (dx / n, dy / n)


def path_length(
    registry: EntityRegistry,
    path_entity_id: int,
) -> float:
    """Returns the total arc length of the guide path entity."""
    entity = registry.get_entity(path_entity_id)
    if entity is None:
        return 0.0
    polyline = _polyline_for(entity, registry)
    if len(polyline) < 2:
        return 0.0
    return _cumulative_lengths(polyline)[-1]


def _resolve_count(
    registry: EntityRegistry, strategy: CurveAlongArrayStrategy
) -> int:
    """
    Returns the number of copies to place along the path.

    When ``spacing`` is positive it drives the count: the usable path
    length (total minus the start offset) is divided by the spacing,
    plus one for the template at the start. Otherwise the explicit
    ``count`` is used. The result is clamped to at least 1 and at most
    360.
    """
    if strategy.spacing > 1e-9 and strategy.path_entity_id >= 0:
        total = path_length(registry, strategy.path_entity_id)
        offset = min(strategy.offset_to_start, total)
        usable = max(total - offset, 0.0)
        if usable <= 0.0:
            return 1
        return max(1, min(360, int(usable / strategy.spacing) + 1))
    return max(1, int(strategy.count))


class CurveAlongArray(Array):
    """
    Persistent definition of an "array along curve" array. Carries the
    mode-specific state: the guide path, orientation and spacing
    parameters, and the template anchor recorded at position 0.
    """

    MODE = "curve_along"
    STRATEGY = CurveAlongArrayStrategy

    def __init__(
        self,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]] | None = None,
        count: int = 6,
        rotate_copies: bool = True,
        path_entity_id: int = -1,
        align_to_tangent: bool = True,
        offset_to_start: float = 0.0,
        spacing: float = 0.0,
        template_anchor: tuple[tuple[float, float], float] | None = None,
    ):
        super().__init__(
            uid,
            guide_circle_id,
            members=members,
            count=count,
            rotate_copies=rotate_copies,
        )
        self.path_entity_id = path_entity_id
        self.align_to_tangent = align_to_tangent
        self.offset_to_start = offset_to_start
        self.spacing = spacing
        self.template_anchor: tuple[tuple[float, float], float] | None = (
            template_anchor
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["path_entity_id"] = self.path_entity_id
        data["align_to_tangent"] = self.align_to_tangent
        data["offset_to_start"] = self.offset_to_start
        data["spacing"] = self.spacing
        if self.template_anchor is not None:
            (ax, ay), angle = self.template_anchor
            data["template_anchor"] = [
                [ax, ay],
                angle,
            ]
        return data

    def snapshot(self) -> dict[str, Any]:
        state = super().snapshot()
        state["path_entity_id"] = self.path_entity_id
        state["align_to_tangent"] = self.align_to_tangent
        state["offset_to_start"] = self.offset_to_start
        state["spacing"] = self.spacing
        state["template_anchor"] = self.template_anchor
        return state

    def restore(self, state: dict[str, Any]) -> None:
        super().restore(state)
        self.path_entity_id = state["path_entity_id"]
        self.align_to_tangent = state["align_to_tangent"]
        self.offset_to_start = state["offset_to_start"]
        self.spacing = state["spacing"]
        self.template_anchor = state["template_anchor"]

    def commit(self, strategy: ArrayStrategy) -> None:
        assert isinstance(strategy, CurveAlongArrayStrategy)
        super().commit(strategy)
        self.path_entity_id = strategy.path_entity_id
        self.align_to_tangent = strategy.align_to_tangent
        self.offset_to_start = strategy.offset_to_start
        self.spacing = strategy.spacing

    def params_changed(self, strategy: ArrayStrategy) -> bool:
        assert isinstance(strategy, CurveAlongArrayStrategy)
        # Note: rotate_copies has no effect on curve-along arrays
        # (orientation is governed by align_to_tangent), so it is
        # intentionally not compared here.
        return (
            super().params_changed(strategy)
            or strategy.path_entity_id != self.path_entity_id
            or strategy.align_to_tangent != self.align_to_tangent
            or strategy.offset_to_start != self.offset_to_start
            or strategy.spacing != self.spacing
        )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> CurveAlongArray:
        return cls(
            uid=data["uid"],
            guide_circle_id=data["guide_circle_id"],
            members=cls._parse_members(data),
            count=data.get("count", 6),
            rotate_copies=data.get("rotate_copies", True),
            path_entity_id=data.get("path_entity_id", -1),
            align_to_tangent=data.get("align_to_tangent", True),
            offset_to_start=data.get("offset_to_start", 0.0),
            spacing=data.get("spacing", 0.0),
            template_anchor=(
                (
                    (
                        float(data["template_anchor"][0][0]),
                        float(data["template_anchor"][0][1]),
                    ),
                    float(data["template_anchor"][1]),
                )
                if data.get("template_anchor") is not None
                else None
            ),
        )

    def make_strategy(
        self, registry: EntityRegistry
    ) -> CurveAlongArrayStrategy:
        """Builds the strategy from the stored fields; the placement
        reads the guide path from the registry at sampling time."""
        return CurveAlongArrayStrategy(
            count=self.count,
            rotate_copies=self.rotate_copies,
            path_entity_id=self.path_entity_id,
            align_to_tangent=self.align_to_tangent,
            offset_to_start=self.offset_to_start,
            spacing=self.spacing,
        )

    def reanchor_template(
        self,
        strategy: ArrayStrategy,
        registry: EntityRegistry,
        template_eids: list[int],
    ) -> None:
        """
        Re-places the template member onto position 0 of the guide.

        The template's position is guide-owned: its center is placed
        ABSOLUTELY onto the guide's position-0 point (not by a delta
        from the stored anchor, which goes stale as soon as the user
        drags the template) and rotated by the tangent change. This
        mirrors the circular array, whose members are re-projected
        onto the orbit. Template shape edits survive; position drags
        do not — so a user-drawn attachment between the template and
        the guide start (e.g. a snapped coincidence) agrees with the
        sync instead of fighting it.
        """
        template_points: list[Point] = []
        for eid in template_eids:
            entity = registry.get_entity(eid)
            if entity is None:
                continue
            for pid in entity.get_point_ids():
                pt = registry.get_point(pid)
                if pt is not None:
                    template_points.append(pt)
        if not template_points:
            return
        assert self.template_anchor is not None, (
            "reanchor requires a created array (template anchor set)"
        )
        current_center = resolve_template_center(
            registry, template_eids, template_points
        )
        placement = strategy.template_placement(current_center, registry)
        if placement.kind is not PlacementKind.CURVE_ALIGNED:
            # No usable path: position 0 is undefined, leave the
            # template where it is.
            logger.info(
                "ArraySync[%s]: reanchor skipped, guide has no usable "
                "path samples",
                self.uid[:8],
            )
            return
        target = placement.target_center
        delta = placement.angle - self.template_anchor[1]
        if (
            abs(target[0] - current_center[0]) < 1e-9
            and abs(target[1] - current_center[1]) < 1e-9
            and abs(delta) < 1e-12
        ):
            logger.info(
                "ArraySync[%s]: reanchor skipped, template already at "
                "position 0 (anchor=%r)",
                self.uid[:8],
                self.template_anchor,
            )
            return
        motion = InstancePlacement(
            kind=PlacementKind.CURVE_ALIGNED,
            angle=delta,
            center=current_center,
            target_center=target,
        )
        logger.info(
            "ArraySync[%s]: reanchor placing template %r -> %r "
            "(delta=%.6f rad)",
            self.uid[:8],
            current_center,
            target,
            delta,
        )
        apply_placement_to_entities(registry, template_eids, motion)
        self.template_anchor = (
            (float(target[0]), float(target[1])),
            float(placement.angle),
        )

    @classmethod
    def _from_strategy(
        cls,
        strategy: ArrayStrategy,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]],
        count: int,
        template_anchor: tuple[tuple[float, float], float] | None = None,
    ) -> CurveAlongArray:
        assert isinstance(strategy, CurveAlongArrayStrategy)
        return cls(
            uid=uid,
            guide_circle_id=guide_circle_id,
            members=members,
            count=count,
            rotate_copies=strategy.rotate_copies,
            path_entity_id=strategy.path_entity_id,
            align_to_tangent=strategy.align_to_tangent,
            offset_to_start=strategy.offset_to_start,
            spacing=strategy.spacing,
            template_anchor=template_anchor,
        )
