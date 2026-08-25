from __future__ import annotations

import copy
import logging
import math
import uuid
from dataclasses import replace
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..constraints import PointOnLineConstraint
from ..entities import Bezier, Circle
from ..entities import Point as SketchPoint
from ..patterns import (
    CircularPatternParams,
    PatternDefinition,
    PatternStrategy,
    SketchArrayMode,
    make_pattern_strategy,
)
from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity, Point
    from ..registry import EntityRegistry
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class CreatePatternCommand(SketchChangeCommand):
    """
    Generic command that duplicates seed entities into a pattern
    (e.g. circular array) using a PatternStrategy.

    Copies are parametrically linked to the template member; the
    strategy's master geometry (e.g. a construction guide circle)
    carries the pattern definition, which is registered on the sketch
    so it can be edited later.
    """

    def __init__(
        self,
        sketch: Sketch,
        mode: SketchArrayMode,
        params: CircularPatternParams,
        seed_entity_ids: list[int],
        name: str | None = None,
    ):
        super().__init__(
            sketch, name if name is not None else _("Create Pattern")
        )
        self.mode = mode
        self.params = params
        self.seed_entity_ids = list(seed_entity_ids)
        self.add_cmd: AddItemsCommand | None = None
        self.pattern: PatternDefinition | None = None
        self.created_entity_ids: list[int] = []
        self.created_member_groups: list[list[int]] = []
        self.guide_circle_id: int | None = None

    @staticmethod
    def collect_seed_point_ids(
        registry: EntityRegistry, seed_entity_ids: list[int]
    ) -> list[int]:
        """Returns all unique point IDs referenced by the seed entities."""
        pids: list[int] = []
        for eid in seed_entity_ids:
            entity = registry.get_entity(eid)
            if entity is None:
                continue
            for pid in entity.get_point_ids():
                if pid not in pids:
                    pids.append(pid)
        return pids

    @staticmethod
    def calculate_geometry(
        registry: EntityRegistry,
        strategy: PatternStrategy,
        seed_entity_ids: list[int],
        params: Any,
        center_pid: int | None = None,
        create_master: bool = True,
    ) -> dict[str, Any] | None:
        """
        Computes points, entities and constraints for the pattern.

        Args:
            registry: The entity registry to read template points from.
            strategy: The pattern strategy driving placements/linkage.
            seed_entity_ids: Entities of the template member.
            params: Mode-specific parameters.
            center_pid: Reuse an existing pattern center point instead of
                creating one (used when regenerating an array).
            create_master: Whether to also build master geometry.

        Returns a dict with 'points', 'entities', 'constraints',
        'instance_maps', 'instance_point_groups',
        'instance_entity_groups', 'center_pid', 'radius_pt_pid' and
        'guide_circle' keys, or None if the pattern cannot be built.
        """
        seed_pids = CreatePatternCommand.collect_seed_point_ids(
            registry, seed_entity_ids
        )
        if not seed_pids or len(seed_entity_ids) < 1:
            return None

        seed_points = [registry.get_point(pid) for pid in seed_pids]
        seed_center = _points_bbox_center(seed_points)

        temp_id = -1

        def next_temp_id() -> int:
            nonlocal temp_id
            temp_id -= 1
            return temp_id

        points: list[Point] = []
        center_is_new = center_pid is None
        if center_is_new and strategy.needs_center_point:
            center_pid = next_temp_id()
            points.append(
                SketchPoint(center_pid, params.center[0], params.center[1])
            )

        entities: list[Entity] = []
        instance_maps: list[dict[int, int]] = []
        instance_point_groups: list[list[Point]] = []
        instance_entity_groups: list[list[Entity]] = []

        for placement in strategy.calculate_placements(seed_center):
            pid_map: dict[int, int] = {}
            instance_points: list[Point] = []
            for seed_pt in seed_points:
                new_pid = next_temp_id()
                nx, ny = placement.transform_point(seed_pt.x, seed_pt.y)
                instance_points.append(SketchPoint(new_pid, nx, ny))
                pid_map[seed_pt.id] = new_pid

            instance_entities: list[Entity] = []
            for eid in seed_entity_ids:
                seed_entity = registry.get_entity(eid)
                if seed_entity is None:
                    continue
                clone = copy.deepcopy(seed_entity)
                clone.id = next_temp_id()
                clone.pattern_copy = True
                _remap_entity(clone, pid_map)
                if isinstance(clone, Bezier):
                    _transform_bezier_offsets(clone, placement)
                instance_entities.append(clone)

            points.extend(instance_points)
            entities.extend(instance_entities)
            instance_maps.append(pid_map)
            instance_point_groups.append(instance_points)
            instance_entity_groups.append(instance_entities)

        constraints: list[Constraint] = strategy.build_linkage_constraints(
            [(j + 1, pid_map) for j, pid_map in enumerate(instance_maps)],
            center_pid,
        )

        radius_pt_pid: int | None = None
        guide_circle: Circle | None = None
        ref_pid: int | None = None
        effective_radius = params.radius

        if (
            create_master
            and strategy.needs_center_point
            and center_pid is not None
        ):
            # Derive the guide circle radius from the anchor point so the
            # pin constraint starts exactly satisfied. An inconsistent
            # start would make the solver fling the geometry around.
            if params.rotate_copies and seed_points:
                ref_pid = _choose_reference_point(seed_points)
                logger.debug("Pattern: anchor point pid=%s", ref_pid)
                ref_pt = next(p for p in seed_points if p.id == ref_pid)
                if center_is_new:
                    cx, cy = params.center
                else:
                    center_pt = registry.get_point(center_pid)
                    cx, cy = center_pt.x, center_pt.y
                d = math.hypot(ref_pt.x - cx, ref_pt.y - cy)
                if d > 1e-6:
                    # Only anchor when consistent: a reference point
                    # sitting on the center cannot be pinned onto the
                    # circle without flinging the geometry.
                    effective_radius = d
                else:
                    ref_pid = None
            if effective_radius > 0.0:
                radius_pt_pid = next_temp_id()

        if create_master:
            master_strategy = strategy
            if effective_radius != params.radius:
                master_params = replace(params, radius=effective_radius)
                master_strategy = type(strategy)(master_params)
            master_points, master_entities, master_constraints = (
                master_strategy.create_master_geometry(
                    center_pid, radius_pt_pid
                )
            )
            points.extend(master_points)
            entities.extend(master_entities)
            constraints.extend(master_constraints)
            guide_circle = next(
                (e for e in master_entities if isinstance(e, Circle)),
                None,
            )

        # Anchor the template member onto the guide circle so the array
        # and its circle form one rigid parametric unit: the circle
        # always crosses every member, the radius dimension resizes the
        # array, and the radius point handle scales it interactively.
        if guide_circle is not None and ref_pid is not None:
            constraints.append(PointOnLineConstraint(ref_pid, guide_circle.id))

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
            "instance_maps": instance_maps,
            "instance_point_groups": instance_point_groups,
            "instance_entity_groups": instance_entity_groups,
            "center_pid": center_pid,
            "radius_pt_pid": radius_pt_pid,
            "guide_circle": guide_circle,
        }

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self._redo()

        result = self.calculate_geometry(
            self.sketch.registry,
            make_pattern_strategy(self.mode, self.params),
            self.seed_entity_ids,
            self.params,
        )
        if not result:
            return

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=result["points"],
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self.add_cmd._do_execute()

        # AddItemsCommand assigns final IDs to the created objects in
        # place; resolve the instance groups into real IDs afterwards.
        self.created_member_groups = [
            [e.id for e in group] for group in result["instance_entity_groups"]
        ]
        self.created_entity_ids = [
            eid for group in self.created_member_groups for eid in group
        ]

        guide_circle = result["guide_circle"]
        self.guide_circle_id = (
            guide_circle.id if guide_circle is not None else None
        )
        logger.info(
            "PatternCreate: mode=%s params=%r members=%d groups=%r "
            "guide_circle=e%s",
            self.mode.value,
            self.params,
            len(self.created_entity_ids),
            [len(g) for g in self.created_member_groups],
            self.guide_circle_id,
        )
        self._register_pattern()

    def _register_pattern(self) -> None:
        if self.guide_circle_id is None:
            return
        if self.pattern is None:
            count = max(int(self.params.count), 1)
            members = [(0, list(self.seed_entity_ids))] + [
                (slot, list(group))
                for slot, group in zip(
                    range(1, count), self.created_member_groups
                )
            ]
            self.pattern = PatternDefinition(
                uid=str(uuid.uuid4()),
                mode=self.mode,
                guide_circle_id=self.guide_circle_id,
                members=members,
                count=count,
                total_angle_deg=self.params.total_angle_deg,
                rotate_copies=self.params.rotate_copies,
            )
        existing = next(
            (p for p in self.sketch.patterns if p.uid == self.pattern.uid),
            None,
        )
        if existing is None:
            self.sketch.patterns.append(self.pattern)

    def _redo(self) -> None:
        assert self.add_cmd is not None
        self.add_cmd._do_execute()
        self._register_pattern()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
        if self.pattern is not None:
            self.sketch.patterns = [
                p for p in self.sketch.patterns if p.uid != self.pattern.uid
            ]


def _points_bbox_center(points: list[Point]) -> tuple[float, float]:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def _choose_reference_point(seed_points: list[Point]) -> int:
    """
    Picks the anchor point that represents a member on the guide
    circle: the unfixed point closest to the member's bbox center.
    """
    pool = [p for p in seed_points if not p.fixed] or seed_points
    cx, cy = _points_bbox_center(seed_points)
    return min(pool, key=lambda p: math.hypot(p.x - cx, p.y - cy)).id


def _remap_entity(entity: Entity, pid_map: dict[int, int]) -> None:
    """Rewrites point ID references on a cloned entity."""
    for attr, value in vars(entity).items():
        # Note: bool is an int subclass; never remap flag attributes.
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in pid_map
        ):
            setattr(entity, attr, pid_map[value])


def _transform_bezier_offsets(clone: Bezier, placement: Any) -> None:
    """
    Rotates bezier control point offsets so curve orientation follows
    rotation placements. Offsets are relative to their anchor endpoints,
    which are already transformed.
    """
    if clone.cp1 is not None:
        clone.cp1 = placement.transform_offset(*clone.cp1)
    if clone.cp2 is not None:
        clone.cp2 = placement.transform_offset(*clone.cp2)
