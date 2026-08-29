from __future__ import annotations

import copy
import logging
import uuid
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..arrays import (
    Array,
    ArrayStrategy,
    CircularArrayStrategy,
    InstancePlacement,
    apply_placement_to_entities,
    resolve_template_center,
)
from ..entities import Bezier, Circle, Ellipse, TextBoxEntity
from ..entities import Point as SketchPoint
from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity, Point
    from ..registry import EntityRegistry
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class CreateArrayCommand(SketchChangeCommand):
    """
    Generic command that turns the selected entities into an array
    (e.g. circular array) using a ArrayStrategy.

    Copies are parametrically linked to the template member; the
    strategy's master geometry (e.g. a construction guide circle)
    carries the array definition, which is registered on the sketch
    so it can be edited later.
    """

    def __init__(
        self,
        sketch: Sketch,
        strategy: ArrayStrategy,
        template_entity_ids: list[int],
        name: str | None = None,
    ):
        super().__init__(
            sketch, name if name is not None else _("Create Array")
        )
        self.strategy = strategy
        self.template_entity_ids = list(template_entity_ids)
        self.add_cmd: AddItemsCommand | None = None
        self.array: Array | None = None
        self.created_entity_ids: list[int] = []
        self.created_member_groups: list[list[int]] = []
        self.guide_circle_id: int | None = None
        self._pre_place_snapshot: list[tuple[Point, float, float]] = []
        self._pre_place_cp_snapshot: list[tuple[Bezier, Any, Any]] = []
        self._template_placement: InstancePlacement | None = None
        self._erased_constraints: list[Constraint] = []
        self._cloned_points: list[Point] = []
        self._clone_pid_map: dict[int, int] = {}
        self._extracted_helper_ids: list[int] = []

    @staticmethod
    def collect_template_point_ids(
        registry: EntityRegistry, template_entity_ids: list[int]
    ) -> list[int]:
        """Returns all unique point IDs referenced by the template entities."""
        pids: list[int] = []
        for eid in template_entity_ids:
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
        strategy: ArrayStrategy,
        template_entity_ids: list[int],
        center_pid: int | None = None,
        create_master: bool = True,
    ) -> dict[str, Any] | None:
        """
        Computes points, entities and constraints for the array.

        Args:
            registry: The entity registry to read template points from.
            strategy: The array strategy driving placements/linkage.
            template_entity_ids: Entities of the template member.
            center_pid: Reuse an existing array center point instead of
                creating one (used when regenerating an array).
            create_master: Whether to also build master geometry.

        Returns a dict with 'points', 'entities', 'constraints',
            'instance_maps', 'instance_point_groups',
            'instance_entity_groups', 'center_pid', 'radius_pt_pid' and
            'guide_circle' keys, or None if the array cannot be built.
        """
        template_pids = CreateArrayCommand.collect_template_point_ids(
            registry, template_entity_ids
        )
        if not template_pids or len(template_entity_ids) < 1:
            return None

        template_points = [registry.get_point(pid) for pid in template_pids]
        # For circles and ellipses, the template center is the shape's
        # explicit center point, not the bbox midpoint of all defining
        # points which would offset every copy.
        template_center = resolve_template_center(
            registry, template_entity_ids, template_points
        )

        temp_id = -1

        def next_temp_id() -> int:
            nonlocal temp_id
            temp_id -= 1
            return temp_id

        points: list[Point] = []
        center_is_new = center_pid is None
        center_point: Point | None = None
        if center_is_new and strategy.needs_center_point:
            assert isinstance(strategy, CircularArrayStrategy)
            center_pid = next_temp_id()
            center_point = SketchPoint(
                center_pid, strategy.center[0], strategy.center[1]
            )
            points.append(center_point)

        entities: list[Entity] = []
        instance_maps: list[dict[int, int]] = []
        instance_point_groups: list[list[Point]] = []
        instance_entity_groups: list[list[Entity]] = []

        constraints: list[Constraint] = []

        for placement in strategy.member_placements(template_center, registry):
            pid_map: dict[int, int] = {}
            instance_points: list[Point] = []
            for tpl_pt in template_points:
                new_pid = next_temp_id()
                nx, ny = placement.transform_point(tpl_pt.x, tpl_pt.y)
                # Copies are static baked geometry: fixed points keep
                # them out of the solver entirely; they are updated
                # only by re-deriving them from the template.
                instance_points.append(
                    SketchPoint(new_pid, nx, ny, fixed=True)
                )
                pid_map[tpl_pt.id] = new_pid

            instance_entities: list[Entity] = []
            eid_map: dict[int, int] = {}
            for eid in template_entity_ids:
                tpl_entity = registry.get_entity(eid)
                if tpl_entity is None:
                    continue
                clone = copy.deepcopy(tpl_entity)
                clone.id = next_temp_id()
                clone.array_copy = True
                eid_map[tpl_entity.id] = clone.id
                _remap_point_refs(clone, pid_map)
                if isinstance(clone, Bezier):
                    _transform_bezier_offsets(clone, placement)
                # Copied Ellipses must not retain the template's
                # helper-line IDs — those reference entities that will
                # be deleted, causing calculate_dependencies to cascade
                # the deletion onto the copies.
                if isinstance(clone, Ellipse):
                    clone.helper_line_ids = []
                instance_entities.append(clone)

            points.extend(instance_points)
            entities.extend(instance_entities)
            instance_maps.append(pid_map)
            instance_point_groups.append(instance_points)
            instance_entity_groups.append(instance_entities)

        radius_pt_pid: int | None = None
        guide_circle: Circle | None = None

        if (
            create_master
            and strategy.needs_center_point
            and center_pid is not None
        ):
            # The dialog's radius is a hard constraint on the guide
            # circle: the single source of truth for its size.
            assert isinstance(strategy, CircularArrayStrategy)
            if strategy.radius > 0.0:
                radius_pt_pid = next_temp_id()

            master_points, master_entities, master_constraints = (
                strategy.create_master_geometry(center_pid, radius_pt_pid)
            )
            points.extend(master_points)
            entities.extend(master_entities)
            constraints.extend(master_constraints)
            guide_circle = next(
                (e for e in master_entities if isinstance(e, Circle)),
                None,
            )

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
            "instance_maps": instance_maps,
            "instance_point_groups": instance_point_groups,
            "instance_entity_groups": instance_entity_groups,
            "center_pid": center_pid,
            "center_point": center_point,
            "radius_pt_pid": radius_pt_pid,
            "guide_circle": guide_circle,
        }

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self._redo()

        # Step 2-3 of the array process: extract the template (its
        # external constraints are erased and it owns its points).
        self._extract_template()

        # Step 4: place the template at position 0 on the guide and
        # bake its position. Step 5: the copies derive from the baked
        # template, so this placement and the strategy's member
        # placements are the single source of truth.
        self._place_template()

        result = self.calculate_geometry(
            self.sketch.registry,
            self.strategy,
            self.template_entity_ids,
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
        # Strategies that generate no master geometry of their own
        # reuse an existing entity (e.g. the pre-drawn guide path of a
        # curve-along array) as the array's master.
        if self.guide_circle_id is None:
            self.guide_circle_id = self.strategy.existing_master_id()
        logger.info(
            "ArrayCreate: mode=%s strategy=%r members=%d groups=%r "
            "guide_circle=e%s",
            self.array.mode if self.array else "<unregistered>",
            self.strategy,
            len(self.created_entity_ids),
            [len(g) for g in self.created_member_groups],
            self.guide_circle_id,
        )
        self._register_array()

    def _register_array(self) -> None:
        if self.guide_circle_id is None:
            return
        if self.array is None:
            count = max(int(self.strategy.count), 1)
            # Uniform structure for every strategy: member 0 is the
            # template, members 1..N-1 are the copies, in placement
            # order.
            members = [(0, list(self.template_entity_ids))] + [
                (slot, list(group))
                for slot, group in zip(
                    range(1, count), self.created_member_groups
                )
            ]
            anchor = None
            if (
                self.strategy.uses_template_anchor
                and self._template_placement is not None
            ):
                target = self._template_placement.target_center
                anchor = (
                    (float(target[0]), float(target[1])),
                    float(self._template_placement.angle),
                )
            self.array = Array.from_strategy(
                self.strategy,
                uid=str(uuid.uuid4()),
                guide_circle_id=self.guide_circle_id,
                members=members,
                count=count,
                template_anchor=anchor,
            )
        existing = next(
            (p for p in self.sketch.arrays if p.uid == self.array.uid),
            None,
        )
        if existing is None:
            self.sketch.arrays.append(self.array)
            # Seed the path-point cache so the first solve doesn't
            # trigger a spurious re-apply.
            self._refresh_sync_caches()

    def _refresh_sync_caches(self) -> None:
        """Caches the guide's and template's geometry signature so the
        first solve after creation doesn't trigger a re-apply."""
        if self.array is None:
            return
        self.array.refresh_caches(self.sketch.registry, self.strategy)

    def _extract_template(self) -> None:
        """
        Step 2 of the array process: makes the selected entities a
        self-contained template.

        The template's helper geometry (an ellipse's construction
        lines, a text box's construction lines) belongs to it: the
        helpers share its points and follow the placement.

        Points shared with entities outside the template (e.g. the
        guide path) are cloned so the template owns its geometry, and
        constraints referencing anything outside the template are
        erased: placing the template must never drag other geometry
        along, nor leave constraints for the solver to fight over.
        """
        registry = self.sketch.registry
        template = set(self.template_entity_ids)
        helper_ids = self._template_helper_ids(registry)
        template.update(helper_ids)
        self._extracted_helper_ids = sorted(helper_ids)
        template_pids = set(
            CreateArrayCommand.collect_template_point_ids(
                registry, self.template_entity_ids
            )
        )

        # Erase external constraints: everything that touches template
        # geometry but is not internal to the template group.
        internal = self.sketch.get_internal_constraints(template)
        internal_ids = {id(c) for c in internal}
        self._erased_constraints = []
        for constr in list(self.sketch.constraints):
            if id(constr) in internal_ids:
                continue
            pids = constr.get_referenced_point_ids()
            eids = constr.get_referenced_entity_ids()
            if pids & template_pids or eids & template:
                self.sketch.constraints.remove(constr)
                self._erased_constraints.append(constr)

        # Clone points the template shares with the rest of the
        # sketch; internal constraints and helper geometry follow the
        # template to the clones.
        self._clone_pid_map = {}
        for entity in registry.entities:
            if entity.id in template:
                continue
            for pid in entity.get_point_ids():
                if pid in template_pids and pid not in self._clone_pid_map:
                    pt = registry.get_point(pid)
                    clone_pid = registry.add_point(pt.x, pt.y)
                    self._clone_pid_map[pid] = clone_pid
                    self._cloned_points.append(registry.get_point(clone_pid))
        if self._clone_pid_map:
            for eid in self.template_entity_ids + self._extracted_helper_ids:
                entity = registry.get_entity(eid)
                if entity is not None:
                    _remap_point_refs(entity, self._clone_pid_map)
            for constr in internal:
                _remap_point_refs(constr, self._clone_pid_map)

    def _template_helper_ids(self, registry: EntityRegistry) -> list[int]:
        """
        Returns the helper geometry belonging to the template entities:
        registered helpers (an ellipse's helper_line_ids, a text box's
        construction lines) plus any construction/invisible entity
        that is fully attached to the template's points (e.g. an
        ellipse's visible axis lines, which are not registered as
        helpers). The helpers share the template's points and must be
        extracted — and placed — along with it.
        """
        helper_ids: list[int] = []
        for eid in self.template_entity_ids:
            entity = registry.get_entity(eid)
            if isinstance(entity, Ellipse):
                helper_ids.extend(entity.helper_line_ids)
            elif isinstance(entity, TextBoxEntity):
                helper_ids.extend(entity.construction_line_ids)
        template_pids = set(
            CreateArrayCommand.collect_template_point_ids(
                registry, self.template_entity_ids
            )
        )
        taken = set(helper_ids) | set(self.template_entity_ids)
        for entity in registry.entities:
            if entity.id in taken:
                continue
            if not (entity.construction or entity.invisible):
                continue
            point_ids = entity.get_point_ids()
            if point_ids and set(point_ids) <= template_pids:
                helper_ids.append(entity.id)
        return helper_ids

    def _rollback_extraction(self) -> None:
        """Undoes _extract_template: restores erased constraints,
        hands the shared points back and removes the clones."""
        if self._clone_pid_map:
            inverse = {v: k for k, v in self._clone_pid_map.items()}
            for eid in self.template_entity_ids + self._extracted_helper_ids:
                entity = self.sketch.registry.get_entity(eid)
                if entity is not None:
                    _remap_point_refs(entity, inverse)
            for constr in self.sketch.get_internal_constraints(
                set(self.template_entity_ids)
            ):
                _remap_point_refs(constr, inverse)
            self.sketch.registry.points = [
                p
                for p in self.sketch.registry.points
                if p.id not in self._clone_pid_map.values()
            ]
        if self._erased_constraints:
            self.sketch.constraints.extend(self._erased_constraints)

    def _reapply_extraction(self) -> None:
        """Redoes _extract_template after an undo rolled it back."""
        if self._clone_pid_map:
            self.sketch.registry.points.extend(self._cloned_points)
            for eid in self.template_entity_ids + self._extracted_helper_ids:
                entity = self.sketch.registry.get_entity(eid)
                if entity is not None:
                    _remap_point_refs(entity, self._clone_pid_map)
            for constr in self.sketch.get_internal_constraints(
                set(self.template_entity_ids)
            ):
                _remap_point_refs(constr, self._clone_pid_map)
        for constr in self._erased_constraints:
            if constr in self.sketch.constraints:
                self.sketch.constraints.remove(constr)

    def _place_template(self) -> None:
        """
        Moves the template entities onto position 0 of the guide
        (slot 0), snapshotting their pre-place positions for undo.
        The placement is kept so redo can re-apply it verbatim.
        """
        registry = self.sketch.registry
        strategy = self.strategy
        pids = CreateArrayCommand.collect_template_point_ids(
            registry, self.template_entity_ids
        )
        pts = [registry.get_point(pid) for pid in pids]
        pts = [p for p in pts if p is not None]
        if not pts:
            return
        template_center = resolve_template_center(
            registry, self.template_entity_ids, pts
        )
        placement = strategy.template_placement(template_center, registry)
        self._pre_place_snapshot = [(p, p.x, p.y) for p in pts]
        self._pre_place_cp_snapshot = [
            (entity, entity.cp1, entity.cp2)
            for entity in (
                registry.get_entity(eid) for eid in self.template_entity_ids
            )
            if isinstance(entity, Bezier)
        ]
        self._template_placement = placement
        apply_placement_to_entities(
            registry, self.template_entity_ids, placement
        )

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
        if self.array is not None:
            self.sketch.arrays = [
                p for p in self.sketch.arrays if p.uid != self.array.uid
            ]
        # Restore the template's pre-place position, then undo the
        # extraction (shared points handed back, external constraints
        # restored).
        for pt, x, y in self._pre_place_snapshot:
            pt.x = x
            pt.y = y
        for entity, cp1, cp2 in self._pre_place_cp_snapshot:
            entity.cp1 = cp1
            entity.cp2 = cp2
        self._rollback_extraction()

    def _redo(self) -> None:
        assert self.add_cmd is not None
        self._reapply_extraction()
        if self._template_placement is not None:
            apply_placement_to_entities(
                self.sketch.registry,
                self.template_entity_ids,
                self._template_placement,
            )
        self.add_cmd._do_execute()
        self._register_array()


def _remap_point_refs(
    obj: Entity | Constraint, pid_map: dict[int, int]
) -> None:
    """Rewrites point ID references on an entity or constraint."""
    for attr, value in vars(obj).items():
        # Note: bool is an int subclass; never remap flag attributes.
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in pid_map
        ):
            setattr(obj, attr, pid_map[value])


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
