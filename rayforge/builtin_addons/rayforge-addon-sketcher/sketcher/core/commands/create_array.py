from __future__ import annotations

import logging
import uuid
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.sketch.entities.bezier import Bezier
from raygeo.sketch.entities.circle import Circle

from ..arrays import (
    Array,
    ArrayStrategy,
    CircularArrayStrategy,
    InstancePlacement,
    resolve_template_center,
)
from ..entity_group import EntityGroup
from ..entity_ops import clone_entity
from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
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
        self._pre_place_snapshot: list[tuple[int, float, float]] = []
        self._pre_place_cp_snapshot: list[tuple[int, Any, Any]] = []
        self._template_placement: InstancePlacement | None = None
        self._erased_constraints: list[Any] = []
        self._clone_pid_map: dict[int, int] = {}
        self._extracted_helper_ids: list[int] = []
        self._extra_template_pids: list[int] = []
        self._standalone_pids: dict[int, list[int]] = {}
        self._clone_point_dicts: list[dict[str, Any]] = []

    @staticmethod
    def calculate_geometry(
        sketch: Sketch,
        add_cmd: AddItemsCommand,
        strategy: ArrayStrategy,
        template_entity_ids: list[int],
        center_pid: int | None = None,
        create_master: bool = True,
        extra_pids: set[int] | None = None,
    ) -> dict[str, Any] | None:
        """
        Builds the array geometry through ``add_cmd``, allocating real
        ids as it goes.

        Args:
            sketch: The sketch receiving the geometry.
            add_cmd: The command recording the added items.
            strategy: The array strategy driving placements/linkage.
            template_entity_ids: Entities of the template member.
            center_pid: Reuse an existing array center point instead of
                creating one (used when regenerating an array).
            create_master: Whether to also build master geometry.
            extra_pids: Standalone point IDs (e.g. a rectangle's
                center) that belong to the template but are not
                referenced by any entity.

        Returns a dict with 'instance_maps', 'instance_entity_groups',
            'instance_extra_pids', 'center_pid', 'radius_pt_pid' and
            'guide_circle_id' keys, or None if the array cannot be
            built. 'instance_extra_pids' holds, per instance index,
            the standalone points in the order they appear in
            ``extra_pids`` within the template's point order (empty
            when no ``extra_pids`` are given).
        """
        registry = sketch.registry
        template_group = EntityGroup(registry, template_entity_ids)
        template_pids = template_group.point_ids()
        if extra_pids:
            for pid in extra_pids:
                if pid not in template_pids:
                    template_pids.append(pid)
        if not template_pids or len(template_entity_ids) < 1:
            return None

        template_points = {
            point.id: (point.x, point.y)
            for point in registry.points()
        }
        # For circles and ellipses, the template center is the shape's
        # explicit center point, not the bbox midpoint of all defining
        # points which would offset every copy.
        template_center = resolve_template_center(
            registry, template_entity_ids
        )

        center_is_new = center_pid is None
        if center_is_new and strategy.needs_center_point:
            assert isinstance(strategy, CircularArrayStrategy)
            center_pid = add_cmd.add_point(
                strategy.center[0], strategy.center[1]
            )

        template_specs = []
        for eid in template_entity_ids:
            entity = registry.get_entity_object(eid)
            if entity is None:
                continue
            template_specs.append((eid, entity))

        instance_maps: list[dict[int, int]] = []
        instance_entity_groups: list[list[int]] = []
        instance_extra_pids: list[list[int]] = []
        extra_order: list[int] = []
        if extra_pids:
            extra_set = set(extra_pids)
            extra_order = [p for p in template_pids if p in extra_set]

        for placement in strategy.member_placements(template_center, registry):
            pid_map: dict[int, int] = {}
            for tpl_pid in template_pids:
                position = template_points.get(tpl_pid)
                if position is None:
                    continue
                nx, ny = placement.transform_point(*position)
                # Copies are static baked geometry: fixed points keep
                # them out of the solver entirely; they are updated
                # only by re-deriving them from the template.
                pid_map[tpl_pid] = add_cmd.add_point(nx, ny, fixed=True)

            instance_entities: list[int] = []
            for tpl_eid, tpl_entity in template_specs:
                clone = clone_entity(registry, tpl_entity, pid_map, placement)
                eid = add_cmd.add_entity(clone)
                sketch.flags.set_array_copy(eid, True)
                sketch.copy_entity_flags(tpl_eid, eid)
                instance_entities.append(eid)

            instance_maps.append(pid_map)
            instance_entity_groups.append(instance_entities)
            instance_extra_pids.append(
                [pid_map[tpl_pid] for tpl_pid in extra_order]
            )

        radius_pt_pid: int | None = None
        guide_circle_id: int | None = None

        if (
            create_master
            and strategy.needs_center_point
            and center_pid is not None
        ):
            # The dialog's radius is a hard constraint on the guide
            # circle: the single source of truth for its size.
            assert isinstance(strategy, CircularArrayStrategy)
            if strategy.radius > 0.0:
                radius_pt_pid = add_cmd.add_point(
                    strategy.center[0] + strategy.radius,
                    strategy.center[1],
                )

            master_points, master_entities, master_constraints = (
                strategy.create_master_geometry(center_pid, radius_pt_pid)
            )
            for point in master_points:
                add_cmd.add_point(point.x, point.y, point.fixed)
            for entity in master_entities:
                eid = add_cmd.add_entity(entity)
                if isinstance(entity, Circle):
                    guide_circle_id = eid
            for constraint_dict in master_constraints:
                resolved = dict(constraint_dict)
                if resolved.get("entity_id") == -1:
                    resolved["entity_id"] = guide_circle_id
                add_cmd.add_constraint(resolved)

        return {
            "instance_maps": instance_maps,
            "instance_entity_groups": instance_entity_groups,
            "instance_extra_pids": instance_extra_pids,
            "center_pid": center_pid,
            "radius_pt_pid": radius_pt_pid,
            "guide_circle_id": guide_circle_id,
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

        self.add_cmd = AddItemsCommand(self.sketch, "")
        result = self.calculate_geometry(
            self.sketch,
            self.add_cmd,
            self.strategy,
            self.template_entity_ids,
            extra_pids=set(self._extra_template_pids) or None,
        )
        if not result:
            return

        self.created_member_groups = result["instance_entity_groups"]
        # Standalone points per copy (e.g. a rectangle's center), in
        # template order. Slot 0 (the template's own standalone
        # points) is attached in _register_array.
        self._standalone_pids = {
            slot: list(pids)
            for slot, pids in enumerate(
                result["instance_extra_pids"], start=1
            )
        }
        self.created_entity_ids = [
            eid for group in self.created_member_groups for eid in group
        ]

        self.guide_circle_id = result["guide_circle_id"]
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
            # The template's standalone points are member data of slot
            # 0; the copies' were resolved in _do_execute.
            self.array.standalone_pids = {
                0: list(self._extra_template_pids),
                **self._standalone_pids,
            }
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
        template_group = EntityGroup(
            registry,
            self.template_entity_ids,
            flags=self.sketch.flags,
        )
        helper_ids = template_group.helper_ids()
        template.update(helper_ids)
        self._extracted_helper_ids = sorted(helper_ids)
        template_pids = set(template_group.point_ids())

        # Pull in standalone Points that belong to the template shape
        # but are not referenced by any entity (e.g. a rectangle's
        # center, referenced only by a SymmetryConstraint between two
        # corner points).
        self._extra_template_pids = self.sketch.find_standalone_point_ids(
            self.template_entity_ids + self._extracted_helper_ids
        )
        template_pids.update(self._extra_template_pids)
        template.update(self._extra_template_pids)

        # Erase external constraints: everything that touches template
        # geometry but is not internal to the template group.
        internal = self.sketch.get_internal_constraints(template)
        internal_ids = {id(c) for c in internal}
        self._erased_constraints = []
        for constr in list(self.sketch.constraints):
            if id(constr) in internal_ids:
                continue
            pids = constr.get_referenced_point_ids()
            if pids & template_pids:
                self.sketch.remove_constraint_object(constr)
                self._erased_constraints.append(constr)

        # Clone points the template shares with the rest of the
        # sketch; internal constraints and helper geometry follow the
        # template to the clones.
        self._clone_pid_map = {}
        for eid in registry.entity_ids():
            if eid in template:
                continue
            for pid in registry.entity_point_ids(eid) or []:
                if pid in template_pids and pid not in self._clone_pid_map:
                    position = registry.get_point(pid)
                    if position is None:
                        continue
                    clone_pid = registry.add_point(*position)
                    self._clone_pid_map[pid] = clone_pid
        if self._clone_pid_map:
            self.sketch.remap_entity_point_refs(
                self._clone_pid_map, entity_ids=template
            )
            self.sketch.rebuild_constraints(internal, self._clone_pid_map)

    def _rollback_extraction(self) -> None:
        """Undoes _extract_template: restores erased constraints,
        hands the shared points back and removes the clones."""
        registry = self.sketch.registry
        if self._clone_pid_map:
            self._clone_point_dicts = [
                {
                    "id": clone_pid,
                    "x": position[0],
                    "y": position[1],
                    "fixed": False,
                }
                for clone_pid, position in (
                    (pid, registry.get_point(pid))
                    for pid in self._clone_pid_map.values()
                )
                if position is not None
            ]
            inverse = {v: k for k, v in self._clone_pid_map.items()}
            internal = self.sketch.get_internal_constraints(
                set(self.template_entity_ids)
                | set(self._extracted_helper_ids)
                | set(self._extra_template_pids)
            )
            self.sketch.rebuild_constraints(internal, inverse)
            self.sketch.remap_entity_point_refs(inverse)
            registry.remove_points(list(self._clone_pid_map.values()))
        if self._erased_constraints:
            self.sketch.add_constraints(self._erased_constraints)

    def _reapply_extraction(self) -> None:
        """Redoes _extract_template after an undo rolled it back."""
        if self._clone_pid_map:
            self.sketch.restore_structure(self._clone_point_dicts, [])
            self.sketch.remap_entity_point_refs(self._clone_pid_map)
            internal = self.sketch.get_internal_constraints(
                set(self.template_entity_ids)
                | set(self._extracted_helper_ids)
                | set(self._extra_template_pids)
            )
            self.sketch.rebuild_constraints(internal, self._clone_pid_map)
        for constr in self._erased_constraints:
            self.sketch.remove_constraint_object(constr)

    def _place_template(self) -> None:
        """
        Moves the template entities onto position 0 of the guide
        (slot 0), snapshotting their pre-place positions for undo.
        The placement is kept so redo can re-apply it verbatim. The
        template's standalone points ride along with the same rigid
        placement, so the shape stays internally consistent (e.g. a
        symmetry center remains the center).
        """
        registry = self.sketch.registry
        strategy = self.strategy
        template_group = EntityGroup(
            registry, self.template_entity_ids, flags=self.sketch.flags
        )
        pts = template_group.points()
        if not pts:
            return
        template_center = resolve_template_center(
            registry, self.template_entity_ids
        )
        placement = strategy.template_placement(template_center, registry)
        # The helper geometry is placed with the member: a helper's
        # points complete the member's shape (a text box's fourth
        # frame corner lives only on its construction lines).
        placement_group = template_group.with_helpers()
        self._pre_place_snapshot = placement_group.snapshot_positions()
        for pid in self._extra_template_pids:
            position = registry.get_point(pid)
            if position is None:
                continue
            self._pre_place_snapshot.append((pid, position[0], position[1]))
            x, y = placement.transform_point(*position)
            registry.set_point_position(pid, x, y)
        self._pre_place_cp_snapshot = [
            (
                entity.id,
                entity.cp1_offset,
                entity.cp2_offset,
            )
            for entity in placement_group.entities()
            if isinstance(entity, Bezier)
        ]
        self._template_placement = placement
        placement_group.apply_placement(placement)

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
        registry = self.sketch.registry
        for entity_id, cp1, cp2 in self._pre_place_cp_snapshot:
            state: dict[str, Any] = {}
            if cp1 is not None:
                state["cp1"] = [cp1[0], cp1[1]]
            if cp2 is not None:
                state["cp2"] = [cp2[0], cp2[1]]
            if state:
                registry.set_state(entity_id, state)
        EntityGroup.restore_positions(
            self._pre_place_snapshot, registry
        )
        self._rollback_extraction()

    def _redo(self) -> None:
        assert self.add_cmd is not None
        self._reapply_extraction()
        if self._template_placement is not None:
            registry = self.sketch.registry
            placement_group = EntityGroup(
                registry,
                self.template_entity_ids,
                flags=self.sketch.flags,
            ).with_helpers()
            placement_group.apply_placement(self._template_placement)
            for pid in self._extra_template_pids:
                position = registry.get_point(pid)
                if position is None:
                    continue
                x, y = self._template_placement.transform_point(*position)
                registry.set_point_position(pid, x, y)
        self.add_cmd._redo()
        self._register_array()
