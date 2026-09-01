from __future__ import annotations

import logging
from collections.abc import Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..arrays import (
    Array,
    ArrayStrategy,
    InstancePlacement,
    resolve_template_center,
)
from ..entities import Circle
from ..entity_group import EntityGroup
from ..params import ParameterContext
from .base import SketchChangeCommand
from .create_array import CreateArrayCommand
from .items import AddItemsCommand, RemoveItemsCommand

if TYPE_CHECKING:
    from ..entities import Entity, Point
    from ..registry import EntityRegistry
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class EditArrayCommand(SketchChangeCommand):
    """
    Regenerates an array's instances from its definition.

    Members are groups of entities. The first surviving member (lowest
    slot) acts as the template. When the array parameters did not
    change, only missing slots are re-created. When they changed, every
    other member is removed and fresh copies are distributed at the new
    placements. All mode-specific behaviour (which params matter, how to
    snapshot/commit them, whether a similarity transform applies) is
    delegated to the array's strategy so this command stays generic.
    """

    def __init__(
        self,
        sketch: Sketch,
        array_def: Array,
        strategy: ArrayStrategy,
        force_full_regen: bool = False,
        capture_snapshot: bool = True,
        old_frame: tuple[tuple[float, float], float] | None = None,
    ):
        super().__init__(sketch, _("Edit Array"))
        # Internal re-applies (sync_arrays) are not undoable and must
        # not pay for a full-sketch snapshot on every solve.
        self._capture_snapshot = capture_snapshot
        self.array = array_def
        self.strategy = strategy
        self.add_cmd: AddItemsCommand | None = None
        self.remove_cmd: RemoveItemsCommand | None = None
        self.created_entity_ids: list[int] = []
        self.created_groups: list[tuple[int, list[int]]] = []
        self._template_group: list[int] = []
        self._full_regen: bool = False
        self._old_frame: tuple[tuple[float, float], float] | None = old_frame
        self._frame_state: dict[str, Any] | None = None
        self._old_array_state: dict[str, Any] | None = None
        self._old_guide_sig: tuple | None = None
        self._old_template_sig: tuple | None = None
        self._template_point_snapshot: list[tuple[Any, float, float]] = []
        self._copy_updates: list[tuple[Any, float, float]] = []
        self._updated_members: list[tuple[int, list[int]]] = []
        self._force_full_regen: bool = force_full_regen
        self._template_standalone: list[int] = []
        self._created_standalone: dict[int, list[int]] = {}

    def _do_execute(self) -> None:
        if self.add_cmd is not None or self.remove_cmd is not None:
            logger.info("ArrayEdit[%s]: redo", self.array.uid[:8])
            self._redo()
            return

        registry = self.sketch.registry
        living = self.array.living_members(registry)
        if not living:
            logger.warning(
                "ArrayEdit[%s]: skipped, no members survive",
                self.array.uid[:8],
            )
            return

        _template_slot, template_eids = living[0]
        self._template_group = list(template_eids)
        logger.info(
            "ArrayEdit[%s]: start params=%r slots=%r member_sizes=%r",
            self.array.uid[:8],
            self.strategy,
            [slot for slot, _e in living],
            [len(eids) for _s, eids in living],
        )

        self._old_array_state = self.array.snapshot()
        self._old_guide_sig = self.array._cached_guide_sig
        self._old_template_sig = self.array._cached_template_sig
        self._template_standalone = self._resolve_template_standalone(
            registry, template_eids
        )
        self._full_regen = (
            self._force_full_regen
            or self.array.params_changed(self.strategy)
            or not any(slot == 0 for slot, _eids in living)
        )
        # Members of one array must be congruent shapes. If deletions
        # left groups of differing sizes, only a full re-distribution
        # can rebuild complete copies everywhere.
        member_sizes = {len(eids) for _slot, eids in living}
        if len(member_sizes) > 1:
            self._full_regen = True
        logger.info(
            "ArrayEdit[%s]: full_regen=%s template=e%s",
            self.array.uid[:8],
            self._full_regen,
            template_eids,
        )

        # Move the whole existing array into the edited frame first
        # (rigid translation + uniform scale about the array center).
        # A similarity transform keeps every linkage/pin/dimension
        # constraint exactly satisfied, so the solver starts from a
        # consistent state instead of having to repair a teleported
        # center point. Curve-along strategies return None (no scalable
        # master), so this is a no-op for them.
        if self._old_frame is None:
            self._old_frame = self.strategy.capture_master_frame(
                registry, self.array
            )
        if self._old_frame is not None:
            (ocx, ocy), old_radius = self._old_frame
            logger.info(
                "ArrayEdit[%s]: frame old_c=(%.3f,%.3f) "
                "old_r=%.3f -> new params=%r",
                self.array.uid[:8],
                ocx,
                ocy,
                old_radius,
                self.strategy,
            )
            self._frame_state = self.strategy.apply_frame(
                registry,
                self.array,
                self._old_frame,
                self.sketch.constraints,
            )

        occupied = {slot for slot, _eids in living}
        count = max(int(self.strategy.count), 1)

        if self._full_regen:
            # Re-derive the copies from the template. The template
            # member (slot 0) is only moved by the re-anchor; existing
            # copies are updated IN PLACE (same ids — recreating them
            # would invalidate history entries and past undo state);
            # groups that no longer match the template (partial
            # deletions) and slots beyond the count are rebuilt/removed.
            self._reanchor_template(registry, template_eids)
            template_group = EntityGroup(registry, template_eids)
            placements = self.strategy.member_placements(
                resolve_template_center(
                    registry, template_eids, template_group.points()
                ),
                registry,
            )
            kept_members = [(0, list(template_eids))]
            stale: list[int] = []
            stale_slots: list[int] = []
            missing_slots: list[int] = []
            updatable = len(template_eids)
            for slot, eids in living[1:]:
                if (
                    slot < count
                    and len(eids) == updatable
                    and slot - 1 < len(placements)
                ):
                    self._update_copy_in_place(
                        registry,
                        template_eids,
                        eids,
                        placements[slot - 1],
                        self._standalone_pairs(slot),
                    )
                    kept_members.append((slot, list(eids)))
                    self._updated_members.append((slot, list(eids)))
                else:
                    stale.extend(eids)
                    stale_slots.append(slot)
            missing_slots = [
                slot
                for slot in range(1, count)
                if slot not in {s for s, _e in kept_members}
            ]
            logger.info(
                "ArrayEdit[%s]: updating %d copies in place, removing "
                "%d stale entities, creating %d slots",
                self.array.uid[:8],
                len(kept_members) - 1,
                len(stale),
                len(missing_slots),
            )
            self._remove_entities(stale, stale_slots)

            self._create_members(missing_slots)
            self._commit_members(kept_members)
            logger.info(
                "ArrayEdit[%s]: done slots=%r sizes=%r",
                self.array.uid[:8],
                [slot for slot, _ in self.array.members],
                [len(eids) for _, eids in self.array.members],
            )
            _log_array_residuals(self.sketch)
            return
        else:
            # Gap fill: keep all surviving members untouched.
            self._remove_entities([])
            missing_slots = [
                slot for slot in range(1, count) if slot not in occupied
            ]
            kept_members = living
            logger.info(
                "ArrayEdit[%s]: gap-fill missing slots %r",
                self.array.uid[:8],
                missing_slots,
            )

        self._create_members(missing_slots)
        self._commit_members(kept_members)

        logger.info(
            "ArrayEdit[%s]: done slots=%r sizes=%r",
            self.array.uid[:8],
            [slot for slot, _e in self.array.members],
            [len(eids) for _s, eids in self.array.members],
        )
        _log_array_residuals(self.sketch)

    def _commit_members(
        self, kept_members: list[tuple[int, list[int]]]
    ) -> None:
        """Writes the post-edit member list and params onto the
        definition."""
        self.array.members = [
            (slot, list(eids)) for slot, eids in kept_members
        ] + [(slot, list(eids)) for slot, eids in self.created_groups]
        member_slots = {slot for slot, _ in self.array.members}
        self.array.standalone_pids = {
            slot: pids
            for slot, pids in self.array.standalone_pids.items()
            if slot in member_slots
        }
        self.array.commit(self.strategy)
        # Refresh the caches so the next solve doesn't
        # trigger a spurious re-apply of this same edit.
        self._refresh_sync_caches()

    def _refresh_sync_caches(self) -> None:
        self.array.refresh_caches(self.sketch.registry, self.strategy)

    def _remove_entities(
        self, stale_ids: list[int], stale_slots: Sequence[int] = ()
    ) -> None:
        """
        Removes entities (with dependent points/constraints) without
        triggering prune_arrays(), so the definition survives.
        ``stale_slots`` are the member slots the entities belonged to;
        their standalone points are array-owned and die with them.
        """
        if not stale_ids:
            return
        sketch = self.sketch
        points, entities, constraints = (
            RemoveItemsCommand.calculate_dependencies_for_ids(
                sketch, set(stale_ids)
            )
        )
        # Static copies' points are fixed (they stay out of the
        # solver) and the dependency calculation never deletes fixed
        # points (the origin must survive cascades). The array
        # machinery owns its copies' points, so collect the fixed
        # orphans of the removed entities here — without this, every
        # re-apply would leak them.
        removed = {e.id for e in entities}
        registry = sketch.registry
        used_by_remaining = {
            pid
            for e in registry.entities
            if e.id not in removed
            for pid in e.get_point_ids()
        }
        known = {p.id for p in points}
        for eid in removed:
            entity = registry.get_entity(eid)
            if entity is None:
                continue
            for pid in entity.get_point_ids():
                if pid in known or pid in used_by_remaining:
                    continue
                try:
                    pt = registry.get_point(pid)
                except IndexError:
                    continue
                if pt.fixed:
                    points.append(pt)
                    known.add(pid)
        for slot in stale_slots:
            for pid in self.array.standalone_pids.get(slot, []):
                if pid in known:
                    continue
                try:
                    pt = registry.get_point(pid)
                except IndexError:
                    continue
                points.append(pt)
                known.add(pid)
        self.remove_cmd = RemoveItemsCommand(
            sketch,
            "",
            points=points,
            entities=entities,
            constraints=constraints,
        )
        self.remove_cmd.apply_direct()

    def _resolve_template_standalone(
        self, registry: EntityRegistry, template_eids: list[int]
    ) -> list[int]:
        """
        The template member's standalone points. Arrays created before
        standalone tracking existed (or loaded from old files) carry
        no stored set; it is rediscovered through the constraint graph
        and persisted so later edits find it.
        """
        stored = self.array.standalone_pids.get(0)
        if stored is not None:
            return stored
        discovered = self.sketch.find_standalone_point_ids(template_eids)
        if discovered:
            self.array.standalone_pids[0] = discovered
        return discovered

    def _standalone_pairs(self, slot: int) -> list[tuple[int, int]]:
        """Pairs the template's standalone points with a copy's, in
        matching order, for ``rewrite_copy_from``."""
        return list(
            zip(
                self._template_standalone,
                self.array.standalone_pids.get(slot, []),
            )
        )

    def _update_copy_in_place(
        self,
        registry: EntityRegistry,
        template_eids: list[int],
        copy_eids: list[int],
        placement: InstancePlacement,
        extra_point_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        Rewrites an existing copy's geometry to the placement applied
        to the current template. The copy keeps its entity and point
        ids, so history entries and undo state stay valid across
        re-derivations. ``extra_point_pairs`` carries the member's
        standalone points, rewritten by the same placement as the
        entity points.
        """
        logger.debug(
            "ArrayEdit[%s]: updating copy slot entities=%r at "
            "target_center=%r angle=%.6f",
            self.array.uid[:8],
            copy_eids,
            placement.target_center,
            placement.angle,
        )
        self._copy_updates.extend(
            EntityGroup(registry, template_eids).rewrite_copy_from(
                EntityGroup(registry, copy_eids),
                placement,
                extra_point_pairs or (),
            )
        )

    def _create_members(self, slots: list[int]) -> None:
        """Creates static copies of the template group at the slots."""
        if not slots:
            return
        assert self._template_group
        registry = self.sketch.registry

        result = CreateArrayCommand.calculate_geometry(
            registry,
            self.strategy,
            list(self._template_group),
            center_pid=self._array_center_pid(registry),
            create_master=False,
            extra_pids=set(self._template_standalone) or None,
        )
        if result is None:
            return

        add_points: list[Point] = []
        add_entities: list[Entity] = []
        pending_groups: list[tuple[int, list[Entity]]] = []
        slot_extras: dict[int, list[Point]] = {}

        # Placements are generated in slot order starting at slot 1;
        # the template member (slot 0) is not a placement.
        for slot in slots:
            i = slot - 1
            if i < 0 or i >= len(result["instance_maps"]):
                continue
            add_points.extend(result["instance_point_groups"][i])
            instance_entities = result["instance_entity_groups"][i]
            add_entities.extend(instance_entities)
            # Entity IDs are assigned by AddItemsCommand during execute;
            # resolve them into real IDs afterwards.
            pending_groups.append((slot, instance_entities))
            if i < len(result["instance_extra_points"]):
                slot_extras[slot] = result["instance_extra_points"][i]

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=add_points,
            entities=add_entities,
            constraints=[],
        )
        self.add_cmd._do_execute()
        self.created_groups = [
            (slot, [e.id for e in entities])
            for slot, entities in pending_groups
        ]
        self.created_entity_ids = [e.id for e in add_entities]
        # Standalone points of the new members; IDs are read after
        # AddItemsCommand executed, when they are final.
        self._created_standalone = {
            slot: [pt.id for pt in extras]
            for slot, extras in slot_extras.items()
        }
        self.array.standalone_pids.update(self._created_standalone)
        logger.info(
            "ArrayEdit[%s]: created %d instances (%d entities)",
            self.array.uid[:8],
            len(self.created_groups),
            len(add_entities),
        )

    def _reanchor_template(
        self,
        registry: EntityRegistry,
        template_eids: list[int],
    ) -> None:
        """
        Re-positions the template member onto position 0 of the guide.

        The template's position is guide-owned (like the circular
        array's members, which are re-projected onto the orbit); the
        rigid motion itself lives on the array. The template point
        positions are snapshotted first so undo can restore them.
        """
        self._snapshot_template_points(registry, template_eids)
        self.array.reanchor_template(
            self.strategy,
            registry,
            template_eids,
            self._template_standalone,
        )

    def _array_center_pid(self, registry: EntityRegistry) -> int | None:
        circle = registry.get_entity(self.array.guide_circle_id)
        if isinstance(circle, Circle):
            return circle.center_idx
        return None

    def _snapshot_template_points(
        self,
        registry: EntityRegistry,
        template_eids: list[int],
    ) -> None:
        """Saves the current positions of all template points — the
        member's entity points and its standalone points — so undo can
        restore them after ``_reanchor_template`` moves them."""
        snapshot = EntityGroup(registry, template_eids).snapshot_positions()
        for pid in self._template_standalone:
            try:
                pt = registry.get_point(pid)
            except IndexError:
                continue
            snapshot.append((pt, pt.x, pt.y))
        self._template_point_snapshot = snapshot

    def _redo(self) -> None:
        if self.remove_cmd is not None:
            self._reexecute_remove()
        if self.add_cmd is not None:
            self.add_cmd._do_execute()
        if self._old_array_state is None:
            return
        registry = self.sketch.registry
        if self._old_frame is not None:
            # Re-apply the similarity relative to the pre-edit frame.
            self.strategy.apply_frame(
                registry,
                self.array,
                self._old_frame,
                self.sketch.constraints,
                self._frame_state,
            )

        assert self._template_group
        if self._full_regen:
            # Undo restored the pre-edit template positions and anchor;
            # re-running the re-anchor and the recorded copy updates
            # reproduces the edited state exactly (the in-place copies
            # kept their ids, so the recorded groups are still valid).
            self._reanchor_template(registry, self._template_group)
            for pt, x, y in self._copy_updates:
                pt.x = x
                pt.y = y
            kept_members = [(0, list(self._template_group))] + [
                (slot, list(eids)) for slot, eids in self._updated_members
            ]
        else:
            kept_members = self.array.living_members(registry)
        # Undo's array.restore() dropped the created slots' standalone
        # points along with their members; re-track them before the
        # commit prunes to surviving members.
        self.array.standalone_pids.update(self._created_standalone)
        self._commit_members(kept_members)

    def _reexecute_remove(self) -> None:
        """Re-runs a removal without triggering prune_arrays()."""
        remove_cmd = self.remove_cmd
        assert remove_cmd is not None
        remove_cmd.apply_direct()

    def _do_undo(self) -> None:
        if self.add_cmd is not None:
            self.add_cmd._do_undo()
        if self.remove_cmd is not None:
            self.remove_cmd._do_undo()
        # Restore template point positions that were moved by
        # _reanchor_template.
        EntityGroup.restore_positions(self._template_point_snapshot)
        self._template_point_snapshot = []
        if self._old_array_state is None:
            return
        self.array.restore(self._old_array_state)
        # Restore the caches so sync_arrays doesn't detect
        # a spurious change and re-apply the edit we just undid.
        self.array._cached_guide_sig = self._old_guide_sig
        self.array._cached_template_sig = self._old_template_sig
        # Restore the radius dimension value captured during apply_frame.
        if (
            self._frame_state is not None
            and self._frame_state.get("radius_constraint") is not None
            and self._frame_state.get("old_radius_value") is not None
        ):
            self._frame_state["radius_constraint"].value = self._frame_state[
                "old_radius_value"
            ]


def _log_array_residuals(sketch: Sketch) -> None:
    """
    Logs the worst constraint residual per type so a broken edit state
    is visible in the log immediately, before any solve runs.
    """
    ctx = ParameterContext()
    worst: dict[str, float] = {}
    for constr in sketch.constraints:
        name = type(constr).__name__
        err = constr.error(sketch.registry, ctx)
        if not isinstance(err, (list, tuple)):
            err = [err]
        for value in err:
            worst[name] = max(worst.get(name, 0.0), abs(value))
    summary = ", ".join(
        f"{name}={value:.3e}" for name, value in sorted(worst.items())
    )
    logger.info("Array residuals after edit: %s", summary or "none")
