from __future__ import annotations

import logging
import math
from gettext import gettext as _
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ..constraints import RadiusConstraint
from ..entities import Circle
from ..params import ParameterContext
from ..patterns import (
    CircularPatternParams,
    PatternDefinition,
    make_pattern_strategy,
)
from .base import SketchChangeCommand
from .create_pattern import CreatePatternCommand
from .items import AddItemsCommand, RemoveItemsCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity, Point
    from ..registry import EntityRegistry
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class EditPatternCommand(SketchChangeCommand):
    """
    Regenerates a pattern's instances from its definition.

    Members are groups of entities. The first surviving member (lowest
    slot) acts as the template. When the pattern parameters (count,
    angle, rotate mode) did not change, only missing slots are
    re-created. When they changed, every other member is removed and
    fresh copies are distributed at the new placements. In both cases
    the existing geometry is first moved into the edited frame via a
    similarity transform so the solver starts fully consistent.
    """

    def __init__(
        self,
        sketch: Sketch,
        pattern: PatternDefinition,
        params: CircularPatternParams,
    ):
        super().__init__(sketch, _("Edit Pattern"))
        self.pattern = pattern
        self.params = params
        self.add_cmd: AddItemsCommand | None = None
        self.remove_cmd: RemoveItemsCommand | None = None
        self.created_entity_ids: list[int] = []
        self.created_groups: list[tuple[int, list[int]]] = []
        self._template_group: list[int] = []
        self._full_regen: bool = False
        self._old_frame: tuple[tuple[float, float], float] | None = None
        self._old_pattern_state: dict[str, Any] | None = None
        self._radius_constraint: RadiusConstraint | None = None
        self._old_radius_value: float | None = None

    @staticmethod
    def _params_changed(
        state: dict[str, Any], params: CircularPatternParams
    ) -> bool:
        return (
            params.count != state["count"]
            or params.total_angle_deg != state["total_angle_deg"]
            or params.rotate_copies != state["rotate_copies"]
        )

    def _do_execute(self) -> None:
        if self.add_cmd is not None or self.remove_cmd is not None:
            logger.info("PatternEdit[%s]: redo", self.pattern.uid[:8])
            self._redo()
            return

        registry = self.sketch.registry
        living = self.pattern.living_members(registry)
        if not living:
            logger.warning(
                "PatternEdit[%s]: skipped, no members survive",
                self.pattern.uid[:8],
            )
            return

        _template_slot, template_eids = living[0]
        self._template_group = list(template_eids)
        logger.info(
            "PatternEdit[%s]: start params=%r slots=%r member_sizes=%r",
            self.pattern.uid[:8],
            self.params,
            [slot for slot, _e in living],
            [len(eids) for _s, eids in living],
        )

        self._old_pattern_state = {
            "members": [
                (slot, list(eids)) for slot, eids in self.pattern.members
            ],
            "count": self.pattern.count,
            "total_angle_deg": self.pattern.total_angle_deg,
            "rotate_copies": self.pattern.rotate_copies,
        }
        self._full_regen = self._params_changed(
            self._old_pattern_state, self.params
        ) or not any(slot == 0 for slot, _eids in living)
        # Members of one pattern must be congruent shapes. If deletions
        # left groups of differing sizes, only a full re-distribution
        # can rebuild complete copies everywhere.
        member_sizes = {len(eids) for _slot, eids in living}
        if len(member_sizes) > 1:
            self._full_regen = True
        logger.info(
            "PatternEdit[%s]: full_regen=%s template=e%s",
            self.pattern.uid[:8],
            self._full_regen,
            template_eids,
        )

        # Move the whole existing pattern into the edited frame first
        # (rigid translation + uniform scale about the pattern center).
        # A similarity transform keeps every linkage/pin/dimension
        # constraint exactly satisfied, so the solver starts from a
        # consistent state instead of having to repair a teleported
        # center point.
        self._old_frame = self._capture_master_frame(registry)
        if self._old_frame is not None:
            (ocx, ocy), old_radius = self._old_frame
            k = self.params.radius / old_radius if old_radius > 1e-9 else 1.0
            logger.info(
                "PatternEdit[%s]: similarity old_c=(%.3f,%.3f) "
                "old_r=%.3f -> new_c=(%.3f,%.3f) new_r=%.3f k=%.4f",
                self.pattern.uid[:8],
                ocx,
                ocy,
                old_radius,
                self.params.center[0],
                self.params.center[1],
                self.params.radius,
                k,
            )
            self._apply_similarity(registry, self._old_frame)

        occupied = {slot for slot, _eids in living}
        count = max(int(self.params.count), 1)

        if self._full_regen:
            # Re-distribute: drop every member except the template.
            stale = [eid for _slot, eids in living[1:] for eid in eids]
            logger.info(
                "PatternEdit[%s]: removing %d stale entities from %d members",
                self.pattern.uid[:8],
                len(stale),
                len(living) - 1,
            )
            self._remove_entities(stale)
            missing_slots = list(range(1, count))
            kept_members = [(0, list(template_eids))]
        else:
            # Gap fill: keep all surviving members untouched.
            self._remove_entities([])
            missing_slots = [
                slot for slot in range(1, count) if slot not in occupied
            ]
            kept_members = living
            logger.info(
                "PatternEdit[%s]: gap-fill missing slots %r",
                self.pattern.uid[:8],
                missing_slots,
            )

        self._create_members(missing_slots)
        self._commit_members(kept_members)

        logger.info(
            "PatternEdit[%s]: done slots=%r sizes=%r",
            self.pattern.uid[:8],
            [slot for slot, _e in self.pattern.members],
            [len(eids) for _s, eids in self.pattern.members],
        )
        _log_pattern_residuals(self.sketch)

    def _commit_members(
        self, kept_members: list[tuple[int, list[int]]]
    ) -> None:
        """Writes the post-edit member list and params onto the
        definition."""
        self.pattern.members = [
            (slot, list(eids)) for slot, eids in kept_members
        ] + [(slot, list(eids)) for slot, eids in self.created_groups]
        self.pattern.count = self.params.count
        self.pattern.total_angle_deg = self.params.total_angle_deg
        self.pattern.rotate_copies = self.params.rotate_copies

    def _remove_entities(self, stale_ids: list[int]) -> None:
        """
        Removes entities (with dependent points/constraints) without
        triggering prune_patterns(), so the definition survives.
        """
        if not stale_ids:
            return
        sketch = self.sketch
        points, entities, constraints = (
            RemoveItemsCommand.calculate_dependencies(
                sketch,
                SimpleNamespace(
                    entity_ids=set(stale_ids),
                    point_ids=set(),
                    constraint_idx=None,
                ),
            )
        )
        self.remove_cmd = RemoveItemsCommand(
            sketch,
            "",
            points=points,
            entities=entities,
            constraints=constraints,
        )
        self._apply_removal(sketch, points, entities, constraints)

    def _create_members(self, slots: list[int]) -> None:
        """Creates linked copies of the template group at the slots."""
        if not slots:
            return
        assert self._template_group
        registry = self.sketch.registry
        strategy = make_pattern_strategy(self.pattern.mode, self.params)

        result = CreatePatternCommand.calculate_geometry(
            registry,
            strategy,
            list(self._template_group),
            self.params,
            center_pid=self._pattern_center_pid(registry),
            create_master=False,
        )
        if result is None:
            return

        add_points: list[Point] = []
        add_entities: list[Entity] = []
        add_constraints = []
        pending_groups: list[tuple[int, list[Entity]]] = []

        for slot in slots:
            # Placements are generated in slot order starting at 1.
            i = slot - 1
            pid_map = result["instance_maps"][i]
            add_points.extend(result["instance_point_groups"][i])
            instance_entities = result["instance_entity_groups"][i]
            add_entities.extend(instance_entities)
            add_constraints.extend(
                strategy.build_linkage_constraints(
                    [(slot, pid_map)], result["center_pid"]
                )
            )
            # Entity IDs are assigned by AddItemsCommand during execute;
            # resolve them into real IDs afterwards.
            pending_groups.append((slot, instance_entities))

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=add_points,
            entities=add_entities,
            constraints=add_constraints,
        )
        self.add_cmd._do_execute()
        self.created_groups = [
            (slot, [e.id for e in entities])
            for slot, entities in pending_groups
        ]
        self.created_entity_ids = [e.id for e in add_entities]
        logger.info(
            "PatternEdit[%s]: created %d instances (%d entities)",
            self.pattern.uid[:8],
            len(self.created_groups),
            len(add_entities),
        )

    def _pattern_center_pid(self, registry: EntityRegistry) -> int | None:
        circle = registry.get_entity(self.pattern.guide_circle_id)
        if isinstance(circle, Circle):
            return circle.center_idx
        return None

    def _capture_master_frame(
        self, registry: EntityRegistry
    ) -> tuple[tuple[float, float], float] | None:
        """Returns the guide circle's (center, radius) before editing."""
        circle = registry.get_entity(self.pattern.guide_circle_id)
        if not isinstance(circle, Circle):
            return None
        try:
            center = registry.get_point(circle.center_idx)
            radius_pt = registry.get_point(circle.radius_pt_idx)
        except IndexError:
            return None
        return (
            center.x,
            center.y,
        ), math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)

    def _apply_similarity(
        self,
        registry: EntityRegistry,
        old_frame: tuple[tuple[float, float], float],
    ) -> None:
        """
        Maps the whole existing pattern into the edited frame via a
        similarity transform: translation to the new center plus a
        uniform scale of new_radius / old_radius. A similarity keeps
        every rotational, pin and radius constraint exactly satisfied,
        so no solver repair is needed.
        """
        (ocx, ocy), old_radius = old_frame
        ncx, ncy = self.params.center
        k = self.params.radius / old_radius if old_radius > 1e-9 else 1.0

        pids: set[int] = set()
        for eid in self.pattern.living_entity_ids(registry):
            entity = registry.get_entity(eid)
            if entity is not None:
                pids.update(entity.get_point_ids())
        circle = registry.get_entity(self.pattern.guide_circle_id)
        if isinstance(circle, Circle):
            pids.update(circle.get_point_ids())

        for pid in pids:
            p = registry.get_point(pid)
            p.x = ncx + k * (p.x - ocx)
            p.y = ncy + k * (p.y - ocy)

        # Pin the master circle geometry exactly onto the target frame.
        if isinstance(circle, Circle):
            radius_pt = registry.get_point(circle.radius_pt_idx)
            radius_pt.x = ncx + self.params.radius
            radius_pt.y = ncy

        for constr in self.sketch.constraints:
            if (
                isinstance(constr, RadiusConstraint)
                and constr.entity_id == self.pattern.guide_circle_id
            ):
                self._radius_constraint = constr
                if self._old_radius_value is None:
                    self._old_radius_value = constr.value
                constr.value = self.params.radius
                break

    def _redo(self) -> None:
        if self.remove_cmd is not None:
            self._reexecute_remove()
        if self.add_cmd is not None:
            self.add_cmd._do_execute()
        if self._old_pattern_state is None:
            return
        registry = self.sketch.registry
        if self._old_frame is not None:
            # Re-apply the similarity relative to the pre-edit frame.
            self._apply_similarity(registry, self._old_frame)

        assert self._template_group
        if self._full_regen:
            kept_members = [(0, list(self._template_group))]
        else:
            kept_members = self.pattern.living_members(registry)
        self._commit_members(kept_members)

    def _reexecute_remove(self) -> None:
        """Re-runs a removal without triggering prune_patterns()."""
        remove_cmd = self.remove_cmd
        assert remove_cmd is not None
        self._apply_removal(
            self.sketch,
            remove_cmd.points,
            remove_cmd.entities,
            remove_cmd.constraints,
        )

    @staticmethod
    def _apply_removal(
        sketch: Sketch,
        points: list[Point],
        entities: list[Entity],
        constraints: list[Constraint],
    ) -> None:
        """
        Removes points, entities and constraints from the sketch
        directly, bypassing RemoveItemsCommand's execute so
        prune_patterns() never sees the pattern mid-edit.
        """
        registry = sketch.registry
        point_ids = {p.id for p in points}
        entity_ids = {e.id for e in entities}
        registry.points = [p for p in registry.points if p.id not in point_ids]
        registry.entities = [
            e for e in registry.entities if e.id not in entity_ids
        ]
        registry._entity_map = {e.id: e for e in registry.entities}
        for c in constraints:
            if c in sketch.constraints:
                sketch.constraints.remove(c)

    def _do_undo(self) -> None:
        if self.add_cmd is not None:
            self.add_cmd._do_undo()
        if self.remove_cmd is not None:
            self.remove_cmd._do_undo()
        if self._old_pattern_state is None:
            return
        state = self._old_pattern_state
        self.pattern.members = [
            (slot, list(eids)) for slot, eids in state["members"]
        ]
        self.pattern.count = state["count"]
        self.pattern.total_angle_deg = state["total_angle_deg"]
        self.pattern.rotate_copies = state["rotate_copies"]
        if (
            self._radius_constraint is not None
            and self._old_radius_value is not None
        ):
            self._radius_constraint.value = self._old_radius_value


def _log_pattern_residuals(sketch: Sketch) -> None:
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
    logger.info("Pattern residuals after edit: %s", summary or "none")
