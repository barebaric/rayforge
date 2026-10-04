from __future__ import annotations

import itertools
import logging
import uuid
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..boolean import BooleanOp, apply_boolean, build_boolean_regions
from ..entities import OffsetPlan
from ..entities.polygon import outline_item
from ..sketch import Fill
from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)

_LABELS = {
    BooleanOp.UNION: _("Union"),
    BooleanOp.DIFFERENCE: _("Difference"),
    BooleanOp.INTERSECTION: _("Intersection"),
    BooleanOp.EXCLUDE: _("Exclude"),
}


class BooleanCommand(SketchChangeCommand):
    """Bakes a boolean operation over the selected shapes.

    The selection is preprocessed into closed regions (see
    ``build_boolean_regions``); the operation result — one solid per
    disjoint piece, holes carried by ring winding — replaces the
    sources with a single multi-ring PolygonEntity per solid. Fills
    bounded by the sources are replaced by fills on the result
    entities, styled after the bottom-most source fill. The command
    either fully applies or does nothing; undo restores the sketch
    snapshot.
    """

    def __init__(
        self,
        sketch: Sketch,
        entity_ids: list[int],
        op: BooleanOp,
    ):
        super().__init__(sketch, _LABELS[op])
        self.entity_ids = list(entity_ids)
        self.op = op
        self.new_entity_ids: list[int] = []
        self._new_entities: list = []
        self._ops: list[tuple[RemoveItemsCommand, AddItemsCommand]] = []
        self._prepared = False
        self._removed_fills: list[tuple[int, Fill]] = []
        self._added_fills: list[Fill] = []
        self._fill_template: Fill | None = None

    @staticmethod
    def prepare_solids(
        sketch: Sketch, entity_ids: list[int], op: BooleanOp
    ) -> list | None:
        """
        Pure function that validates the selection and computes the
        boolean result. Returns None when the operation cannot run or
        produces no geometry. No sketch state is modified.
        """
        regions = build_boolean_regions(sketch, entity_ids)
        if regions is None:
            return None
        solids = apply_boolean(op, regions)
        if not solids:
            logger.warning("Boolean operation produced no geometry.")
            return None
        return solids

    def _prepare(self) -> bool:
        if self._prepared:
            return True
        regions = build_boolean_regions(self.sketch, self.entity_ids)
        if regions is None:
            return False
        solids = apply_boolean(self.op, regions)
        if not solids:
            logger.warning("Boolean operation produced no geometry.")
            return False

        allocate_id = itertools.count(-1, -1).__next__
        plan = OffsetPlan()
        for solid in solids:
            center_pt, handle_pt, entity = outline_item(
                solid.outer, True, allocate_id, solid.holes
            )
            plan.points.extend((center_pt, handle_pt))
            plan.entities.append(entity)

        source_ids = {eid for region in regions for eid in region.entity_ids}
        self._removed_fills = self._collect_removed_fills(source_ids)
        points, entities, constraints = (
            RemoveItemsCommand.calculate_dependencies_for_ids(
                self.sketch, source_ids
            )
        )
        remove_cmd = RemoveItemsCommand(
            self.sketch,
            "",
            points=points,
            entities=entities,
            constraints=constraints,
        )
        add_cmd = AddItemsCommand(
            self.sketch, "", points=plan.points, entities=plan.entities
        )
        self._new_entities = list(plan.entities)
        self._ops = [(remove_cmd, add_cmd)]
        self._prepared = True
        return True

    def _collect_removed_fills(
        self, source_ids: set[int]
    ) -> list[tuple[int, Fill]]:
        """Finds fills whose boundary references a removed source
        entity, recorded with their position in ``sketch.fills`` so
        undo can restore them in place. Also selects the style
        template: the bottom-most affected fill (registry stacking
        order)."""
        z_of = {
            entity.id: i
            for i, entity in enumerate(self.sketch.registry.entities)
        }
        affected = [
            (index, fill)
            for index, fill in enumerate(self.sketch.fills)
            if any(eid in source_ids for eid, _ in fill.boundary)
        ]
        if affected:
            self._fill_template = min(
                affected,
                key=lambda item: min(
                    z_of.get(eid, 0) for eid, _ in item[1].boundary
                ),
            )[1]
        return affected

    def _transfer_fills(self) -> None:
        """Drops the fills of the removed sources and re-creates the
        bottom-most one's style on every result entity. Re-runnable
        for redo: the created Fill objects are kept and re-attached."""
        for _index, fill in self._removed_fills:
            if fill in self.sketch.fills:
                self.sketch.fills.remove(fill)
        template = self._fill_template
        if template is None:
            return
        if not self._added_fills:
            self._added_fills = [
                self._new_fill(entity, template)
                for entity in self._new_entities
            ]
        self.sketch.fills.extend(self._added_fills)

    @staticmethod
    def _new_fill(entity, template: Fill) -> Fill:
        return Fill(
            uid=str(uuid.uuid4()),
            boundary=[(entity.id, True)],
            style=template.style,
            color=template.color,
            gradient_stops=template.gradient_stops,
            gradient_angle=template.gradient_angle,
        )

    def _do_execute(self) -> None:
        if not self._prepare():
            return
        for remove_cmd, add_cmd in self._ops:
            remove_cmd._do_execute()
            add_cmd._do_execute()
        self.new_entity_ids = [entity.id for entity in self._new_entities]
        self._transfer_fills()

    def _do_undo(self) -> None:
        for fill in reversed(self._added_fills):
            if fill in self.sketch.fills:
                self.sketch.fills.remove(fill)
        for remove_cmd, add_cmd in reversed(self._ops):
            add_cmd._do_undo()
            remove_cmd._do_undo()
        for index, fill in self._removed_fills:
            if fill not in self.sketch.fills:
                self.sketch.fills.insert(index, fill)
