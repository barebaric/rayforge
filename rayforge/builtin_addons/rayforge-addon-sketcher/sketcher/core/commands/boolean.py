from __future__ import annotations

import itertools
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..boolean import BooleanOp, apply_boolean, build_boolean_regions
from ..entities import OffsetPlan
from ..entities.polygon import outline_item
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
    sources with a single multi-ring PolygonEntity per solid. The
    command either fully applies or does nothing; undo restores the
    sketch snapshot.
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

    def _do_execute(self) -> None:
        if not self._prepare():
            return
        for remove_cmd, add_cmd in self._ops:
            remove_cmd._do_execute()
            add_cmd._do_execute()
        self.new_entity_ids = [entity.id for entity in self._new_entities]

    def _do_undo(self) -> None:
        for remove_cmd, add_cmd in reversed(self._ops):
            add_cmd._do_undo()
            remove_cmd._do_undo()
