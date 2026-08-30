from __future__ import annotations

import itertools
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..contour import OffsetItem, build_offset_items
from ..entities import OffsetPlan
from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class OffsetCommand(SketchChangeCommand):
    """Offsets the selected contours in place.

    The selection is preprocessed into offsettable items: lone
    circles, arcs and ellipses are updated in place and keep their
    entity type; every other connected component is replaced by the
    offset result (a PolygonEntity per result outline). A plan is
    validated for the whole selection before anything is applied, so
    the command either fully applies or does nothing. Point moves are
    undone by the command snapshot; replacements by add/remove
    commands.
    """

    def __init__(
        self,
        sketch: Sketch,
        entity_ids: list[int],
        distance: float,
    ):
        super().__init__(sketch, _("Offset"))
        self.entity_ids = list(entity_ids)
        self.distance = distance
        self._point_moves: dict[int, tuple[float, float]] = {}
        self._ops: list[tuple[RemoveItemsCommand | None, AddItemsCommand]] = []
        self._prepared = False

    @staticmethod
    def prepare_items(
        sketch: Sketch, entity_ids: list[int]
    ) -> list[OffsetItem] | None:
        """
        Pure function that partitions the selection into offsettable
        items. Returns None when the selection cannot be offset.
        """
        items = build_offset_items(sketch, entity_ids)
        if not items:
            logger.warning("Selection contains no offsettable geometry.")
            return None
        return items

    def _prepare(self) -> bool:
        if self._prepared:
            return True
        items = self.prepare_items(self.sketch, self.entity_ids)
        if items is None:
            return False

        allocate_id = itertools.count(-1, -1).__next__
        point_moves: dict[int, tuple[float, float]] = {}
        ops: list[tuple[RemoveItemsCommand | None, AddItemsCommand]] = []
        for item in items:
            plan: OffsetPlan | None = item.plan_offset(
                self.sketch.registry, self.distance, allocate_id
            )
            if plan is None:
                logger.warning(
                    "Offset collapsed an item; aborting the whole operation."
                )
                return False
            point_moves.update(plan.point_moves)
            if not plan.entities:
                continue
            remove_cmd = None
            if plan.removed_entity_ids:
                points, entities, constraints = (
                    RemoveItemsCommand.calculate_dependencies_for_ids(
                        self.sketch, set(plan.removed_entity_ids)
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
            ops.append((remove_cmd, add_cmd))

        self._point_moves = point_moves
        self._ops = ops
        self._prepared = True
        return True

    def _do_execute(self) -> None:
        if not self._prepare():
            return
        registry = self.sketch.registry
        for pid, (x, y) in self._point_moves.items():
            point = registry.get_point(pid)
            point.x, point.y = x, y
        for remove_cmd, add_cmd in self._ops:
            if remove_cmd:
                remove_cmd._do_execute()
            add_cmd._do_execute()

    def _do_undo(self) -> None:
        for remove_cmd, add_cmd in reversed(self._ops):
            add_cmd._do_undo()
            if remove_cmd:
                remove_cmd._do_undo()

    @staticmethod
    def preview_polylines(
        items: list[OffsetItem],
        registry,
        distance: float,
    ) -> list[list[tuple[float, float]]] | None:
        """
        Polylines of the offset result for live preview. Each item
        previews its own offset polymorphically (preview_polylines on
        the item protocol); no geometry is committed. Takes
        preprocessed items (see prepare_items) so the selection has to
        be validated only once.
        """
        if not items:
            return None
        polylines: list[list[tuple[float, float]]] = []
        for item in items:
            polylines.extend(item.preview_polylines(registry, distance))
        return polylines or None
