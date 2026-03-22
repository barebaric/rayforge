from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Tuple

from ..entities import Bezier, Line
from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..sketch import Sketch


class StraightenBezierCommand(SketchChangeCommand):
    """
    Command to convert a Bezier curve to a straight Line.

    Removes control points and replaces the Bezier entity with a Line entity.
    """

    def __init__(self, sketch: "Sketch", bezier_id: EntityID):
        label = _("Straighten")
        super().__init__(sketch, label)
        self.bezier_id = bezier_id
        self._old_bezier: Optional[
            Tuple[
                EntityID,
                EntityID,
                EntityID,
                bool,
                Optional[Tuple[float, float]],
                Optional[Tuple[float, float]],
            ]
        ] = None

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        bezier = registry.get_entity(self.bezier_id)
        if not isinstance(bezier, Bezier):
            return

        self._old_bezier = (
            bezier.id,
            bezier.start_idx,
            bezier.end_idx,
            bezier.construction,
            bezier.cp1,
            bezier.cp2,
        )

        line = Line(
            bezier.id,
            bezier.start_idx,
            bezier.end_idx,
            bezier.construction,
        )

        registry.remove_entities_by_id([bezier.id])
        registry.entities.append(line)
        registry._entity_map[line.id] = line

    def _do_undo(self) -> None:
        if self._old_bezier is None:
            return

        registry = self.sketch.registry
        (
            bezier_id,
            start_idx,
            end_idx,
            construction,
            cp1,
            cp2,
        ) = self._old_bezier

        line = registry.get_entity(bezier_id)
        if not isinstance(line, Line):
            return

        bezier = Bezier(bezier_id, start_idx, end_idx, construction, cp1, cp2)

        registry.entities.remove(line)
        del registry._entity_map[bezier_id]
        registry.entities.append(bezier)
        registry._entity_map[bezier_id] = bezier
