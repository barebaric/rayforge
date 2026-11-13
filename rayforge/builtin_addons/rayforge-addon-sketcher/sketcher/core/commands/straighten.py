from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.sketch.entities.bezier import Bezier

from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..sketch import Sketch


class StraightenBezierCommand(SketchChangeCommand):
    """
    Command to convert a Bezier curve to a straight Line.

    Removes control points and replaces the Bezier entity with a Line
    entity keeping the same id.
    """

    def __init__(self, sketch: Sketch, bezier_id: EntityID):
        label = _("Straighten")
        super().__init__(sketch, label)
        self.bezier_id = bezier_id
        self._old_bezier_dict: dict[str, Any] | None = None

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        bezier = registry.get_entity_object(self.bezier_id)
        if not isinstance(bezier, Bezier):
            return

        self._old_bezier_dict = registry.entity_to_dict(bezier.id)

        line_dict = {
            "id": bezier.id,
            "type": "line",
            "p1_idx": bezier.start_idx,
            "p2_idx": bezier.end_idx,
            "construction": bezier.construction,
        }
        registry.remove_entities([bezier.id])
        self.sketch.restore_structure([], [line_dict])

    def _do_undo(self) -> None:
        if self._old_bezier_dict is None:
            return

        registry = self.sketch.registry
        entity = registry.get_entity_object(self.bezier_id)
        if entity is None or entity.type_name() != "line":
            return

        registry.remove_entities([self.bezier_id])
        self.sketch.restore_structure([], [self._old_bezier_dict])
