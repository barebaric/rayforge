from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Dict, Optional

from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class ToggleConstructionCommand(SketchChangeCommand):
    """Command to toggle the construction state of multiple entities."""

    def __init__(self, sketch: "Sketch", name: str, entity_ids: List[int]):
        super().__init__(sketch, name)
        self.entity_ids = entity_ids
        self.original_states: Dict[int, bool] = {}
        self.new_state: Optional[bool] = None

    def _do_execute(self) -> None:
        self.original_states.clear()
        entities_to_modify = []
        for eid in self.entity_ids:
            ent = self.sketch.registry.get_entity(eid)
            if ent:
                entities_to_modify.append(ent)
                self.original_states[eid] = ent.construction

        if not entities_to_modify:
            return

        # Logic: If any selected entity is NOT construction, set all to
        # construction.
        # Otherwise (all are construction), set all to normal.
        if self.new_state is None:
            has_normal = any(not e.construction for e in entities_to_modify)
            self.new_state = has_normal

        for e in entities_to_modify:
            e.construction = self.new_state

    def _do_undo(self) -> None:
        for eid, old_state in self.original_states.items():
            ent = self.sketch.registry.get_entity(eid)
            if ent:
                ent.construction = old_state
