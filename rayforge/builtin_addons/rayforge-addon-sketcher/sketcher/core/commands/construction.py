from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class ToggleConstructionCommand(SketchChangeCommand):
    """Command to toggle the construction state of multiple entities."""

    def __init__(self, sketch: Sketch, name: str, entity_ids: list[int]):
        super().__init__(sketch, name)
        self.entity_ids = entity_ids
        self.original_states: dict[int, bool] = {}
        self.new_state: bool | None = None

    def _do_execute(self) -> None:
        self.original_states.clear()
        registry = self.sketch.registry
        for eid in self.entity_ids:
            entity = registry.get_entity_object(eid)
            if entity is not None:
                self.original_states[eid] = entity.construction

        if not self.original_states:
            return

        # Logic: If any selected entity is NOT construction, set all to
        # construction.
        # Otherwise (all are construction), set all to normal.
        if self.new_state is None:
            has_normal = any(
                not entity.construction
                for entity in (
                    registry.get_entity_object(eid)
                    for eid in self.original_states
                )
            )
            self.new_state = has_normal

        for eid in self.original_states:
            registry.set_construction(eid, self.new_state)

    def _do_undo(self) -> None:
        registry = self.sketch.registry
        for eid, old_state in self.original_states.items():
            registry.set_construction(eid, old_state)
