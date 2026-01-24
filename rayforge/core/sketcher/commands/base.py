from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any

from ...undo.command import Command

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class SketchChangeCommand(Command):
    """
    Base class for commands that modify a sketch and need to trigger a solve.
    Includes functionality to snapshot geometry state for precise undo.
    """

    def __init__(self, sketch: "Sketch", name: str):
        super().__init__(name)
        self.sketch = sketch
        # Stores ( {point_id: (x, y)}, {entity_id: state_dict} )
        self._snapshot: Optional[
            Tuple[Dict[int, Tuple[float, float]], Dict[int, Any]]
        ] = None

    def capture_snapshot(self):
        """Captures the current coordinates of all points and entity states."""
        points = {p.id: (p.x, p.y) for p in self.sketch.registry.points}
        entities = {}
        for e in self.sketch.registry.entities:
            state = e.get_state()
            if state is not None:
                entities[e.id] = state

        self._snapshot = (points, entities)

    def restore_snapshot(self):
        """Restores coordinates and entity states from the snapshot."""
        if self._snapshot is None:
            return

        points, entities = self._snapshot
        registry = self.sketch.registry

        # Restore Points
        for pid, (x, y) in points.items():
            try:
                p = registry.get_point(pid)
                p.x = x
                p.y = y
            except IndexError:
                pass

        # Restore Entities
        for eid, state in entities.items():
            entity = registry.get_entity(eid)
            if entity:
                entity.set_state(state)

    def execute(self) -> None:
        # If a snapshot wasn't provided during initialization, capture it now.
        if self._snapshot is None:
            self.capture_snapshot()

        self._do_execute()
        self.sketch.notify_update()

    def undo(self) -> None:
        self._do_undo()
        # Restore the exact geometric positions from before the command.
        # This prevents the solver from jumping to an alternative solution
        # (e.g., triangle flip) when constraints are reapplied.
        self.restore_snapshot()
        self.sketch.notify_update()

    def _do_execute(self) -> None:
        raise NotImplementedError

    def _do_undo(self) -> None:
        raise NotImplementedError
