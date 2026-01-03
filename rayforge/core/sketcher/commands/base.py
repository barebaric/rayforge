from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Tuple

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
        # Stores {point_id: (x, y)} for all points in the sketch
        self._state_snapshot: Dict[int, Tuple[float, float]] = {}

    def capture_snapshot(self):
        """Captures the current coordinates of all points."""
        self._state_snapshot = {
            p.id: (p.x, p.y) for p in self.sketch.registry.points
        }

    def restore_snapshot(self):
        """Restores coordinates from the snapshot."""
        registry = self.sketch.registry
        for pid, (x, y) in self._state_snapshot.items():
            try:
                p = registry.get_point(pid)
                p.x = x
                p.y = y
            except IndexError:
                # Point might not exist if _do_undo didn't restore it,
                # but typically this shouldn't happen.
                pass

    def execute(self) -> None:
        # If a snapshot wasn't provided during initialization (e.g. by a tool
        # that moved points before creating the command), capture it now.
        if not self._state_snapshot:
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
