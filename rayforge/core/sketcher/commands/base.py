from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from ...geo import Point
from ...undo.command import Command

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch
    from .dimension import DimensionData

logger = logging.getLogger(__name__)


class PreviewState:
    """
    Base class for preview state returned by start_preview().

    Subclass this to store command-specific preview data.
    All preview state should be stored as attributes for the tool to read
    after calling cleanup_preview().
    """

    def get_preview_point_ids(self) -> set[int]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Subclasses should override this to return the IDs of points that
        were created during preview and should be excluded from hit testing.
        The start/center point is typically excluded since it may be permanent.

        Returns:
            Set of temporary point IDs that tools should ignore during
            hit testing for snap purposes.
        """
        return set()

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List["DimensionData"]:
        """
        Returns dimension data for live preview rendering.

        Subclasses should override this to provide dimension information
        (lengths, radii, angles, etc.) that should be displayed during
        the preview phase.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List of DimensionData objects representing dimensions to display.
        """
        return []


class SketchChangeCommand(Command):
    """
    Base class for commands that modify a sketch and need to trigger a solve.
    Includes functionality to snapshot geometry state for precise undo.
    """

    def __init__(self, sketch: "Sketch", name: str):
        super().__init__(name)
        self.sketch = sketch
        # Stores ( {point_id: (x, y)}, {entity_id: state_dict} )
        self._snapshot: Optional[tuple[Dict[int, Point], Dict[int, Any]]] = (
            None
        )

    @staticmethod
    def start_preview(
        registry: "EntityRegistry",
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
        **kwargs,
    ) -> PreviewState:
        """
        Creates initial preview state with start point(s).

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.
            **kwargs: Additional command-specific parameters.

        Returns:
            PreviewState containing preview state for use with update_preview
            and cleanup_preview.

        Raises:
            NotImplementedError: If the command does not support preview.
        """
        raise NotImplementedError("This command does not support preview")

    @staticmethod
    def update_preview(
        registry: "EntityRegistry",
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        """
        Updates the preview geometry based on new cursor position.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.
            x, y: The new cursor coordinates.

        Raises:
            NotImplementedError: If the command does not support preview.
        """
        raise NotImplementedError("This command does not support preview")

    @staticmethod
    def cleanup_preview(
        registry: "EntityRegistry", preview_state: PreviewState
    ) -> None:
        """
        Removes all preview entities and points from the registry.

        The preview_state is modified in place if needed (e.g., to store
        final computed values like direction). The tool reads all necessary
        values from preview_state after calling this method.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.

        Raises:
            NotImplementedError: If the command does not support preview.
        """
        raise NotImplementedError("This command does not support preview")

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
