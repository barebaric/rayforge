from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Optional, Dict, Any
from .base import SketchChangeCommand
from .items import AddItemsCommand, RemoveItemsCommand
from ..entities import Line, Point
from ..constraints import (
    EqualDistanceConstraint,
    CollinearConstraint,
)

if TYPE_CHECKING:
    from ..sketch import Sketch
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


class ChamferCommand(SketchChangeCommand):
    """Command to add a chamfer to a corner by replacing the corner lines."""

    def __init__(
        self,
        sketch: Sketch,
        corner_pid: int,
        line1_id: int,
        line2_id: int,
        distance: float,
    ):
        super().__init__(sketch, _("Add Chamfer"))
        self.corner_pid = corner_pid
        self.line1_id = line1_id
        self.line2_id = line2_id
        self.distance = distance

        # State for undo/redo
        self.add_cmd: Optional[AddItemsCommand] = None
        self.remove_cmd: Optional[RemoveItemsCommand] = None
        self._prepared = False

    @staticmethod
    def calculate_geometry(
        reg: "EntityRegistry",
        corner_pid: int,
        line1_id: int,
        line2_id: int,
        distance: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates the points, entities, and constraints for a chamfer.
        This is a pure function for testability and reusability.
        """
        line1 = reg.get_entity(line1_id)
        line2 = reg.get_entity(line2_id)
        if not isinstance(line1, Line) or not isinstance(line2, Line):
            return None

        try:
            corner_point = reg.get_point(corner_pid)
            other1_pid = (
                line1.p2_idx if line1.p1_idx == corner_pid else line1.p1_idx
            )
            other2_pid = (
                line2.p2_idx if line2.p1_idx == corner_pid else line2.p1_idx
            )
            other1_pt = reg.get_point(other1_pid)
            other2_pt = reg.get_point(other2_pid)
        except IndexError:
            return None

        v1 = (other1_pt.x - corner_point.x, other1_pt.y - corner_point.y)
        len1 = math.hypot(v1[0], v1[1])
        if len1 < distance:  # Not enough length for chamfer
            return None
        u1 = (v1[0] / len1, v1[1] / len1) if len1 > 1e-9 else (0.0, 0.0)
        p_new1_pos = (
            corner_point.x + distance * u1[0],
            corner_point.y + distance * u1[1],
        )

        v2 = (other2_pt.x - corner_point.x, other2_pt.y - corner_point.y)
        len2 = math.hypot(v2[0], v2[1])
        if len2 < distance:  # Not enough length for chamfer
            return None
        u2 = (v2[0] / len2, v2[1] / len2) if len2 > 1e-9 else (0.0, 0.0)
        p_new2_pos = (
            corner_point.x + distance * u2[0],
            corner_point.y + distance * u2[1],
        )

        # Define new items with temporary IDs
        p1 = Point(-1, p_new1_pos[0], p_new1_pos[1])
        p2 = Point(-2, p_new2_pos[0], p_new2_pos[1])

        added_entities = [
            Line(-3, p1.id, p2.id),  # chamfer_line
            Line(-4, other1_pid, p1.id),  # new_segment1
            Line(-5, other2_pid, p2.id),  # new_segment2
        ]

        added_constraints = [
            CollinearConstraint(other1_pid, corner_pid, p1.id),
            CollinearConstraint(other2_pid, corner_pid, p2.id),
            EqualDistanceConstraint(corner_pid, p1.id, corner_pid, p2.id),
        ]

        return {
            "points": [p1, p2],
            "entities": added_entities,
            "constraints": added_constraints,
            "removed_entities": [line1, line2],
        }

    def _prepare(self) -> bool:
        """Prepares internal commands on first execution."""
        if self._prepared:
            return True

        result = self.calculate_geometry(
            self.sketch.registry,
            self.corner_pid,
            self.line1_id,
            self.line2_id,
            self.distance,
        )

        if result is None:
            return False

        self.remove_cmd = RemoveItemsCommand(
            self.sketch, "", entities=result["removed_entities"]
        )
        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=result["points"],
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self._prepared = True
        return True

    def _do_execute(self) -> None:
        if not self._prepare():
            return

        # Use composition to apply changes
        if self.remove_cmd:
            self.remove_cmd._do_execute()
        if self.add_cmd:
            self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if not self.add_cmd or not self.remove_cmd:
            return

        # Undo in reverse order
        self.add_cmd._do_undo()
        self.remove_cmd._do_undo()
