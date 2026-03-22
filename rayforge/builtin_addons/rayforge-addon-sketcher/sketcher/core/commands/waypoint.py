from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rayforge.core.geo.types import Point as GeoPoint
from .base import SketchChangeCommand
from ..entities import Bezier, Line
from ..entities.point import WaypointType
from ..types import EntityID

if TYPE_CHECKING:
    from ..entities.point import Point
    from ..registry import EntityRegistry
    from ..sketch import Sketch

logger = logging.getLogger(__name__)

DEFAULT_CP_LENGTH = 30.0


class SetWaypointTypeCommand(SketchChangeCommand):
    """
    Command to change a waypoint's type (sharp/smooth/symmetric).

    When converting from SHARP to SMOOTH or SYMMETRIC:
    - Creates control point offsets on connected beziers if they don't exist
    - Positions them appropriately based on connected segments
    - Converts connected Line entities to Bezier entities

    When converting from SMOOTH/SYMMETRIC to SHARP:
    - Control points are preserved, but co-linearity is no longer enforced
    """

    def __init__(
        self,
        sketch: "Sketch",
        waypoint_id: EntityID,
        new_type: "WaypointType",
    ):
        label = _("Set Waypoint Type")
        super().__init__(sketch, label)
        self.waypoint_id = waypoint_id
        self.new_type = new_type
        self._old_waypoint_type: Optional[WaypointType] = None
        self._old_bezier_states: Optional[
            Dict[int, Tuple[Optional[GeoPoint], Optional[GeoPoint]]]
        ] = None
        self._converted_lines: Optional[List[Tuple[int, int, int]]] = None
        self._added_bezier_ids: Optional[List[int]] = None

    def _get_segment_directions(
        self, registry: "EntityRegistry", waypoint: "Point"
    ) -> tuple[Optional[GeoPoint], Optional[GeoPoint]]:
        """
        Get the incoming and outgoing direction vectors at this waypoint.

        Returns (incoming_dir, outgoing_dir) as normalized vectors.
        """
        incoming_dir: Optional[GeoPoint] = None
        outgoing_dir: Optional[GeoPoint] = None

        for entity in registry.entities:
            if isinstance(entity, Line):
                if entity.p2_idx == waypoint.id:
                    p1 = registry.get_point(entity.p1_idx)
                    if p1:
                        dx = waypoint.x - p1.x
                        dy = waypoint.y - p1.y
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            incoming_dir = (dx / length, dy / length)
                elif entity.p1_idx == waypoint.id:
                    p2 = registry.get_point(entity.p2_idx)
                    if p2:
                        dx = p2.x - waypoint.x
                        dy = p2.y - waypoint.y
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            outgoing_dir = (dx / length, dy / length)
            elif isinstance(entity, Bezier):
                if entity.end_idx == waypoint.id:
                    if entity.cp2 is not None:
                        cp_abs = (
                            waypoint.x + entity.cp2[0],
                            waypoint.y + entity.cp2[1],
                        )
                        dx = waypoint.x - cp_abs[0]
                        dy = waypoint.y - cp_abs[1]
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            incoming_dir = (dx / length, dy / length)
                    else:
                        start = registry.get_point(entity.start_idx)
                        if start:
                            dx = waypoint.x - start.x
                            dy = waypoint.y - start.y
                            length = math.hypot(dx, dy)
                            if length > 1e-9:
                                incoming_dir = (dx / length, dy / length)
                elif entity.start_idx == waypoint.id:
                    if entity.cp1 is not None:
                        cp_abs = (
                            waypoint.x + entity.cp1[0],
                            waypoint.y + entity.cp1[1],
                        )
                        dx = cp_abs[0] - waypoint.x
                        dy = cp_abs[1] - waypoint.y
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            outgoing_dir = (dx / length, dy / length)
                    else:
                        end = registry.get_point(entity.end_idx)
                        if end:
                            dx = end.x - waypoint.x
                            dy = end.y - waypoint.y
                            length = math.hypot(dx, dy)
                            if length > 1e-9:
                                outgoing_dir = (dx / length, dy / length)

        return incoming_dir, outgoing_dir

    def _find_connected_lines(
        self, registry: "EntityRegistry", waypoint_id: EntityID
    ) -> List[Tuple[EntityID, EntityID, EntityID]]:
        """Find Line entities connected to this waypoint.

        Returns list of (line_id, p1_idx, p2_idx).
        """
        connected = []
        for entity in registry.entities:
            if isinstance(entity, Line):
                if (
                    entity.p1_idx == waypoint_id
                    or entity.p2_idx == waypoint_id
                ):
                    connected.append((entity.id, entity.p1_idx, entity.p2_idx))
        return connected

    def _convert_lines_to_beziers(
        self,
        registry: "EntityRegistry",
        lines: List[Tuple[EntityID, EntityID, EntityID]],
    ) -> List[EntityID]:
        """Remove Line entities and add Bezier entities in their place."""
        bezier_ids = []
        line_ids_to_remove = [lid for lid, unused1, unused2 in lines]
        registry.remove_entities_by_id(line_ids_to_remove)

        for unused, p1_idx, p2_idx in lines:
            bezier_id = registry.add_bezier(p1_idx, p2_idx)
            bezier_ids.append(bezier_id)

        return bezier_ids

    def _restore_lines(
        self,
        registry: "EntityRegistry",
        lines: List[Tuple[EntityID, EntityID, EntityID]],
        bezier_ids: List[EntityID],
    ):
        """Remove Bezier entities and restore Line entities."""
        registry.remove_entities_by_id(bezier_ids)
        for line_id, p1_idx, p2_idx in lines:
            new_line = Line(line_id, p1_idx, p2_idx)
            registry.entities.append(new_line)
            registry._entity_map[line_id] = new_line

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        try:
            waypoint = registry.get_point(self.waypoint_id)
        except IndexError:
            return

        connected_beziers = waypoint.get_connected_beziers(registry)
        self._old_bezier_states = {}
        for b in connected_beziers:
            self._old_bezier_states[b.id] = (b.cp1, b.cp2)

        self._old_waypoint_type = waypoint.waypoint_type

        if self.new_type in (WaypointType.SMOOTH, WaypointType.SYMMETRIC):
            connected_lines = self._find_connected_lines(registry, waypoint.id)
            if connected_lines:
                self._converted_lines = connected_lines
                self._added_bezier_ids = self._convert_lines_to_beziers(
                    registry, connected_lines
                )
                connected_beziers = waypoint.get_connected_beziers(registry)

            incoming_dir, outgoing_dir = self._get_segment_directions(
                registry, waypoint
            )

            avg_dir: Optional[GeoPoint] = None
            if incoming_dir is not None and outgoing_dir is not None:
                avg_dir = (
                    (incoming_dir[0] + outgoing_dir[0]) / 2,
                    (incoming_dir[1] + outgoing_dir[1]) / 2,
                )
                length = math.hypot(avg_dir[0], avg_dir[1])
                if length > 1e-9:
                    avg_dir = (avg_dir[0] / length, avg_dir[1] / length)
                else:
                    avg_dir = outgoing_dir
            elif outgoing_dir is not None:
                avg_dir = outgoing_dir
            elif incoming_dir is not None:
                avg_dir = (-incoming_dir[0], -incoming_dir[1])

            if avg_dir is None:
                avg_dir = (1.0, 0.0)

            cp_length = DEFAULT_CP_LENGTH / 3.0
            cp_in = (-avg_dir[0] * cp_length, -avg_dir[1] * cp_length)
            cp_out = (avg_dir[0] * cp_length, avg_dir[1] * cp_length)

            for b in connected_beziers:
                if b.start_idx == waypoint.id and b.cp1 is None:
                    b.cp1 = cp_out
                if b.end_idx == waypoint.id and b.cp2 is None:
                    b.cp2 = cp_in

        waypoint.waypoint_type = self.new_type
        waypoint.enforce_constraint(registry)

    def _do_undo(self) -> None:
        if self._old_waypoint_type is None:
            return

        registry = self.sketch.registry
        try:
            waypoint = registry.get_point(self.waypoint_id)
        except IndexError:
            return

        waypoint.waypoint_type = self._old_waypoint_type

        if self._old_bezier_states is not None:
            for bezier_id, (cp1, cp2) in self._old_bezier_states.items():
                bezier = registry.get_entity(bezier_id)
                if isinstance(bezier, Bezier):
                    bezier.cp1 = cp1
                    bezier.cp2 = cp2

        if (
            self._converted_lines is not None
            and self._added_bezier_ids is not None
        ):
            self._restore_lines(
                registry, self._converted_lines, self._added_bezier_ids
            )
