from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo.types import Point as GeoPoint

from ..entity_flags import WaypointType
from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch

logger = logging.getLogger(__name__)

DEFAULT_CP_LENGTH = 30.0


def get_connected_beziers(
    registry: EntityRegistry,
    sketch: Sketch,
    waypoint_id: EntityID,
) -> list[Any]:
    """Bezier entities connected to the waypoint or any point in its
    coincident group."""
    point_ids = {waypoint_id}
    point_ids.update(sketch.get_coincident_points(waypoint_id))
    connected = []
    for eid in registry.entity_ids():
        if registry.entity_type(eid) != "bezier":
            continue
        entity = registry.get_entity_object(eid)
        if entity is None:
            continue
        if entity.start_idx in point_ids or entity.end_idx in point_ids:
            connected.append(entity)
    return connected


def enforce_waypoint_constraint(
    registry: EntityRegistry,
    sketch: Sketch,
    waypoint_id: EntityID,
    waypoint_type: WaypointType,
) -> None:
    """Applies the waypoint's smoothing semantics to the control
    offsets of its connected beziers: symmetric waypoints get equal
    and opposite offsets of common length."""
    if waypoint_type is WaypointType.SHARP:
        return

    beziers = get_connected_beziers(registry, sketch, waypoint_id)
    if len(beziers) < 2:
        return

    point_ids = {waypoint_id}
    point_ids.update(sketch.get_coincident_points(waypoint_id))

    cp_data = []
    for bezier in beziers:
        if bezier.start_idx in point_ids and bezier.cp1_offset is not None:
            cp_data.append((bezier, "cp1", bezier.cp1_offset))
        elif bezier.end_idx in point_ids and bezier.cp2_offset is not None:
            cp_data.append((bezier, "cp2", bezier.cp2_offset))

    if len(cp_data) < 2:
        return

    if waypoint_type is WaypointType.SYMMETRIC:
        avg_length = sum(
            math.sqrt(cp[0] ** 2 + cp[1] ** 2) for _, _, cp in cp_data
        ) / len(cp_data)
        if avg_length < 1e-10:
            return

        sum_dx = 0.0
        sum_dy = 0.0
        for _pid, attr, cp in cp_data:
            sign = 1.0 if attr == "cp1" else -1.0
            sum_dx += sign * cp[0]
            sum_dy += sign * cp[1]
        length = math.sqrt(sum_dx**2 + sum_dy**2)
        if length < 1e-10:
            return
        direction = (sum_dx / length, sum_dy / length)

        for bezier, attr, _cp in cp_data:
            sign = 1.0 if attr == "cp1" else -1.0
            _set_cp_offset(
                registry,
                bezier.id,
                attr,
                (
                    sign * direction[0] * avg_length,
                    sign * direction[1] * avg_length,
                ),
            )


def _set_cp_offset(
    registry: EntityRegistry,
    bezier_id: EntityID,
    attr: str,
    offset: tuple[float, float],
) -> None:
    registry.set_state(bezier_id, {attr: [offset[0], offset[1]]})


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
        sketch: Sketch,
        waypoint_id: EntityID,
        new_type: WaypointType,
    ):
        label = _("Set Waypoint Type")
        super().__init__(sketch, label)
        self.waypoint_id = waypoint_id
        self.new_type = new_type
        self._old_waypoint_type: WaypointType | None = None
        self._old_bezier_states: (
            dict[int, tuple[GeoPoint | None, GeoPoint | None]] | None
        ) = None
        self._converted_lines: (
            list[tuple[EntityID, EntityID, EntityID, bool]] | None
        ) = None
        self._added_bezier_ids: list[int] | None = None

    def _get_segment_directions(
        self, registry: EntityRegistry, waypoint_id: EntityID
    ) -> tuple[GeoPoint | None, GeoPoint | None]:
        """
        Get the incoming and outgoing direction vectors at this waypoint.

        Returns (incoming_dir, outgoing_dir) as normalized vectors.
        """
        incoming_dir: GeoPoint | None = None
        outgoing_dir: GeoPoint | None = None

        waypoint = registry.get_point(waypoint_id)
        if waypoint is None:
            return None, None
        wx, wy = waypoint

        point_ids = {waypoint_id}
        point_ids.update(self.sketch.get_coincident_points(waypoint_id))

        for eid in registry.entity_ids():
            entity_type = registry.entity_type(eid)
            entity = registry.get_entity_object(eid)
            if entity is None:
                continue
            if entity_type == "line":
                if entity.p2_idx in point_ids:
                    p1 = registry.get_point(entity.p1_idx)
                    if p1:
                        dx = wx - p1[0]
                        dy = wy - p1[1]
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            incoming_dir = (dx / length, dy / length)
                elif entity.p1_idx in point_ids:
                    p2 = registry.get_point(entity.p2_idx)
                    if p2:
                        dx = p2[0] - wx
                        dy = p2[1] - wy
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            outgoing_dir = (dx / length, dy / length)
            elif entity_type == "bezier":
                if entity.end_idx in point_ids:
                    if entity.cp2_offset is not None:
                        cp_abs = (
                            wx + entity.cp2_offset[0],
                            wy + entity.cp2_offset[1],
                        )
                        dx = wx - cp_abs[0]
                        dy = wy - cp_abs[1]
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            incoming_dir = (dx / length, dy / length)
                    else:
                        start = registry.get_point(entity.start_idx)
                        if start:
                            dx = wx - start[0]
                            dy = wy - start[1]
                            length = math.hypot(dx, dy)
                            if length > 1e-9:
                                incoming_dir = (dx / length, dy / length)
                elif entity.start_idx in point_ids:
                    if entity.cp1_offset is not None:
                        cp_abs = (
                            wx + entity.cp1_offset[0],
                            wy + entity.cp1_offset[1],
                        )
                        dx = cp_abs[0] - wx
                        dy = cp_abs[1] - wy
                        length = math.hypot(dx, dy)
                        if length > 1e-9:
                            outgoing_dir = (dx / length, dy / length)
                    else:
                        end = registry.get_point(entity.end_idx)
                        if end:
                            dx = end[0] - wx
                            dy = end[1] - wy
                            length = math.hypot(dx, dy)
                            if length > 1e-9:
                                outgoing_dir = (dx / length, dy / length)

        return incoming_dir, outgoing_dir

    def _find_connected_lines(
        self, registry: EntityRegistry, waypoint_id: EntityID
    ) -> list[tuple[EntityID, EntityID, EntityID, bool]]:
        """Find Line entities connected to this waypoint.

        Returns list of (line_id, p1_idx, p2_idx, construction).
        """
        point_ids = {waypoint_id}
        point_ids.update(self.sketch.get_coincident_points(waypoint_id))

        connected = []
        for eid in registry.entity_ids():
            if registry.entity_type(eid) != "line":
                continue
            entity = registry.get_entity_object(eid)
            if entity is None:
                continue
            if entity.p1_idx in point_ids or entity.p2_idx in point_ids:
                connected.append(
                    (
                        entity.id,
                        entity.p1_idx,
                        entity.p2_idx,
                        entity.construction,
                    )
                )
        return connected

    def _convert_lines_to_beziers(
        self,
        registry: EntityRegistry,
        lines: list[tuple[EntityID, EntityID, EntityID, bool]],
    ) -> list[EntityID]:
        """Remove Line entities and add Bezier entities in their place."""
        bezier_ids = []
        line_ids_to_remove = [lid for lid, *_ in lines]
        registry.remove_entities(line_ids_to_remove)

        for entry in lines:
            _, p1_idx, p2_idx, construction = entry
            bezier_id = registry.add_bezier(
                p1_idx, p2_idx, construction=construction
            )
            bezier_ids.append(bezier_id)

        return bezier_ids

    def _restore_lines(
        self,
        registry: EntityRegistry,
        lines: list[tuple[EntityID, EntityID, EntityID, bool]],
        bezier_ids: list[EntityID],
    ):
        """Remove Bezier entities and restore Line entities with their
        original ids."""
        registry.remove_entities(bezier_ids)
        line_dicts = [
            {
                "id": line_id,
                "type": "line",
                "p1_idx": p1_idx,
                "p2_idx": p2_idx,
                "construction": construction,
            }
            for line_id, p1_idx, p2_idx, construction in lines
        ]
        self.sketch.restore_structure([], line_dicts)

    def _do_execute(self) -> None:
        registry = self.sketch.registry
        waypoint = registry.get_point(self.waypoint_id)
        if waypoint is None:
            return

        connected_beziers = get_connected_beziers(
            registry, self.sketch, self.waypoint_id
        )
        self._old_bezier_states = {}
        for bezier in connected_beziers:
            self._old_bezier_states[bezier.id] = (
                bezier.cp1_offset,
                bezier.cp2_offset,
            )

        self._old_waypoint_type = self.sketch.flags.waypoint_type(
            self.waypoint_id
        )

        if self.new_type in (WaypointType.SMOOTH, WaypointType.SYMMETRIC):
            connected_lines = self._find_connected_lines(
                registry, self.waypoint_id
            )
            if connected_lines:
                self._converted_lines = connected_lines
                self._added_bezier_ids = self._convert_lines_to_beziers(
                    registry, connected_lines
                )
                connected_beziers = get_connected_beziers(
                    registry, self.sketch, self.waypoint_id
                )

            incoming_dir, outgoing_dir = self._get_segment_directions(
                registry, self.waypoint_id
            )

            avg_dir: GeoPoint | None = None
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

            point_ids = {self.waypoint_id}
            point_ids.update(
                self.sketch.get_coincident_points(self.waypoint_id)
            )

            for bezier in connected_beziers:
                if bezier.start_idx in point_ids and bezier.cp1_offset is None:
                    _set_cp_offset(registry, bezier.id, "cp1", cp_out)
                if bezier.end_idx in point_ids and bezier.cp2_offset is None:
                    _set_cp_offset(registry, bezier.id, "cp2", cp_in)

        self.sketch.flags.set_waypoint_type(self.waypoint_id, self.new_type)
        enforce_waypoint_constraint(
            registry, self.sketch, self.waypoint_id, self.new_type
        )

    def _do_undo(self) -> None:
        if self._old_waypoint_type is None:
            return

        registry = self.sketch.registry
        if registry.get_point(self.waypoint_id) is None:
            return

        self.sketch.flags.set_waypoint_type(
            self.waypoint_id, self._old_waypoint_type
        )

        if self._old_bezier_states is not None:
            for bezier_id, (cp1, cp2) in self._old_bezier_states.items():
                if registry.entity_type(bezier_id) != "bezier":
                    continue
                state: dict[str, Any] = {}
                if cp1 is not None:
                    state["cp1"] = [cp1[0], cp1[1]]
                if cp2 is not None:
                    state["cp2"] = [cp2[0], cp2[1]]
                if state:
                    registry.set_state(bezier_id, state)

        if (
            self._converted_lines is not None
            and self._added_bezier_ids is not None
        ):
            self._restore_lines(
                registry, self._converted_lines, self._added_bezier_ids
            )
