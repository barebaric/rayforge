from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional

from ...geo.types import Point as GeoPoint
from ..entities import Bezier, Line, Point
from ..entities.point import WaypointType
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class BezierPreviewState(PreviewState):
    """
    Preview state for unified line/bezier tool workflow.

    Workflow:
    - Click once: starts line preview from start to cursor
    - Click without drag: creates line segment, starts next preview
    - Click with drag: creates bezier segment where drag controls the "bow"

    Control points are stored on Bezier entity (cp1, cp2 as relative offsets).
    virtual_cp tracks the "outgoing" handle for the end point - it will become
    cp1 of the next bezier segment.
    """

    def __init__(
        self,
        start_id: EntityID,
        start_temp: bool,
        end_id: Optional[EntityID] = None,
        end_temp: bool = False,
        temp_entity_id: Optional[EntityID] = None,
        is_line_preview: bool = True,
        virtual_cp: Optional[GeoPoint] = None,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.end_id = end_id
        self.end_temp = end_temp
        self.temp_entity_id = temp_entity_id
        self.is_line_preview = is_line_preview
        self.virtual_cp = virtual_cp

    def get_preview_point_ids(self) -> set[EntityID]:
        result: set[int] = set()
        if self.end_temp and self.end_id is not None:
            result.add(self.end_id)
        return result

    def get_virtual_cp_absolute(
        self, registry: "EntityRegistry"
    ) -> Optional[GeoPoint]:
        if self.virtual_cp is None or self.end_id is None:
            return None
        end_pt = registry.get_point(self.end_id)
        if end_pt is None:
            return None
        return (end_pt.x + self.virtual_cp[0], end_pt.y + self.virtual_cp[1])

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List[DimensionData]:
        return []


class BezierCommand(SketchChangeCommand):
    """
    A command to create a cubic bezier curve or line segment.

    Control points are stored on Bezier entities (cp1, cp2).
    This command only creates the segment entity - it does not modify CPs.
    """

    def __init__(
        self,
        sketch: Sketch,
        start_id: EntityID,
        end_pos: GeoPoint,
        end_pid: Optional[EntityID] = None,
        is_start_temp: bool = False,
        is_line: bool = True,
        cp1: Optional[GeoPoint] = None,
        cp2: Optional[GeoPoint] = None,
    ):
        label = _("Add Line") if is_line else _("Add Bezier")
        super().__init__(sketch, label)
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.is_line = is_line
        self.cp1 = cp1
        self.cp2 = cp2
        self.add_cmd: Optional[AddItemsCommand] = None
        self._committed_end_id: Optional[EntityID] = None

    @property
    def committed_end_id(self) -> Optional[EntityID]:
        return self._committed_end_id

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
        virtual_cp: Optional[GeoPoint] = None,
        **kwargs,
    ) -> BezierPreviewState:
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
            try:
                start_pt = registry.get_point(snapped_pid)
                x, y = start_pt.x, start_pt.y
            except IndexError:
                pass
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        end_id = registry.add_point(x, y)

        effective_virtual_cp = virtual_cp
        if snapped_pid is not None and virtual_cp is None:
            try:
                start_pt = registry.get_point(snapped_pid)
                if not start_pt.is_sharp():
                    connected = start_pt.get_connected_beziers(registry)
                    for other_b in connected:
                        if other_b.end_idx == snapped_pid:
                            if other_b.cp2 is not None:
                                effective_virtual_cp = (
                                    -other_b.cp2[0],
                                    -other_b.cp2[1],
                                )
                        elif other_b.start_idx == snapped_pid:
                            if other_b.cp1 is not None:
                                effective_virtual_cp = (
                                    -other_b.cp1[0],
                                    -other_b.cp1[1],
                                )
                        break
            except (IndexError, ValueError):
                pass

        if effective_virtual_cp is not None:
            entity_id = registry.add_bezier(start_id, end_id)
            temp_entity = registry.get_entity(entity_id)
            if isinstance(temp_entity, Bezier):
                temp_entity.cp1 = effective_virtual_cp
            is_line = False
        else:
            entity_id = registry.add_line(start_id, end_id)
            is_line = True

        return BezierPreviewState(
            start_id=start_id,
            start_temp=start_temp,
            end_id=end_id,
            end_temp=True,
            temp_entity_id=entity_id,
            is_line_preview=is_line,
            virtual_cp=None,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        if not isinstance(preview_state, BezierPreviewState):
            raise AttributeError("Expected BezierPreviewState")

        if preview_state.end_id is None:
            return

        try:
            end_pt = registry.get_point(preview_state.end_id)
        except IndexError:
            return

        end_pt.x = x
        end_pt.y = y

    @staticmethod
    def convert_to_bezier(
        registry: EntityRegistry,
        preview_state: BezierPreviewState,
        waypoint_x: float,
        waypoint_y: float,
        drag_x: float,
        drag_y: float,
        mirror_cp_offset: Optional[GeoPoint] = None,
    ) -> None:
        """
        Converts a line preview to a bezier preview.

        Control points belong to BEZIER entity:
        - Drag controls the bezier's cp2 (incoming to end point)
        - virtual_cp tracks the outgoing handle (will be cp1 of next)
        - cp1 comes from previous segment's virtual_cp or segment default
        """
        if not preview_state.is_line_preview:
            return

        try:
            start_pt = registry.get_point(preview_state.start_id)
        except IndexError:
            return

        preview_state.is_line_preview = False

        registry.entities = [
            e
            for e in registry.entities
            if e.id != preview_state.temp_entity_id
        ]
        registry._entity_map = {
            k: v
            for k, v in registry._entity_map.items()
            if k != preview_state.temp_entity_id
        }

        if preview_state.end_id is None:
            return
        end_pt = registry.get_point(preview_state.end_id)
        end_pt.x = waypoint_x
        end_pt.y = waypoint_y

        drag_offset = (drag_x - waypoint_x, drag_y - waypoint_y)

        preview_state.virtual_cp = drag_offset

        preview_state.temp_entity_id = registry.add_bezier(
            preview_state.start_id,
            preview_state.end_id,
        )
        temp_entity = registry.get_entity(preview_state.temp_entity_id)
        if isinstance(temp_entity, Bezier):
            cp2_val = (-drag_offset[0], -drag_offset[1])
            temp_entity.cp2 = cp2_val

            cp1_val: Optional[GeoPoint] = None
            if mirror_cp_offset is not None:
                cp1_val = mirror_cp_offset
            elif not start_pt.is_sharp():
                connected = start_pt.get_connected_beziers(registry)
                for other_b in connected:
                    if other_b.id == preview_state.temp_entity_id:
                        continue
                    if other_b.end_idx == preview_state.start_id:
                        if other_b.cp2 is not None:
                            cp1_val = (-other_b.cp2[0], -other_b.cp2[1])
                    elif other_b.start_idx == preview_state.start_id:
                        if other_b.cp1 is not None:
                            cp1_val = (-other_b.cp1[0], -other_b.cp1[1])
                    break

            if cp1_val is None:
                seg_dx = waypoint_x - start_pt.x
                seg_dy = waypoint_y - start_pt.y
                seg_len = (seg_dx * seg_dx + seg_dy * seg_dy) ** 0.5
                if seg_len > 1e-9:
                    third = seg_len / 3.0
                    cp1_val = (
                        seg_dx / seg_len * third,
                        seg_dy / seg_len * third,
                    )
                else:
                    cp1_val = (drag_offset[0], drag_offset[1])
            temp_entity.cp1 = cp1_val

            logger.debug(
                f"convert_to_bezier: cp1={cp1_val}, "
                f"cp2={cp2_val}, virtual_cp={preview_state.virtual_cp}"
            )

    @staticmethod
    def update_control_point(
        registry: EntityRegistry,
        preview_state: BezierPreviewState,
        x: float,
        y: float,
    ) -> None:
        """Update control points during bezier drag.

        The drag controls the bezier's cp2 (incoming to end point).
        virtual_cp tracks the outgoing handle for the next segment.
        """
        if preview_state.is_line_preview or preview_state.end_id is None:
            return

        try:
            end_pt = registry.get_point(preview_state.end_id)
        except IndexError:
            return

        drag_offset = (x - end_pt.x, y - end_pt.y)
        preview_state.virtual_cp = drag_offset

        if preview_state.temp_entity_id is not None:
            temp_entity = registry.get_entity(preview_state.temp_entity_id)
            if isinstance(temp_entity, Bezier):
                temp_entity.cp2 = (-drag_offset[0], -drag_offset[1])

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        if not isinstance(preview_state, BezierPreviewState):
            raise AttributeError("Expected BezierPreviewState")

        logger.debug(
            f"cleanup_preview: temp_entity_id={preview_state.temp_entity_id}, "
            f"end_id={preview_state.end_id}, end_temp={preview_state.end_temp}"
        )

        if preview_state.temp_entity_id is not None:
            registry.entities = [
                e
                for e in registry.entities
                if e.id != preview_state.temp_entity_id
            ]
            registry._entity_map = {e.id: e for e in registry.entities}

        if preview_state.end_temp and preview_state.end_id is not None:
            registry.points = [
                p for p in registry.points if p.id != preview_state.end_id
            ]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        registry = self.sketch.registry

        final_x, final_y = self.end_pos
        if self.end_pid is not None:
            try:
                end_p = registry.get_point(self.end_pid)
                final_x, final_y = end_p.x, end_p.y
            except IndexError:
                pass

        new_point = None
        end_pid = self.end_pid

        if end_pid is None:
            temp_id = registry._id_counter
            end_pid = temp_id
            new_point = Point(temp_id, final_x, final_y)

        points_to_add: List[Point] = [new_point] if new_point else []

        if self.is_start_temp:
            try:
                p = registry.get_point(self.start_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        if self.is_line:
            temp_entity_id = registry._id_counter + (1 if new_point else 0)
            new_entity = Line(temp_entity_id, self.start_id, end_pid)
        else:
            temp_entity_id = registry._id_counter + (1 if new_point else 0)
            new_entity = Bezier(
                temp_entity_id,
                self.start_id,
                end_pid,
                cp1=self.cp1,
                cp2=self.cp2,
            )

            try:
                start_pt = registry.get_point(self.start_id)
                start_pt.waypoint_type = WaypointType.SYMMETRIC
            except (IndexError, ValueError):
                pass

            if new_point:
                new_point.waypoint_type = WaypointType.SYMMETRIC
            elif end_pid is not None:
                try:
                    end_pt = registry.get_point(end_pid)
                    end_pt.waypoint_type = WaypointType.SYMMETRIC
                except (IndexError, ValueError):
                    pass

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=[new_entity],
            constraints=[],
        )
        self.add_cmd._do_execute()
        self._committed_end_id = end_pid

        if not self.is_line and isinstance(new_entity, Bezier):
            try:
                start_pt = registry.get_point(self.start_id)
                if not start_pt.is_sharp():
                    connected = start_pt.get_connected_beziers(registry)
                    for other_b in connected:
                        if other_b.id != new_entity.id:
                            if other_b.end_idx == self.start_id:
                                if other_b.cp2 is not None:
                                    new_entity.cp1 = (
                                        -other_b.cp2[0],
                                        -other_b.cp2[1],
                                    )
                            elif other_b.start_idx == self.start_id:
                                if other_b.cp1 is not None:
                                    new_entity.cp1 = (
                                        -other_b.cp1[0],
                                        -other_b.cp1[1],
                                    )
                            break
            except (IndexError, ValueError):
                pass

            try:
                end_pt = registry.get_point(end_pid)
                if end_pt is not None and not end_pt.is_sharp():
                    connected = end_pt.get_connected_beziers(registry)
                    for other_b in connected:
                        if other_b.id != new_entity.id:
                            if other_b.start_idx == end_pid:
                                if other_b.cp1 is not None:
                                    new_entity.cp2 = (
                                        -other_b.cp1[0],
                                        -other_b.cp1[1],
                                    )
                            elif other_b.end_idx == end_pid:
                                if other_b.cp2 is not None:
                                    new_entity.cp2 = (
                                        -other_b.cp2[0],
                                        -other_b.cp2[1],
                                    )
                            break
            except (IndexError, ValueError):
                pass

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
