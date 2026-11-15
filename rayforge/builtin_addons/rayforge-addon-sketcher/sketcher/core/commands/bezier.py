from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.entities.bezier import Bezier
from raygeo.sketch.entities.line import Line
from raygeo.sketch.types import EntityID

from ..entity_flags import WaypointType
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch

logger = logging.getLogger(__name__)


def _registry_connected_beziers(
    registry: EntityRegistry, sketch: Sketch, pid: EntityID
):
    from .waypoint import get_connected_beziers

    return get_connected_beziers(registry, sketch, pid)


def _mirror_cp(
    registry: EntityRegistry,
    sketch: Sketch,
    pid: EntityID,
    self_bezier_id: EntityID | None,
    from_end: bool,
) -> GeoPoint | None:
    """The negated control offset of a neighbor bezier at the point,
    for smooth continuation."""
    for other_b in _registry_connected_beziers(registry, sketch, pid):
        if self_bezier_id is not None and other_b.id == self_bezier_id:
            continue
        if (
            from_end
            and other_b.end_idx == pid
            and (other_b.cp2_offset is not None)
        ):
            return (-other_b.cp2_offset[0], -other_b.cp2_offset[1])
        if (
            not from_end
            and other_b.start_idx == pid
            and (other_b.cp1_offset is not None)
        ):
            return (-other_b.cp1_offset[0], -other_b.cp1_offset[1])
    return None


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
        end_id: EntityID | None = None,
        end_temp: bool = False,
        temp_entity_id: EntityID | None = None,
        is_line_preview: bool = True,
        virtual_cp: GeoPoint | None = None,
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
        self, registry: EntityRegistry
    ) -> GeoPoint | None:
        if self.virtual_cp is None or self.end_id is None:
            return None
        end_pt = registry.get_point(self.end_id)
        if end_pt is None:
            return None
        return (end_pt[0] + self.virtual_cp[0], end_pt[1] + self.virtual_cp[1])

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
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
        end_pid: EntityID | None = None,
        is_start_temp: bool = False,
        is_line: bool = True,
        cp1: GeoPoint | None = None,
        cp2: GeoPoint | None = None,
        constraints: list | None = None,
    ):
        label = _("Add Line") if is_line else _("Add Bezier")
        super().__init__(sketch, label)
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.is_line = is_line
        self.cp1_offset = cp1
        self.cp2_offset = cp2
        self.constraints = constraints or []
        self.add_cmd: AddItemsCommand | None = None
        self._committed_end_id: EntityID | None = None

    @property
    def committed_end_id(self) -> EntityID | None:
        return self._committed_end_id

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: int | None = None,
        virtual_cp: GeoPoint | None = None,
        **kwargs,
    ) -> BezierPreviewState:
        sketch = kwargs.pop("sketch", None)
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
            start_pt = registry.get_point(snapped_pid)
            if start_pt is not None:
                x, y = start_pt
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        end_id = registry.add_point(x, y)

        effective_virtual_cp = virtual_cp
        if snapped_pid is not None and virtual_cp is None and sketch:
            mirrored = _mirror_cp(
                registry, sketch, snapped_pid, None, from_end=True
            )
            if mirrored is not None:
                effective_virtual_cp = mirrored

        if effective_virtual_cp is not None:
            entity_id = registry.add_bezier(start_id, end_id)
            registry.set_state(
                entity_id,
                {"cp1": [effective_virtual_cp[0], effective_virtual_cp[1]]},
            )
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
            raise TypeError("Expected BezierPreviewState")

        if preview_state.end_id is None:
            return

        registry.set_point_position(preview_state.end_id, x, y)

    @staticmethod
    def convert_to_bezier(
        registry: EntityRegistry,
        preview_state: BezierPreviewState,
        waypoint_x: float,
        waypoint_y: float,
        drag_x: float,
        drag_y: float,
        mirror_cp_offset: GeoPoint | None = None,
        sketch: Sketch | None = None,
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

        start_pt = registry.get_point(preview_state.start_id)
        if start_pt is None:
            return

        preview_state.is_line_preview = False

        registry.remove_entities([preview_state.temp_entity_id])

        if preview_state.end_id is None:
            return
        registry.set_point_position(
            preview_state.end_id, waypoint_x, waypoint_y
        )

        drag_offset = (drag_x - waypoint_x, drag_y - waypoint_y)

        preview_state.virtual_cp = drag_offset

        preview_state.temp_entity_id = registry.add_bezier(
            preview_state.start_id,
            preview_state.end_id,
        )
        temp_entity_id = preview_state.temp_entity_id
        cp2_val = (-drag_offset[0], -drag_offset[1])
        state: dict = {"cp2": [cp2_val[0], cp2_val[1]]}

        cp1_val: GeoPoint | None = mirror_cp_offset
        if cp1_val is None and sketch is not None:
            cp1_val = _mirror_cp(
                registry,
                sketch,
                preview_state.start_id,
                temp_entity_id,
                from_end=True,
            )

        if cp1_val is None:
            seg_dx = waypoint_x - start_pt[0]
            seg_dy = waypoint_y - start_pt[1]
            seg_len = (seg_dx * seg_dx + seg_dy * seg_dy) ** 0.5
            if seg_len > 1e-9:
                third = seg_len / 3.0
                cp1_val = (
                    seg_dx / seg_len * third,
                    seg_dy / seg_len * third,
                )
            else:
                cp1_val = (drag_offset[0], drag_offset[1])
        state["cp1"] = [cp1_val[0], cp1_val[1]]
        registry.set_state(temp_entity_id, state)

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

        end_pt = registry.get_point(preview_state.end_id)
        if end_pt is None:
            return

        drag_offset = (x - end_pt[0], y - end_pt[1])
        preview_state.virtual_cp = drag_offset

        if preview_state.temp_entity_id is not None:
            registry.set_state(
                preview_state.temp_entity_id,
                {"cp2": [-drag_offset[0], -drag_offset[1]]},
            )

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        if not isinstance(preview_state, BezierPreviewState):
            raise TypeError("Expected BezierPreviewState")

        logger.debug(
            f"cleanup_preview: temp_entity_id={preview_state.temp_entity_id}, "
            f"end_id={preview_state.end_id}, end_temp={preview_state.end_temp}"
        )

        if preview_state.temp_entity_id is not None:
            registry.remove_entities([preview_state.temp_entity_id])

        if preview_state.end_temp and preview_state.end_id is not None:
            registry.remove_points([preview_state.end_id])

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        registry = self.sketch.registry

        final_x, final_y = self.end_pos
        if self.end_pid is not None:
            position = registry.get_point(self.end_pid)
            if position is not None:
                final_x, final_y = position

        end_pid = self.end_pid
        add_cmd = AddItemsCommand(self.sketch, "")
        new_pid = False
        if end_pid is None:
            end_pid = add_cmd.add_point(final_x, final_y)
            new_pid = True

        if self.is_start_temp and registry.get_point(self.start_id):
            add_cmd.adopt_point(self.start_id)

        if self.is_line:
            add_cmd.add_entity(Line(self.start_id, end_pid))
            self.add_cmd = add_cmd
            self._committed_end_id = end_pid
            return

        start_cp1 = self.cp1_offset
        if start_cp1 is None:
            start_cp1 = _mirror_cp(
                registry,
                self.sketch,
                self.start_id,
                None,
                from_end=True,
            )

        end_cp2 = self.cp2_offset
        if end_cp2 is None:
            mirror_source_pid = self.start_id if new_pid else end_pid
            end_cp2 = _mirror_cp(
                registry,
                self.sketch,
                mirror_source_pid,
                None,
                from_end=False,
            )
            if end_cp2 is not None and end_cp2 == start_cp1:
                end_cp2 = None

        add_cmd.add_entity(
            Bezier(
                self.start_id,
                end_pid,
                cp1=start_cp1,
                cp2=end_cp2,
            )
        )

        # Bezier endpoints become smooth waypoints so connected
        # segments continue each other.
        self.sketch.flags.set_waypoint_type(
            self.start_id, WaypointType.SYMMETRIC
        )
        if new_pid:
            self.sketch.flags.set_waypoint_type(
                end_pid, WaypointType.SYMMETRIC
            )

        self.add_cmd = add_cmd
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
