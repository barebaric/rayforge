from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.shape.arc import get_arc_direction, get_arc_sweep
from raygeo.geo.shape.circle import project_point_onto_circle
from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.entities.arc import Arc

from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch


class ArcPreviewState(PreviewState):
    """Preview state for arc tool's 3-click workflow."""

    def __init__(
        self,
        center_id: EntityID,
        center_temp: bool,
        start_id: EntityID | None = None,
        start_temp: bool = False,
        temp_end_id: EntityID | None = None,
        temp_entity_id: EntityID | None = None,
    ):
        self.center_id = center_id
        self.center_temp = center_temp
        self.start_id = start_id
        self.start_temp = start_temp
        self.temp_end_id = temp_end_id
        self.temp_entity_id = temp_entity_id
        self.clockwise = False
        self.locked_radius: float | None = None

    def get_preview_point_ids(self) -> set[EntityID]:
        """
        Returns IDs of temp preview points that shouldn't be snapped to.

        Excludes center and start points since they may be permanent.
        """
        result: set[EntityID] = set()
        if self.temp_end_id is not None:
            result.add(self.temp_end_id)
        return result

    @property
    def has_start_point(self) -> bool:
        """Returns True if start point has been set."""
        return self.start_id is not None

    def set_radius(self, registry: EntityRegistry, radius: float) -> None:
        """
        Sets the arc radius from numeric input.

        Args:
            registry: The entity registry to modify.
            radius: The radius to apply.
        """
        if self.start_id is None or self.temp_end_id is None:
            return

        self.locked_radius = radius

        center = registry.get_point(self.center_id)
        start = registry.get_point(self.start_id)
        end = registry.get_point(self.temp_end_id)
        if center is None or start is None or end is None:
            return

        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])

        registry.set_point_position(
            self.start_id,
            center[0] + radius * math.cos(start_angle),
            center[1] + radius * math.sin(start_angle),
        )
        registry.set_point_position(
            self.temp_end_id,
            center[0] + radius * math.cos(end_angle),
            center[1] + radius * math.sin(end_angle),
        )

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
        """
        Returns the arc radius dimension for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing a single DimensionData for the arc radius.
        """
        if self.start_id is None or self.temp_end_id is None:
            return []
        center = registry.get_point(self.center_id)
        start = registry.get_point(self.start_id)
        end = registry.get_point(self.temp_end_id)
        if center is None or start is None or end is None:
            return []
        radius = math.hypot(start[0] - center[0], start[1] - center[1])
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        sweep = get_arc_sweep(start_angle, end_angle, self.clockwise)
        mid_angle = start_angle + sweep / 2
        arc_mid_x = center[0] + radius * math.cos(mid_angle)
        arc_mid_y = center[1] + radius * math.sin(mid_angle)
        return [
            DimensionData(
                label=f"R{DimensionData.format_length(radius)}",
                position=(arc_mid_x, arc_mid_y),
            )
        ]


class ArcCommand(SketchChangeCommand):
    """A command to create an arc with center, start, and end points."""

    def __init__(
        self,
        sketch: Sketch,
        center_id: EntityID,
        start_id: EntityID,
        end_pos: GeoPoint,
        end_pid: EntityID | None = None,
        is_center_temp: bool = False,
        is_start_temp: bool = False,
        clockwise: bool = False,
        fixed_radius: float | None = None,
    ):
        super().__init__(sketch, _("Add Arc"))
        self.center_id = center_id
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_center_temp = is_center_temp
        self.is_start_temp = is_start_temp
        self.clockwise = clockwise
        self.fixed_radius = fixed_radius
        self.add_cmd: AddItemsCommand | None = None
        self._committed_end_id: EntityID | None = None

    @property
    def committed_end_id(self) -> EntityID | None:
        """The final end point ID after execute(), or None."""
        return self._committed_end_id

    @staticmethod
    def start_center_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
        **kwargs,
    ) -> ArcPreviewState:
        """
        Creates preview state after first click (center point).

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.

        Returns:
            ArcPreviewState with center point set.
        """
        if snapped_pid is not None:
            center_id = snapped_pid
            center_temp = False
        else:
            center_id = registry.add_point(x, y)
            center_temp = True

        return ArcPreviewState(
            center_id=center_id,
            center_temp=center_temp,
        )

    @staticmethod
    def set_start_point(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
    ) -> None:
        """
        Sets the start point and creates the preview arc entity.

        Raises:
            TypeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise TypeError("Expected ArcPreviewState")

        if snapped_pid is not None and snapped_pid != preview_state.center_id:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        temp_end_id = registry.add_point(x, y)
        temp_entity_id = registry.add_arc(
            start_id, temp_end_id, preview_state.center_id
        )

        preview_state.start_id = start_id
        preview_state.start_temp = start_temp
        preview_state.temp_end_id = temp_end_id
        preview_state.temp_entity_id = temp_entity_id

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
        **kwargs,
    ) -> ArcPreviewState:
        """
        Creates preview arc entity after start point is set.

        This method expects center_id, center_temp, start_id, and start_temp
        to be passed via kwargs since the arc tool has a 3-click workflow
        where center and start are already established.

        Args:
            registry: The entity registry to modify.
            x, y: Initial end point coordinates.
            snapped_pid: Not used for arc preview.
            **kwargs: Must include center_id, center_temp,
                start_id, start_temp.

        Returns:
            ArcPreviewState for use with update_preview and cleanup_preview.
        """
        center_id = kwargs["center_id"]
        center_temp = kwargs["center_temp"]
        start_id = kwargs["start_id"]
        start_temp = kwargs["start_temp"]

        temp_end_id = registry.add_point(x, y)
        temp_entity_id = registry.add_arc(start_id, temp_end_id, center_id)

        return ArcPreviewState(
            center_id=center_id,
            center_temp=center_temp,
            start_id=start_id,
            start_temp=start_temp,
            temp_end_id=temp_end_id,
            temp_entity_id=temp_entity_id,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        """
        Updates the preview arc's end point position and direction.

        Raises:
            TypeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise TypeError("Expected ArcPreviewState")
        if (
            preview_state.temp_end_id is None
            or preview_state.temp_entity_id is None
            or preview_state.start_id is None
        ):
            return

        center = registry.get_point(preview_state.center_id)
        start = registry.get_point(preview_state.start_id)
        if center is None or start is None:
            return
        if registry.entity_type(preview_state.temp_entity_id) != "arc":
            return

        if preview_state.locked_radius is not None:
            cursor_radius = preview_state.locked_radius
        else:
            cursor_radius = math.hypot(x - center[0], y - center[1])

        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        registry.set_point_position(
            preview_state.start_id,
            center[0] + cursor_radius * math.cos(start_angle),
            center[1] + cursor_radius * math.sin(start_angle),
        )
        end_angle = math.atan2(y - center[1], x - center[0])
        registry.set_point_position(
            preview_state.temp_end_id,
            center[0] + cursor_radius * math.cos(end_angle),
            center[1] + cursor_radius * math.sin(end_angle),
        )

        clockwise = get_arc_direction(center, start, (x, y))
        registry.set_state(
            preview_state.temp_entity_id, {"clockwise": clockwise}
        )
        preview_state.clockwise = clockwise

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes preview entities from the registry.

        Stores the final clockwise direction in preview_state.clockwise
        before cleanup for the tool to read.

        Raises:
            TypeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise TypeError("Expected ArcPreviewState")

        if preview_state.temp_entity_id is not None:
            entity = registry.get_entity_object(preview_state.temp_entity_id)
            if isinstance(entity, Arc):
                preview_state.clockwise = entity.clockwise
            registry.remove_entities([preview_state.temp_entity_id])

        if preview_state.temp_end_id is not None:
            registry.remove_points([preview_state.temp_end_id])

    @staticmethod
    def cleanup_center_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes center point preview (when only center is set).

        Raises:
            TypeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise TypeError("Expected ArcPreviewState")

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        registry = self.sketch.registry

        center_p = registry.get_point(self.center_id)
        start_p = registry.get_point(self.start_id)
        if center_p is None or start_p is None:
            return

        final_x, final_y = self.end_pos
        if self.end_pid is not None:
            position = registry.get_point(self.end_pid)
            if position is not None:
                final_x, final_y = position

        end_pid = self.end_pid
        add_cmd = AddItemsCommand(self.sketch, "")

        if end_pid is None:
            radius = math.hypot(
                start_p[0] - center_p[0], start_p[1] - center_p[1]
            )
            projected = project_point_onto_circle(
                (final_x, final_y), center_p, radius
            )
            if projected:
                final_x, final_y = projected
            end_pid = add_cmd.add_point(final_x, final_y)

        if end_pid == self.start_id or end_pid == self.center_id:
            if self.is_center_temp:
                self.sketch.remove_point_if_unused(self.center_id)
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_id)
            return

        if self.is_center_temp and registry.get_point(self.center_id):
            add_cmd.adopt_point(self.center_id)
        if self.is_start_temp and registry.get_point(self.start_id):
            add_cmd.adopt_point(self.start_id)

        arc_id = add_cmd.add_entity(
            Arc(
                self.start_id,
                end_pid,
                self.center_id,
                clockwise=self.clockwise,
            )
        )
        add_cmd.add_constraint(
            EqualDistanceConstraint(
                self.center_id, self.start_id, self.center_id, end_pid
            )
        )
        if self.fixed_radius is not None:
            add_cmd.add_constraint(
                {
                    "type": "RadiusConstraint",
                    "entity_id": arc_id,
                    "value": self.fixed_radius,
                }
            )

        self.add_cmd = add_cmd
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
