from __future__ import annotations
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional

from ...geo import Point as GeoPoint, primitives
from ..constraints import EqualDistanceConstraint, RadiusConstraint
from ..entities import Arc, Point
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class ArcPreviewState(PreviewState):
    """Preview state for arc tool's 3-click workflow."""

    def __init__(
        self,
        center_id: int,
        center_temp: bool,
        start_id: Optional[int] = None,
        start_temp: bool = False,
        temp_end_id: Optional[int] = None,
        temp_entity_id: Optional[int] = None,
    ):
        self.center_id = center_id
        self.center_temp = center_temp
        self.start_id = start_id
        self.start_temp = start_temp
        self.temp_end_id = temp_end_id
        self.temp_entity_id = temp_entity_id
        self.clockwise = False
        self.locked_radius: Optional[float] = None

    def get_preview_point_ids(self) -> set[int]:
        """
        Returns IDs of temp preview points that shouldn't be snapped to.

        Excludes center and start points since they may be permanent.
        """
        result: set[int] = set()
        if self.temp_end_id is not None:
            result.add(self.temp_end_id)
        return result

    @property
    def has_start_point(self) -> bool:
        """Returns True if start point has been set."""
        return self.start_id is not None

    def set_radius(self, registry: "EntityRegistry", radius: float) -> None:
        """
        Sets the arc radius from numeric input.

        Args:
            registry: The entity registry to modify.
            radius: The radius to apply.
        """
        if self.start_id is None or self.temp_end_id is None:
            return

        self.locked_radius = radius

        try:
            center = registry.get_point(self.center_id)
            start = registry.get_point(self.start_id)
            end = registry.get_point(self.temp_end_id)
        except IndexError:
            return

        start_angle = math.atan2(start.y - center.y, start.x - center.x)
        end_angle = math.atan2(end.y - center.y, end.x - center.x)

        start.x = center.x + radius * math.cos(start_angle)
        start.y = center.y + radius * math.sin(start_angle)
        end.x = center.x + radius * math.cos(end_angle)
        end.y = center.y + radius * math.sin(end_angle)

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List[DimensionData]:
        """
        Returns the arc radius dimension for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing a single DimensionData for the arc radius.
        """
        if self.start_id is None or self.temp_end_id is None:
            return []
        try:
            center = registry.get_point(self.center_id)
            start = registry.get_point(self.start_id)
            end = registry.get_point(self.temp_end_id)
        except IndexError:
            return []
        radius = math.hypot(start.x - center.x, start.y - center.y)
        arc = Arc(
            -1,
            self.start_id,
            self.temp_end_id,
            self.center_id,
            clockwise=self.clockwise,
        )
        midpoint = arc.get_midpoint(registry)
        if not midpoint:
            start_angle = math.atan2(start.y - center.y, start.x - center.x)
            end_angle = math.atan2(end.y - center.y, end.x - center.x)
            mid_angle = (start_angle + end_angle) / 2
        else:
            mid_angle = math.atan2(
                midpoint[1] - center.y, midpoint[0] - center.x
            )
        arc_mid_x = center.x + radius * math.cos(mid_angle)
        arc_mid_y = center.y + radius * math.sin(mid_angle)
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
        center_id: int,
        start_id: int,
        end_pos: GeoPoint,
        end_pid: Optional[int] = None,
        is_center_temp: bool = False,
        is_start_temp: bool = False,
        clockwise: bool = False,
        fixed_radius: Optional[float] = None,
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
        self.add_cmd: Optional[AddItemsCommand] = None
        self._committed_end_id: Optional[int] = None

    @property
    def committed_end_id(self) -> Optional[int]:
        """The final end point ID after execute(), or None."""
        return self._committed_end_id

    @staticmethod
    def start_center_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
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
        snapped_pid: Optional[int] = None,
    ) -> None:
        """
        Sets the start point and creates the preview arc entity.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_center_preview.
            x, y: The start point coordinates.
            snapped_pid: An existing point ID to snap to, or None.

        Raises:
            AttributeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise AttributeError("Expected ArcPreviewState")

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
        snapped_pid: Optional[int] = None,
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

        Args:
            registry: The entity registry.
            preview_state: The preview state from start_preview.
            x, y: The new cursor coordinates.

        Raises:
            AttributeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise AttributeError("Expected ArcPreviewState")
        if (
            preview_state.temp_end_id is None
            or preview_state.temp_entity_id is None
            or preview_state.start_id is None
        ):
            return

        try:
            center = registry.get_point(preview_state.center_id)
            start = registry.get_point(preview_state.start_id)
            end = registry.get_point(preview_state.temp_end_id)
            arc_ent = registry.get_entity(preview_state.temp_entity_id)
        except IndexError:
            return

        if not isinstance(arc_ent, Arc):
            return

        if preview_state.locked_radius is not None:
            cursor_radius = preview_state.locked_radius
        else:
            cursor_radius = math.hypot(x - center.x, y - center.y)

        start_angle = math.atan2(start.y - center.y, start.x - center.x)
        start.x = center.x + cursor_radius * math.cos(start_angle)
        start.y = center.y + cursor_radius * math.sin(start_angle)
        end.x = center.x + cursor_radius * math.cos(
            math.atan2(y - center.y, x - center.x)
        )
        end.y = center.y + cursor_radius * math.sin(
            math.atan2(y - center.y, x - center.x)
        )

        arc_ent.clockwise = primitives.determine_arc_direction(
            (center.x, center.y), (start.x, start.y), (x, y)
        )
        preview_state.clockwise = arc_ent.clockwise

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes preview entities from the registry.

        Stores the final clockwise direction in preview_state.clockwise
        before cleanup for the tool to read.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.

        Raises:
            AttributeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise AttributeError("Expected ArcPreviewState")

        if preview_state.temp_entity_id is not None:
            try:
                arc_ent = registry.get_entity(preview_state.temp_entity_id)
                if isinstance(arc_ent, Arc):
                    preview_state.clockwise = arc_ent.clockwise
            except IndexError:
                pass

            registry.entities = [
                e
                for e in registry.entities
                if e.id != preview_state.temp_entity_id
            ]
            registry._entity_map = {e.id: e for e in registry.entities}

        if preview_state.temp_end_id is not None:
            registry.points = [
                p for p in registry.points if p.id != preview_state.temp_end_id
            ]

    @staticmethod
    def cleanup_center_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes center point preview (when only center is set).

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_center_preview.

        Raises:
            AttributeError: If preview_state is not an ArcPreviewState.
        """
        if not isinstance(preview_state, ArcPreviewState):
            raise AttributeError("Expected ArcPreviewState")

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        registry = self.sketch.registry

        try:
            center_p = registry.get_point(self.center_id)
            start_p = registry.get_point(self.start_id)
        except IndexError:
            return

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
            radius = math.hypot(start_p.x - center_p.x, start_p.y - center_p.y)
            projected = primitives.project_point_onto_circle(
                (final_x, final_y), (center_p.x, center_p.y), radius
            )
            if projected:
                final_x, final_y = projected

            temp_id = registry._id_counter
            end_pid = temp_id
            new_point = Point(temp_id, final_x, final_y)

        if end_pid == self.start_id or end_pid == self.center_id:
            if self.is_center_temp:
                self.sketch.remove_point_if_unused(self.center_id)
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_id)
            return

        temp_arc_id = registry._id_counter + (1 if new_point else 0)
        new_arc = Arc(
            temp_arc_id,
            self.start_id,
            end_pid,
            self.center_id,
            clockwise=self.clockwise,
        )

        constraints: List = [
            EqualDistanceConstraint(
                self.center_id, self.start_id, self.center_id, end_pid
            )
        ]

        if self.fixed_radius is not None:
            constraints.append(
                RadiusConstraint(temp_arc_id, self.fixed_radius)
            )

        points_to_add: List[Point] = [new_point] if new_point else []

        if self.is_center_temp:
            try:
                p = registry.get_point(self.center_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        if self.is_start_temp:
            try:
                p = registry.get_point(self.start_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=[new_arc],
            constraints=constraints,
        )
        self.add_cmd._do_execute()
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
