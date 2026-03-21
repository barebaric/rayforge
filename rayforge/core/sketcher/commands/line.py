from __future__ import annotations
from gettext import gettext as _
import math
from typing import TYPE_CHECKING, List, Optional, Set

from ...geo import Point as GeoPoint
from ..constraints import DistanceConstraint
from ..entities import Line, Point
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class LinePreviewState(PreviewState):
    """Preview state for line tool's 2-click workflow."""

    def __init__(
        self,
        start_id: EntityID,
        start_temp: bool,
        end_id: EntityID,
        entity_id: EntityID,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.end_id = end_id
        self.entity_id = entity_id
        self.locked_length: Optional[float] = None

    def get_preview_point_ids(self) -> Set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the start point since it may be permanent.
        """
        return {self.end_id}

    def set_length(self, registry: "EntityRegistry", length: float) -> None:
        """
        Sets the line length from numeric input.

        Args:
            registry: The entity registry to modify.
            length: The length to apply.
        """
        self.locked_length = length

        try:
            start_p = registry.get_point(self.start_id)
            end_p = registry.get_point(self.end_id)
        except IndexError:
            return

        dx = end_p.x - start_p.x
        dy = end_p.y - start_p.y
        current_length = math.hypot(dx, dy)

        if current_length < 1e-9:
            end_p.x = start_p.x + length
            return

        scale = length / current_length
        end_p.x = start_p.x + dx * scale
        end_p.y = start_p.y + dy * scale

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List["DimensionData"]:
        """
        Returns the line length dimension for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing a single DimensionData for the line length.
        """
        try:
            p1 = registry.get_point(self.start_id)
            p2 = registry.get_point(self.end_id)
        except IndexError:
            return []
        length = math.hypot(p2.x - p1.x, p2.y - p1.y)
        mid_x = (p1.x + p2.x) / 2
        mid_y = (p1.y + p2.y) / 2
        return [
            DimensionData(
                label=DimensionData.format_length(length),
                position=(mid_x, mid_y),
            )
        ]


class LineCommand(SketchChangeCommand):
    """A command to create a line between two points."""

    def __init__(
        self,
        sketch: Sketch,
        start_id: EntityID,
        end_pos: GeoPoint,
        end_pid: Optional[EntityID] = None,
        is_start_temp: bool = False,
        fixed_length: Optional[float] = None,
    ):
        super().__init__(sketch, _("Add Line"))
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.fixed_length = fixed_length
        self.add_cmd: Optional[AddItemsCommand] = None
        self._committed_end_id: Optional[EntityID] = None

    @property
    def committed_end_id(self) -> Optional[EntityID]:
        """
        The final end point ID after execute(), or None if not applicable.
        """
        return self._committed_end_id

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: Optional[EntityID] = None,
        **kwargs,
    ) -> LinePreviewState:
        """
        Creates initial preview state with start point, end point, and line.

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.

        Returns:
            LinePreviewState for use with update_preview and cleanup_preview.
        """
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        end_id = registry.add_point(x, y)
        entity_id = registry.add_line(start_id, end_id)

        return LinePreviewState(
            start_id=start_id,
            start_temp=start_temp,
            end_id=end_id,
            entity_id=entity_id,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        """
        Updates the preview end point position.

        Args:
            registry: The entity registry.
            preview_state: The preview state from start_preview.
            x, y: The new cursor coordinates.

        Raises:
            AttributeError: If preview_state is not a LinePreviewState.
        """
        if not isinstance(preview_state, LinePreviewState):
            raise AttributeError("Expected LinePreviewState")

        if preview_state.locked_length is not None:
            return

        try:
            end_p = registry.get_point(preview_state.end_id)
        except IndexError:
            return
        end_p.x = x
        end_p.y = y

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes preview entities from the registry.

        Note: This does NOT remove the start point if start_temp=True.
        The tool is responsible for removing it if the user cancels.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.

        Raises:
            AttributeError: If preview_state is not a LinePreviewState.
        """
        if not isinstance(preview_state, LinePreviewState):
            raise AttributeError("Expected LinePreviewState")

        if preview_state.entity_id is not None:
            registry.entities = [
                e for e in registry.entities if e.id != preview_state.entity_id
            ]
            registry._entity_map = {
                k: v
                for k, v in registry._entity_map.items()
                if k != preview_state.entity_id
            }

        if preview_state.end_id is not None:
            registry.points = [
                p for p in registry.points if p.id != preview_state.end_id
            ]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        registry = self.sketch.registry

        try:
            registry.get_point(self.start_id)
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
            temp_id = registry._id_counter
            end_pid = temp_id
            new_point = Point(temp_id, final_x, final_y)

        if end_pid == self.start_id:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_id)
            return

        temp_line_id = registry._id_counter + (1 if new_point else 0)
        new_line = Line(temp_line_id, self.start_id, end_pid)

        points_to_add: List[Point] = [new_point] if new_point else []

        if self.is_start_temp:
            try:
                p = registry.get_point(self.start_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        constraints = []
        if self.fixed_length is not None:
            constraints.append(
                DistanceConstraint(self.start_id, end_pid, self.fixed_length)
            )

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=[new_line],
            constraints=constraints,
        )
        self.add_cmd._do_execute()
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
