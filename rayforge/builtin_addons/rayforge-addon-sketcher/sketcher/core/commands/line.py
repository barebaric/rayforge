from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.entities.line import Line

from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

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
        self.locked_length: float | None = None

    def get_preview_point_ids(self) -> set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the start point since it may be permanent.
        """
        return {self.end_id}

    def set_length(self, registry: EntityRegistry, length: float) -> None:
        """
        Sets the line length from numeric input.

        Args:
            registry: The entity registry to modify.
            length: The length to apply.
        """
        self.locked_length = length

        start_p = registry.get_point(self.start_id)
        end_p = registry.get_point(self.end_id)
        if start_p is None or end_p is None:
            return

        dx = end_p[0] - start_p[0]
        dy = end_p[1] - start_p[1]
        current_length = math.hypot(dx, dy)

        if current_length < 1e-9:
            registry.set_point_position(
                self.end_id, start_p[0] + length, start_p[1]
            )
            return

        scale = length / current_length
        registry.set_point_position(
            self.end_id,
            start_p[0] + dx * scale,
            start_p[1] + dy * scale,
        )

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
        """
        Returns the line length dimension for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing a single DimensionData for the line length.
        """
        p1 = registry.get_point(self.start_id)
        p2 = registry.get_point(self.end_id)
        if p1 is None or p2 is None:
            return []
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
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
        end_pid: EntityID | None = None,
        is_start_temp: bool = False,
        fixed_length: float | None = None,
    ):
        super().__init__(sketch, _("Add Line"))
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.fixed_length = fixed_length
        self.add_cmd: AddItemsCommand | None = None
        self._committed_end_id: EntityID | None = None

    @property
    def committed_end_id(self) -> EntityID | None:
        """
        The final end point ID after execute(), or None if not applicable.
        """
        return self._committed_end_id

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
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
            TypeError: If preview_state is not a LinePreviewState.
        """
        if not isinstance(preview_state, LinePreviewState):
            raise TypeError("Expected LinePreviewState")

        if preview_state.locked_length is not None:
            return

        registry.set_point_position(preview_state.end_id, x, y)

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

        Raises:
            TypeError: If preview_state is not a LinePreviewState.
        """
        if not isinstance(preview_state, LinePreviewState):
            raise TypeError("Expected LinePreviewState")

        if preview_state.entity_id is not None:
            registry.remove_entities([preview_state.entity_id])

        if preview_state.end_id is not None:
            registry.remove_points([preview_state.end_id])

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        registry = self.sketch.registry

        if registry.get_point(self.start_id) is None:
            return

        final_x, final_y = self.end_pos
        if self.end_pid is not None:
            position = registry.get_point(self.end_pid)
            if position is not None:
                final_x, final_y = position

        end_pid = self.end_pid
        add_cmd = AddItemsCommand(self.sketch, "")
        if end_pid is None:
            end_pid = add_cmd.add_point(final_x, final_y)

        if end_pid == self.start_id:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_id)
            return

        start_id = self.start_id
        if self.is_start_temp:
            if registry.get_point(self.start_id) is not None:
                # The preview start point belongs to no command yet;
                # replace it with a freshly allocated point so undo
                # removes it with the line.
                position = registry.get_point(self.start_id)
                self.sketch.remove_point_if_unused(self.start_id)
                start_id = add_cmd.add_point(position[0], position[1])

        add_cmd.add_entity(Line(start_id, end_pid))
        if self.fixed_length is not None:
            add_cmd.add_constraint(
                DistanceConstraint(start_id, end_pid, self.fixed_length)
            )

        self.add_cmd = add_cmd
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
