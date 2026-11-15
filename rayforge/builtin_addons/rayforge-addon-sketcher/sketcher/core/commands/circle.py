from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.entities.circle import Circle
from raygeo.sketch.types import EntityID

from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch


class CirclePreviewState(PreviewState):
    """Preview state for circle tool's 2-click workflow."""

    def __init__(
        self,
        center_id: EntityID,
        center_temp: bool,
        radius_id: EntityID,
        entity_id: EntityID,
    ):
        self.center_id = center_id
        self.center_temp = center_temp
        self.radius_id = radius_id
        self.entity_id = entity_id
        self.locked_diameter: float | None = None

    def get_preview_point_ids(self) -> set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the center point since it may be permanent.
        """
        return {self.radius_id}

    def set_diameter(self, registry: EntityRegistry, diameter: float) -> None:
        """
        Sets the circle diameter from numeric input.

        Args:
            registry: The entity registry to modify.
            diameter: The diameter to apply.
        """
        self.locked_diameter = diameter

        center = registry.get_point(self.center_id)
        radius_pt = registry.get_point(self.radius_id)
        if center is None or radius_pt is None:
            return

        dx = radius_pt[0] - center[0]
        dy = radius_pt[1] - center[1]
        current_radius = math.hypot(dx, dy)
        target_radius = diameter / 2.0

        if current_radius < 1e-9:
            registry.set_point_position(
                self.radius_id, center[0] + target_radius, center[1]
            )
            return

        scale = target_radius / current_radius
        registry.set_point_position(
            self.radius_id,
            center[0] + dx * scale,
            center[1] + dy * scale,
        )

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
        """
        Returns the circle diameter dimension for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing a single DimensionData for the circle diameter.
        """
        center = registry.get_point(self.center_id)
        radius_pt = registry.get_point(self.radius_id)
        if center is None or radius_pt is None:
            return []
        radius = math.hypot(radius_pt[0] - center[0], radius_pt[1] - center[1])
        diameter = radius * 2
        dx = radius_pt[0] - center[0]
        dy = radius_pt[1] - center[1]
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if dist > 1e-9 else 0
        arc_mid_x = center[0] + radius * math.cos(angle)
        arc_mid_y = center[1] + radius * math.sin(angle)
        return [
            DimensionData(
                label=f"Ø{DimensionData.format_length(diameter)}",
                position=(arc_mid_x, arc_mid_y),
            )
        ]


class CircleCommand(SketchChangeCommand):
    """A command to create a circle with center and radius points."""

    def __init__(
        self,
        sketch: Sketch,
        center_id: EntityID,
        end_pos: GeoPoint,
        end_pid: EntityID | None = None,
        is_center_temp: bool = False,
        fixed_diameter: float | None = None,
    ):
        super().__init__(sketch, _("Add Circle"))
        self.center_id = center_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_center_temp = is_center_temp
        self.fixed_diameter = fixed_diameter
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
    ) -> CirclePreviewState:
        """
        Creates initial preview state with center, radius point, and circle.

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.

        Returns:
            CirclePreviewState for use with update_preview and cleanup_preview.
        """
        if snapped_pid is not None:
            center_id = snapped_pid
            center_temp = False
        else:
            center_id = registry.add_point(x, y)
            center_temp = True

        radius_id = registry.add_point(x, y)
        entity_id = registry.add_circle(center_id, radius_id)

        return CirclePreviewState(
            center_id=center_id,
            center_temp=center_temp,
            radius_id=radius_id,
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
        Updates the preview radius point position.

        Raises:
            TypeError: If preview_state is not a CirclePreviewState.
        """
        if not isinstance(preview_state, CirclePreviewState):
            raise TypeError("Expected CirclePreviewState")

        if preview_state.locked_diameter is not None:
            return

        registry.set_point_position(preview_state.radius_id, x, y)

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes preview entities from the registry.

        Note: This does NOT remove the center point if center_temp=True.
        The tool is responsible for removing it if the user cancels.

        Raises:
            TypeError: If preview_state is not a CirclePreviewState.
        """
        if not isinstance(preview_state, CirclePreviewState):
            raise TypeError("Expected CirclePreviewState")

        if preview_state.entity_id is not None:
            registry.remove_entities([preview_state.entity_id])

        if preview_state.radius_id is not None:
            registry.remove_points([preview_state.radius_id])

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        registry = self.sketch.registry

        if registry.get_point(self.center_id) is None:
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

        if end_pid == self.center_id:
            if self.is_center_temp:
                self.sketch.remove_point_if_unused(self.center_id)
            return

        center_id = self.center_id
        if self.is_center_temp and registry.get_point(self.center_id):
            position = registry.get_point(self.center_id)
            self.sketch.remove_point_if_unused(self.center_id)
            center_id = add_cmd.add_point(position[0], position[1])

        circle_id = add_cmd.add_entity(Circle(center_id, end_pid))
        if self.fixed_diameter is not None:
            add_cmd.add_constraint(
                {
                    "type": "DiameterConstraint",
                    "circle_id": circle_id,
                    "value": self.fixed_diameter,
                }
            )

        self.add_cmd = add_cmd
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
