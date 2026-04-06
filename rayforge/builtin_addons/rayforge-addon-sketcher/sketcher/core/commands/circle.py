from __future__ import annotations
from gettext import gettext as _
import math
from typing import TYPE_CHECKING, List, Optional, Set

from rayforge.core.geo import Point as GeoPoint
from ..constraints import DiameterConstraint
from ..entities import Circle, Point
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
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
        self.locked_diameter: Optional[float] = None

    def get_preview_point_ids(self) -> Set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the center point since it may be permanent.
        """
        return {self.radius_id}

    def set_diameter(
        self, registry: "EntityRegistry", diameter: float
    ) -> None:
        """
        Sets the circle diameter from numeric input.

        Args:
            registry: The entity registry to modify.
            diameter: The diameter to apply.
        """
        self.locked_diameter = diameter

        try:
            center = registry.get_point(self.center_id)
            radius_pt = registry.get_point(self.radius_id)
        except IndexError:
            return

        dx = radius_pt.x - center.x
        dy = radius_pt.y - center.y
        current_radius = math.hypot(dx, dy)
        target_radius = diameter / 2.0

        if current_radius < 1e-9:
            radius_pt.x = center.x + target_radius
            return

        scale = target_radius / current_radius
        radius_pt.x = center.x + dx * scale
        radius_pt.y = center.y + dy * scale

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List["DimensionData"]:
        """
        Returns the circle diameter dimension for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing a single DimensionData for the circle diameter.
        """
        try:
            center = registry.get_point(self.center_id)
            radius_pt = registry.get_point(self.radius_id)
        except IndexError:
            return []
        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        diameter = radius * 2
        dx = radius_pt.x - center.x
        dy = radius_pt.y - center.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if dist > 1e-9 else 0
        arc_mid_x = center.x + radius * math.cos(angle)
        arc_mid_y = center.y + radius * math.sin(angle)
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
        end_pid: Optional[EntityID] = None,
        is_center_temp: bool = False,
        fixed_diameter: Optional[float] = None,
    ):
        super().__init__(sketch, _("Add Circle"))
        self.center_id = center_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_center_temp = is_center_temp
        self.fixed_diameter = fixed_diameter
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

        Args:
            registry: The entity registry.
            preview_state: The preview state from start_preview.
            x, y: The new cursor coordinates.

        Raises:
            AttributeError: If preview_state is not a CirclePreviewState.
        """
        if not isinstance(preview_state, CirclePreviewState):
            raise AttributeError("Expected CirclePreviewState")

        if preview_state.locked_diameter is not None:
            return

        try:
            radius_p = registry.get_point(preview_state.radius_id)
        except IndexError:
            return
        radius_p.x = x
        radius_p.y = y

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes preview entities from the registry.

        Note: This does NOT remove the center point if center_temp=True.
        The tool is responsible for removing it if the user cancels.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.

        Raises:
            AttributeError: If preview_state is not a CirclePreviewState.
        """
        if not isinstance(preview_state, CirclePreviewState):
            raise AttributeError("Expected CirclePreviewState")

        if preview_state.entity_id is not None:
            registry.entities = [
                e for e in registry.entities if e.id != preview_state.entity_id
            ]
            registry._entity_map = {e.id: e for e in registry.entities}

        if preview_state.radius_id is not None:
            registry.points = [
                p for p in registry.points if p.id != preview_state.radius_id
            ]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        registry = self.sketch.registry

        try:
            registry.get_point(self.center_id)
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

        if end_pid == self.center_id:
            if self.is_center_temp:
                self.sketch.remove_point_if_unused(self.center_id)
            return

        temp_circle_id = registry._id_counter + (1 if new_point else 0)
        new_circle = Circle(temp_circle_id, self.center_id, end_pid)

        points_to_add: List[Point] = [new_point] if new_point else []

        if self.is_center_temp:
            try:
                p = registry.get_point(self.center_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        constraints_to_add = []
        if self.fixed_diameter is not None:
            diameter_constr = DiameterConstraint(
                temp_circle_id, self.fixed_diameter, user_visible=True
            )
            constraints_to_add.append(diameter_constr)

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=[new_circle],
            constraints=constraints_to_add,
        )
        self.add_cmd._do_execute()
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
