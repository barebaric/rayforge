from __future__ import annotations
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from rayforge.core.geo import Point as GeoPoint
from ..constraints import (
    DistanceConstraint,
    HorizontalConstraint,
    VerticalConstraint,
)
from ..entities import Line, Point
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class RectanglePreviewState(PreviewState):
    """Preview state for rectangle tool's 2-click workflow."""

    def __init__(
        self,
        start_id: EntityID,
        start_temp: bool,
        p_end_id: EntityID,
        preview_ids: Dict[str, EntityID],
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.p_end_id = p_end_id
        self.preview_ids = preview_ids
        self.locked_width: Optional[float] = None
        self.locked_height: Optional[float] = None

    def get_preview_point_ids(self) -> Set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the start point since that may be permanent.
        """
        result = {self.p_end_id}
        for key in ["p2", "p4"]:
            pid = self.preview_ids.get(key)
            if pid is not None:
                result.add(pid)
        return result

    def set_dimensions(
        self,
        registry: "EntityRegistry",
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        """
        Sets the rectangle dimensions from numeric input.

        Args:
            registry: The entity registry to modify.
            width: The width to apply (or None to keep current).
            height: The height to apply (or None to keep current).
        """
        if width is not None:
            self.locked_width = width
        if height is not None:
            self.locked_height = height

        try:
            start_p = registry.get_point(self.start_id)
            end_p = registry.get_point(self.p_end_id)
        except IndexError:
            return

        dx = end_p.x - start_p.x
        dy = end_p.y - start_p.y

        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0

        new_width = (
            self.locked_width if self.locked_width is not None else abs(dx)
        )
        new_height = (
            self.locked_height if self.locked_height is not None else abs(dy)
        )

        end_p.x = start_p.x + sign_x * new_width
        end_p.y = start_p.y + sign_y * new_height

        RectangleCommand.create_preview(
            registry,
            self.start_id,
            self.p_end_id,
            preview_ids=self.preview_ids,
        )

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List["DimensionData"]:
        """
        Returns width and height dimensions for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing DimensionData for width and height.
        """
        try:
            p1 = registry.get_point(self.start_id)
            p2 = registry.get_point(self.p_end_id)
        except IndexError:
            return []
        width = abs(p2.x - p1.x)
        height = abs(p2.y - p1.y)
        mid_x = (p1.x + p2.x) / 2
        top_y = min(p1.y, p2.y)
        right_x = max(p1.x, p2.x)
        return [
            DimensionData(
                label=DimensionData.format_length(width),
                position=(mid_x, top_y),
            ),
            DimensionData(
                label=DimensionData.format_length(height),
                position=(right_x, (p1.y + p2.y) / 2),
            ),
        ]


class RectangleCommand(SketchChangeCommand):
    """A smart command to create a fully constrained rectangle."""

    def __init__(
        self,
        sketch: Sketch,
        start_pid: EntityID,
        end_pos: GeoPoint,
        end_pid: Optional[EntityID] = None,
        is_start_temp: bool = False,
        fixed_width: Optional[float] = None,
        fixed_height: Optional[float] = None,
    ):
        super().__init__(sketch, _("Add Rectangle"))
        self.start_pid = start_pid
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        self.add_cmd: Optional[AddItemsCommand] = None
        self._committed_end_id: Optional[EntityID] = None

    @property
    def committed_end_id(self) -> Optional[EntityID]:
        """
        The final end point ID after execute(), or None if not applicable.
        """
        return self._committed_end_id

    @staticmethod
    def calculate_geometry(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        start_pid: EntityID,
        end_pid: Optional[EntityID],
        fixed_width: Optional[float] = None,
        fixed_height: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Calculates the points, entities, and constraints for a rectangle."""
        if abs(x2 - x1) < 1e-6 or abs(y2 - y1) < 1e-6:
            return None

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        p3_id = end_pid if end_pid is not None else next_temp_id()

        points = {
            "p1_id": start_pid,
            "p2": Point(next_temp_id(), x2, y1),
            "p3": Point(p3_id, x2, y2),
            "p4": Point(next_temp_id(), x1, y2),
        }

        entities = [
            Line(next_temp_id(), points["p1_id"], points["p2"].id),
            Line(next_temp_id(), points["p2"].id, points["p3"].id),
            Line(next_temp_id(), points["p3"].id, points["p4"].id),
            Line(next_temp_id(), points["p4"].id, points["p1_id"]),
        ]

        constraints: List[Any] = [
            HorizontalConstraint(points["p1_id"], points["p2"].id),
            VerticalConstraint(points["p2"].id, points["p3"].id),
            HorizontalConstraint(points["p4"].id, points["p3"].id),
            VerticalConstraint(points["p1_id"], points["p4"].id),
        ]

        top_edge_y = min(y1, y2)
        right_edge_x = max(x1, x2)

        if top_edge_y == y1:
            top_edge_p1 = points["p1_id"]
            top_edge_p2 = points["p2"].id
        else:
            top_edge_p1 = points["p4"].id
            top_edge_p2 = points["p3"].id

        if right_edge_x == x2:
            right_edge_p1 = points["p2"].id
            right_edge_p2 = points["p3"].id
        else:
            right_edge_p1 = points["p1_id"]
            right_edge_p2 = points["p4"].id

        if fixed_width is not None:
            constraints.append(
                DistanceConstraint(top_edge_p1, top_edge_p2, fixed_width)
            )

        if fixed_height is not None:
            constraints.append(
                DistanceConstraint(right_edge_p1, right_edge_p2, fixed_height)
            )

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
        }

    @staticmethod
    def create_preview(
        registry: EntityRegistry,
        start_pid: EntityID,
        end_pid: EntityID,
        preview_ids: Optional[Dict[str, EntityID]] = None,
    ) -> Optional[Dict[str, EntityID]]:
        """
        Creates or updates preview geometry in the registry.

        Args:
            registry: The entity registry to modify.
            start_pid: The ID of the start corner point.
            end_pid: The ID of the end corner point (preview corner).
            preview_ids: Existing preview IDs to update, or None to create new.

        Returns:
            Dict of preview IDs, or None if geometry is invalid.
        """
        try:
            start_p = registry.get_point(start_pid)
            end_p = registry.get_point(end_pid)
        except IndexError:
            return None

        coords = {
            "p2": (end_p.x, start_p.y),
            "p4": (start_p.x, end_p.y),
        }

        if preview_ids is None:
            # Create new preview geometry
            preview_ids = {}
            for name, (px, py) in coords.items():
                preview_ids[name] = registry.add_point(px, py)

            # Create lines
            preview_ids["line1"] = registry.add_line(
                start_pid, preview_ids["p2"]
            )
            preview_ids["line2"] = registry.add_line(
                preview_ids["p2"], end_pid
            )
            preview_ids["line3"] = registry.add_line(
                end_pid, preview_ids["p4"]
            )
            preview_ids["line4"] = registry.add_line(
                preview_ids["p4"], start_pid
            )
        else:
            # Update existing preview geometry
            for name, (px, py) in coords.items():
                p = registry.get_point(preview_ids[name])
                p.x, p.y = px, py

        return preview_ids

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: Optional[EntityID] = None,
        **kwargs,
    ) -> RectanglePreviewState:
        """
        Creates initial preview state with start and end points.

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.

        Returns:
            RectanglePreviewState for use with update_preview and
            cleanup_preview.
        """
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        p_end_id = registry.add_point(x, y)

        preview_ids = RectangleCommand.create_preview(
            registry, start_id, p_end_id
        )
        assert preview_ids is not None

        return RectanglePreviewState(
            start_id=start_id,
            start_temp=start_temp,
            p_end_id=p_end_id,
            preview_ids=preview_ids,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        """
        Updates the end point position and refreshes preview geometry.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.
            x, y: The new end point coordinates.

        Raises:
            AttributeError: If preview_state is not a RectanglePreviewState.
        """
        if not isinstance(preview_state, RectanglePreviewState):
            raise AttributeError("Expected RectanglePreviewState")
        try:
            p_end = registry.get_point(preview_state.p_end_id)
            p_start = registry.get_point(preview_state.start_id)
        except IndexError:
            return

        if (
            preview_state.locked_width is not None
            or preview_state.locked_height is not None
        ):
            dx = p_end.x - p_start.x
            dy = p_end.y - p_start.y
            sign_x = 1.0 if dx >= 0 else -1.0
            sign_y = 1.0 if dy >= 0 else -1.0

            new_x = (
                p_start.x + sign_x * preview_state.locked_width
                if preview_state.locked_width is not None
                else x
            )
            new_y = (
                p_start.y + sign_y * preview_state.locked_height
                if preview_state.locked_height is not None
                else y
            )
            p_end.x = new_x
            p_end.y = new_y
        else:
            p_end.x = x
            p_end.y = y

        RectangleCommand.create_preview(
            registry,
            preview_state.start_id,
            preview_state.p_end_id,
            preview_ids=preview_state.preview_ids,
        )

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes all preview entities and points from the registry.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.

        Raises:
            AttributeError: If preview_state is not a RectanglePreviewState.
        """
        if not isinstance(preview_state, RectanglePreviewState):
            raise AttributeError("Expected RectanglePreviewState")
        preview_ids = preview_state.preview_ids
        p_end_id = preview_state.p_end_id

        # Collect all point IDs to remove
        point_ids = set(preview_ids.values())
        point_ids.add(p_end_id)

        # Find and remove entities that use these points
        entity_ids_to_remove = {
            e.id
            for e in registry.entities
            if any(pid in point_ids for pid in e.get_point_ids())
        }
        registry.remove_entities_by_id(list(entity_ids_to_remove))

        # Remove points
        registry.points = [p for p in registry.points if p.id not in point_ids]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        reg = self.sketch.registry
        try:
            start_p = reg.get_point(self.start_pid)
        except IndexError:
            return

        final_mx, final_my = self.end_pos
        if self.end_pid is not None:
            try:
                end_p = reg.get_point(self.end_pid)
                final_mx, final_my = end_p.x, end_p.y
            except IndexError:
                pass  # Use mouse coords if pid is invalid

        result = self.calculate_geometry(
            start_p.x,
            start_p.y,
            final_mx,
            final_my,
            self.start_pid,
            self.end_pid,
            fixed_width=self.fixed_width,
            fixed_height=self.fixed_height,
        )
        if not result:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_pid)
            return

        points_dict = result["points"]
        points_to_add = []
        # These points are always new
        points_to_add.extend([points_dict["p2"], points_dict["p4"]])

        # Add p3 only if it wasn't an existing snapped point
        if self.end_pid is None:
            points_to_add.append(points_dict["p3"])

        # If the start point was temporary, remove it from the registry
        # and add its object to the command to be re-added properly.
        if self.is_start_temp:
            reg.points.remove(start_p)
            points_to_add.append(start_p)

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self.add_cmd._do_execute()
        self._committed_end_id = self.end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
