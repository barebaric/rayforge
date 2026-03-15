from __future__ import annotations
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, List, Tuple

from ...geo import primitives
from ..constraints import EqualDistanceConstraint
from ..entities import Arc, Point
from .base import PreviewState, SketchChangeCommand
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
        start_id: int,
        start_temp: bool,
        temp_end_id: int,
        temp_entity_id: int,
    ):
        self.center_id = center_id
        self.center_temp = center_temp
        self.start_id = start_id
        self.start_temp = start_temp
        self.temp_end_id = temp_end_id
        self.temp_entity_id = temp_entity_id
        self.clockwise = False


class ArcCommand(SketchChangeCommand):
    """A command to create an arc with center, start, and end points."""

    def __init__(
        self,
        sketch: Sketch,
        center_id: int,
        start_id: int,
        end_pos: Tuple[float, float],
        end_pid: Optional[int] = None,
        is_center_temp: bool = False,
        is_start_temp: bool = False,
        clockwise: bool = False,
    ):
        super().__init__(sketch, _("Add Arc"))
        self.center_id = center_id
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_center_temp = is_center_temp
        self.is_start_temp = is_start_temp
        self.clockwise = clockwise
        self.add_cmd: Optional[AddItemsCommand] = None

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
        try:
            center = registry.get_point(preview_state.center_id)
            start = registry.get_point(preview_state.start_id)
            end = registry.get_point(preview_state.temp_end_id)
            arc_ent = registry.get_entity(preview_state.temp_entity_id)
        except IndexError:
            return

        if not isinstance(arc_ent, Arc):
            return

        radius = math.hypot(start.x - center.x, start.y - center.y)
        projected = primitives.project_point_onto_circle(
            (x, y), (center.x, center.y), radius
        )
        if projected:
            end.x, end.y = projected
        else:
            end.x, end.y = x, y

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
        arc_ent = registry.get_entity(preview_state.temp_entity_id)
        if isinstance(arc_ent, Arc):
            preview_state.clockwise = arc_ent.clockwise

        if preview_state.temp_entity_id is not None:
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

        geom_constr = EqualDistanceConstraint(
            self.center_id, self.start_id, self.center_id, end_pid
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
            constraints=[geom_constr],
        )
        self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
