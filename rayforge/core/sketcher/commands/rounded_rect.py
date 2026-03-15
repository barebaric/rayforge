from __future__ import annotations
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from .base import PreviewState, SketchChangeCommand
from .items import AddItemsCommand
from ..entities import Point, Line, Arc
from ..constraints import (
    HorizontalConstraint,
    VerticalConstraint,
    TangentConstraint,
    EqualLengthConstraint,
    EqualDistanceConstraint,
)

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class RoundedRectPreviewState(PreviewState):
    """Preview state for rounded rectangle tool's 2-click workflow."""

    def __init__(
        self,
        start_id: int,
        start_temp: bool,
        p_end_id: int,
        preview_ids: Dict[str, int],
        radius: float,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.p_end_id = p_end_id
        self.preview_ids = preview_ids
        self.radius = radius


class RoundedRectCommand(SketchChangeCommand):
    """A smart command to create a fully constrained rounded rectangle."""

    def __init__(
        self,
        sketch: Sketch,
        start_pid: int,
        end_pos: Tuple[float, float],
        radius: float,
        is_start_temp: bool = False,
    ):
        super().__init__(sketch, _("Add Rounded Rectangle"))
        self.start_pid = start_pid
        self.end_pos = end_pos
        self.radius = radius
        self.is_start_temp = is_start_temp
        self.add_cmd: Optional[AddItemsCommand] = None

    @staticmethod
    def calculate_geometry(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
    ) -> Optional[Dict[str, Any]]:
        """Calculates geometry for a rounded rectangle."""
        width, height = abs(x2 - x1), abs(y2 - y1)
        if width < 1e-6 or height < 1e-6:
            return None

        radius = min(radius, width / 2.0, height / 2.0)
        sx, sy = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1)

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        points = {
            "t1": Point(next_temp_id(), x1 + sx * radius, y1),
            "t2": Point(next_temp_id(), x2 - sx * radius, y1),
            "t3": Point(next_temp_id(), x2, y1 + sy * radius),
            "t4": Point(next_temp_id(), x2, y2 - sy * radius),
            "t5": Point(next_temp_id(), x2 - sx * radius, y2),
            "t6": Point(next_temp_id(), x1 + sx * radius, y2),
            "t7": Point(next_temp_id(), x1, y2 - sy * radius),
            "t8": Point(next_temp_id(), x1, y1 + sy * radius),
            "c1": Point(next_temp_id(), x1 + sx * radius, y1 + sy * radius),
            "c2": Point(next_temp_id(), x2 - sx * radius, y1 + sy * radius),
            "c3": Point(next_temp_id(), x2 - sx * radius, y2 - sy * radius),
            "c4": Point(next_temp_id(), x1 + sx * radius, y2 - sy * radius),
        }

        is_cw = sx * sy < 0
        entities = [
            Line(next_temp_id(), points["t1"].id, points["t2"].id),
            Line(next_temp_id(), points["t3"].id, points["t4"].id),
            Line(next_temp_id(), points["t5"].id, points["t6"].id),
            Line(next_temp_id(), points["t7"].id, points["t8"].id),
            Arc(
                next_temp_id(),
                points["t8"].id,
                points["t1"].id,
                points["c1"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t2"].id,
                points["t3"].id,
                points["c2"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t4"].id,
                points["t5"].id,
                points["c3"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t6"].id,
                points["t7"].id,
                points["c4"].id,
                clockwise=is_cw,
            ),
        ]

        constraints = [
            HorizontalConstraint(points["t1"].id, points["t2"].id),
            VerticalConstraint(points["t3"].id, points["t4"].id),
            HorizontalConstraint(points["t5"].id, points["t6"].id),
            VerticalConstraint(points["t7"].id, points["t8"].id),
            TangentConstraint(entities[0].id, entities[4].id),
            TangentConstraint(entities[3].id, entities[4].id),
            TangentConstraint(entities[0].id, entities[5].id),
            TangentConstraint(entities[1].id, entities[5].id),
            TangentConstraint(entities[1].id, entities[6].id),
            TangentConstraint(entities[2].id, entities[6].id),
            TangentConstraint(entities[2].id, entities[7].id),
            TangentConstraint(entities[3].id, entities[7].id),
            EqualLengthConstraint([e.id for e in entities[4:]]),
            EqualDistanceConstraint(
                points["c1"].id,
                points["t8"].id,
                points["c1"].id,
                points["t1"].id,
            ),
            EqualDistanceConstraint(
                points["c2"].id,
                points["t2"].id,
                points["c2"].id,
                points["t3"].id,
            ),
            EqualDistanceConstraint(
                points["c3"].id,
                points["t4"].id,
                points["c3"].id,
                points["t5"].id,
            ),
            EqualDistanceConstraint(
                points["c4"].id,
                points["t6"].id,
                points["c4"].id,
                points["t7"].id,
            ),
        ]
        return {
            "points": list(points.values()),
            "entities": entities,
            "constraints": constraints,
        }

    @staticmethod
    def create_preview(
        registry: EntityRegistry,
        start_pid: int,
        end_pid: int,
        radius: float,
        preview_ids: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, int]]:
        """
        Creates or updates preview geometry in the registry.

        Args:
            registry: The entity registry to modify.
            start_pid: The ID of the start corner point.
            end_pid: The ID of the end corner point (preview corner).
            radius: The corner radius.
            preview_ids: Existing preview IDs to update, or None to create new.

        Returns:
            Dict of preview IDs, or None if geometry is invalid.
        """
        try:
            start_p = registry.get_point(start_pid)
            end_p = registry.get_point(end_pid)
        except IndexError:
            return None

        x1, y1 = start_p.x, start_p.y
        x2, y2 = end_p.x, end_p.y
        width, height = abs(x2 - x1), abs(y2 - y1)

        if width > 1e-6 and height > 1e-6:
            radius = min(radius, width / 2.0, height / 2.0)
        else:
            radius = 0.0

        sx, sy = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1)
        is_cw = sx * sy < 0

        coords = {
            "t1": (x1 + sx * radius, y1),
            "t2": (x2 - sx * radius, y1),
            "t3": (x2, y1 + sy * radius),
            "t4": (x2, y2 - sy * radius),
            "t5": (x2 - sx * radius, y2),
            "t6": (x1 + sx * radius, y2),
            "t7": (x1, y2 - sy * radius),
            "t8": (x1, y1 + sy * radius),
            "c1": (x1 + sx * radius, y1 + sy * radius),
            "c2": (x2 - sx * radius, y1 + sy * radius),
            "c3": (x2 - sx * radius, y2 - sy * radius),
            "c4": (x1 + sx * radius, y2 - sy * radius),
        }

        if preview_ids is None:
            # Create new preview geometry
            preview_ids = {}

            # Create all points
            for name, (px, py) in coords.items():
                preview_ids[name] = registry.add_point(px, py)

            # Lines
            preview_ids["line1"] = registry.add_line(
                preview_ids["t1"], preview_ids["t2"]
            )
            preview_ids["line2"] = registry.add_line(
                preview_ids["t3"], preview_ids["t4"]
            )
            preview_ids["line3"] = registry.add_line(
                preview_ids["t5"], preview_ids["t6"]
            )
            preview_ids["line4"] = registry.add_line(
                preview_ids["t7"], preview_ids["t8"]
            )

            # Arcs
            preview_ids["arc1"] = registry.add_arc(
                preview_ids["t8"],
                preview_ids["t1"],
                preview_ids["c1"],
                cw=is_cw,
            )
            preview_ids["arc2"] = registry.add_arc(
                preview_ids["t2"],
                preview_ids["t3"],
                preview_ids["c2"],
                cw=is_cw,
            )
            preview_ids["arc3"] = registry.add_arc(
                preview_ids["t4"],
                preview_ids["t5"],
                preview_ids["c3"],
                cw=is_cw,
            )
            preview_ids["arc4"] = registry.add_arc(
                preview_ids["t6"],
                preview_ids["t7"],
                preview_ids["c4"],
                cw=is_cw,
            )
        else:
            # Update existing preview geometry
            for name, (px, py) in coords.items():
                p = registry.get_point(preview_ids[name])
                p.x, p.y = px, py

            # Update arc directions
            for key in ["arc1", "arc2", "arc3", "arc4"]:
                arc_entity = registry.get_entity(preview_ids[key])
                if isinstance(arc_entity, Arc):
                    arc_entity.clockwise = is_cw

        return preview_ids

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
        radius: float = 10.0,
        **kwargs,
    ) -> RoundedRectPreviewState:
        """
        Creates initial preview state with start and end points.

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.
            radius: The corner radius.

        Returns:
            RoundedRectPreviewState for use with update_preview and
            cleanup_preview.
        """
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        p_end_id = registry.add_point(x, y)

        preview_ids = RoundedRectCommand.create_preview(
            registry, start_id, p_end_id, radius
        )
        assert preview_ids is not None

        return RoundedRectPreviewState(
            start_id=start_id,
            start_temp=start_temp,
            p_end_id=p_end_id,
            preview_ids=preview_ids,
            radius=radius,
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
            AttributeError: If preview_state is not a RoundedRectPreviewState.
        """
        if not isinstance(preview_state, RoundedRectPreviewState):
            raise AttributeError("Expected RoundedRectPreviewState")
        try:
            p_end = registry.get_point(preview_state.p_end_id)
        except IndexError:
            return
        p_end.x = x
        p_end.y = y

        RoundedRectCommand.create_preview(
            registry,
            preview_state.start_id,
            preview_state.p_end_id,
            preview_state.radius,
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
            AttributeError: If preview_state is not a RoundedRectPreviewState.
        """
        if not isinstance(preview_state, RoundedRectPreviewState):
            raise AttributeError("Expected RoundedRectPreviewState")
        preview_ids = preview_state.preview_ids
        p_end_id = preview_state.p_end_id

        point_ids = set(preview_ids.values())
        point_ids.add(p_end_id)

        entity_ids_to_remove = {
            e.id
            for e in registry.entities
            if any(pid in point_ids for pid in e.get_point_ids())
        }
        registry.remove_entities_by_id(list(entity_ids_to_remove))

        registry.points = [p for p in registry.points if p.id not in point_ids]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        reg = self.sketch.registry
        try:
            start_p = reg.get_point(self.start_pid)
        except IndexError:
            return

        result = self.calculate_geometry(
            start_p.x, start_p.y, self.end_pos[0], self.end_pos[1], self.radius
        )
        if not result:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_pid)
            return

        points_to_add = result["points"]
        if self.is_start_temp:
            reg.points.remove(start_p)
            # Unlike Rectangle, RoundedRect doesn't use the start_pid in its
            # final geometry, so we just remove it.

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
