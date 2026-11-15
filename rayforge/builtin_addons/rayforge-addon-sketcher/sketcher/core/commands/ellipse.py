from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.entities.line import Line
from raygeo.sketch.types import EntityID

from .base import PreviewState, SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch


class EllipsePreviewState(PreviewState):
    """Preview state for ellipse tool's drag-to-create workflow."""

    def __init__(
        self,
        start_id: EntityID,
        start_temp: bool,
        center_id: EntityID,
        radius_x_id: EntityID,
        radius_y_id: EntityID,
        entity_id: EntityID,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.center_id = center_id
        self.radius_x_id = radius_x_id
        self.radius_y_id = radius_y_id
        self.entity_id = entity_id

    def get_preview_point_ids(self) -> set[EntityID]:
        return {self.center_id, self.radius_x_id, self.radius_y_id}

    def get_hidden_point_ids(self) -> set[EntityID]:
        return {self.start_id}


class EllipseCommand(SketchChangeCommand):
    """A command to create an ellipse."""

    def __init__(
        self,
        sketch: Sketch,
        start_id: EntityID,
        end_pos: GeoPoint,
        end_pid: EntityID | None = None,
        is_start_temp: bool = False,
        center_on_start: bool = False,
        constrain_circle: bool = False,
    ):
        super().__init__(sketch, _("Add Ellipse"))
        self.start_id = start_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.center_on_start = center_on_start
        self.constrain_circle = constrain_circle
        self.add_cmd: AddItemsCommand | None = None

    @staticmethod
    def _calculate_ellipse_params(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        center_on_start: bool,
        constrain_circle: bool,
    ) -> tuple[float, float, float, float]:
        if center_on_start:
            cx, cy = x1, y1
            rx = abs(x2 - x1)
            ry = abs(y2 - y1)
            if constrain_circle:
                r = min(rx, ry)
                rx = r
                ry = r
        else:
            if constrain_circle:
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                size = min(width, height)
                rx = size / 2
                ry = rx
                dx = 1 if x2 >= x1 else -1
                dy = 1 if y2 >= y1 else -1
                cx = x1 + dx * rx
                cy = y1 + dy * ry
            else:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rx = abs(x2 - x1) / 2
                ry = abs(y2 - y1) / 2

        return cx, cy, rx, ry

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
        **kwargs,
    ) -> EllipsePreviewState:
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        center_id = registry.add_point(x, y)
        radius_x_id = registry.add_point(x, y)
        radius_y_id = registry.add_point(x, y)
        entity_id = registry.add_ellipse(center_id, radius_x_id, radius_y_id)

        return EllipsePreviewState(
            start_id=start_id,
            start_temp=start_temp,
            center_id=center_id,
            radius_x_id=radius_x_id,
            radius_y_id=radius_y_id,
            entity_id=entity_id,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
        center_on_start: bool = False,
        constrain_circle: bool = False,
    ) -> None:
        if not isinstance(preview_state, EllipsePreviewState):
            raise TypeError("Expected EllipsePreviewState")

        start_p = registry.get_point(preview_state.start_id)
        if start_p is None:
            return

        cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
            start_p[0], start_p[1], x, y, center_on_start, constrain_circle
        )

        registry.set_point_position(preview_state.center_id, cx, cy)
        registry.set_point_position(preview_state.radius_x_id, cx + rx, cy)
        registry.set_point_position(preview_state.radius_y_id, cx, cy + ry)

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        if not isinstance(preview_state, EllipsePreviewState):
            raise TypeError("Expected EllipsePreviewState")

        if preview_state.entity_id is not None:
            registry.remove_entities([preview_state.entity_id])

        point_ids = {
            preview_state.center_id,
            preview_state.radius_x_id,
            preview_state.radius_y_id,
        }
        registry.remove_points(list(point_ids))

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        registry = self.sketch.registry

        start_p = registry.get_point(self.start_id)
        if start_p is None:
            return

        final_x, final_y = self.end_pos
        if self.end_pid is not None:
            position = registry.get_point(self.end_pid)
            if position is not None:
                final_x, final_y = position

        cx, cy, rx, ry = self._calculate_ellipse_params(
            start_p[0],
            start_p[1],
            final_x,
            final_y,
            self.center_on_start,
            self.constrain_circle,
        )

        if rx < 1e-6 or ry < 1e-6:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_id)
            return

        add_cmd = AddItemsCommand(self.sketch, "")

        center_id = add_cmd.add_point(cx, cy)
        radius_x_id = add_cmd.add_point(cx + rx, cy)
        radius_y_id = add_cmd.add_point(cx, cy + ry)

        line_x_id = add_cmd.add_entity(Line(center_id, radius_x_id))
        line_y_id = add_cmd.add_entity(Line(center_id, radius_y_id))
        self.sketch.flags.set_invisible(line_x_id, True)
        self.sketch.flags.set_invisible(line_y_id, True)

        add_cmd.add_entity(Line(center_id, radius_x_id, construction=True))
        add_cmd.add_entity(Line(center_id, radius_y_id, construction=True))

        ellipse_dict = {
            "type": "ellipse",
            "center_idx": center_id,
            "radius_x_pt_idx": radius_x_id,
            "radius_y_pt_idx": radius_y_id,
            "helper_line_ids": [line_x_id, line_y_id],
        }
        add_cmd.add_entity_from_dict(ellipse_dict)

        add_cmd.add_constraint(
            {
                "type": "PerpendicularConstraint",
                "e1_id": line_x_id,
                "e2_id": line_y_id,
                "user_visible": False,
            }
        )
        if self.constrain_circle:
            add_cmd.add_constraint(
                EqualDistanceConstraint(
                    center_id,
                    radius_x_id,
                    center_id,
                    radius_y_id,
                )
            )

        if self.is_start_temp and registry.get_point(self.start_id):
            self.sketch.remove_point_if_unused(self.start_id)

        self.add_cmd = add_cmd

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
