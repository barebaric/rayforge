from __future__ import annotations
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, Set, Tuple

from rayforge.core.geo import Point as GeoPoint
from ..constraints import PerpendicularConstraint
from ..entities import Ellipse, Line, Point
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
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

    def get_preview_point_ids(self) -> Set[EntityID]:
        return {self.center_id, self.radius_x_id, self.radius_y_id}

    def get_hidden_point_ids(self) -> Set[EntityID]:
        return {self.start_id}


class EllipseCommand(SketchChangeCommand):
    """A command to create an ellipse."""

    def __init__(
        self,
        sketch: Sketch,
        start_id: EntityID,
        end_pos: GeoPoint,
        end_pid: Optional[EntityID] = None,
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
        self.add_cmd: Optional[AddItemsCommand] = None

    @staticmethod
    def _calculate_ellipse_params(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        center_on_start: bool,
        constrain_circle: bool,
    ) -> Tuple[float, float, float, float]:
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
        snapped_pid: Optional[EntityID] = None,
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
            raise AttributeError("Expected EllipsePreviewState")

        try:
            start_p = registry.get_point(preview_state.start_id)
            center_p = registry.get_point(preview_state.center_id)
            radius_x_p = registry.get_point(preview_state.radius_x_id)
            radius_y_p = registry.get_point(preview_state.radius_y_id)
        except IndexError:
            return

        cx, cy, rx, ry = EllipseCommand._calculate_ellipse_params(
            start_p.x, start_p.y, x, y, center_on_start, constrain_circle
        )

        center_p.x = cx
        center_p.y = cy
        radius_x_p.x = cx + rx
        radius_x_p.y = cy
        radius_y_p.x = cx
        radius_y_p.y = cy + ry

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        if not isinstance(preview_state, EllipsePreviewState):
            raise AttributeError("Expected EllipsePreviewState")

        if preview_state.entity_id is not None:
            registry.entities = [
                e for e in registry.entities if e.id != preview_state.entity_id
            ]
            registry._entity_map = {e.id: e for e in registry.entities}

        point_ids = {
            preview_state.center_id,
            preview_state.radius_x_id,
            preview_state.radius_y_id,
        }
        registry.points = [p for p in registry.points if p.id not in point_ids]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        registry = self.sketch.registry

        try:
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

        cx, cy, rx, ry = self._calculate_ellipse_params(
            start_p.x,
            start_p.y,
            final_x,
            final_y,
            self.center_on_start,
            self.constrain_circle,
        )

        if rx < 1e-6 or ry < 1e-6:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_id)
            return

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        center_id = next_temp_id()
        radius_x_id = next_temp_id()
        radius_y_id = next_temp_id()

        new_center = Point(center_id, cx, cy)
        new_radius_x = Point(radius_x_id, cx + rx, cy)
        new_radius_y = Point(radius_y_id, cx, cy + ry)

        line_x_id = next_temp_id()
        line_x = Line(line_x_id, center_id, radius_x_id)
        line_x.invisible = True

        line_y_id = next_temp_id()
        line_y = Line(line_y_id, center_id, radius_y_id)
        line_y.invisible = True

        visible_line_x_id = next_temp_id()
        visible_line_x = Line(
            visible_line_x_id, center_id, radius_x_id, construction=True
        )

        visible_line_y_id = next_temp_id()
        visible_line_y = Line(
            visible_line_y_id, center_id, radius_y_id, construction=True
        )

        ellipse_id = next_temp_id()
        new_ellipse = Ellipse(
            ellipse_id,
            center_id,
            radius_x_id,
            radius_y_id,
            helper_line_ids=[line_x_id, line_y_id],
        )

        perp_constraint = PerpendicularConstraint(
            line_x_id, line_y_id, user_visible=False
        )

        points_to_add = [new_center, new_radius_x, new_radius_y]

        if self.is_start_temp:
            registry.points.remove(start_p)

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=[
                new_ellipse,
                line_x,
                line_y,
                visible_line_x,
                visible_line_y,
            ],
            constraints=[perp_constraint],
        )
        self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
