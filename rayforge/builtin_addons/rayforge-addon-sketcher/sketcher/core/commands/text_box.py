from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.constraints.aspect_ratio import AspectRatioConstraint
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.parallelogram import ParallelogramConstraint

from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..sketch import Sketch


class TextBoxCommand(SketchChangeCommand):
    """A command to create a text box with its default constraints."""

    def __init__(
        self,
        sketch: Sketch,
        origin: GeoPoint,
        width: float = 10.0,
        height: float = 10.0,
    ):
        super().__init__(sketch, _("Add Text Box"))
        self.origin = origin
        self.width = width
        self.height = height
        self.add_cmd: AddItemsCommand | None = None
        self.text_box_id: int | None = None

    @staticmethod
    @staticmethod
    def calculate_geometry(
        origin: GeoPoint,
        width: float,
        height: float,
    ) -> dict[str, Any] | None:
        """Pure geometry of the text box frame: points, construction
        lines, the text box entity and its constraints, with temporary
        sequential point ids."""
        from raygeo.sketch.constraints.perpendicular import (
            PerpendicularConstraint,
            PerpendicularTarget,
        )
        from raygeo.sketch.entities.line import Line
        from raygeo.sketch.entities.point import SketchPoint
        from raygeo.sketch.entities.text_box import TextBox

        mx, my = origin
        p_origin, p_width, p_height, p4 = 0, 1, 2, 3
        points = [
            SketchPoint(p_origin, mx, my),
            SketchPoint(p_width, mx + width, my),
            SketchPoint(p_height, mx, my + height),
            SketchPoint(p4, mx + width, my + height),
        ]
        bottom_line = Line(p_origin, p_width, construction=True)
        right_line = Line(p_width, p4, construction=True)
        top_line = Line(p4, p_height, construction=True)
        left_line = Line(p_height, p_origin, construction=True)
        text_box = TextBox(p_origin, p_width, p_height)
        entities = [bottom_line, right_line, top_line, left_line, text_box]
        constraints: list[Any] = [
            AspectRatioConstraint(
                p_origin, p_width, p_origin, p_height, 1.0
            ),
            ParallelogramConstraint(
                p_origin, p_width, p_height, p4, user_visible=False
            ),
            HorizontalConstraint(p_origin, p_width),
            PerpendicularConstraint(
                PerpendicularTarget.LineLine(
                    p_origin, p_width, p_height, p_origin
                )
            ),
        ]
        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
            "text_box_id": -9,
        }

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        result = self.calculate_geometry(
            self.origin, self.width, self.height
        )
        if result is None:
            return

        from .. import constraint_ops
        from ..entity_ops import clone_entity

        add_cmd = AddItemsCommand(self.sketch, "")
        registry = self.sketch.registry
        pid_map = {
            point.id: add_cmd.add_point(point.x, point.y)
            for point in result["points"]
        }
        line_ids = []
        for entity in result["entities"][:4]:
            line_ids.append(
                add_cmd.add_entity(clone_entity(registry, entity, pid_map))
            )
        text_box_dict = {
            "type": "text_box",
            "origin_id": pid_map[0],
            "width_id": pid_map[1],
            "height_id": pid_map[2],
            "content": "",
            "construction_line_ids": line_ids,
        }
        self.text_box_id = add_cmd.add_entity_from_dict(text_box_dict)
        for constraint in result["constraints"]:
            add_cmd.add_constraint(
                constraint_ops.rebuild(constraint, pid_map)
            )
        self.add_cmd = add_cmd

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
