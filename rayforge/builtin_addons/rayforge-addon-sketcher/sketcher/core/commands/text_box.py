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
    def calculate_geometry(
        sketch,
        add_cmd: AddItemsCommand,
        origin: GeoPoint,
        width: float,
        height: float,
    ) -> int:
        """Builds the text box and its construction lines through
        ``add_cmd`` and returns the text box entity id."""
        from raygeo.sketch.entities.line import Line

        mx, my = origin

        p_origin = add_cmd.add_point(mx, my)
        p_width = add_cmd.add_point(mx + width, my)
        p_height = add_cmd.add_point(mx, my + height)
        p4 = add_cmd.add_point(mx + width, my + height)

        # Construction lines
        bottom_line = add_cmd.add_entity(
            Line(p_origin, p_width, construction=True)
        )
        right_line = add_cmd.add_entity(
            Line(p_width, p4, construction=True)
        )
        top_line = add_cmd.add_entity(Line(p4, p_height, construction=True))
        left_line = add_cmd.add_entity(
            Line(p_height, p_origin, construction=True)
        )

        text_box_dict = {
            "type": "text_box",
            "origin_id": p_origin,
            "width_id": p_width,
            "height_id": p_height,
            "content": "",
            "construction_line_ids": [
                bottom_line,
                right_line,
                top_line,
                left_line,
            ],
        }
        text_box_id = add_cmd.add_entity_from_dict(text_box_dict)

        constraints: list[Any] = [
            # Aspect ratio constraint for live text resizing
            AspectRatioConstraint(
                p_origin,
                p_width,
                p_origin,
                p_height,
                1.0,
            ),
            # Structural integrity constraint (hidden)
            ParallelogramConstraint(
                p_origin,
                p_width,
                p_height,
                p4,
            ),
            # Default constraints for user interaction (visible)
            HorizontalConstraint(p_origin, p_width),
            {
                "type": "PerpendicularConstraint",
                "e1_id": bottom_line,
                "e2_id": left_line,
            },
        ]
        for constraint in constraints:
            add_cmd.add_constraint(constraint)
        return text_box_id

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        add_cmd = AddItemsCommand(self.sketch, "")
        self.text_box_id = self.calculate_geometry(
            self.sketch,
            add_cmd,
            self.origin,
            self.width,
            self.height,
        )
        self.add_cmd = add_cmd

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
