from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional, Dict, Any

from .base import SketchChangeCommand
from .items import AddItemsCommand
from ..entities import Point, Line, TextBoxEntity
from ..constraints import (
    AspectRatioConstraint,
    HorizontalConstraint,
    PerpendicularConstraint,
    ParallelogramConstraint,
)

if TYPE_CHECKING:
    from ..sketch import Sketch


class TextBoxCommand(SketchChangeCommand):
    """A command to create a text box with its default constraints."""

    def __init__(
        self,
        sketch: Sketch,
        origin: Tuple[float, float],
        width: float = 10.0,
        height: float = 10.0,
    ):
        super().__init__(sketch, _("Add Text Box"))
        self.origin = origin
        self.width = width
        self.height = height
        self.add_cmd: Optional[AddItemsCommand] = None
        self.text_box_id: Optional[int] = None

    @staticmethod
    def calculate_geometry(
        origin: Tuple[float, float], width: float, height: float
    ) -> Dict[str, Any]:
        """Calculates all points, entities, and constraints for a text box."""
        mx, my = origin

        # Use temporary negative IDs
        p_origin = Point(-1, mx, my)
        p_width = Point(-2, mx + width, my)
        p_height = Point(-3, mx, my + height)
        p4 = Point(-4, mx + width, my + height)

        points_to_add = [p_origin, p_width, p_height, p4]

        # Construction lines
        bottom_line = Line(-5, p_origin.id, p_width.id, construction=True)
        right_line = Line(-6, p_width.id, p4.id, construction=True)
        top_line = Line(-7, p4.id, p_height.id, construction=True)
        left_line = Line(-8, p_height.id, p_origin.id, construction=True)

        lines_to_add = [bottom_line, right_line, top_line, left_line]
        line_ids = [line.id for line in lines_to_add]

        text_box = TextBoxEntity(
            -9,
            p_origin.id,
            p_width.id,
            p_height.id,
            content="",
            construction_line_ids=line_ids,
        )

        entities_to_add = [*lines_to_add, text_box]

        constraints_to_add = [
            # Internal constraint for live text resizing (hidden)
            AspectRatioConstraint(
                p_origin.id,
                p_width.id,
                p_origin.id,
                p_height.id,
                1.0,
                user_visible=False,
            ),
            # Structural integrity constraint (hidden)
            ParallelogramConstraint(
                p_origin.id,
                p_width.id,
                p_height.id,
                p4.id,
                user_visible=False,
            ),
            # Default constraints for user interaction (visible)
            HorizontalConstraint(p_origin.id, p_width.id),
            PerpendicularConstraint(bottom_line.id, left_line.id),
        ]

        return {
            "points": points_to_add,
            "entities": entities_to_add,
            "constraints": constraints_to_add,
            "text_box_id": text_box.id,
        }

    def _do_execute(self) -> None:
        if not self.add_cmd:
            geom = self.calculate_geometry(
                self.origin, self.width, self.height
            )
            self.add_cmd = AddItemsCommand(
                self.sketch,
                "",
                points=geom["points"],
                entities=geom["entities"],
                constraints=geom["constraints"],
            )

        self.add_cmd._do_execute()

        # After execution, the temporary ID is resolved. Find the new ID.
        for entity in reversed(self.sketch.registry.entities):
            if isinstance(entity, TextBoxEntity):
                self.text_box_id = entity.id
                break

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
