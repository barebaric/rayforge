from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional
from ....core.geo.geometry import Geometry
from ..constraints import AspectRatioConstraint
from ..entities.text_box import TextBoxEntity
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..sketch import Sketch


class ModifyTextPropertyCommand(SketchChangeCommand):
    def __init__(
        self,
        sketch: Sketch,
        text_entity_id: int,
        new_content: str,
        new_font_params: Dict[str, Any],
    ):
        super().__init__(sketch, _("Modify Text Property"))
        self.text_entity_id = text_entity_id
        self.new_content = new_content
        self.new_font_params = new_font_params
        self.old_content = ""
        self.old_font_params: Dict[str, Any] = {}
        self.old_point_positions: Dict[int, Tuple[float, float]] = {}
        self.old_aspect_ratio: Optional[float] = None
        self.aspect_ratio_constraint_idx: Optional[int] = None

    def _do_execute(self) -> None:
        entity = self.sketch.registry.get_entity(self.text_entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        text_entity = entity

        if not self.old_content and not self.old_point_positions:
            self.old_content = text_entity.content
            self.old_font_params = text_entity.font_params.copy()
            p_width = self.sketch.registry.get_point(text_entity.width_id)
            p_height = self.sketch.registry.get_point(text_entity.height_id)
            self.old_point_positions = {
                text_entity.width_id: (p_width.x, p_width.y),
                text_entity.height_id: (p_height.x, p_height.y),
            }

            # Find and store the old aspect ratio constraint
            for idx, constr in enumerate(self.sketch.constraints or []):
                if isinstance(constr, AspectRatioConstraint):
                    if (
                        constr.p1 == text_entity.origin_id
                        and constr.p2 == text_entity.width_id
                        and constr.p3 == text_entity.origin_id
                        and constr.p4 == text_entity.height_id
                    ):
                        self.aspect_ratio_constraint_idx = idx
                        self.old_aspect_ratio = constr.ratio
                        break

        text_entity.content = self.new_content
        text_entity.font_params = self.new_font_params.copy()

        # Resize box to fit new content
        ascent, descent, font_height = text_entity.get_font_metrics()

        if not text_entity.content:
            # Provide a minimal size for empty text to avoid collapse
            natural_width = 10.0
        else:
            natural_geo = Geometry.from_text(
                text_entity.content, **text_entity.font_params
            )
            natural_geo.flip_y()
            min_x, _, max_x, _ = natural_geo.rect()
            natural_width = max(max_x - min_x, 1.0)

        p_origin = self.sketch.registry.get_point(text_entity.origin_id)
        p_width = self.sketch.registry.get_point(text_entity.width_id)
        p_height = self.sketch.registry.get_point(text_entity.height_id)

        # Update point positions as a starting guess for the solver
        # Origin is at bottom-left corner of box
        # Text baseline is offset by descent within the box
        # Box extends from origin.y to origin.y + font_height
        p_width.x = p_origin.x + natural_width
        p_width.y = p_origin.y
        p_height.x = p_origin.x
        p_height.y = p_origin.y + font_height

        # Update aspect ratio constraint with new text dimensions
        if self.aspect_ratio_constraint_idx is not None:
            new_ratio = natural_width / font_height
            constr = self.sketch.constraints[self.aspect_ratio_constraint_idx]
            assert isinstance(constr, AspectRatioConstraint)
            constr.ratio = new_ratio

        # The base class execute() will call notify_update(), which
        # triggers the final solve.

    def _do_undo(self) -> None:
        entity = self.sketch.registry.get_entity(self.text_entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        text_entity = entity

        text_entity.content = self.old_content
        text_entity.font_params = self.old_font_params.copy()

        for pid, (x, y) in self.old_point_positions.items():
            p = self.sketch.registry.get_point(pid)
            p.x = x
            p.y = y

        # Restore old aspect ratio
        if (
            self.aspect_ratio_constraint_idx is not None
            and self.old_aspect_ratio is not None
        ):
            constr = self.sketch.constraints[self.aspect_ratio_constraint_idx]
            assert isinstance(constr, AspectRatioConstraint)
            constr.ratio = self.old_aspect_ratio
